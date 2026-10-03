"""Integration regression for the DiT/all-in-one port and local extensions."""

import contextlib
import io
import json
import pickle
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
import lightning.pytorch as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_all_in_one import configure_smoke_model, acoustic_sample, variance_sample
from modules.backbones import build_backbone
from modules.backbones.dit import DiT
from modules.core.reflow import RectifiedFlow
from modules.core.ddpm import GaussianDiffusion
from modules.toplevel import DiffSingerAcoustic, DiffSingerVariance
from deployment.modules.dit import compile_backbone_for_onnx
from training.all_in_one_task import AllInOneTask
from utils import load_ckpt
from utils.hparams import hparams, set_hparams
from utils.indexed_datasets import IndexedDatasetBuilder
from utils.training_utils import DsTensorBoardLogger


def make_task():
    with contextlib.redirect_stdout(io.StringIO()):
        return AllInOneTask()


def validate_dual_time():
    for kind in ('wavenet', 'lynxnet', 'lynxnet2', 'dit'):
        configure_smoke_model()
        hparams['use_dual_timestep'] = True
        args = dict(num_layers=1, num_channels=16)
        if kind == 'dit':
            args.update(num_heads=2, time_embed_dim=16, use_gradient_checkpointing=True)
        flow = RectifiedFlow(4, backbone_type=kind, backbone_args=args,
                             spec_min=[-1], spec_max=[1], use_shallow_diffusion=False)
        prediction, target, time = flow(torch.randn(2, 5, 16), torch.randn(2, 5, 4),
                                       infer=False, valid_mask=torch.ones(2, 5, dtype=torch.bool))
        assert prediction.shape == target.shape == (2, 1, 4, 5)
        assert time.shape == (2, 5)
        prediction.square().mean().backward()
        assert torch.isfinite(prediction).all()
    # A mask selecting only t2 is equivalent to evaluating the same model at t2.
    model = flow.velocity_fn.eval()
    with torch.no_grad():
        model.output_proj.weight.normal_()
        for block in model.blocks:
            block.adaLN_modulation[1].bias.normal_()
        spec, cond = torch.randn(2, 1, 4, 5), torch.randn(2, 16, 5)
        t1, t2 = torch.tensor([1., 2.]), torch.tensor([3., 4.])
        torch.testing.assert_close(model(spec, t2, cond), model(
            spec, t1, cond, diffusion_step_2=t2, mask=torch.ones(2, 5)))


def validate_sampling_masks():
    configure_smoke_model()
    hparams.update(use_shallow_diffusion=False, K_step_infer=8, infer=False)
    flow = RectifiedFlow(4, backbone_type='dit', backbone_args=hparams['backbone_args'],
                         spec_min=[-1], spec_max=[1], use_shallow_diffusion=False)
    diffusion = GaussianDiffusion(4, timesteps=8, k_step=8, backbone_type='dit',
                                  backbone_args=hparams['backbone_args'], spec_min=[-1], spec_max=[1])
    condition = torch.randn(1, 6, 16)
    mask = torch.tensor([[True, True, True, False, False, False]])
    for core, backbone, algorithms in (
        (flow, flow.velocity_fn, ('euler', 'rk2', 'rk4', 'rk5')),
        (diffusion, diffusion.denoise_fn, ('ddpm', 'ddim', 'pndm', 'dpm-solver', 'unipc')),
    ):
        seen = []

        def capture(module, args, kwargs):
            torch.testing.assert_close(kwargs['valid_mask'], mask)
            seen.append(True)

        handle = backbone.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            for algorithm in algorithms:
                seen.clear()
                hparams.update(sampling_algorithm=algorithm, sampling_steps=4,
                               diff_speedup=1 if algorithm == 'ddpm' else 2)
                output = core(condition, infer=True, valid_mask=mask)
                assert output.shape == (1, 6, 4) and torch.isfinite(output).all()
                assert seen, algorithm
        finally:
            handle.remove()


def validate_local_features():
    for objective in ('ddpm', 'reflow'):
        for kind in ('wavenet', 'lynxnet', 'lynxnet2', 'dit'):
            configure_smoke_model()
            hparams.update(diffusion_type=objective, use_stretch_embed=True,
                           use_variance_scaling=True, use_dual_timestep=objective == 'reflow',
                           use_spk_id=True, num_spk=2, use_lang_id=True, num_lang=1)
            args = dict(num_layers=1, num_channels=16)
            if kind == 'dit':
                args.update(num_heads=2, time_embed_dim=16, use_gradient_checkpointing=True)
            hparams['backbone_type'], hparams['backbone_args'] = kind, args
            for key in ('pitch_prediction_args', 'variances_prediction_args'):
                hparams[key].update(backbone_type=kind, backbone_args=args, timesteps=4, K_step=4)
            hparams.update(timesteps=4, K_step=4, K_step_infer=4, diff_speedup=1)
            task = make_task()
            acoustic, variance = acoustic_sample(), variance_sample()
            for sample in (acoustic, variance):
                sample['spk_ids'] = torch.tensor([1])
                sample['languages'] = torch.zeros_like(sample['tokens'])
            loss, _ = task._training_step(dict(acoustic=acoustic, variance=variance))
            assert torch.isfinite(loss)
            loss.backward()
            assert task.model.variance.stretch_embed_rnn.weight_hh_l0.grad is not None
            task.model.eval()
            with torch.no_grad():
                from training.acoustic_task import AcousticTask
                from training.variance_task import VarianceTask
                mel = AcousticTask.run_model(task, acoustic, infer=True, model=task.model.acoustic)
                curves = VarianceTask.run_model(task, variance, infer=True, model=task.model.variance)
                assert torch.isfinite(mel.diff_out).all()
                assert torch.isfinite(curves[1]).all()
            print(f'Local features: {kind}/{objective} passed.')


def validate_checkpoint():
    configure_smoke_model()
    task = make_task()
    with tempfile.TemporaryDirectory() as directory:
        filename = Path(directory) / 'joint.ckpt'
        torch.save(dict(category='all_in_one', state_dict={
            f'model.{key}': value for key, value in task.model.state_dict().items()
        }), filename)
        for kind, model in (('acoustic', DiffSingerAcoustic(65, 4)),
                            ('variance', DiffSingerVariance(65))):
            load_ckpt(model, filename, strict=True)
            original = getattr(task.model, kind).state_dict()
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, original[key])
        load_ckpt(task.model, filename, strict=True)


def write_dataset(directory, prefix, batch, count):
    directory.mkdir(parents=True, exist_ok=True)
    item = {key: value[0].numpy() for key, value in batch.items()
            if isinstance(value, torch.Tensor) and value.ndim >= 2}
    item['uv'] = torch.zeros(4, dtype=torch.bool).numpy()
    for key in ('key_shift', 'speed'):
        if key in item:
            item[key] = float(item[key][0])
    builder = IndexedDatasetBuilder(str(directory), prefix)
    for _ in range(count):
        builder.add_item(item)
    builder.finalize()
    with (directory / f'{prefix}.meta').open('wb') as stream:
        pickle.dump({'lengths': [4] * count}, stream)


def close_datasets(task):
    for name in ('acoustic_train_dataset', 'acoustic_valid_dataset',
                 'variance_train_dataset', 'variance_valid_dataset'):
        dataset = getattr(task, name)
        if dataset.indexed_ds.dset is not None:
            dataset.indexed_ds.dset.close()
            dataset.indexed_ds.dset = None


def validate_trainer():
    configure_smoke_model()
    hparams.update(num_valid_plots=0, ds_workers=0, accumulate_grad_batches=1,
                   max_batch_frames=16, max_batch_size=2, max_val_batch_frames=8,
                   max_val_batch_size=1, val_with_vocoder=False)
    with tempfile.TemporaryDirectory() as directory:
        hparams['binary_data_dir'] = directory
        for prefix in ('train', 'valid'):
            write_dataset(Path(directory) / 'acoustic', prefix, acoustic_sample(), 3)
            write_dataset(Path(directory) / 'variance', prefix, variance_sample(), 2)
        task = make_task()
        trainer = pl.Trainer(accelerator='cpu', devices=1, max_steps=2,
                             logger=DsTensorBoardLogger(directory, name='logs', version='smoke'), enable_checkpointing=False,
                             enable_progress_bar=False, enable_model_summary=False,
                             num_sanity_val_steps=1, limit_val_batches=1)
        trainer.fit(task)
        close_datasets(task)
        assert trainer.global_step == 2
        assert set(task.all_in_one_samplers) == {'acoustic', 'variance'}
        filename = Path(directory) / 'resume.ckpt'
        trainer.save_checkpoint(filename)
        saved = torch.load(filename, weights_only=False)
        assert saved['category'] == 'all_in_one'
        assert any(key.startswith('model.acoustic.') for key in saved['state_dict'])
        assert any(key.startswith('model.variance.') for key in saved['state_dict'])
        # Lightning resumes weights and both-stream optimizer state.
        resumed = make_task()
        trainer = pl.Trainer(accelerator='cpu', devices=1, max_steps=3,
                             logger=DsTensorBoardLogger(directory, name='logs', version='smoke'), enable_checkpointing=False,
                             enable_progress_bar=False, enable_model_summary=False,
                             num_sanity_val_steps=0, limit_val_batches=0)
        trainer.fit(resumed, ckpt_path=str(filename))
        close_datasets(resumed)
        assert trainer.global_step == 3


def validate_standalone_training():
    from training.acoustic_task import AcousticTask
    from training.variance_task import VarianceTask
    for kind, task_class, sample in (('acoustic', AcousticTask, acoustic_sample),
                                     ('variance', VarianceTask, variance_sample)):
        configure_smoke_model()
        hparams.update(num_valid_plots=0, ds_workers=0, accumulate_grad_batches=1,
                       max_batch_frames=8, max_batch_size=1, max_val_batch_frames=8,
                       max_val_batch_size=1, val_with_vocoder=False)
        hparams['all_in_one']['enabled'] = False
        with tempfile.TemporaryDirectory() as directory:
            hparams['binary_data_dir'] = directory
            for prefix in ('train', 'valid'):
                write_dataset(Path(directory), prefix, sample(), 2)
            with contextlib.redirect_stdout(io.StringIO()):
                task = task_class()
            trainer = pl.Trainer(accelerator='cpu', devices=1, max_steps=1,
                                 logger=DsTensorBoardLogger(directory, name='logs', version='standalone'),
                                 enable_checkpointing=False, enable_progress_bar=False,
                                 enable_model_summary=False, num_sanity_val_steps=1, limit_val_batches=1)
            trainer.fit(task)
            for dataset in (task.train_dataset, task.valid_dataset):
                if dataset.indexed_ds.dset is not None:
                    dataset.indexed_ds.dset.close()
                    dataset.indexed_ds.dset = None
            assert trainer.global_step == 1
            print(f'Standalone {kind} training passed.')


def validate_onnx():
    import onnx
    import onnxruntime as ort
    import numpy as np
    configure_smoke_model()
    for features in (1, 3):
        model = DiT(4, features, num_layers=1, num_channels=16, num_heads=2,
                    time_embed_dim=16, use_gradient_checkpointing=False).eval()
        with torch.no_grad():
            model.output_proj.weight.normal_()
            model.blocks[0].adaLN_modulation[1].bias.normal_()
        inputs = (torch.randn(1, features, 4, 5), torch.tensor([1.]), torch.randn(1, 16, 5))
        traced = compile_backbone_for_onnx(model, inputs)
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / 'dit.onnx')
            torch.onnx.export(traced, inputs, filename, input_names=['spec', 'time', 'cond'],
                              output_names=['output'], opset_version=17, dynamo=False,
                              dynamic_axes={'spec': {3: 'frames'}, 'cond': {2: 'frames'},
                                            'output': {3: 'frames'}})
            onnx.checker.check_model(onnx.load(filename))
            session = ort.InferenceSession(filename, providers=['CPUExecutionProvider'])
            for frames in (3, 5, 11):
                values = (torch.randn(1, features, 4, frames), torch.tensor([2.]),
                          torch.randn(1, 16, frames))
                with torch.no_grad():
                    expected = model(*values).numpy()
                actual = session.run(None, dict(zip(['spec', 'time', 'cond'],
                                                    [value.numpy() for value in values])))[0]
                np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)


def validate_templates():
    for name in ('config_acoustic_dit', 'config_variance_dit', 'all_in_one_dit',
                 'all_in_one_lynxnet2_muon', 'all_in_one_wavenet_adamw'):
        set_hparams(f'configs/templates/{name}.yaml', print_hparams=False)
        for config in [hparams] + ([hparams[key] for key in
                                   ('pitch_prediction_args', 'variances_prediction_args')]
                                  if 'all_in_one' in name or 'variance' in name else []):
            with torch.device('meta'):
                build_backbone(128, 1, config.get('backbone_type', hparams.get('backbone_type', 'wavenet')), config.get('backbone_args', hparams.get('backbone_args', {})))


def validate_cuda_and_fused_routing():
    from modules.kernels import integration
    configure_smoke_model()
    hparams['use_fused_kernels'] = True
    calls = []
    with patch.object(integration, 'patch_diffusion_module',
                      side_effect=lambda module, **kwargs: calls.append(module) or 0):
        task = make_task()
    assert calls == [task.model.acoustic.diffusion, task.model.variance.pitch_predictor,
                     task.model.variance.variance_predictor]
    if not torch.cuda.is_available():
        print('CUDA validation skipped: no CUDA device.')
        return
    configure_smoke_model()
    hparams['use_dual_timestep'] = True
    for key in ('backbone_args',):
        hparams[key]['use_gradient_checkpointing'] = True
    for key in ('pitch_prediction_args', 'variances_prediction_args'):
        hparams[key]['backbone_args']['use_gradient_checkpointing'] = True
    for dtype in (torch.float16, torch.bfloat16):
        task = make_task().cuda()
        acoustic, variance = acoustic_sample(), variance_sample()
        for sample in (acoustic, variance):
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    sample[key] = value.cuda()
        with torch.autocast('cuda', dtype=dtype):
            loss, _ = task._training_step(dict(acoustic=acoustic, variance=variance))
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(parameter.grad).all() for parameter in task.parameters()
                   if parameter.grad is not None)
        print(f'CUDA joint training {dtype} passed.')


def validate_full_exports():
    import onnx
    import onnxruntime as ort
    from deployment.exporters.acoustic_exporter import DiffSingerAcousticExporter
    from deployment.exporters.variance_exporter import DiffSingerVarianceExporter
    for objective in ('ddpm', 'reflow'):
        configure_smoke_model()
        hparams['diffusion_type'] = objective
        hparams['use_stretch_embed'] = True
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            hparams.update(work_dir=str(directory), exp_name='port-smoke')
            (directory / 'lang_map.json').write_text(json.dumps({'zh': 0}))
            shutil.copyfile('dictionaries/opencpop-extension.txt', directory / 'dictionary-zh.txt')
            task = make_task()
            torch.save(dict(category='all_in_one', state_dict={
                f'model.{key}': value for key, value in task.model.state_dict().items()
            }), directory / 'model_ckpt_steps_1.ckpt')
            for kind, exporter_class in (('acoustic', DiffSingerAcousticExporter),
                                         ('variance', DiffSingerVarianceExporter)):
                exporter = exporter_class(device='cpu', cache_dir=directory / f'{kind}-cache')
                destination = directory / kind
                with contextlib.redirect_stdout(io.StringIO()):
                    exporter.export(destination)
                artifacts = list(destination.glob('*.onnx'))
                assert artifacts
                for artifact in artifacts:
                    onnx.checker.check_model(onnx.load(artifact))
                    ort.InferenceSession(str(artifact), providers=['CPUExecutionProvider'])
                print(f'Full {kind}/{objective} export from joint checkpoint passed.')


def main():
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    validate_templates()
    validate_dual_time()
    validate_sampling_masks()
    validate_local_features()
    validate_checkpoint()
    validate_trainer()
    validate_standalone_training()
    validate_onnx()
    validate_cuda_and_fused_routing()
    validate_full_exports()
    print('DiT/all-in-one port integration passed.')


if __name__ == '__main__':
    main()
