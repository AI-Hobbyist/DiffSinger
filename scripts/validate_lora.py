"""Validate LoRA updates, strict merged loading, and Lightning checkpoint resume."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_training_modes import configure, create_data, trainer_for, close_datasets
from scripts.validate_all_in_one import acoustic_sample, variance_sample
from training.acoustic_task import AcousticTask
from training.variance_task import VarianceTask
from training.all_in_one_task import AllInOneTask
from modules.toplevel import DiffSingerAcoustic, DiffSingerVariance
from utils import load_ckpt
from utils.hparams import hparams
from utils.lora import lora_metadata

def validate_training_and_loading():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for kind in ('wavenet', 'lynxnet', 'lynxnet2', 'dit'):
            for objective in ('ddpm', 'reflow'):
                for (cls, category) in ((AcousticTask, 'acoustic'), (VarianceTask, 'variance'), (AllInOneTask, 'all_in_one')):
                    configure(kind, objective, joint=category == 'all_in_one', auxiliary=False)
                    hparams['shallow_diffusion_args']['aux_decoder_args'].update(num_channels=16, num_layers=1)
                    hparams.update(work_dir=str(root / 'work'), lora={'enabled': False}, use_fused_kernels=False)
                    with contextlib.redirect_stdout(io.StringIO()):
                        base = cls()
                    path = root / 'base.ckpt'
                    torch.save({'category': category, 'state_dict': base.state_dict()}, path)
                    baseline = {k: v.clone() for (k, v) in base.model.state_dict().items()}
                    hparams['lora'] = {'enabled': True, 'base_ckpt': str(path), 'rank': 2, 'alpha': 4, 'target_modules': ['linear'], 'train_bias': False}
                    with contextlib.redirect_stdout(io.StringIO()):
                        task = cls()
                    optimizer = task.build_optimizer(task.model)
                    sample = {'acoustic': acoustic_sample(), 'variance': variance_sample()} if category == 'all_in_one' else acoustic_sample() if category == 'acoustic' else variance_sample()
                    for _ in range(2):
                        optimizer.zero_grad()
                        loss = task._training_step(sample)[0]
                        assert torch.isfinite(loss)
                        loss.backward()
                        optimizer.step()
                    for (k, v) in baseline.items():
                        torch.testing.assert_close(task.model.state_dict()[k], v, rtol=0, atol=0)
                    assert any((p.abs().sum() > 0 for (n, p) in task.model.named_parameters() if n.endswith('.lora_B')))
                    checkpoint = root / 'lora.ckpt'
                    torch.save({'category': category, 'state_dict': task.state_dict(), 'lora': lora_metadata(task.model)}, checkpoint)
                    with contextlib.redirect_stdout(io.StringIO()):
                        load_ckpt(base.model.eval(), checkpoint)
                    if category == 'all_in_one':
                        for (comp, factory) in (('acoustic', lambda : DiffSingerAcoustic(len(task.phoneme_dictionary), 4)), ('variance', lambda : DiffSingerVariance(len(task.phoneme_dictionary)))):
                            branch = factory().eval()
                            with contextlib.redirect_stdout(io.StringIO()):
                                load_ckpt(branch, checkpoint)
                            for (k, v) in branch.state_dict().items():
                                torch.testing.assert_close(v, getattr(base.model, comp).state_dict()[k], rtol=0, atol=0)
                    print('PASS', kind, objective, category, 'LoRA updates; frozen base exact; merged checkpoint strict load')

def validate_resume():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        configure('dit', auxiliary=False)
        hparams['shallow_diffusion_args']['aux_decoder_args'].update(num_layers=1, num_channels=16)
        hparams.update(lora={'enabled': False}, work_dir=str(root / 'work'), binary_data_dir=str(root / 'binary'))
        create_data(root / 'binary', auxiliary=False)
        with contextlib.redirect_stdout(io.StringIO()):
            base = AllInOneTask()
        torch.save({'category': 'all_in_one', 'state_dict': base.state_dict()}, root / 'base.ckpt')
        hparams['lora'] = {'enabled': True, 'base_ckpt': str(root / 'base.ckpt'), 'rank': 2, 'alpha': 4, 'target_modules': ['linear'], 'train_bias': False}
        with contextlib.redirect_stdout(io.StringIO()):
            task = AllInOneTask()
        trainer = trainer_for(root, max_steps=1, enable_checkpointing=False)
        trainer.fit(task)
        checkpoint = root / 'work/model_ckpt_steps_1.ckpt'
        checkpoint.parent.mkdir()
        trainer.save_checkpoint(checkpoint)
        saved = torch.load(checkpoint, map_location='cpu')
        assert saved['lora'] == lora_metadata(task.model)
        assert saved['optimizer_states'] and saved['lr_schedulers']
        close_datasets(task)
        (root / 'base.ckpt').unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            resumed = AllInOneTask()
        resumed_trainer = trainer_for(root, max_steps=2, enable_checkpointing=False)
        resumed_trainer.fit(resumed, ckpt_path=str(checkpoint))
        assert resumed_trainer.global_step == 2
        close_datasets(resumed)
        hparams['lora']['rank'] = 3
        with contextlib.redirect_stdout(io.StringIO()):
            wrong = AllInOneTask()
        try:
            wrong.on_load_checkpoint(saved)
        except ValueError as error:
            assert 'same adapter' in str(error)
        else:
            raise AssertionError('Mismatched rank resumed silently')
        print('PASS: Lightning LoRA train/save/resume; optimizer and scheduler retained; original base removed; rank mismatch rejected')
if __name__ == '__main__':
    validate_training_and_loading()
    validate_resume()
