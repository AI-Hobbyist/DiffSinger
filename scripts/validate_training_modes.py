"""Exercise auxiliary streams, all backbones, audio preview, and ep/step resume."""

import contextlib
import io
import json
import pickle
import sys
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

import lightning.pytorch as pl
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_all_in_one import configure_smoke_model, acoustic_sample, variance_sample
from scripts.validate_port import write_dataset
from training.all_in_one_task import AllInOneTask
from training.variance_task import VarianceTask
from utils.hparams import hparams
from utils.training_schedule import training_schedule_options
from utils.training_utils import DsTensorBoardLogger, DsModelCheckpoint


def configure(kind, objective='reflow', joint=True, auxiliary=True, accumulation=1):
    configure_smoke_model()
    hparams.update(diffusion_type=objective, num_valid_plots=0, ds_workers=0,
                   accumulate_grad_batches=accumulation, max_batch_frames=24,
                   max_batch_size=3 if joint else 1, max_val_batch_frames=8,
                   max_val_batch_size=1, val_with_vocoder=False,
                   timesteps=4, K_step=4, K_step_infer=4, diff_speedup=1,
                   sampling_algorithm='euler' if objective == 'reflow' else 'ddpm')
    hparams['all_in_one']['enabled'] = joint
    hparams['aux_datasets'] = {'enable': auxiliary, 'module': ['pitch', 'voicing'],
                              'datasets': {'data': [{'zh': 'data/aux'}], 'val': {'zh': ['valid']}}}
    args = dict(num_layers=1, num_channels=16)
    if kind == 'dit':
        args.update(num_heads=2, time_embed_dim=16)
    hparams.update(backbone_type=kind, backbone_args=args)
    for key in ('pitch_prediction_args', 'variances_prediction_args'):
        hparams[key].update(backbone_type=kind, backbone_args=args, timesteps=4, K_step=4)


def create_data(root, joint=True, auxiliary=True):
    for prefix in ('train', 'valid'):
        if joint:
            write_dataset(root / 'acoustic', prefix, acoustic_sample(), 3)
        write_dataset(root / 'variance' if joint else root, prefix, variance_sample(), 2)
        if auxiliary:
            write_dataset(root / 'aux', prefix, variance_sample(), 3)


def close_datasets(task):
    for value in vars(task).values():
        if hasattr(value, 'indexed_ds') and hasattr(value.indexed_ds, 'dset'):
            if value.indexed_ds.dset is not None:
                value.indexed_ds.dset.close()
                value.indexed_ds.dset = None


class RecordValidation(pl.Callback):
    def __init__(self):
        self.events = []

    def on_validation_end(self, trainer, module):
        if not trainer.sanity_checking:
            self.events.append((trainer.current_epoch + 1, trainer.global_step))


def trainer_for(root, **kwargs):
    return pl.Trainer(accelerator='cpu', devices=1,
                      logger=DsTensorBoardLogger(str(root), name='logs', version='smoke'),
                      enable_progress_bar=False, enable_model_summary=False,
                      accumulate_grad_batches=hparams['accumulate_grad_batches'],
                      num_sanity_val_steps=0, limit_val_batches=1, **kwargs)


def make_task(joint=True):
    with contextlib.redirect_stdout(io.StringIO()):
        return AllInOneTask() if joint else VarianceTask()


def validate_aux_backbones():
    for objective in ('ddpm', 'reflow'):
        for kind in ('wavenet', 'lynxnet', 'lynxnet2', 'dit'):
            for joint in (False, True):
                configure(kind, objective, joint)
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    hparams['binary_data_dir'] = directory
                    create_data(root, joint)
                    task = make_task(joint)
                    trainer = trainer_for(root, max_steps=1, enable_checkpointing=False)
                    try:
                        trainer.fit(task)
                        # Exercise both/all three validation streams even though training ends mid-epoch.
                        close_datasets(task)
                        trainer.validate(task)
                        assert trainer.global_step == 1
                        assert torch.isfinite(task.valid_losses['pitch_loss'].compute())
                        assert torch.isfinite(task.valid_losses['var_loss'].compute())
                        if joint:
                            assert set(task.all_in_one_samplers) == {'acoustic', 'variance', 'aux'}
                            assert task.training_stream_count == 3
                        else:
                            assert task.aux_training_sampler is not None
                        # Main and auxiliary supervision select disjoint module groups.
                        model = task.model.variance if joint else task.model
                        main = VarianceTask.run_model(task, variance_sample(),
                                                      modules={'dur', 'energy', 'breathiness', 'tension'}, model=model)
                        aux = VarianceTask.run_model(task, variance_sample(), modules={'pitch', 'voicing'}, model=model)
                        assert set(main) == {'dur_loss', 'var_loss'}
                        assert set(aux) == {'pitch_loss', 'var_loss'}
                    finally:
                        close_datasets(task)
                print(f'Auxiliary {objective}/{kind}/joint={joint} passed.')


def validate_ep_step_resume():
    for max_updates, interval, expected_events in (
        ('2ep', '1ep', [(1, 2), (2, 4)]),
        ('4step', '1step', [(1, 1), (1, 2), (2, 3), (2, 4)]),
    ):
        configure('dit', accumulation=2)
        hparams.update(max_updates=max_updates, val_check_interval=interval)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hparams['binary_data_dir'] = directory
            create_data(root)
            recorder = RecordValidation()
            checkpoint = DsModelCheckpoint(dirpath=root, filename='model_ckpt_steps_{step}',
                                            auto_insert_metric_name=False, monitor='step', mode='max',
                                            save_top_k=2, save_on_train_epoch_end=False,
                                            permanent_ckpt_start=0, permanent_ckpt_interval=0)
            task = make_task()
            trainer = trainer_for(root, callbacks=[recorder, checkpoint], **training_schedule_options(hparams))
            try:
                trainer.fit(task)
                assert trainer.global_step == 4
                assert recorder.events == expected_events, recorder.events
                assert checkpoint.best_model_path and Path(checkpoint.best_model_path).is_file()
            finally:
                close_datasets(task)
            # Change units while resuming the same model and optimizer state.
            hparams.update(max_updates='3ep' if max_updates.endswith('step') else '6step')
            resumed = make_task()
            recorder = RecordValidation()
            trainer = trainer_for(root, callbacks=[recorder], enable_checkpointing=False,
                                  **training_schedule_options(hparams))
            try:
                trainer.fit(resumed, ckpt_path=checkpoint.best_model_path)
                assert trainer.global_step == 6
            finally:
                close_datasets(resumed)
        print(f'{max_updates}/{interval} validation and unit-switch resume passed.')


def validate_variance_preview():
    # Real duration -> pitch/curves -> acoustic inference; replace only waveform synthesis/logging.
    class Vocoder:
        def spec2wav_torch(self, mel, f0):
            assert mel.shape[1] == f0.shape[1]
            assert torch.isfinite(mel).all() and torch.isfinite(f0).all()
            return mel.mean(-1).repeat_interleave(hparams['hop_size'], dim=-1)

    for kind in ('wavenet', 'lynxnet', 'lynxnet2', 'dit', 'mixed'):
        configure('dit' if kind == 'mixed' else kind, auxiliary=False)
        if kind == 'mixed':
            hparams['pitch_prediction_args'].update(backbone_type='wavenet',
                                                    backbone_args=dict(num_layers=1, num_channels=16))
            hparams['variances_prediction_args'].update(backbone_type='lynxnet2',
                                                        backbone_args=dict(num_layers=1, num_channels=16))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'preview.ds'
            source.write_text(json.dumps([dict(text='SP SP', note_seq='C4 D4', note_dur='0.06 0.06')]), encoding='utf-8')
            hparams['val_with_variance'] = {'enable': True, 'zh': [str(source)]}
            with patch('training.acoustic_task.get_vocoder_cls', return_value=Vocoder):
                task = make_task()
            from modules.fastspeech.tts_modules import DurationPredictor
            with torch.no_grad():
                for module in task.model.modules():
                    if isinstance(module, DurationPredictor):
                        module.linear.weight.zero_()
                        module.linear.bias.fill_(1.)
            task.model.eval()
            task.variance_validation_speakers = [('smoke', 0)]
            trainer = trainer_for(root, max_steps=1, enable_checkpointing=False)
            task._trainer = trainer
            experiment = trainer.logger.all_rank_experiment
            with patch.object(experiment, 'add_audio') as audio, patch.object(experiment, 'add_figure') as figure:
                task.use_vocoder = False  # Stub exposes only synthesis; no device-management interface.
                task._on_validation_start()
                task._on_validation_epoch_end()
                audio.assert_called_once()
                figure.assert_called_once()
                # Randomly initialized duration heads can collapse every phone.
                with torch.no_grad():
                    for module in task.model.modules():
                        if isinstance(module, DurationPredictor):
                            module.linear.bias.fill_(-1.)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    task._on_validation_epoch_end()
                assert any('Skipping variance preview' in str(w.message) for w in caught)
                audio.assert_called_once()
        print(f'Variance preview {kind} passed.')


if __name__ == '__main__':
    torch.set_num_threads(1)
    validate_aux_backbones()
    validate_ep_step_resume()
    validate_variance_preview()
    print('Auxiliary datasets, generic previews and ep/step validation passed.')
