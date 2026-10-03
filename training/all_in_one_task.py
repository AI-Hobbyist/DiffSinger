import pathlib
import warnings

import matplotlib.pyplot as plt
import torch
from lightning.pytorch.utilities.combined_loader import CombinedLoader

from inference.ds_acoustic import DiffSingerAcousticInfer
from inference.ds_variance import DiffSingerVarianceInfer
from modules.toplevel import DiffSingerAllInOne
from modules.backbones.dit import EmptyValidMaskError
from training.acoustic_task import AcousticDataset, AcousticTask
from training.variance_task import VarianceDataset, VarianceTask
from utils.hparams import hparams
from utils.aux_dataset import AUX_MODULES, parse_aux_dataset_config
from utils.plot import spec_to_figure
from utils.training_utils import DsBatchSampler
from utils.variance_validation import (
    load_validation_sources, validate_variance_validation_config
)


class AllInOneTask(AcousticTask):
    training_stream_count = 2

    def __init__(self):
        all_in_one_config = hparams.get('all_in_one', {})
        if not isinstance(all_in_one_config, dict) or not all_in_one_config.get('enabled', False):
            raise ValueError('AllInOneTask requires all_in_one.enabled: true.')

        self.val_with_variance = hparams.get('val_with_variance', {})
        self.val_with_variance_enabled = (
            isinstance(self.val_with_variance, dict)
            and self.val_with_variance.get('enable', False)
        )
        if self.val_with_variance_enabled:
            validate_variance_validation_config(
                self.val_with_variance, hparams['dictionaries']
            )
            if not hparams['predict_dur'] or not hparams['predict_pitch']:
                raise ValueError(
                    'val_with_variance requires predict_dur and predict_pitch.'
                )
            hparams['val_with_vocoder'] = True

        self.use_spk_id = hparams['use_spk_id']
        self.use_lang_id = hparams['use_lang_id']
        self.predict_dur = hparams['predict_dur']
        if self.predict_dur:
            self.lambda_dur_loss = hparams['lambda_dur_loss']
        self.predict_pitch = hparams['predict_pitch']
        if self.predict_pitch:
            self.lambda_pitch_loss = hparams['lambda_pitch_loss']
        self.variance_prediction_list = [
            name for name in ('energy', 'breathiness', 'voicing', 'tension')
            if hparams[f'predict_{name}']
        ]
        self.predict_variances = bool(self.variance_prediction_list)
        if not (self.predict_dur or self.predict_pitch or self.predict_variances):
            raise ValueError('All-in-one training requires at least one variance predictor.')
        self.lambda_var_loss = hparams['lambda_var_loss']
        self.aux_config = parse_aux_dataset_config(
            hparams.get('aux_datasets', {}),
            enabled_modules={name for name in AUX_MODULES if hparams.get(f'predict_{name}', False)}
        )
        self.training_stream_count = 3 if self.aux_config else 2
        self.all_in_one_samplers = {}
        self.variance_validation_language = None
        self.variance_validation_speakers = []
        super().__init__()
        VarianceTask._patch_variance_fused_kernels(self)

    def on_fit_start(self):
        super().on_fit_start()
        VarianceTask.on_fit_start(self)

    def _build_model(self):
        return DiffSingerAllInOne(
            vocab_size=len(self.phoneme_dictionary),
            out_dims=hparams['audio_num_mel_bins']
        )

    def build_losses_and_metrics(self):
        AcousticTask.build_losses_and_metrics(self)
        VarianceTask.build_losses_and_metrics(self)

    def setup(self, stage):
        binary_data_dir = pathlib.Path(hparams['binary_data_dir'])
        self.acoustic_train_dataset = AcousticDataset(
            'train', data_dir=binary_data_dir / 'acoustic'
        )
        self.acoustic_valid_dataset = AcousticDataset(
            'valid', data_dir=binary_data_dir / 'acoustic'
        )
        self.variance_train_dataset = VarianceDataset(
            'train', data_dir=binary_data_dir / 'variance'
        )
        self.variance_valid_dataset = VarianceDataset(
            'valid', data_dir=binary_data_dir / 'variance'
        )
        if self.aux_config:
            self.aux_train_dataset = VarianceTask._build_aux_dataset(self, 'train')
            self.aux_valid_dataset = VarianceTask._build_aux_dataset(self, 'valid')
        if self.val_with_variance_enabled:
            speaker_names = self.acoustic_valid_dataset.metadata.get('spk_names', [])
            speaker_ids = self.acoustic_valid_dataset.metadata.get('spk_ids', [])
            self.variance_validation_speakers = list(dict.fromkeys(
                zip(speaker_names, speaker_ids)
            ))
        self.num_replicas = (self.trainer.distributed_sampler_kwargs or {}).get('num_replicas', 1)

    def _build_dataloader(self, name, dataset, training):
        if training:
            max_batch_frames = self._split_training_batch_limit(
                self.max_batch_frames, 'max_batch_frames'
            )
            max_batch_size = self._split_training_batch_limit(
                self.max_batch_size, 'max_batch_size'
            )
        else:
            max_batch_frames = self.max_val_batch_frames
            max_batch_size = self.max_val_batch_size
        sampler = DsBatchSampler(
            dataset,
            max_batch_frames=max_batch_frames,
            max_batch_size=max_batch_size,
            num_replicas=self.num_replicas,
            rank=self.global_rank,
            sort_by_similar_size=hparams['sort_by_len'] if training else False,
            size_reversed=training,
            required_batch_count_multiple=hparams['accumulate_grad_batches'] if training else 1,
            shuffle_sample=training,
            shuffle_batch=training,
            disallow_empty_batch=training,
            pad_batch_assignment=training
        )
        if training:
            self.all_in_one_samplers[name] = sampler
        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collater,
            batch_sampler=sampler,
            num_workers=hparams['ds_workers'],
            prefetch_factor=(hparams['dataloader_prefetch_factor'] if hparams['ds_workers'] > 0 else None),
            pin_memory=True,
            persistent_workers=(hparams['ds_workers'] > 0)
        )

    def _split_training_batch_limit(self, total_limit, config_name):
        if total_limit < self.training_stream_count:
            raise ValueError(
                f'All-in-one training requires {config_name} >= '
                f'{self.training_stream_count}, got {total_limit}.'
            )
        return total_limit // self.training_stream_count

    def train_dataloader(self):
        loaders = {
            'acoustic': self._build_dataloader(
                'acoustic', self.acoustic_train_dataset, training=True
            ),
            'variance': self._build_dataloader(
                'variance', self.variance_train_dataset, training=True
            )
        }
        if self.aux_config:
            loaders['aux'] = self._build_dataloader('aux', self.aux_train_dataset, training=True)
        return CombinedLoader(loaders, mode='max_size_cycle')

    def val_dataloader(self):
        loaders = [
            self._build_dataloader('acoustic', self.acoustic_valid_dataset, training=False),
            self._build_dataloader('variance', self.variance_valid_dataset, training=False)
        ]
        if self.aux_config:
            loaders.append(self._build_dataloader('aux', self.aux_valid_dataset, training=False))
        return loaders

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        for sampler in self.all_in_one_samplers.values():
            sampler.set_epoch(self.current_epoch)

    def _training_step(self, sample):
        acoustic_losses = AcousticTask.run_model(
            self, sample['acoustic'], model=self.model.acoustic
        )
        variance_losses = VarianceTask.run_model(
            self, sample['variance'], model=self.model.variance,
            modules=AUX_MODULES - self.aux_config['modules'] if self.aux_config else None
        )
        losses = {**acoustic_losses, **variance_losses}
        if self.aux_config:
            aux_losses = VarianceTask.run_model(
                self, sample['aux'], model=self.model.variance, modules=self.aux_config['modules']
            )
            for name, value in aux_losses.items():
                losses[name] = losses.get(name, 0) + value
        batch_size = sum(batch['size'] for batch in sample.values())
        return sum(losses.values()), {**losses, 'batch_size': float(batch_size)}

    def validation_step(self, sample, batch_idx, dataloader_idx=0):
        if sample['size'] == 0:
            return
        with torch.autocast(self.device.type, enabled=False):
            if dataloader_idx == 0:
                losses, weight = self._run_validation_step(
                    AcousticTask._validation_step,
                    self.acoustic_valid_dataset, sample, batch_idx,
                    model=self.model.acoustic
                )
            else:
                losses, weight = self._run_validation_step(
                    VarianceTask._validation_step,
                    self.variance_valid_dataset if dataloader_idx == 1 else self.aux_valid_dataset,
                    sample, batch_idx, model=self.model.variance,
                    modules=(AUX_MODULES - self.aux_config['modules'] if dataloader_idx == 1
                             else self.aux_config['modules']) if self.aux_config else None,
                    plot=dataloader_idx == 1
                )
        if not losses:
            return
        losses = {'total_loss': sum(losses.values()), **losses}
        for name, value in losses.items():
            self.valid_losses[name].update(value, weight=weight)

    def _run_validation_step(self, validation_step, dataset, sample, batch_idx, **kwargs):
        previous_dataset = getattr(self, 'valid_dataset', None)
        self.valid_dataset = dataset
        try:
            return validation_step(self, sample, batch_idx, **kwargs)
        finally:
            self.valid_dataset = previous_dataset

    def _on_validation_start(self):
        super()._on_validation_start()
        if not self.val_with_variance_enabled or self.global_rank != 0:
            return
        speaker_map = {
            name: speaker_id for name, speaker_id in self.variance_validation_speakers
        }
        lang_map = {
            language: index for index, language in enumerate(sorted(hparams['dictionaries']), start=1)
        }
        self.variance_validation_infer = DiffSingerVarianceInfer(
            device=self.device, predictions=set(), model=self.model.variance,
            spk_map=speaker_map, lang_map=lang_map
        )
        self.acoustic_validation_infer = DiffSingerAcousticInfer(
            device=self.device, load_vocoder=False, model=self.model.acoustic,
            spk_map=speaker_map, lang_map=lang_map
        )

    @torch.inference_mode()
    def _on_validation_epoch_end(self):
        if (
                not self.val_with_variance_enabled
                or self.global_rank != 0
                or not self.variance_validation_speakers
        ):
            return
        language, source = load_validation_sources(
            self.val_with_variance, hparams['dictionaries'],
            previous_language=self.variance_validation_language
        )
        self.variance_validation_language = language
        timestep = hparams['hop_size'] / hparams['audio_sample_rate']
        for speaker_name, _ in self.variance_validation_speakers:
            param = dict(source)
            if hparams['use_spk_id']:
                param['ph_spk_mix'] = param['spk_mix'] = {speaker_name: 1.0}
            variance_batch = self.variance_validation_infer.preprocess_input(param)
            try:
                durations, pitch, variances = self.variance_validation_infer.forward_model(
                    variance_batch
                )
            except EmptyValidMaskError:
                warnings.warn(f'Skipping variance preview for {speaker_name}: no valid predicted duration frames.')
                continue
            if durations.sum() <= 0:
                warnings.warn(f'Skipping variance preview for {speaker_name}: predicted durations are all zero.')
                continue
            param['ph_dur'] = ' '.join(
                str(round(value * timestep, 6)) for value in durations[0].cpu().tolist()
            )
            f0 = 440.0 * torch.pow(2.0, (pitch[0] - 69.0) / 12.0)
            param['f0_seq'] = ' '.join(str(round(value, 1)) for value in f0.cpu().tolist())
            param['f0_timestep'] = str(timestep)
            for name, values in variances.items():
                param[name] = ' '.join(
                    str(round(value, 4)) for value in values[0].cpu().tolist()
                )
                param[f'{name}_timestep'] = str(timestep)

            acoustic_batch = self.acoustic_validation_infer.preprocess_input(param)
            mel = self.acoustic_validation_infer.forward_model(acoustic_batch)
            waveform = self.vocoder.spec2wav_torch(mel, f0=acoustic_batch['f0'])
            tag = f'variance_validate/{speaker_name}'
            title = f'{language} - {speaker_name} - Variance Validate'
            self.logger.all_rank_experiment.add_audio(
                f'{tag}/audio', waveform,
                sample_rate=hparams['audio_sample_rate'], global_step=self.global_step
            )
            figure = spec_to_figure(
                mel[0], hparams['mel_vmin'], hparams['mel_vmax'], title
            )
            self.logger.all_rank_experiment.add_figure(
                f'{tag}/mel', figure, global_step=self.global_step
            )
            plt.close(figure)
