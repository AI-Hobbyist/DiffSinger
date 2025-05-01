import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn

from modules.decoder import ShallowDiffusionOutput
from modules.losses import RectifiedFlowLoss
from modules.toplevel import DiffSingerAcoustic
from modules.vocoder import Vocoder
from utils.plot import spec_to_figure
from .pl_module_base import BaseLightningModule


class AcousticLightningModule(BaseLightningModule):
    def build_model(self):
        return DiffSingerAcoustic(self.model_config)

    # noinspection PyAttributeOutsideInit
    def post_init(self):
        if self.training_config.validation.use_vocoder:
            self.vocoder = Vocoder(self.training_config.validation.vocoder)
        else:
            self.vocoder = None
        self.logged_gt_wav_indices = set()

    def setup(self, stage: str) -> None:
        super().setup(stage)
        if self.vocoder is not None:
            self.vocoder.to(self.device)

    def build_losses_and_metrics(self):
        if self.model_config.spec_decoder.use_shallow_diffusion:
            aux_loss_type = self.training_config.loss.spec_decoder.aux_loss_type
            if aux_loss_type == "L1":
                aux_spec_loss = nn.L1Loss()
            elif aux_loss_type == "L2":
                aux_spec_loss = nn.MSELoss()
            else:
                raise ValueError("Invalid spec_decoder.aux_loss_type")
            self.register_loss("aux_spec_loss", aux_spec_loss)
        main_loss_type = self.training_config.loss.spec_decoder.main_loss_type
        if main_loss_type not in ["L1", "L2"]:
            raise ValueError("Invalid spec_decoder.main_loss_type")
        diff_spec_loss = RectifiedFlowLoss(
            loss_type=main_loss_type,
            log_norm=self.training_config.loss.spec_decoder.main_loss_log_norm
        )
        self.register_loss("diff_spec_loss", diff_spec_loss)

    def forward_model(self, sample: dict[str, torch.Tensor], infer: bool) -> dict[str, torch.Tensor]:
        sample = sample.copy()
        tokens = sample.pop("tokens")
        languages = sample.pop("languages")
        durations = sample.pop("ph_dur")
        f0 = sample.pop("f0")
        spk_id = sample.pop("spk_id")
        mel = sample.pop("mel")
        key_shift = sample.pop("key_shift").unsqueeze(1)
        speed = sample.pop("speed").unsqueeze(1)
        model_out, mask = self.model(
            tokens=tokens, durations=durations, languages=languages, spk_ids=spk_id,
            f0=f0, key_shift=key_shift, speed=speed, spec_gt=mel, infer=infer, **sample
        )
        model_out: ShallowDiffusionOutput
        if infer:
            outputs = {
                "aux_spec": model_out.aux_out,
                "diff_spec": model_out.diff_out,
            }
            return {k: v for k, v in outputs.items() if v is not None}
        else:
            losses = {}
            if model_out.aux_out is not None:
                aux_spec_loss = self.losses["aux_spec_loss"](model_out.aux_out, model_out.norm_gt)
                losses["aux_spec_loss"] = aux_spec_loss * self.training_config.loss.spec_decoder.aux_loss_lambda
            v_pred, v_gt, t = model_out.diff_out
            diff_spec_loss = self.losses["diff_spec_loss"](v_pred, v_gt, t=t, non_padding=mask.unsqueeze(-1).to(t))
            losses["diff_spec_loss"] = diff_spec_loss
            return losses

    def plot_validation_results(self, sample: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]):
        for i in range(len(sample["indices"])):
            data_idx = sample['indices'][i].item()
            spec_len = self.valid_dataset.info["mel"][data_idx]
            f0_len = self.valid_dataset.info["f0"][data_idx]
            if data_idx < self.training_config.validation.max_plots:
                gt_spec = sample["mel"][i, :spec_len]
                f0 = sample["f0"][i, :f0_len]
                if self.vocoder is not None and data_idx not in self.logged_gt_wav_indices:
                    gt_wav = self.vocoder.run(gt_spec.unsqueeze(0), f0=f0.unsqueeze(0)).squeeze(0)
                    self.plot_wav(data_idx, gt_wav, name_prefix="gt")
                    self.logged_gt_wav_indices.add(data_idx)
                aux_spec = outputs.get("aux_spec")
                diff_spec = outputs.get("diff_spec")
                if aux_spec is not None:
                    self.plot_spec(
                        data_idx,
                        sample["mel"][i, :spec_len],
                        outputs["aux_spec"][i, :spec_len],
                        name_prefix="aux_spec",
                    )
                    if self.vocoder is not None:
                        aux_wav = self.vocoder.run(outputs["aux_spec"][i].unsqueeze(0), f0=f0.unsqueeze(0)).squeeze(0)
                        self.plot_wav(data_idx, aux_wav, name_prefix="aux")
                if diff_spec is not None:
                    self.plot_spec(
                        data_idx,
                        sample["mel"][i, :spec_len],
                        outputs["diff_spec"][i, :spec_len],
                        name_prefix="diff_spec",
                    )
                    if self.vocoder is not None:
                        diff_wav = self.vocoder.run(outputs["diff_spec"][i].unsqueeze(0), f0=f0.unsqueeze(0)).squeeze(0)
                        self.plot_wav(data_idx, diff_wav, name_prefix="diff")

    def plot_spec(self, data_idx: int, spec_gt: torch.Tensor, spec_pred: torch.Tensor, name_prefix="spec", title=None):
        vmin = self.training_config.validation.spec_vmin
        vmax = self.training_config.validation.spec_vmax
        spec_compare = torch.cat([(spec_pred - spec_gt).abs() + vmin, spec_gt, spec_pred], -1)
        logger: TensorBoardLogger = self.logger
        logger.experiment.add_figure(f"{name_prefix}_{data_idx}", spec_to_figure(
            spec_compare, vmin=vmin, vmax=vmax, title=title
        ), global_step=self.global_step)

    def plot_wav(self, data_idx: int, wav: torch.Tensor, name_prefix="gt"):
        logger: TensorBoardLogger = self.logger
        logger.experiment.add_audio(
            f"{name_prefix}_{data_idx}", wav.cpu().numpy(),
            sample_rate=self.vocoder.sample_rate, global_step=self.global_step
        )
