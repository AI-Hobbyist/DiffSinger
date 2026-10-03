import torch.nn as nn
from torch import Tensor


class DiffusionLoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type
        if self.loss_type == 'l1':
            self.loss = nn.L1Loss(reduction='none')
        elif self.loss_type == 'l2':
            self.loss = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError()

    @staticmethod
    def _mask_non_padding(x_recon, noise, non_padding=None):
        if non_padding is not None:
            non_padding = non_padding.transpose(1, 2).unsqueeze(1)
            return x_recon * non_padding, noise * non_padding
        else:
            return x_recon, noise

    def _forward(self, x_recon, noise):
        return self.loss(x_recon, noise)

    def forward(self, x_recon: Tensor, noise: Tensor, non_padding: Tensor = None, feature_mask: Tensor = None) -> Tensor:
        """
        :param x_recon: [B, 1, M, T]
        :param noise: [B, 1, M, T]
        :param non_padding: [B, T, M]
        """
        x_recon, noise = self._mask_non_padding(x_recon, noise, non_padding)
        loss = self._forward(x_recon, noise)
        if feature_mask is None:
            return loss.mean()
        # [B,F] selects curve generators' feature axis, never mel bins or frames.
        mask = feature_mask.to(device=loss.device, dtype=loss.dtype)
        if mask.ndim != 2 or mask.shape[1] != loss.shape[1] or mask.shape[0] not in (1, loss.shape[0]):
            raise ValueError('feature_mask must have shape [1,F] or [B,F].')
        mask = mask.expand(loss.shape[0], -1)
        # Preserve the legacy mean over padded frames, averaging only selected curves.
        return (loss * mask[:, :, None, None]).sum() / (mask.sum().clamp_min(1) * loss.shape[2] * loss.shape[3])
