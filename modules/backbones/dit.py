import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from utils.hparams import hparams


class EmptyValidMaskError(ValueError):
    """An input item has no valid frames for attention."""


class DiTAttention(nn.Module):
    def __init__(self, dim, num_heads, rope_base=10000.0, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        inv_freq = 1.0 / (
            rope_base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    @staticmethod
    def _rotate_half(x):
        x_1 = x[..., 0::2]
        x_2 = x[..., 1::2]
        return torch.stack((-x_2, x_1), dim=-1).reshape_as(x)

    def _apply_rope(self, q, k):
        positions = torch.arange(q.shape[-2], device=q.device, dtype=self.inv_freq.dtype)
        angles = torch.outer(positions, self.inv_freq)
        cos = angles.cos().repeat_interleave(2, dim=-1).to(q.dtype)[None, None, :, :]
        sin = angles.sin().repeat_interleave(2, dim=-1).to(q.dtype)[None, None, :, :]
        return q * cos + self._rotate_half(q) * sin, k * cos + self._rotate_half(k) * sin

    def forward(self, x, valid_mask):
        batch, frames, channels = x.shape
        qkv = self.qkv(x).reshape(batch, frames, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self._apply_rope(q, k)
        q = F.normalize(q, dim=-1) * math.sqrt(self.head_dim)
        k = F.normalize(k, dim=-1) * math.sqrt(self.head_dim)

        attention_mask = None
        if valid_mask is not None:
            attention_mask = torch.zeros(
                batch, 1, 1, frames, dtype=q.dtype, device=q.device
            ).masked_fill(~valid_mask[:, None, None, :], float('-inf'))
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        x = x.transpose(1, 2).reshape(batch, frames, channels)
        return self.proj(x)


class DiTMLP(nn.Module):
    def __init__(self, dim, mlp_ratio, dropout=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.fc1(x), approximate='tanh')
        return self.dropout(self.fc2(x))


class DiTBlock(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio, rope_base, layer_norm_eps,
            attention_dropout, mlp_dropout
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps, elementwise_affine=False)
        self.attn = DiTAttention(dim, num_heads, rope_base=rope_base, dropout=attention_dropout)
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps, elementwise_affine=False)
        self.mlp = DiTMLP(dim, mlp_ratio, dropout=mlp_dropout)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    @staticmethod
    def _modulate(x, shift, scale):
        if shift.ndim == 2:
            shift, scale = shift[:, None, :], scale[:, None, :]
        return x * (1 + scale) + shift

    @staticmethod
    def _stabilize_modulation(modulation):
        return torch.tanh(modulation)

    def forward(self, x, time_embedding, valid_mask):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self._stabilize_modulation(
                self.adaLN_modulation(time_embedding)
            ).chunk(6, dim=-1)
        )
        if gate_attn.ndim == 2:
            gate_attn, gate_mlp = gate_attn[:, None, :], gate_mlp[:, None, :]
        x = x + gate_attn * self.attn(
            self._modulate(self.norm1(x), shift_attn, scale_attn), valid_mask
        )
        x = x + gate_mlp * self.mlp(
            self._modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        if valid_mask is not None:
            x = x * valid_mask[:, :, None]
        return x


class DiT(nn.Module):
    supports_valid_mask = True
    supports_dual_timestep = True

    def __init__(
            self, in_dims, n_feats, *, num_layers=8, num_channels=384,
            num_heads=6, mlp_ratio=4, time_embed_dim=256, patch_size=1,
            rope_base=10000.0, layer_norm_eps=1e-6, attention_dropout=0.0,
            mlp_dropout=0.0, use_gradient_checkpointing=True
    ):
        super().__init__()
        self._validate_config(
            num_layers=num_layers,
            num_channels=num_channels,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            time_embed_dim=time_embed_dim,
            patch_size=patch_size,
            rope_base=rope_base,
            layer_norm_eps=layer_norm_eps,
            attention_dropout=attention_dropout,
            mlp_dropout=mlp_dropout,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        self.in_dims = in_dims
        self.n_feats = n_feats
        self.num_channels = num_channels
        self.time_embed_dim = time_embed_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        input_dims = in_dims * n_feats

        self.input_proj = nn.Linear(input_dims, num_channels)
        self.cond_proj = nn.Linear(hparams['hidden_size'], num_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, num_channels),
            nn.SiLU(),
            nn.Linear(num_channels, num_channels),
        )
        self.blocks = nn.ModuleList([
            DiTBlock(
                num_channels,
                num_heads,
                mlp_ratio,
                rope_base,
                layer_norm_eps,
                attention_dropout,
                mlp_dropout,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(
            num_channels, eps=layer_norm_eps, elementwise_affine=False
        )
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(num_channels, num_channels * 2))
        self.output_proj = nn.Linear(num_channels, input_dims)
        nn.init.zeros_(self.final_modulation[1].weight)
        nn.init.zeros_(self.final_modulation[1].bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    @staticmethod
    def _validate_config(**config):
        integer_fields = ('num_layers', 'num_channels', 'num_heads', 'time_embed_dim', 'patch_size')
        for name in integer_fields:
            if not isinstance(config[name], int) or isinstance(config[name], bool) or config[name] <= 0:
                raise ValueError(f'{name} must be a positive integer, got {config[name]!r}.')
        if config['patch_size'] != 1:
            raise ValueError('DiT currently requires patch_size=1.')
        if config['num_channels'] % config['num_heads'] != 0:
            raise ValueError('num_channels must be divisible by num_heads.')
        if (config['num_channels'] // config['num_heads']) % 2 != 0:
            raise ValueError('The DiT attention head dimension must be even for RoPE.')
        if not isinstance(config['mlp_ratio'], (int, float)) or config['mlp_ratio'] <= 0:
            raise ValueError('mlp_ratio must be positive.')
        if int(config['num_channels'] * config['mlp_ratio']) <= 0:
            raise ValueError('mlp_ratio produces an invalid hidden dimension.')
        for name in ('rope_base', 'layer_norm_eps'):
            if not isinstance(config[name], (int, float)) or config[name] <= 0:
                raise ValueError(f'{name} must be positive.')
        for name in ('attention_dropout', 'mlp_dropout'):
            if not isinstance(config[name], (int, float)) or not 0 <= config[name] < 1:
                raise ValueError(f'{name} must be in [0, 1).')
        if not isinstance(config['use_gradient_checkpointing'], bool):
            raise ValueError('use_gradient_checkpointing must be a boolean.')

    def _time_embedding(self, diffusion_step, batch, device):
        step = diffusion_step.to(device=device, dtype=torch.float32)
        if step.numel() == 1:
            step = step.reshape(1).expand(batch)
        elif step.shape == (batch,) or step.shape == (batch, 1):
            step = step.reshape(batch)
        else:
            raise ValueError(
                f'diffusion_step must contain one value or one value per batch item, got {tuple(step.shape)}.'
            )
        half_dim = self.time_embed_dim // 2
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=device, dtype=torch.float32) / half_dim
        )
        angles = step[:, None] * frequencies[None, :]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if embedding.shape[-1] < self.time_embed_dim:
            embedding = F.pad(embedding, (0, 1))
        projection = self.time_mlp[0]
        dtype = projection.compute_dtype if hasattr(projection, 'compute_dtype') else projection.weight.dtype
        return self.time_mlp(embedding.to(dtype))

    @staticmethod
    def _validate_mask(valid_mask, batch, frames, device):
        if valid_mask is None:
            return None
        if valid_mask.shape != (batch, frames):
            raise ValueError(
                f'valid_mask must have shape {(batch, frames)}, got {tuple(valid_mask.shape)}.'
            )
        if valid_mask.dtype != torch.bool:
            raise ValueError('valid_mask must use boolean dtype with True indicating valid frames.')
        valid_mask = valid_mask.to(device=device)
        if not torch.all(valid_mask.any(dim=1)):
            raise EmptyValidMaskError('Each DiT request must contain at least one valid frame.')
        return valid_mask

    def forward(self, spec, diffusion_step, cond, diffusion_step_2=None, mask=None, valid_mask=None):
        if spec.ndim != 4:
            raise ValueError(f'spec must have shape [B, F, M, T], got {tuple(spec.shape)}.')
        batch, num_feats, in_dims, frames = spec.shape
        if num_feats != self.n_feats or in_dims != self.in_dims:
            raise ValueError(
                f'spec feature shape must be {(self.n_feats, self.in_dims)}, got {(num_feats, in_dims)}.'
            )
        if cond.shape != (batch, self.cond_proj.in_features, frames):
            raise ValueError(
                f'cond must have shape {(batch, self.cond_proj.in_features, frames)}, got {tuple(cond.shape)}.'
            )
        valid_mask = self._validate_mask(valid_mask, batch, frames, spec.device)
        query_mask = 1 if valid_mask is None else valid_mask[:, :, None]

        x = spec.flatten(1, 2).transpose(1, 2)
        x = (self.input_proj(x) + self.cond_proj(cond.transpose(1, 2))) * query_mask
        time_embedding = self._time_embedding(diffusion_step, batch, spec.device).to(x.dtype)
        if mask is not None:
            if diffusion_step_2 is None or mask.shape != (batch, frames):
                raise ValueError('Dual timestep requires diffusion_step_2 and mask shaped [B, T].')
            second_embedding = self._time_embedding(diffusion_step_2, batch, spec.device).to(x.dtype)
            time_embedding = time_embedding[:, None, :] + (
                second_embedding - time_embedding
            )[:, None, :] * mask.to(x)[:, :, None]
        for block in self.blocks:
            if self.training and self.use_gradient_checkpointing:
                x = checkpoint(block, x, time_embedding, valid_mask, use_reentrant=False)
            else:
                x = block(x, time_embedding, valid_mask)

        shift, scale = torch.tanh(self.final_modulation(time_embedding)).chunk(2, dim=-1)
        x = DiTBlock._modulate(self.final_norm(x), shift, scale)
        x = self.output_proj(x) * query_mask
        return x.transpose(1, 2).reshape(batch, num_feats, in_dims, frames)
