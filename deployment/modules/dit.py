import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from modules.backbones.dit import DiT, DiTAttention, DiTBlock


class DiTAttentionONNX(nn.Module):
    def __init__(self, source: DiTAttention):
        super().__init__()
        self.num_heads = source.num_heads
        self.head_dim = source.head_dim
        self.qkv = copy.deepcopy(source.qkv)
        self.proj = copy.deepcopy(source.proj)
        self.register_buffer('inv_freq', source.inv_freq.detach().clone(), persistent=False)

    @staticmethod
    def _rotate_half(x):
        x_1 = x[..., 0::2]
        x_2 = x[..., 1::2]
        return torch.stack((-x_2, x_1), dim=-1).reshape_as(x)

    def forward(self, x, valid_mask):
        batch, frames, channels = x.shape
        qkv = self.qkv(x).reshape(batch, frames, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        positions = torch.arange(q.shape[-2], device=q.device, dtype=torch.float32)
        angles = torch.outer(positions, self.inv_freq)
        cos = angles.cos().repeat_interleave(2, dim=1)[None, None, :, :]
        sin = angles.sin().repeat_interleave(2, dim=1)[None, None, :, :]
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        q = F.normalize(q, dim=-1) * math.sqrt(self.head_dim)
        k = F.normalize(k, dim=-1) * math.sqrt(self.head_dim)

        scores = torch.matmul(q, k.permute(0, 1, 3, 2)) * (self.head_dim ** -0.5)
        scores = scores.masked_fill(~valid_mask[:, None, None, :], -1e4)
        attention = torch.softmax(scores, dim=3)
        x = torch.matmul(attention, v)
        x = x.transpose(1, 2).reshape(batch, frames, channels)
        return self.proj(x)


class DiTBlockONNX(nn.Module):
    def __init__(self, source: DiTBlock):
        super().__init__()
        self.norm1 = copy.deepcopy(source.norm1)
        self.attn = DiTAttentionONNX(source.attn)
        self.norm2 = copy.deepcopy(source.norm2)
        self.mlp = copy.deepcopy(source.mlp)
        self.adaLN_modulation = copy.deepcopy(source.adaLN_modulation)

    @staticmethod
    def _modulate(x, shift, scale):
        return x * (1 + scale[:, None, :]) + shift[:, None, :]

    @staticmethod
    def _stabilize_modulation(modulation):
        return torch.tanh(modulation)

    def forward(self, x, time_embedding, valid_mask):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self._stabilize_modulation(
                self.adaLN_modulation(time_embedding)
            ).chunk(6, dim=1)
        )
        x = x + gate_attn[:, None, :] * self.attn(
            self._modulate(self.norm1(x), shift_attn, scale_attn), valid_mask
        )
        x = x + gate_mlp[:, None, :] * self.mlp(
            self._modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x * valid_mask[:, :, None]


class DiTONNXAdapter(nn.Module):
    """Three-argument, opset-15-compatible view of a mask-aware DiT."""

    def __init__(self, backbone: DiT):
        super().__init__()
        self.in_dims = backbone.in_dims
        self.n_feats = backbone.n_feats
        self.time_embed_dim = backbone.time_embed_dim
        self.input_proj = copy.deepcopy(backbone.input_proj)
        self.cond_proj = copy.deepcopy(backbone.cond_proj)
        self.time_mlp = copy.deepcopy(backbone.time_mlp)
        self.blocks = nn.ModuleList([DiTBlockONNX(block) for block in backbone.blocks])
        self.final_norm = copy.deepcopy(backbone.final_norm)
        self.final_modulation = copy.deepcopy(backbone.final_modulation)
        self.output_proj = copy.deepcopy(backbone.output_proj)
        half_dim = self.time_embed_dim // 2
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer('time_frequencies', frequencies, persistent=False)

    def forward(self, spec, diffusion_step, cond):
        batch = spec.shape[0]
        frames = spec.shape[-1]
        valid_mask = torch.ones_like(cond[:, 0, :], dtype=torch.bool)
        query_mask = valid_mask[:, :, None]
        x = spec.flatten(1, 2).transpose(1, 2)
        x = (self.input_proj(x) + self.cond_proj(cond.transpose(1, 2))) * query_mask

        step = diffusion_step.to(dtype=torch.float32).reshape(-1)
        angles = step[:, None] * self.time_frequencies[None, :]
        time_embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        if self.time_embed_dim % 2:
            time_embedding = F.pad(time_embedding, (0, 1))
        time_embedding = self.time_mlp(time_embedding).to(x.dtype)
        for block in self.blocks:
            x = block(x, time_embedding, valid_mask)

        shift, scale = torch.tanh(self.final_modulation(time_embedding)).chunk(2, dim=1)
        x = self.final_norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]
        x = self.output_proj(x) * query_mask
        return x.transpose(1, 2).reshape(batch, self.n_feats, self.in_dims, frames)


def prepare_backbone_for_onnx(backbone: nn.Module) -> nn.Module:
    if bool(getattr(backbone, 'supports_valid_mask', False)):
        if not isinstance(backbone, DiT):
            raise TypeError(
                f'Unsupported mask-aware ONNX backbone: {type(backbone).__name__}'
            )
        return DiTONNXAdapter(backbone)
    return backbone


def compile_backbone_for_onnx(backbone: nn.Module, example_inputs) -> torch.jit.ScriptModule:
    prepared = prepare_backbone_for_onnx(backbone).to(example_inputs[0].device)
    from deployment.modules.quantization import prepare_portable_modules
    prepare_portable_modules(prepared, 'int8', compile_integer=True)
    return torch.jit.trace(prepared, example_inputs)
