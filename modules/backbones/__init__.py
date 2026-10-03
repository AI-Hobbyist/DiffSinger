import torch.nn
from modules.backbones.dit import DiT
from modules.backbones.wavenet import WaveNet
from modules.backbones.lynxnet import LYNXNet
from modules.backbones.lynxnet2 import LYNXNet2
from utils import filter_kwargs

BACKBONES = {
    'wavenet': WaveNet,
    'lynxnet': LYNXNet,
    'lynxnet2': LYNXNet2,
    'dit': DiT,
}


def build_backbone(
        out_dims: int, num_feats: int,
        backbone_type: str, backbone_args: dict
) -> torch.nn.Module:
    backbone = BACKBONES[backbone_type]
    if backbone_type == 'dit':
        if not isinstance(backbone_args, dict):
            raise ValueError('DiT requires backbone_args to be a mapping.')
        allowed_args = {
            'num_layers', 'num_channels', 'num_heads', 'mlp_ratio', 'time_embed_dim',
            'patch_size', 'rope_base', 'layer_norm_eps', 'attention_dropout',
            'mlp_dropout', 'use_gradient_checkpointing'
        }
        unknown_args = set(backbone_args) - allowed_args
        if unknown_args:
            raise ValueError(f'Unknown DiT backbone arguments: {sorted(unknown_args)}')
        return backbone(out_dims, num_feats, **backbone_args)
    kwargs = filter_kwargs(backbone_args, backbone)
    return backbone(out_dims, num_feats, **kwargs)


def run_backbone(backbone, spec, diffusion_step, cond, valid_mask=None):
    if getattr(backbone, 'supports_valid_mask', False):
        return backbone(spec, diffusion_step, cond, valid_mask=valid_mask)
    return backbone(spec, diffusion_step, cond)
