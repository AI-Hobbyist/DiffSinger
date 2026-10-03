"""Count actual backbone parameters without allocating their weights (PyTorch meta)."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.backbones import build_backbone
from utils.hparams import hparams, set_hparams


# Each family spans a small model through approximately two billion parameters.
PRESETS = {
    'wavenet': [(12, 128), (20, 384), (24, 512), (32, 896), (40, 1472), (40, 2304)],
    'lynxnet': [(6, 128), (12, 384), (16, 512), (24, 768), (32, 2048), (40, 2560)],
    'lynxnet2': [(6, 256), (12, 512), (16, 768), (24, 1024), (32, 2560), (40, 3072)],
    'dit': [(4, 128), (8, 384), (12, 640), (16, 1024), (24, 1536), (27, 2048)],
}


def preset_args(kind, layers, width):
    args = {'num_layers': layers, 'num_channels': width}
    if kind == 'wavenet':
        args['dilation_cycle_length'] = 4
    elif kind == 'lynxnet':
        args.update(expansion_factor=2, kernel_size=31)
    elif kind == 'lynxnet2':
        args.update(expansion_factor=1, kernel_size=31, dropout_rate=0.0,
                    glu_type='softsign_glu', use_conditioner_cache=False)
    else:
        args.update(num_heads=width // 64, mlp_ratio=4, time_embed_dim=256,
                    patch_size=1, use_gradient_checkpointing=True)
    return args


def scaling_rows(hidden_size=384, out_dims=128, num_feats=1):
    previous = hparams.copy()
    try:
        hparams['hidden_size'] = hidden_size
        rows = []
        for kind, presets in PRESETS.items():
            for layers, width in presets:
                args = preset_args(kind, layers, width)
                with torch.device('meta'):
                    model = build_backbone(out_dims, num_feats, kind, args)
                count = sum(parameter.numel() for parameter in model.parameters())
                # A budgeting envelope, not a benchmark or a minimum requirement.
                weights_gib = count * 2 / 2**30
                rows.append(dict(backbone=kind, args=args, parameters=count,
                                 weights_fp16_gib=round(weights_gib, 3),
                                 inference_estimate_gib=[round(weights_gib + 1, 2),
                                                         round(weights_gib + 3, 2)]))
        return rows
    finally:
        hparams.clear()
        hparams.update(previous)


def standalone_rows(kind, hidden_size=384, vocab_size=65):
    """Count a complete standalone model; variance scales both generators."""
    from modules.toplevel import DiffSingerAcoustic, DiffSingerVariance
    previous = hparams.copy()
    try:
        set_hparams(str(ROOT / 'configs/templates' / f'config_{kind}_dit.yaml'),
                    print_hparams=False)
        hparams['hidden_size'] = hidden_size
        rows = []
        for backbone, presets in PRESETS.items():
            for layers, width in presets:
                # Two generators share the variance model's parameter budget.
                if kind == 'variance':
                    layers = max(1, layers // 2)
                args = preset_args(backbone, layers, width)
                if kind == 'acoustic':
                    hparams.update(backbone_type=backbone, backbone_args=args)
                    with torch.device('meta'):
                        model = DiffSingerAcoustic(vocab_size, 128)
                    generators = [model.diffusion.velocity_fn]
                else:
                    for key in ('pitch_prediction_args', 'variances_prediction_args'):
                        hparams[key].update(backbone_type=backbone, backbone_args=args)
                    with torch.device('meta'):
                        model = DiffSingerVariance(vocab_size)
                    generators = [model.pitch_predictor.velocity_fn,
                                  model.variance_predictor.velocity_fn]
                count = sum(parameter.numel() for parameter in model.parameters())
                backbone_count = sum(parameter.numel() for generator in generators
                                     for parameter in generator.parameters())
                weights_gib = count * 2 / 2**30
                rows.append(dict(scope=kind, backbone=backbone, args=args,
                                 parameters=count, generator_parameters=backbone_count,
                                 weights_fp16_gib=round(weights_gib, 3),
                                 inference_estimate_gib=[round(weights_gib + 1, 2),
                                                         round(weights_gib + 3, 2)]))
        return rows
    finally:
        hparams.clear()
        hparams.update(previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--scope', choices=('backbone', 'acoustic', 'variance'), default='backbone')
    parser.add_argument('--vocab-size', type=int, default=65)
    parser.add_argument('--hidden-size', type=int, default=384)
    parser.add_argument('--out-dims', type=int, default=128)
    parser.add_argument('--num-feats', type=int, default=1)
    args = parser.parse_args()
    if args.scope == 'backbone':
        rows = scaling_rows(args.hidden_size, args.out_dims, args.num_feats)
    else:
        rows = standalone_rows(args.scope, args.hidden_size, args.vocab_size)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print(f'# Scope: {args.scope}; VRAM values are estimates for reference only.')
        print('| Backbone | Layers | Width | Parameters (M) | FP16 weights (GiB) | Inference estimate (GiB) |')
        print('| --- | ---: | ---: | ---: | ---: | ---: |')
        for row in rows:
            config = row['args']
            low, high = row['inference_estimate_gib']
            print(f"| {row['backbone']} | {config['num_layers']} | {config['num_channels']} "
                  f"| {row['parameters'] / 1e6:.2f} | {row['weights_fp16_gib']:.3f} | {low:.2f}–{high:.2f} |")


if __name__ == '__main__':
    main()
