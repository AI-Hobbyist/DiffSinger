"""Count complete acoustic/variance state storage on meta; do not allocate large weights."""
import argparse
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.toplevel import DiffSingerAcoustic, DiffSingerVariance
from scripts.model_scaling import PRESETS, preset_args
from utils.checkpoint_optimization import prepare_inference_model, tensor_bytes
from utils.hparams import hparams, set_hparams


def estimate(scope, backbone, index):
    set_hparams(str(ROOT / 'configs/templates' / f'config_{scope}_dit.yaml'), print_hparams=False)
    hparams['hidden_size'] = 384
    layers, width = PRESETS[backbone][index]
    if scope == 'variance':
        layers = max(1, layers // 2)
    args = preset_args(backbone, layers, width)
    if scope == 'acoustic':
        hparams.update(backbone_type=backbone, backbone_args=args)
    else:
        for key in ('pitch_prediction_args', 'variances_prediction_args'):
            hparams[key].update(backbone_type=backbone, backbone_args=args)
    with torch.device('meta'):
        model = (DiffSingerAcoustic(65, 128) if scope == 'acoustic' else DiffSingerVariance(65)).eval()
    parameters = sum(p.numel() for p in model.parameters())
    groups = {name: sum(p.numel() for p in module.parameters()) for name, module in model.named_children()}
    results = {}
    for precision in ('fp32', 'fp16', 'int8'):
        with torch.device('meta'):
            converted = prepare_inference_model(copy.deepcopy(model), precision)
        state = converted.state_dict()
        size = tensor_bytes(state)
        results[precision] = {'state_bytes': size, 'state_gib': size / 2**30,
                              'int8_parameter_coverage': sum(t.numel() for t in state.values()
                                                            if t.dtype == torch.int8) / parameters,
                              'module_bytes': {name: tensor_bytes({k: v for k, v in state.items()
                                                                  if k.startswith(name + '.')})
                                               for name in groups},
                              'inference_budget_gib': [size / 2**30 + (1.5 if precision == 'int8' else 1),
                                                       size / 2**30 + (4 if precision == 'int8' else 3)]}
    return {'scope': scope, 'backbone': backbone, 'preset_index': index, 'args': args,
            'parameters': parameters, 'module_parameters': groups, 'precisions': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--all-presets', action='store_true', help='Count all 6 presets; default smallest/largest.')
    args = parser.parse_args()
    rows = [estimate(scope, backbone, index) for scope in ('acoustic', 'variance')
            for backbone in PRESETS for index in (range(6) if args.all_presets else (0, 5))]
    output = {'method': 'Exact state tensor bytes on meta, including FS2, auxiliary decoder, predictors, '
                        'embeddings, buffers and quantization scales. Runtime budgets are estimates, not measured peaks.',
              'hidden_size': 384, 'vocab_size': 65, 'acoustic_mel_bins': 128, 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    for row in rows:
        p = row['precisions']
        print(f"{row['scope']}/{row['backbone']}/{row['preset_index']}: {row['parameters']/1e6:.2f}M "
              f"FP32={p['fp32']['state_gib']:.3f} FP16={p['fp16']['state_gib']:.3f} INT8={p['int8']['state_gib']:.3f} GiB")


if __name__ == '__main__':
    main()
