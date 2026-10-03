"""Strip training state and convert original DiffSinger weights for native inference."""
import argparse
import copy
import json
import re
import shutil
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.toplevel import DiffSingerAcoustic, DiffSingerVariance, DiffSingerAllInOne
from utils import load_ckpt
from utils.checkpoint_optimization import FORMAT_VERSION, prepare_inference_model, tensor_bytes
from utils.hparams import hparams, set_hparams
from utils.phoneme_utils import load_phoneme_dictionary


def optimize(checkpoint, config, output_dir, precision='int8', component='auto'):
    checkpoint, config, output_dir = Path(checkpoint).resolve(), Path(config).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f'Use a new output directory: {output_dir}')
    original = torch.load(checkpoint, map_location='cpu')
    if original.get('inference_optimization'):
        raise ValueError('Convert original training weights, not an already optimized checkpoint.')
    source_category = original.get('category')
    category = source_category if component == 'auto' else component
    if category not in ('acoustic', 'variance', 'all_in_one'):
        raise ValueError('Specify --component when the checkpoint has no category.')
    if source_category not in (None, category, 'all_in_one'):
        raise ValueError(f'Cannot extract {category} from {source_category}.')
    set_hparams(str(config), print_hparams=False)
    vocab = len(load_phoneme_dictionary())
    factory = {'acoustic': lambda: DiffSingerAcoustic(vocab, hparams['audio_num_mel_bins']),
               'variance': lambda: DiffSingerVariance(vocab),
               'all_in_one': lambda: DiffSingerAllInOne(vocab, hparams['audio_num_mel_bins'])}
    model = factory[category]().eval()
    load_ckpt(model, checkpoint, strict=True)
    original_state = model.state_dict()
    original_model_bytes = tensor_bytes(original_state)
    original_parameters = sum(p.numel() for p in model.parameters())
    group_models = []
    branches = [('acoustic', model.acoustic), ('variance', model.variance)] if category == 'all_in_one' else [('', model)]
    for branch_name, branch in branches:
        for name, module in branch.named_children():
            group_name = '.'.join(part for part in (branch_name, name) if part)
            group_models.append((group_name, sum(p.numel() for p in module.parameters())))
    prefix = 'model.' + (category + '.' if source_category == 'all_in_one' and category != 'all_in_one' else '')
    kept_original_keys = {prefix + k for k in original_state}
    removed_keys = sorted(set(original['state_dict']) - kept_original_keys)
    prepare_inference_model(model, precision)
    state = model.state_dict()
    removed_buffers = [prefix + k for k in original_state
                       if k.endswith(('.alphas_cumprod_prev', '.log_one_minus_alphas_cumprod'))]
    removed_keys = sorted(set(removed_keys + removed_buffers))
    int8_elements = sum(v.numel() for v in state.values() if v.dtype == torch.int8)
    storage_groups = {}
    for name, parameters in group_models:
        original_group = {k: v for k, v in original_state.items() if k.startswith(name + '.')}
        optimized_group = {k: v for k, v in state.items() if k.startswith(name + '.')}
        integer_count = sum(v.numel() for v in optimized_group.values() if v.dtype == torch.int8)
        storage_groups[name] = {'original_parameters': parameters,
                               'original_state_bytes': tensor_bytes(original_group),
                               'optimized_state_bytes': tensor_bytes(optimized_group),
                               'int8_elements': integer_count,
                               'int8_parameter_coverage': integer_count / max(1, parameters)}
    metadata = {'format_version': FORMAT_VERSION, 'precision': precision,
                'source_category': source_category, 'component': category,
                'quantize_small_parameters': precision == 'int8'}
    artifact = {'category': category, 'state_dict': {'model.' + k: v for k, v in state.items()},
                'inference_optimization': metadata}
    step = original.get('global_step', 0)
    match = re.fullmatch(r'model_ckpt_steps_(\d+)\.ckpt', checkpoint.name)
    if match:
        step = int(match.group(1))
    manifest = {**metadata, 'source_checkpoint': str(checkpoint),
                'source_file_bytes': checkpoint.stat().st_size,
                'original_selected_model_bytes': original_model_bytes,
                'optimized_model_bytes': tensor_bytes(state),
                'original_parameters': original_parameters,
                'int8_weight_elements': int8_elements,
                'int8_parameter_coverage': int8_elements / max(1, original_parameters),
                'module_storage': storage_groups,
                'remaining_float_parameters': {k: {'shape': list(p.shape), 'dtype': str(p.dtype)}
                                               for k, p in model.named_parameters() if p.is_floating_point()},
                'memory_budget_gib': {
                    'resident_state': tensor_bytes(state) / 2**30,
                    'estimated_runtime_low': tensor_bytes(state) / 2**30 + (1.5 if precision == 'int8' else 1),
                    'estimated_runtime_high': tensor_bytes(state) / 2**30 + (4 if precision == 'int8' else 3),
                    'note': 'Budget only: batch1/~768 frames, excludes vocoder/context/other processes; not measured peak.'},
                'removed_state_keys': removed_keys,
                'removed_training_buffer_keys': removed_buffers,
                'removed_checkpoint_fields': sorted(set(original) - {'category', 'state_dict'}),
                'torch_version': str(torch.__version__)}
    saved_config = copy.deepcopy(hparams)
    saved_config.update(base_config=[], work_dir=str(output_dir), infer=True)
    saved_config['all_in_one'] = {**saved_config.get('all_in_one', {}), 'enabled': category == 'all_in_one'}
    # Resolve source dictionary files before copying so output configs are portable.
    dictionaries = {}
    for language, path in saved_config.get('dictionaries', {}).items():
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        dictionaries[language] = source
    output_dir.mkdir(parents=True)
    for language, source in dictionaries.items():
        destination = output_dir / 'dictionaries' / f'{language}{source.suffix}'
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(source, destination)
        saved_config['dictionaries'][language] = str(destination)
    for filename in ('spk_map.json', 'lang_map.json'):
        source = config.parent / filename
        if source.exists():
            shutil.copy2(source, output_dir / filename)
    path = output_dir / f'model_ckpt_steps_{int(step)}.ckpt'
    torch.save(artifact, path)
    manifest['optimized_file_bytes'] = path.stat().st_size
    (output_dir / 'config.yaml').write_text(yaml.safe_dump(saved_config, allow_unicode=True), encoding='utf-8')
    (output_dir / 'optimization.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--config', type=Path, help='Defaults to config.yaml next to checkpoint.')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--precision', choices=('int8', 'fp16', 'fp32'), default='int8')
    parser.add_argument('--component', choices=('auto', 'acoustic', 'variance', 'all_in_one'), default='auto')
    args = parser.parse_args()
    optimize(args.checkpoint, args.config or args.checkpoint.parent / 'config.yaml',
             args.output_dir, args.precision, args.component)


if __name__ == '__main__':
    main()
