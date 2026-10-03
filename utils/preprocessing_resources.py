"""Device selection and bounded GPU concurrency for dataset preprocessing."""

import gc
import sys

import torch


def preprocessing_device(config):
    value = config.get('device', 'auto')
    if value == 'auto':
        value = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(value)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('binarization_args.device must be auto, cpu, or cuda[:index].')
    return device


def preprocessing_worker_devices(config, num_workers, device):
    if num_workers < 0:
        raise ValueError('binarization_args.num_workers must be non-negative.')
    if num_workers == 0 or torch.device(device).type == 'cpu':
        return num_workers, None
    per_gpu = config.get('num_workers_per_gpu')
    if per_gpu is None:
        per_gpu = num_workers if config.get('workers_per_gpu', False) else 1
    if type(per_gpu) is not int or per_gpu < 1:
        raise ValueError('binarization_args.num_workers_per_gpu must be a positive integer.')
    index = torch.device(device).index
    devices = [index] if index is not None else list(range(torch.cuda.device_count()))
    assignments = [gpu for gpu in devices for _ in range(per_gpu)]
    # The explicit per-GPU option follows the reference project's worker-count semantics.
    if 'num_workers_per_gpu' not in config and not config.get('workers_per_gpu', False):
        assignments = assignments[:num_workers]
    return len(assignments), assignments


def release_preprocessing_cache(device):
    from utils.hparams import hparams
    if torch.device(device).type == 'cuda' and hparams.get('binarization_args', {}).get('clear_cuda_cache', True):
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def release_binarizer_resources(device=None):
    # Global operators work around PyTorch's Windows shared-memory limitation.
    # Drop live models too: empty_cache alone cannot release their allocations.
    for name in ('preprocessing.acoustic_binarizer', 'preprocessing.variance_binarizer'):
        module = sys.modules.get(name)
        if module is not None:
            for attribute in ('pitch_extractor', 'midi_smooth', 'energy_smooth',
                              'breathiness_smooth', 'voicing_smooth', 'tension_smooth'):
                if hasattr(module, attribute):
                    setattr(module, attribute, None)
    gc.collect()
    if (device is None or torch.device(device).type == 'cuda') and torch.cuda.is_available():
        from utils.hparams import hparams
        if hparams.get('binarization_args', {}).get('clear_cuda_cache', True):
            torch.cuda.empty_cache()
