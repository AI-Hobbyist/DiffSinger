"""Synthetic CUDA benchmark; random weights measure runtime, not singing quality."""
import argparse
import copy
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.toplevel import DiffSingerAllInOne
from scripts.validate_training_modes import configure
from scripts.validate_checkpoint_optimization import infer
from utils.checkpoint_optimization import prepare_inference_model, tensor_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=5)
    args = parser.parse_args()
    if args.iterations < 1 or not torch.cuda.is_available():
        raise ValueError('CUDA and positive --iterations are required.')
    torch.set_num_threads(2)
    results = []
    for kind in ('wavenet', 'lynxnet', 'lynxnet2', 'dit'):
        configure(kind, 'reflow')
        torch.manual_seed(123)
        base = DiffSingerAllInOne(65, 4).eval()
        for precision in ('fp32', 'fp16', 'int8'):
            model = prepare_inference_model(copy.deepcopy(base), precision).cuda()
            with torch.inference_mode():
                for _ in range(3):
                    infer(model, 'cuda')
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                timings = []
                for _ in range(args.iterations):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    infer(model, 'cuda')
                    torch.cuda.synchronize()
                    timings.append((time.perf_counter() - start) * 1000)
            record = {'backbone': kind, 'precision': precision,
                      'weight_mib': tensor_bytes(model.state_dict()) / 2**20,
                      'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                      'median_ms': statistics.median(timings)}
            results.append(record)
            print(record, flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    output = {'device': torch.cuda.get_device_name(), 'torch': str(torch.__version__),
              'configuration': 'synthetic all-in-one, hidden16, backbone L1 D16, mel4, frames4, '
                               'Reflow Euler 1 step, batch1, shallow decoder enabled, no vocoder',
              'warmup': 3, 'iterations': args.iterations, 'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
