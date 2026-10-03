"""Export original or optimized DiffSinger checkpoints for LibTorch."""
import argparse
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deployment.exporters.acoustic_exporter import DiffSingerAcousticExporter
from deployment.exporters.variance_exporter import DiffSingerVarianceExporter
from deployment.exporters.torchscript_exporter import export_torchscript_stages
from utils.hparams import hparams, set_hparams


def export(checkpoint, config, output_dir, component='auto', device='cpu'):
    checkpoint, config, output_dir = Path(checkpoint).resolve(), Path(config).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f'Use a new output directory: {output_dir}')
    metadata = torch.load(checkpoint, map_location='cpu')
    category = metadata.get('category') if component == 'auto' else component
    if category not in ('acoustic', 'variance', 'all_in_one'):
        raise ValueError('Specify --component for checkpoints without category metadata.')
    precision = metadata.get('inference_optimization', {}).get('precision', 'fp32')
    del metadata
    if device == 'cpu' and precision == 'fp16':
        raise ValueError('For CPU deployment select FP32 or INT8; export FP16 with --device cuda.')
    set_hparams(str(config), print_hparams=False)
    hparams.update(work_dir=str(config.parent), exp_name=checkpoint.parent.name, infer=True)
    components = ('acoustic', 'variance') if category == 'all_in_one' else (category,)
    manifests = []
    with tempfile.TemporaryDirectory(prefix='diffsinger-torchscript-') as directory:
        for name in components:
            factory = DiffSingerAcousticExporter if name == 'acoustic' else DiffSingerVarianceExporter
            exporter = factory(device=torch.device(device), cache_dir=Path(directory) / name,
                               checkpoint_path=checkpoint)
            manifests.append(export_torchscript_stages(exporter, output_dir / name, name))
    return manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--component', choices=('auto', 'acoustic', 'variance', 'all_in_one'), default='auto')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    export(args.checkpoint, args.config or args.checkpoint.parent / 'config.yaml',
           args.output_dir, args.component, args.device)


if __name__ == '__main__':
    main()
