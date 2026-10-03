"""Exercise optimized checkpoint exports, reload TorchScript and run ONNX stages."""
import argparse
import copy
import json
import itertools
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_training_modes import configure
from modules.toplevel import DiffSingerAllInOne
from utils.hparams import hparams
from utils.phoneme_utils import load_phoneme_dictionary
from utils.checkpoint_optimization import prepare_inference_model
from deployment.exporters.acoustic_exporter import DiffSingerAcousticExporter
from deployment.exporters.variance_exporter import DiffSingerVarianceExporter
from deployment.exporters.torchscript_exporter import TorchScriptWriter, compile_stage


def tensor_outputs(value):
    if torch.is_tensor(value):
        return [value]
    return sum((tensor_outputs(v) for v in value), [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbones', nargs='+', default=['wavenet', 'lynxnet', 'lynxnet2', 'dit'])
    parser.add_argument('--objectives', nargs='+', default=['ddpm', 'reflow'])
    parser.add_argument('--precisions', nargs='+')
    parser.add_argument('--lora', action='store_true', help='Validate export from a full LoRA checkpoint after merging.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.precisions = args.precisions or (['fp32'] if args.lora else ['int8'])
    if args.lora and args.precisions != ['fp32']:
        parser.error('--lora validates original FP32 LoRA checkpoints; use fp32 precision.')
    cases = list(itertools.product(args.backbones, args.objectives, args.precisions))
    if len(cases) > 1:
        # Match the command-line deployment workflow: a fresh exporter process
        # per checkpoint/configuration, avoiding global JIT/ONNX cache reuse.
        records = []
        with tempfile.TemporaryDirectory(prefix='diffsinger-export-results-') as directory:
            for index, (kind, objective, precision) in enumerate(cases):
                result = Path(directory) / f'{index}.json'
                subprocess.run([sys.executable, str(Path(__file__)), *(['--lora'] if args.lora else []), '--backbones', kind,
                                '--objectives', objective, '--precisions', precision,
                                '--output', str(result)], check=True)
                records.extend(json.loads(result.read_text(encoding='utf-8')))
        if args.output:
            args.output.write_text(json.dumps(records, indent=2), encoding='utf-8')
        print(f'PASS {len(records)} component exports', flush=True)
        return
    torch.set_num_threads(2)
    records = []
    with tempfile.TemporaryDirectory(prefix='diffsinger-export-check-') as directory:
        root = Path(directory)
        for kind in args.backbones:
            for objective in args.objectives:
                for precision in args.precisions:
                    configure(kind, objective)
                    hparams['use_stretch_embed'] = True
                    hparams['shallow_diffusion_args']['aux_decoder_args'].update(num_channels=16, num_layers=1)
                    case = root / f'{kind}-{objective}-{precision}'
                    case.mkdir()
                    hparams.update(work_dir=str(case), exp_name=case.name, infer=True)
                    saved = copy.deepcopy(hparams)
                    saved['base_config'] = []
                    (case / 'config.yaml').write_text(yaml.safe_dump(saved), encoding='utf-8')
                    model = DiffSingerAllInOne(len(load_phoneme_dictionary()), 4).eval()
                    if not args.lora:
                        prepare_inference_model(model, precision)
                    if args.lora:
                        from utils.lora import inject_lora
                        inject_lora(model, rank=2, alpha=4)
                        with torch.no_grad():
                            for name, parameter in model.named_parameters():
                                if name.endswith('.lora_B'):
                                    parameter.normal_(std=0.02)
                    if precision == 'int8':
                        assert not any(p.is_floating_point() for p in model.parameters())
                    checkpoint = case / 'model_ckpt_steps_1.ckpt'
                    artifact = {'category': 'all_in_one',
                                'inference_optimization': {'format_version': 1, 'precision': precision,
                                                          'quantize_small_parameters': precision == 'int8'},
                                'state_dict': {'model.' + k: v for k, v in model.state_dict().items()}}
                    if args.lora:
                        from utils.lora import lora_metadata
                        artifact.pop('inference_optimization')
                        artifact['lora'] = lora_metadata(model)
                    torch.save(artifact, checkpoint)
                    del model
                    for component, factory in [('acoustic', DiffSingerAcousticExporter),
                                               ('variance', DiffSingerVarianceExporter)]:
                        device = torch.device('cuda' if precision == 'fp16' else 'cpu')
                        exporter = factory(device=device, cache_dir=case / component / 'cache',
                                           checkpoint_path=checkpoint)
                        output = case / component / 'output'
                        output.mkdir(parents=True)
                        writer = TorchScriptWriter(output, component)
                        stage_count = [0]
                        stage_inputs = {}

                        def both(model, inputs, path, **options):
                            writer(model, inputs, path, **options)
                            compiled, flat, reference = compile_stage(model, inputs, freeze=precision != 'fp16')
                            torch.onnx.export(compiled, flat, str(path), **options)
                            exporter.scope_onnx_graph(path)
                            onnx.checker.check_model(str(path))
                            if precision == 'fp16':
                                # This environment has only ORT CPU providers.
                                # FP16 targets CUDA; validate the graph structure
                                # and the CUDA TorchScript round trip above.
                                stage_count[0] += 1
                                return
                            session_options = ort.SessionOptions()
                            session_options.intra_op_num_threads = 2
                            if precision != 'int8':
                                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                            session = ort.InferenceSession(str(path), session_options,
                                                           providers=['CPUExecutionProvider'])
                            named = dict(zip(options['input_names'], flat))
                            feed = {v.name: (named[v.name].detach().cpu().numpy()
                                            if torch.is_tensor(named[v.name]) else
                                            np.asarray(named[v.name], dtype=np.int64))
                                    for v in session.get_inputs()}
                            stage_inputs[Path(path).stem] = feed
                            actual = session.run(None, feed)
                            assert all(np.isfinite(v).all() for v in actual)
                            with torch.inference_mode():
                                expected = tensor_outputs(reference(*flat))
                            assert [tuple(v.shape) for v in actual] == [tuple(v.shape) for v in expected]
                            if Path(path).stem not in ('diffusion', 'pitch', 'variance'):
                                for a, b in zip(actual, expected):
                                    np.testing.assert_allclose(a, b.detach().cpu().numpy(), rtol=.03, atol=.03)
                            if component == 'acoustic' and Path(path).stem in ('fs2_aux', 'fs2'):
                                dynamic = []
                                for name, value in zip(options['input_names'], flat):
                                    if name == 'durations':
                                        value = torch.tensor([[3, 3, 13]], device=device)
                                    elif name in ('tokens', 'languages'):
                                        value = value.repeat(1, 3)
                                    elif value.dim() >= 2 and value.size(1) == 10:
                                        repeat = [1] * value.dim()
                                        repeat[1] = 19
                                        value = value[:, :1].repeat(*repeat)
                                    dynamic.append(value)
                                mapping = dict(zip(options['input_names'], dynamic))
                                changed = session.run(None, {v.name: mapping[v.name].cpu().numpy()
                                                             for v in session.get_inputs()})
                                with torch.inference_mode():
                                    expected = tensor_outputs(reference(*dynamic))
                                    loaded = torch.jit.load(str(output / writer.records[-1]['file']), map_location=device)
                                    torch.testing.assert_close(loaded(*dynamic), reference(*dynamic))
                                for a, b in zip(changed, expected):
                                    np.testing.assert_allclose(a, b.cpu().numpy(), rtol=.03, atol=.03)
                            if Path(path).stem in ('diffusion', 'pitch', 'variance'):
                                loaded = torch.jit.load(str(output / writer.records[-1]['file']), map_location=device)
                                for steps in (1, 3):
                                    varied = dict(zip(options['input_names'], flat))
                                    varied['steps'] = steps
                                    if 'depth' in varied:
                                        varied['depth'] = torch.tensor(.75, device=device)
                                    inputs = tuple(varied[name] for name in options['input_names'])
                                    with torch.inference_mode():
                                        torch.manual_seed(321)
                                        expected = reference(*inputs)
                                        torch.manual_seed(321)
                                        torch.testing.assert_close(loaded(*inputs), expected, rtol=.003, atol=.003)
                                    changed = session.run(None, {v.name: (varied[v.name].cpu().numpy()
                                                                         if torch.is_tensor(varied[v.name]) else
                                                                         np.asarray(varied[v.name], dtype=np.int64))
                                                                for v in session.get_inputs()})
                                    assert all(np.isfinite(v).all() for v in changed)
                            stage_count[0] += 1

                        exporter.graph_writer = both
                        exporter.export(output)
                        for path in output.glob('*.onnx'):
                            onnx.checker.check_model(str(path))
                            if precision == 'fp16':
                                continue
                            session_options = ort.SessionOptions()
                            session_options.intra_op_num_threads = 2
                            if precision != 'int8':
                                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                            session = ort.InferenceSession(str(path), session_options, providers=['CPUExecutionProvider'])
                            suffix = path.stem.split('.')[-1]
                            names = (('pitch_pre', 'pitch', 'pitch_post') if suffix == 'pitch' else
                                     ('variance_pre', 'variance', 'variance_post') if suffix == 'variance' else
                                     (suffix,) if component == 'variance' else ('fs2_aux', 'fs2', 'diffusion'))
                            public_inputs = {key: value for name in names
                                             for key, value in stage_inputs.get(name, {}).items()}
                            actual = session.run(None, {v.name: public_inputs[v.name] for v in session.get_inputs()})
                            assert all(np.isfinite(v).all() for v in actual)
                        records.append({'backbone': kind, 'objective': objective, 'precision': precision,
                                        'lora_merged': args.lora,
                                        'onnx_execution': precision != 'fp16',
                                        'ort_optimization': ('not_run' if precision == 'fp16' else
                                                             'default' if precision == 'int8' else 'disabled'),
                                        'component': component, 'stages': stage_count[0]})
                        print('PASS EXPORT', records[-1], flush=True)
    if args.output:
        args.output.write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(f'PASS {len(records)} component exports', flush=True)


if __name__ == '__main__':
    main()
