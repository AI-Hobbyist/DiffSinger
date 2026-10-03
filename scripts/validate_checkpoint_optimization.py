"""Validate native optimized checkpoints on all backbones and diffusion objectives."""
import copy
import json
import sys
import tempfile
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.toplevel import DiffSingerAllInOne, DiffSingerAcoustic, DiffSingerVariance
from scripts.validate_training_modes import configure
from scripts.validate_all_in_one import acoustic_sample, variance_sample
from scripts.optimize_checkpoint import optimize
from utils import load_ckpt
from utils.checkpoint_optimization import prepare_inference_model, Int8Linear, Int8Conv1d, Int8GRU
from utils.hparams import hparams
from utils.phoneme_utils import load_phoneme_dictionary
from inference.ds_acoustic import DiffSingerAcousticInfer
from inference.ds_variance import DiffSingerVarianceInfer


def infer(model, device):
    acoustic = {k: v.to(device) for k, v in acoustic_sample().items() if torch.is_tensor(v)}
    acoustic.pop('mel')
    acoustic['txt_tokens'] = acoustic.pop('tokens')
    variance = {k: v.to(device) for k, v in variance_sample().items() if torch.is_tensor(v)}
    variance.pop('ph_dur')
    variance.pop('pitch')
    variance['txt_tokens'] = variance.pop('tokens')
    variance['word_dur'] = torch.tensor([[4, 0]], device=device)
    with torch.inference_mode():
        mel = model.acoustic(**acoustic, infer=True).diff_out
        dur, pitch, curves = model.variance(**variance, infer=True)
    values = [mel, dur, pitch, *curves.values()]
    assert all(torch.isfinite(value).all() for value in values if value is not None)
    return values


def validate_operators(device):
    torch.manual_seed(17)
    for module, quantized, shape in (
        (torch.nn.Linear(37, 19), Int8Linear, (2, 7, 37)),
        (torch.nn.Conv1d(4, 6, 3, padding=2, dilation=2), Int8Conv1d, (2, 4, 9)),
        (torch.nn.Conv1d(4, 4, 3, padding=1, groups=4), Int8Conv1d, (2, 4, 9)),
        (torch.nn.Conv1d(4, 6, 3, padding=1, groups=2), Int8Conv1d, (2, 4, 9)),
        (torch.nn.Conv1d(4, 6, 3, padding=2, dilation=2), Int8Conv1d, (2, 4, 513)),
        (torch.nn.Conv1d(4, 4, 7, padding=3, groups=4), Int8Conv1d, (2, 4, 513)),
        (torch.nn.Conv1d(4, 6, 3, padding=1, groups=2), Int8Conv1d, (2, 4, 513)),
        (torch.nn.GRU(7, 9, 2, batch_first=True, bidirectional=True), Int8GRU, (2, 5, 7)),
    ):
        module = module.eval().to(device)
        x = torch.randn(shape, device=device)
        with torch.inference_mode():
            expected, actual = module(x), quantized(module).to(device)(x)
        if isinstance(expected, tuple):
            for a, b in zip(expected, actual):
                torch.testing.assert_close(a, b, atol=0.025, rtol=0.06)
        else:
            torch.testing.assert_close(expected, actual, atol=0.04, rtol=0.06)
        if quantized is Int8Conv1d:
            scripted = torch.jit.script(quantized(module).to(device).eval())
            torch.testing.assert_close(scripted(x), actual, rtol=0, atol=0)


def main():
    torch.set_num_threads(2)
    devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    results = []
    for device in devices:
        validate_operators(device)
        for kind in ('wavenet', 'lynxnet', 'lynxnet2', 'dit'):
            for objective in ('ddpm', 'reflow'):
                configure(kind, objective)
                hparams['use_stretch_embed'] = True
                base = DiffSingerAllInOne(65, 4).eval()
                for precision in ('fp32', 'fp16', 'int8'):
                    model = copy.deepcopy(base).to(device)
                    prepare_inference_model(model, precision)
                    start = time.perf_counter()
                    infer(model, device)
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    results.append({'device': device, 'backbone': kind, 'objective': objective,
                                    'precision': precision, 'seconds': time.perf_counter() - start})
                    print('PASS', results[-1], flush=True)
    # Actual converter + native strict loader, including branch extraction.
    configure('dit', 'reflow')
    vocab = len(load_phoneme_dictionary())
    base = DiffSingerAllInOne(vocab, 4).eval()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = root / 'config.yaml'
        saved = copy.deepcopy(hparams)
        saved['base_config'] = []
        config.write_text(yaml.safe_dump(saved), encoding='utf-8')
        checkpoint = root / 'model_ckpt_steps_10.ckpt'
        torch.save({'category': 'all_in_one', 'global_step': 10,
                    'optimizer_states': [{'unused': torch.ones(100)}],
                    'state_dict': {'model.' + k: v for k, v in base.state_dict().items()}}, checkpoint)
        for precision in ('fp32', 'fp16', 'int8'):
            path = optimize(checkpoint, config, root / precision, precision)
            if precision == 'int8':
                manifest = json.loads((path.parent / 'optimization.json').read_text(encoding='utf-8'))
                assert manifest['int8_parameter_coverage'] == 1.0
                assert manifest['remaining_float_parameters'] == {}
                assert all(group['int8_parameter_coverage'] == 1.0
                           for group in manifest['module_storage'].values() if group['original_parameters'])
            fresh = DiffSingerAllInOne(vocab, 4).eval()
            load_ckpt(fresh, path)
            infer(fresh, 'cpu')
            if precision == 'fp32':
                torch.manual_seed(123)
                expected = infer(base, 'cpu')
                torch.manual_seed(123)
                actual = infer(fresh, 'cpu')
                for a, b in zip(expected, actual):
                    if a is not None:
                        torch.testing.assert_close(a, b, atol=0, rtol=0)
            assert all(torch.equal(v, fresh.state_dict()[k]) for k, v in
                       ((k.removeprefix('model.'), v) for k, v in torch.load(path)['state_dict'].items()))
            # Joint checkpoints can also load directly through standalone infer classes.
            for factory in (lambda: DiffSingerAcoustic(vocab, 4), lambda: DiffSingerVariance(vocab)):
                load_ckpt(factory().eval(), path)
            hparams['work_dir'] = str(path.parent)
            acoustic_infer = DiffSingerAcousticInfer(device='cpu', load_vocoder=False)
            variance_infer = DiffSingerVarianceInfer(device='cpu', predictions=set())
            infer(type('Models', (), {'acoustic': acoustic_infer.model, 'variance': variance_infer.model})(), 'cpu')
            try:
                load_ckpt(DiffSingerAllInOne(vocab, 4), path)
            except ValueError as error:
                assert 'inference' in str(error).lower()
            else:
                raise AssertionError('Training loaded an inference-only artifact.')
        for component in ('acoustic', 'variance'):
            path = optimize(checkpoint, config, root / component, 'int8', component)
            artifact = torch.load(path)
            assert artifact['category'] == component and 'optimizer_states' not in artifact
            factory = (lambda: DiffSingerAcoustic(vocab, 4)) if component == 'acoustic' else (lambda: DiffSingerVariance(vocab))
            load_ckpt(factory().eval(), path)
    print(json.dumps({'checks': len(results), 'results': results}, indent=2))


if __name__ == '__main__':
    main()
