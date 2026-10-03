"""Write LibTorch modules using the same deployment stages as ONNX."""
import inspect
import json
from pathlib import Path

import torch
import yaml
from torch import nn


def describe(value):
    if torch.is_tensor(value):
        return {'type': 'Tensor', 'dtype': str(value.dtype), 'example_shape': list(value.shape)}
    if isinstance(value, dict):
        return {'type': 'Dict', 'items': {key: describe(item) for key, item in value.items()}}
    if isinstance(value, (tuple, list)):
        return {'type': 'Tuple', 'items': [describe(item) for item in value]}
    return {'type': type(value).__name__, 'example_value': value}


class FlatInputs(nn.Module):
    """Flatten tensor dictionaries into a positional LibTorch tensor interface."""
    def __init__(self, module, args, kwargs):
        super().__init__()
        self.module = module
        self.template = (args, kwargs)

    def forward(self, *inputs):
        iterator = iter(inputs)
        def rebuild(value):
            if torch.is_tensor(value):
                return next(iterator)
            if isinstance(value, dict):
                return {key: rebuild(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return tuple(rebuild(item) for item in value)
            return value
        args, kwargs = rebuild(self.template)
        return self.module(*args, **kwargs)


def flatten_tensors(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        return sum((flatten_tensors(item) for item in value.values()), [])
    if isinstance(value, (tuple, list)):
        return sum((flatten_tensors(item) for item in value), [])
    return []


class TorchScriptWriter:
    def __init__(self, output_dir, component):
        self.output_dir = Path(output_dir)
        self.component = component
        self.records = []

    def __call__(self, model, args, path, **options):
        compiled, inputs, reference = compile_stage(model, args)
        name = f'{self.component}.{Path(path).stem}.pt'
        destination = self.output_dir / name
        torch.jit.save(compiled, str(destination))
        # Reload before publishing the record; no Python implementation is saved.
        tensors = flatten_tensors(inputs)
        device = tensors[0].device if tensors else torch.device('cpu')
        loaded = torch.jit.load(str(destination), map_location=device).eval()
        with torch.inference_mode():
            torch.manual_seed(123)
            expected = reference(*inputs)
            torch.manual_seed(123)
            actual = loaded(*inputs)
        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)
        self.records.append({'file': name, 'forward_schema': str(loaded.forward.schema),
                             'input_names': options.get('input_names', []),
                             'output_names': options.get('output_names', []),
                             'arguments': [describe(value) for value in inputs],
                             'dynamic_axes': options.get('dynamic_axes', {}),
                             'outputs': describe(actual),
                             'operators': sorted(torch.jit.export_opnames(loaded))})
        print(f'| export TorchScript => {destination}')


def compile_stage(model, args, freeze=True):
    args = args if isinstance(args, tuple) else (args,)
    if isinstance(model, torch.jit.ScriptModule):
        return (torch.jit.freeze(model.eval()) if freeze else model), args, model
    kwargs = {}
    parameters = inspect.signature(model.forward).parameters
    if args and isinstance(args[-1], dict) and set(args[-1]).issubset(parameters):
        args, kwargs = args[:-1], args[-1]
    inputs = tuple(flatten_tensors((args, kwargs)))
    wrapped = FlatInputs(model, args, kwargs).eval()
    compiled = torch.jit.trace(wrapped, inputs, strict=False, check_trace=False)
    if freeze:
        compiled = torch.jit.freeze(compiled.eval())
    return compiled, inputs, wrapped


def export_torchscript_stages(exporter, output_dir, component):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = TorchScriptWriter(output_dir, component)
    exporter.graph_writer = writer
    try:
        exporter._torch_export_model()
    finally:
        del exporter.graph_writer
    exporter.export_attachments(output_dir)
    # The shared attachment writer targets ONNX hosts. Publish separate runtime
    # metadata rather than a dsconfig pointing at nonexistent ONNX files.
    onnx_config = output_dir / 'dsconfig.yaml'
    metadata = yaml.safe_load(onnx_config.read_text(encoding='utf-8'))
    metadata = {key: value for key, value in metadata.items()
                if not isinstance(value, str) or not value.endswith('.onnx')}
    metadata.update(runtime='torchscript', stages={Path(record['file']).stem: record['file']
                                                  for record in writer.records})
    (output_dir / 'metadata.yaml').write_text(yaml.safe_dump(metadata, sort_keys=False), encoding='utf-8')
    onnx_config.unlink()
    manifest = {'format': 'torchscript', 'component': component,
                'target_device': str(exporter.device),
                'precision': getattr(exporter.model, '_checkpoint_precision', 'fp32'),
                'torch_version': str(torch.__version__), 'modules': writer.records}
    (output_dir / f'{component}.torchscript.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest
