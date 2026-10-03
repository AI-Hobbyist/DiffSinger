"""Native inference checkpoint precision conversion (not ONNX quantization).

INT8 matrices remain integer buffers throughout inference. Float activations
are quantized per row immediately before integer GEMM; only its accumulator
and embedding lookup outputs are converted to floating point.
"""
import functools
import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize


FORMAT_VERSION = 1


class Int8VectorScale(nn.Module):
    """Integer storage for small affine vectors consumed by float operators."""
    def __init__(self, scale):
        super().__init__()
        self.register_buffer('scale', scale)

    def forward(self, value):
        return value.float() * self.scale


def scaled_bias(bias: Optional[torch.Tensor], scale: Optional[torch.Tensor]):
    if bias is None:
        result = torch.tensor(0., dtype=torch.float32)
    elif scale is None:
        result = bias.float()
    else:
        result = bias.float() * scale
    return result


def quantize_small_parameters(model):
    # Matrix kernels already store their large weights as INT8. Quantize small
    # affine vectors too, without ever reconstructing a whole float matrix.
    for module in list(model.modules()):
        for name, value in list(module.named_parameters(recurse=False)):
            if not value.is_floating_point():
                continue
            if value.ndim > 1:
                raise ValueError(f'Unsupported remaining matrix: {type(module).__name__}.{name}')
            scale = value.detach().float().abs().amax().clamp_min(1e-12) / 127
            q = (value.detach().float() / scale).round().clamp(-127, 127).to(torch.int8)
            value.requires_grad_(False)
            parametrize.register_parametrization(module, name, Int8VectorScale(scale), unsafe=True)
            getattr(module.parametrizations, name).original.data = q
        if isinstance(module, (Int8Linear, Int8Conv1d)) and module.bias is not None:
            scale = module.bias.float().abs().amax().clamp_min(1e-12) / 127
            module.bias = (module.bias.float() / scale).round().clamp(-127, 127).to(torch.int8)
            module.bias_scale = scale


def quantize_rows(weight):
    flat = weight.detach().float().reshape(weight.shape[0], -1)
    scale = flat.abs().amax(1).clamp_min(1e-12) / 127
    return (flat / scale[:, None]).round().clamp(-127, 127).to(torch.int8), scale


def integer_linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
                   bias: torch.Tensor, chunk_size: int = 256):
    """Bound scratch space, including for long utterances and wide networks."""
    rows = x.reshape(-1, x.size(-1)).float()
    outputs = []
    # CUDA _int_mm requires aligned matrices and a minimum row count.
    reduction = ((weight.size(1) + 31) // 32) * 32
    columns = ((weight.size(0) + 31) // 32) * 32
    w = F.pad(weight, (0, reduction - weight.size(1), 0, columns - weight.size(0)))
    transposed_weight = w.t().contiguous()
    for block in rows.split(chunk_size):
        s = block.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127
        q = (block / s).round().clamp(-127, 127).to(torch.int8)
        count = ((q.size(0) + 31) // 32) * 32
        q = F.pad(q, (0, reduction - q.size(1), 0, count - q.size(0)))
        y = torch._int_mm(q.contiguous(), transposed_weight)
        y = y[:block.size(0), :weight.size(0)].float() * s * scale
        outputs.append(y + bias)
    result = torch.cat(outputs) if outputs else x.new_empty((0, weight.size(0)), dtype=torch.float32)
    if x.dim() == 1:
        shaped = result.reshape(weight.size(0))
    elif x.dim() == 2:
        shaped = result.reshape(x.size(0), weight.size(0))
    elif x.dim() == 3:
        shaped = result.reshape(x.size(0), x.size(1), weight.size(0))
    else:
        shaped = result.reshape(x.size(0), x.size(1), x.size(2), weight.size(0))
    return shaped


class Int8Linear(nn.Module):
    compute_dtype = torch.float32

    def __init__(self, source):
        super().__init__()
        self.in_features, self.out_features = source.in_features, source.out_features
        q, s = quantize_rows(source.weight)
        self.register_buffer('qweight', q)
        self.register_buffer('weight_scale', s)
        self.register_buffer('bias', None if source.bias is None else source.bias.detach().float())
        self.register_buffer('bias_scale', None)

    def forward(self, x):
        return integer_linear(x, self.qweight, self.weight_scale, scaled_bias(self.bias, self.bias_scale))


class Int8Conv1d(nn.Module):
    def __init__(self, source):
        super().__init__()
        if isinstance(source.padding, str) or source.padding_mode != 'zeros':
            raise ValueError('INT8 Conv1d requires numeric zero padding.')
        self.in_channels, self.out_channels = source.in_channels, source.out_channels
        self.groups = source.groups
        self.kernel_size, self.stride = source.kernel_size[0], source.stride[0]
        self.padding, self.dilation = source.padding[0], source.dilation[0]
        q, s = quantize_rows(source.weight)
        self.register_buffer('qweight', q)
        self.register_buffer('weight_scale', s)
        self.register_buffer('bias', None if source.bias is None else source.bias.detach().float())
        self.register_buffer('bias_scale', None)

    def forward(self, x):
        bias = scaled_bias(self.bias, self.bias_scale).to(device=x.device)
        padded = F.pad(x, (self.padding, self.padding))
        extent = self.dilation * (self.kernel_size - 1) + 1
        frames = (padded.size(2) - extent) // self.stride + 1
        frame_outputs = []
        inputs, outputs = self.in_channels // self.groups, self.out_channels // self.groups
        # Gather at most 256 output frames, including for depthwise kernels.
        # index_select materializes data; gathering the entire utterance first
        # would defeat the bounded integer GEMM scratch space.
        for start in range(0, frames, 256):
            count = min(256, frames - start)
            positions = (torch.arange(start, start + count, device=x.device)[:, None] * self.stride
                         + torch.arange(self.kernel_size, device=x.device)[None, :] * self.dilation)
            windows = padded.index_select(2, positions.flatten()).reshape(
                x.size(0), self.in_channels, count, self.kernel_size)
            if self.groups == self.in_channels == self.out_channels:
                scale = windows.float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127
                q = (windows / scale).round().clamp(-127, 127).to(torch.int8)
                acc = (q.int() * self.qweight[None, :, None, :].int()).sum(-1)
                y = acc.float() * scale.squeeze(-1) * self.weight_scale[None, :, None]
                frame_outputs.append(y + bias.reshape(1, -1, 1))
            else:
                groups = []
                for group in range(self.groups):
                    first, last = group * outputs, (group + 1) * outputs
                    window = windows[:, group * inputs:(group + 1) * inputs]
                    rows = window.permute(0, 2, 1, 3).reshape(-1, inputs * self.kernel_size)
                    y = integer_linear(rows, self.qweight[first:last], self.weight_scale[first:last],
                                       bias if bias.dim() == 0 else bias[first:last])
                    groups.append(y.reshape(x.size(0), count, outputs).transpose(1, 2))
                frame_outputs.append(torch.cat(groups, dim=1))
        return torch.cat(frame_outputs, dim=2)


class Int8Embedding(nn.Module):
    def __init__(self, source):
        super().__init__()
        if source.max_norm is not None:
            raise ValueError('INT8 Embedding does not support max_norm mutation.')
        self.num_embeddings, self.embedding_dim = source.num_embeddings, source.embedding_dim
        self.padding_idx = source.padding_idx
        q, s = quantize_rows(source.weight)
        self.register_buffer('qweight', q)
        self.register_buffer('weight_scale', s)

    def forward(self, indices):
        # Scale selected outputs only; never rebuild a floating embedding table.
        rows = F.embedding(indices, self.qweight)
        scales = F.embedding(indices, self.weight_scale.unsqueeze(-1))
        return rows.float() * scales


class Int8GRU(nn.Module):
    """Evaluation GRU with integer gate projections; no float matrix fallback."""
    def __init__(self, source):
        super().__init__()
        self.hidden_size, self.num_layers = source.hidden_size, source.num_layers
        self.batch_first, self.bidirectional = source.batch_first, source.bidirectional
        self.gates = nn.ModuleDict()
        for layer in range(source.num_layers):
            for suffix in ('', '_reverse') if source.bidirectional else ('',):
                for kind in ('ih', 'hh'):
                    name = f'{kind}_l{layer}{suffix}'
                    weight = getattr(source, f'weight_{name}')
                    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=source.bias,
                                       device=weight.device)
                    linear.weight.data.copy_(weight)
                    if source.bias:
                        linear.bias.data.copy_(getattr(source, f'bias_{name}'))
                    self.gates[name] = Int8Linear(linear)

    def flatten_parameters(self):
        pass

    def forward(self, x, hx=None):
        if x.ndim != 3:
            raise ValueError('INT8 GRU expects a batched dense 3D tensor.')
        x = x if self.batch_first else x.transpose(0, 1)
        directions = 2 if self.bidirectional else 1
        if hx is None:
            hx = x.new_zeros(self.num_layers * directions, x.shape[0], self.hidden_size)
        states = []
        for layer in range(self.num_layers):
            sequences = []
            for direction in range(directions):
                suffix = '_reverse' if direction else ''
                h = hx[layer * directions + direction].float()
                ys = []
                frames = x.flip(1) if direction else x
                for frame in frames.unbind(1):
                    ir, iz, inn = self.gates[f'ih_l{layer}{suffix}'](frame).chunk(3, -1)
                    hr, hz, hn = self.gates[f'hh_l{layer}{suffix}'](h).chunk(3, -1)
                    r, z = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
                    n = torch.tanh(inn + r * hn)
                    h = (1 - z) * n + z * h
                    ys.append(h)
                seq = torch.stack(ys, 1)
                sequences.append(seq.flip(1) if direction else seq)
                states.append(h)
            x = torch.cat(sequences, -1)
        return (x if self.batch_first else x.transpose(0, 1)), torch.stack(states)


def convert_int8(model):
    replacements = {nn.Linear: Int8Linear, nn.Conv1d: Int8Conv1d,
                    nn.Embedding: Int8Embedding, nn.GRU: Int8GRU}
    for name, child in list(model.named_children()):
        for base, replacement in replacements.items():
            if isinstance(child, base):
                setattr(model, name, replacement(child).eval())
                break
        else:
            convert_int8(child)
    # Unsupported *matrices* must not silently remain floating point.
    for name, parameter in model.named_parameters(recurse=False):
        if parameter.ndim > 1:
            raise ValueError(f'Unsupported INT8 matrix: {type(model).__name__}.{name}')
        parameter.requires_grad_(False)
    return model


def prepare_inference_model(model, precision, for_export=False, quantize_vectors=True):
    if precision not in ('int8', 'fp16', 'fp32'):
        raise ValueError(f'Unknown checkpoint precision: {precision}')
    if model.training:
        raise ValueError('Inference-only checkpoints require model.eval(); they cannot resume training.')
    if getattr(model, '_checkpoint_precision', None) is not None:
        raise ValueError('Load an optimized checkpoint into a fresh model.')
    from modules.core.ddpm import GaussianDiffusion
    for child in model.modules():
        if isinstance(child, GaussianDiffusion):
            # prev is only used to construct other buffers; log is only read by
            # the training q_mean_variance helper. Keep all sampler buffers.
            for name in ('alphas_cumprod_prev', 'log_one_minus_alphas_cumprod'):
                child._buffers.pop(name, None)
    if precision == 'int8':
        convert_int8(model)
        if quantize_vectors:
            quantize_small_parameters(model)
    else:
        dtype = torch.float16 if precision == 'fp16' else torch.float32
        # Keep schedule and positional buffers at their original stable precision.
        for parameter in model.parameters():
            parameter.data = parameter.data.to(dtype)
            parameter.requires_grad_(False)
        if precision == 'fp16' and not for_export:
            # Residuals and positional encodings may promote activations to
            # FP32. CPU layer_norm cannot mix those with FP16 affine weights.
            for child in model.modules():
                if isinstance(child, (nn.LayerNorm, nn.PReLU)):
                    def cast_normalization_input(module, args):
                        return (args[0].to(torch.float16), *args[1:])
                    child.register_forward_pre_hook(cast_normalization_input)
    def install_guard(module):
        original_forward = module.forward
        @functools.wraps(original_forward)
        def forward(*args, **kwargs):
            if module.training or kwargs.get('infer') is False:
                raise ValueError('This checkpoint supports inference only.')
            device = next(module.buffers(), None)
            if device is None:
                device = next(module.parameters())
            with torch.autocast(device_type=device.device.type, dtype=torch.float16,
                                enabled=precision == 'fp16'):
                return original_forward(*args, **kwargs)
        module.forward = forward
        module._checkpoint_precision = precision
    if for_export:
        model._checkpoint_precision = precision
        return model
    install_guard(model)
    if getattr(model, 'category', None) == 'all_in_one':
        install_guard(model.acoustic)
        install_guard(model.variance)
    return model


def load_optimized_state(model, state, metadata):
    if metadata.get('format_version') != FORMAT_VERSION:
        raise ValueError('Unsupported optimized checkpoint format version.')
    for_export = model.__class__.__module__.startswith('deployment.')
    # Older INT8 checkpoints kept small affine vectors in FP32. Preserve their
    # format on load; new converters mark expanded quantization explicitly.
    quantize_vectors = metadata.get('quantize_small_parameters',
                                   any('.parametrizations.' in key for key in state))
    prepare_inference_model(model, metadata['precision'], for_export=for_export,
                            quantize_vectors=quantize_vectors)
    model.load_state_dict(state, strict=True)
    if for_export:
        from deployment.modules.quantization import prepare_portable_modules, register_integer_onnx_symbolics
        prepare_portable_modules(model, metadata['precision'])
        if metadata['precision'] == 'int8':
            register_integer_onnx_symbolics()


def tensor_bytes(state):
    return sum(t.numel() * t.element_size() for t in state.values())
