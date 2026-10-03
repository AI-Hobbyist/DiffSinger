"""Portable precision modules shared by ONNX and LibTorch exports."""
from typing import Optional, List

import torch
from torch import nn, Tensor

from utils.checkpoint_optimization import Int8Linear, Int8Conv1d, Int8Embedding, Int8GRU


class IntegerGRUDirection(nn.Module):
    def __init__(self, ih, hh, reverse):
        super().__init__()
        self.ih, self.hh = ih, hh
        self.reverse = reverse

    def forward(self, x: Tensor, h: Tensor):
        frames = x.flip(1) if self.reverse else x
        outputs = torch.jit.annotate(List[Tensor], [])
        for index in range(frames.shape[1]):
            ir, iz, inn = self.ih(frames[:, index]).chunk(3, -1)
            hr, hz, hn = self.hh(h).chunk(3, -1)
            r, z = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
            n = torch.tanh(inn + r * hn)
            h = (1 - z) * n + z * h
            outputs.append(h)
        sequence = torch.stack(outputs, 1)
        return (sequence.flip(1) if self.reverse else sequence), h


class IntegerGRULayer(nn.Module):
    def __init__(self, source, layer):
        super().__init__()
        self.directions = nn.ModuleList([
            IntegerGRUDirection(source.gates[f'ih_l{layer}{suffix}'],
                                source.gates[f'hh_l{layer}{suffix}'], bool(suffix))
            for suffix in (('', '_reverse') if source.bidirectional else ('',))
        ])

    def forward(self, x: Tensor, states: Tensor):
        sequences = torch.jit.annotate(List[Tensor], [])
        outputs = torch.jit.annotate(List[Tensor], [])
        for index, direction in enumerate(self.directions):
            sequence, state = direction(x, states[index])
            sequences.append(sequence)
            outputs.append(state)
        return torch.cat(sequences, -1), torch.stack(outputs)


class IntegerGRU(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.hidden_size, self.num_layers = source.hidden_size, source.num_layers
        self.batch_first = source.batch_first
        self.num_directions = 2 if source.bidirectional else 1
        self.layers = nn.ModuleList([IntegerGRULayer(source, i) for i in range(source.num_layers)])

    @torch.jit.export
    def flatten_parameters(self):
        pass

    def forward(self, x: Tensor, hx: Optional[Tensor] = None):
        x = x if self.batch_first else x.transpose(0, 1)
        if hx is None:
            hx = x.new_zeros(self.num_layers * self.num_directions, x.shape[0], self.hidden_size)
        states = torch.jit.annotate(List[Tensor], [])
        for index, layer in enumerate(self.layers):
            start = index * self.num_directions
            x, state = layer(x, hx[start:start + self.num_directions])
            states.append(state)
        return (x if self.batch_first else x.transpose(0, 1)), torch.cat(states)


class HalfCompute(nn.Module):
    """Preserve half parameters while exposing float activation interfaces."""
    def __init__(self, source):
        super().__init__()
        self.layer = source

    def forward(self, x):
        return self.layer(x.half()).float()


class HalfGRU(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.layer = source

    def flatten_parameters(self):
        self.layer.flatten_parameters()

    def forward(self, x, hx=None):
        x, h = self.layer(x.half(), None if hx is None else hx.half())
        return x.float(), h.float()


class HalfLayerNorm(nn.Module):
    """Accumulate normalization in FP32 while retaining half source vectors."""
    def __init__(self, source):
        super().__init__()
        self.layer = source
        self.dim = getattr(source, 'dim', -1)

    def forward(self, x):
        if self.dim != -1:
            x = x.transpose(1, -1)
        x = torch.nn.functional.layer_norm(
            x.float(), self.layer.normalized_shape,
            None if self.layer.weight is None else self.layer.weight.float(),
            None if self.layer.bias is None else self.layer.bias.float(), self.layer.eps)
        return x if self.dim == -1 else x.transpose(1, -1)


class RankedIntegerLinear(nn.Module):
    """Trace rank-specific views while retaining dynamic scripted GEMM loops."""
    compute_dtype = torch.float32

    def __init__(self, source):
        super().__init__()
        self.in_features, self.out_features = source.in_features, source.out_features
        self.kernel = torch.jit.script(source.eval())

    def forward(self, x):
        y = self.kernel(x).float()
        if x.dim() == 2:
            return y.reshape(x.shape[0], self.out_features)
        if x.dim() == 3:
            return y.reshape(x.shape[0], x.shape[1], self.out_features)
        return y.reshape(*x.shape[:-1], self.out_features)


class RankedIntegerConv(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.kernel = torch.jit.script(source.eval())
        self.out_channels = source.out_channels
        self.padding, self.stride = source.padding, source.stride
        self.dilation, self.kernel_size = source.dilation, source.kernel_size

    def forward(self, x):
        frames = (x.shape[2] + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        return self.kernel(x).float().reshape(x.shape[0], self.out_channels, frames)


def prepare_portable_modules(model, precision, compile_integer=False):
    for name, child in list(model.named_children()):
        if isinstance(child, torch.jit.ScriptModule):
            continue
        if isinstance(child, (RankedIntegerLinear, RankedIntegerConv)):
            continue
        if precision == 'int8' and isinstance(child, Int8Linear):
            if compile_integer:
                setattr(model, name, RankedIntegerLinear(child).eval())
        elif precision == 'int8' and isinstance(child, Int8Conv1d):
            if compile_integer:
                setattr(model, name, RankedIntegerConv(child).eval())
        elif precision == 'int8' and isinstance(child, Int8Embedding):
            if compile_integer:
                setattr(model, name, torch.jit.script(child.eval()))
        elif precision == 'int8' and isinstance(child, Int8GRU):
            gru = IntegerGRU(child).eval()
            setattr(model, name, torch.jit.script(gru) if compile_integer else gru)
        elif precision == 'int8' and isinstance(child, IntegerGRU):
            if compile_integer:
                setattr(model, name, torch.jit.script(child.eval()))
        elif precision == 'fp16' and isinstance(child, nn.LayerNorm):
            setattr(model, name, HalfLayerNorm(child).eval())
        elif precision == 'fp16' and isinstance(child, (nn.Linear, nn.Conv1d, nn.PReLU)):
            setattr(model, name, HalfCompute(child).eval())
        elif precision == 'fp16' and isinstance(child, nn.GRU):
            setattr(model, name, HalfGRU(child).eval())
        else:
            prepare_portable_modules(child, precision, compile_integer=compile_integer)


def register_integer_onnx_symbolics():
    from torch.onnx._internal.torchscript_exporter import symbolic_opset11
    def int_mm(graph, a, b):
        # ORT CPU supports U8/S8 MatMulInteger. Shift signed activation bytes
        # by 128 and declare that zero point; the integer dot product is exact.
        a = graph.op('Cast', a, to_i=6)
        a = graph.op('Add', a, graph.op('Constant', value_t=torch.tensor(128, dtype=torch.int32)))
        a = graph.op('Cast', a, to_i=2)
        zero = graph.op('Constant', value_t=torch.tensor(128, dtype=torch.uint8))
        weight_zero = graph.op('Constant', value_t=torch.tensor(0, dtype=torch.int8))
        result = graph.op('MatMulInteger', a, b, zero, weight_zero)
        result.setType(a.type().with_dtype(torch.int32).with_sizes([None, None]))
        return result
    torch.onnx.register_custom_op_symbolic('aten::_int_mm', int_mm, 17)
    def floating_clamp(graph, value, minimum, maximum):
        # Dynamic scripted integer kernels return FP32 activations. Some
        # exporter shape passes lose their scalar type across If/Squeeze.
        if value.type().scalarType() is None:
            value.setType(value.type().with_dtype(torch.float32))
        return symbolic_opset11.clamp(graph, value, minimum, maximum)
    torch.onnx.register_custom_op_symbolic('aten::clip', floating_clamp, 17)
    torch.onnx.register_custom_op_symbolic('aten::clamp', floating_clamp, 17)
