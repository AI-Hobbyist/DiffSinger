"""Training-only Triton Linear + tanh GELU for DiT, with eager fallback."""
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _forward(X, W, Bias, Y, Z, M: tl.constexpr, N: tl.constexpr,
                 K: tl.constexpr, HAS_BIAS: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        cols = tl.program_id(1) * BN + tl.arange(0, BN)
        inner = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for start in range(tl.cdiv(K, BK)):
            ks = start * BK + inner
            x = tl.load(X + rows[:, None] * K + ks[None, :],
                        (rows[:, None] < M) & (ks[None, :] < K), other=0)
            w = tl.load(W + cols[None, :] * K + ks[:, None],
                        (cols[None, :] < N) & (ks[:, None] < K), other=0)
            acc += tl.dot(x, w)
        if HAS_BIAS:
            acc += tl.load(Bias + cols, cols < N, other=0)[None, :]
        # Match the eager Linear output rounding before evaluating GELU.
        z = acc.to(Y.dtype.element_ty).to(tl.float32)
        u = 0.7978845608028654 * (z + 0.044715 * z * z * z)
        t = 2.0 * tl.sigmoid(2.0 * u) - 1.0
        y = 0.5 * z * (1.0 + t)
        offsets = rows[:, None] * N + cols[None, :]
        valid = (rows[:, None] < M) & (cols[None, :] < N)
        tl.store(Y + offsets, y, valid)
        tl.store(Z + offsets, z, valid)

    @triton.jit
    def _backward(G, Z, DZ, SIZE: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        z = tl.load(Z + offsets, offsets < SIZE, other=0).to(tl.float32)
        g = tl.load(G + offsets, offsets < SIZE, other=0).to(tl.float32)
        u = 0.7978845608028654 * (z + 0.044715 * z * z * z)
        t = 2.0 * tl.sigmoid(2.0 * u) - 1.0
        derivative = 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t * t) * (
            0.7978845608028654 * (1.0 + 3.0 * 0.044715 * z * z))
        tl.store(DZ + offsets, g * derivative, offsets < SIZE)


class _LinearGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        shape = x.shape
        x = x.reshape(-1, shape[-1]).contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous() if bias is not None else None
        m, k = x.shape
        n = weight.shape[0]
        y = torch.empty((m, n), device=x.device, dtype=x.dtype)
        z = torch.empty_like(y)
        _forward[(triton.cdiv(m, 32), triton.cdiv(n, 64))](
            x, weight, bias if bias is not None else x, y, z,
            m, n, k, bias is not None, 32, 64, 32)
        ctx.save_for_backward(x, weight, z)
        ctx.input_shape = shape
        ctx.has_bias = bias is not None
        return y.reshape(*shape[:-1], n)

    @staticmethod
    def backward(ctx, grad):
        x, weight, z = ctx.saved_tensors
        grad = grad.contiguous().reshape_as(z)
        dz = torch.empty_like(z)
        _backward[(triton.cdiv(z.numel(), 256),)](grad, z, dz, z.numel(), 256)
        dx = (dz @ weight).reshape(ctx.input_shape) if ctx.needs_input_grad[0] else None
        dw = dz.t() @ x if ctx.needs_input_grad[1] else None
        db = dz.float().sum(0).to(dz.dtype) if ctx.has_bias and ctx.needs_input_grad[2] else None
        return dx, dw, db


def fused_linear_gelu(x, weight, bias=None):
    """Fuse only supported CUDA fp16/bf16; retain eager math elsewhere."""
    if triton is None or x.device.type != 'cuda':
        return F.gelu(F.linear(x, weight, bias), approximate='tanh')
    if torch.is_autocast_enabled():
        dtype = torch.get_autocast_dtype('cuda')
        x, weight = x.to(dtype), weight.to(dtype)
        bias = bias.to(dtype) if bias is not None else None
    if (x.dtype not in (torch.float16, torch.bfloat16)
            or weight.dtype != x.dtype or x.numel() == 0
            or torch.cuda.get_device_capability(x.device) < (7, 0)
            or (x.dtype == torch.bfloat16 and torch.cuda.get_device_capability(x.device) < (8, 0))):
        return F.gelu(F.linear(x, weight, bias), approximate='tanh')
    return _LinearGELU.apply(x, weight, bias)
