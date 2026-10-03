import copy
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from modules.backbones.dit import DiT
from modules.kernels.dit import FusedDiTMLP
from modules.kernels.fused_linear_gelu import fused_linear_gelu, triton
from modules.kernels.integration import patch_diffusion_module
from utils.hparams import hparams


class DiTFusionTests(unittest.TestCase):
    def make_model(self, checkpointing=False):
        with patch.dict(hparams, {'hidden_size': 16}):
            model = DiT(4, 1, num_layers=1, num_channels=16, num_heads=2,
                        mlp_ratio=2, time_embed_dim=16,
                        use_gradient_checkpointing=checkpointing)
        # Avoid zero-initialized gates hiding MLP errors and gradients.
        with torch.no_grad():
            model.output_proj.weight.normal_(std=0.1)
            model.blocks[0].adaLN_modulation[1].weight.normal_(std=0.1)
        return model

    def test_patch_preserves_checkpoint_and_eval(self):
        reference = self.make_model()
        fused = copy.deepcopy(reference)
        with patch('modules.kernels.integration.is_triton_available', return_value=True):
            for attr in ('denoise_fn', 'velocity_fn'):
                wrapper = torch.nn.Module()
                setattr(wrapper, attr, fused)
                self.assertEqual(patch_diffusion_module(wrapper), 1)
        self.assertIsInstance(fused.blocks[0].mlp, FusedDiTMLP)
        self.assertEqual(list(fused.state_dict()), list(reference.state_dict()))
        reference.load_state_dict(fused.state_dict(), strict=True)
        reference.eval()
        fused.eval()
        args = (torch.randn(2, 1, 4, 7), torch.ones(2), torch.randn(2, 16, 7))
        torch.testing.assert_close(fused(*args), reference(*args), rtol=0, atol=0)

    def test_cpu_fallback_gradients(self):
        x = torch.randn(2, 3, 17, requires_grad=True)
        weight = torch.randn(19, 17, requires_grad=True)
        bias = torch.randn(19, requires_grad=True)
        actual = fused_linear_gelu(x, weight, bias)
        expected = F.gelu(F.linear(x, weight, bias), approximate='tanh')
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        gradient = torch.randn_like(actual)
        a = torch.autograd.grad(actual, (x, weight, bias), gradient)
        b = torch.autograd.grad(expected, (x, weight, bias), gradient)
        for left, right in zip(a, b):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_missing_triton_leaves_dit_unpatched(self):
        wrapper = torch.nn.Module()
        wrapper.velocity_fn = self.make_model()
        with patch('modules.kernels.integration.is_triton_available', return_value=False):
            with self.assertWarnsRegex(UserWarning, 'running eager'):
                self.assertEqual(patch_diffusion_module(wrapper), 0)
        self.assertNotIsInstance(wrapper.velocity_fn.blocks[0].mlp, FusedDiTMLP)

    @unittest.skipUnless(torch.cuda.is_available() and triton is not None,
                         'CUDA and Triton are required')
    def test_cuda_kernel_forward_backward(self):
        for dtype in (torch.float16, torch.bfloat16):
            for bias_enabled in (False, True):
                with self.subTest(dtype=dtype, bias=bias_enabled):
                    x = torch.randn(2, 7, 33, device='cuda', dtype=dtype, requires_grad=True)
                    w = (torch.randn(65, 33, device='cuda', dtype=dtype) * 0.1).requires_grad_()
                    bias = torch.randn(65, device='cuda', dtype=dtype, requires_grad=True) if bias_enabled else None
                    variables = (x, w, bias) if bias_enabled else (x, w)
                    actual = fused_linear_gelu(x, w, bias)
                    self.assertIn('_LinearGELUBackward', type(actual.grad_fn).__name__)
                    expected = F.gelu(F.linear(x, w, bias), approximate='tanh')
                    tolerance = 0.003 if dtype == torch.float16 else 0.025
                    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                    gradient = torch.randn_like(actual)
                    a = torch.autograd.grad(actual, variables, gradient)
                    b = torch.autograd.grad(expected, variables, gradient)
                    for left, right in zip(a, b):
                        torch.testing.assert_close(left, right, rtol=tolerance, atol=tolerance)

    @unittest.skipUnless(torch.cuda.is_available() and triton is not None,
                         'CUDA and Triton are required')
    def test_cuda_autocast_checkpoint_dual_time(self):
        for dtype in (torch.float16, torch.bfloat16):
            reference = self.make_model(checkpointing=True).cuda().train()
            fused = copy.deepcopy(reference)
            wrapper = torch.nn.Module()
            wrapper.velocity_fn = fused
            self.assertEqual(patch_diffusion_module(wrapper), 1)
            spec = torch.randn(2, 1, 4, 9, device='cuda')
            cond = torch.randn(2, 16, 9, device='cuda')
            time = torch.ones(2, device='cuda')
            mask = torch.rand(2, 9, device='cuda')
            valid = torch.ones(2, 9, device='cuda', dtype=torch.bool)
            valid[0, -2:] = False
            with torch.autocast('cuda', dtype=dtype):
                a = fused(spec, time, cond, time * 2, mask, valid)
                b = reference(spec, time, cond, time * 2, mask, valid)
            tolerance = 0.003 if dtype == torch.float16 else 0.03
            torch.testing.assert_close(a, b, rtol=tolerance, atol=tolerance)
            a.float().square().mean().backward()
            b.float().square().mean().backward()
            for (_, left), (_, right) in zip(fused.named_parameters(), reference.named_parameters()):
                torch.testing.assert_close(left.grad, right.grad, rtol=tolerance, atol=tolerance)


if __name__ == '__main__':
    unittest.main()
