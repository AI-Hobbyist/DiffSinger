import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from utils import load_ckpt
from utils.lora import (LoRALinear, inject_lora, lora_metadata,
                        mark_only_lora_as_trainable, merge_lora_into_model,
                        merge_lora_state_dict)


class LoRATests(unittest.TestCase):
    def test_adapter_training_freezes_base_and_merges(self):
        model = nn.Sequential(nn.Linear(5, 7), nn.GELU(), nn.Linear(7, 3))
        original = copy.deepcopy(model.state_dict())
        inject_lora(model, rank=2, alpha=3)
        mark_only_lora_as_trainable(model)
        x = torch.randn(2, 4, 5)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
        for _ in range(2):
            optimizer.zero_grad()
            model(x).square().mean().backward()
            optimizer.step()
        self.assertGreater(model[0].lora_B.abs().sum().item(), 0)
        for name, value in original.items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
        expected = model(x)
        merged_state = merge_lora_state_dict(model.state_dict(), lora_metadata(model), prefix='model')
        plain = nn.Sequential(nn.Linear(5, 7), nn.GELU(), nn.Linear(7, 3))
        plain.load_state_dict(merged_state, strict=True)
        torch.testing.assert_close(plain(x), expected)
        merge_lora_into_model(model)
        self.assertFalse(any(isinstance(m, LoRALinear) for m in model.modules()))
        torch.testing.assert_close(model(x), expected)

    def test_strict_loader_merges_full_lora(self):
        source = nn.Sequential(nn.Linear(5, 3)).eval()
        inject_lora(source, rank=2, alpha=7)
        with torch.no_grad():
            source[0].lora_B.normal_()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.ckpt'
            torch.save({'state_dict': {'model.' + k: v for k, v in source.state_dict().items()},
                        'lora': lora_metadata(source)}, path)
            model = nn.Sequential(nn.Linear(5, 3)).eval()
            load_ckpt(model, path, strict=True)
            x = torch.randn(2, 5)
            torch.testing.assert_close(model(x), source(x))
            artifact = torch.load(path, map_location='cpu')
            artifact['state_dict'] = {k: v for k, v in artifact['state_dict'].items()
                                      if not k.endswith(('.lora_A', '.lora_B'))}
            torch.save(artifact, path)
            with self.assertRaisesRegex(ValueError, 'metadata'):
                load_ckpt(model, path, strict=True)

    def test_targeting_bias_and_dtype(self):
        model = nn.Sequential(nn.Linear(5, 7), nn.Linear(7, 3)).double()
        names = inject_lora(model, rank=2, alpha=3, target_modules=['^1$'])
        self.assertEqual(names, ['1'])
        self.assertEqual(model[1].lora_A.dtype, torch.float64)
        self.assertNotIsInstance(model[0], LoRALinear)
        mark_only_lora_as_trainable(model, train_bias=True)
        self.assertTrue(model[0].bias.requires_grad)
        self.assertTrue(model[1].bias.requires_grad)
        self.assertFalse(model[0].weight.requires_grad)
        self.assertFalse(model[1].weight.requires_grad)

    def test_legacy_checkpoint_requires_saved_config(self):
        from utils.hparams import hparams
        source = nn.Sequential(nn.Linear(5, 3)).eval()
        inject_lora(source, rank=2, alpha=7)
        with torch.no_grad():
            source[0].lora_B.normal_()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'legacy.ckpt'
            torch.save({'state_dict': {'model.' + k: v for k, v in source.state_dict().items()}}, path)
            model = nn.Sequential(nn.Linear(5, 3)).eval()
            with patch.dict(hparams, {'lora': {'enabled': False, 'alpha': 16}}):
                with self.assertRaisesRegex(ValueError, 'alpha'):
                    load_ckpt(model, path)
            with patch.dict(hparams, {'lora': {'enabled': True, 'alpha': 7}}):
                load_ckpt(model, path)
            x = torch.randn(2, 5)
            torch.testing.assert_close(model(x), source(x))

    def test_validation_and_missing_metadata(self):
        for rank in (0, True, 1.5):
            with self.assertRaises(ValueError):
                inject_lora(nn.Sequential(nn.Linear(2, 3)), rank=rank)
        with self.assertRaisesRegex(ValueError, 'matched no'):
            inject_lora(nn.Sequential(nn.Linear(2, 3)), target_modules=['no_match'])
        model = nn.Sequential(nn.Linear(2, 3))
        inject_lora(model, rank=2)
        with self.assertRaisesRegex(ValueError, 'alpha'):
            merge_lora_state_dict(model.state_dict())
        state = dict(model.state_dict())
        del state['0.lora_B']
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            merge_lora_state_dict(state, lora_metadata(model))

    def test_muon_routes_adapters_to_adamw_and_fusion_does_not_bypass(self):
        from modules.optimizer.muon import Muon_AdamW, get_params_for_muon
        from modules.kernels.integration import patch_diffusion_module
        model = nn.Sequential(nn.Linear(2, 3))
        inject_lora(model, rank=2)
        mark_only_lora_as_trainable(model)
        self.assertEqual(get_params_for_muon(model), [])
        optimizer = Muon_AdamW(model)
        self.assertEqual(len(optimizer.optimizers), 1)
        model(torch.randn(2, 2)).sum().backward()
        optimizer.step()
        wrapper = nn.Module()
        wrapper.velocity_fn = model
        with self.assertWarnsRegex(UserWarning, 'preserve adapter gradients'):
            self.assertEqual(patch_diffusion_module(wrapper), 0)


if __name__ == '__main__':
    unittest.main()
