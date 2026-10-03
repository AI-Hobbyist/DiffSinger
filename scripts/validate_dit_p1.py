import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.backbones import build_backbone, run_backbone
from modules.backbones.dit import DiT, DiTAttention, DiTBlock
from modules.core.ddpm import GaussianDiffusion
from modules.core.reflow import RectifiedFlow
from modules.losses import DiffusionLoss, RectifiedFlowLoss
from modules.optimizer.muon import Muon_AdamW, get_params_for_muon
from basics.base_dataset import validate_sample_lengths
from utils.hparams import hparams


def expect_error(error_type, fn, message: str) -> None:
    try:
        fn()
    except error_type:
        return
    raise AssertionError(message)


def build_default_dit(use_gradient_checkpointing: bool = False) -> DiT:
    hparams.clear()
    hparams.update({"hidden_size": 256})
    return DiT(
        128,
        1,
        num_layers=8,
        num_channels=384,
        num_heads=6,
        mlp_ratio=4,
        time_embed_dim=256,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )


def validate_shapes_and_masks() -> None:
    hparams.clear()
    hparams.update({"hidden_size": 8})
    model = DiT(
        3,
        2,
        num_layers=2,
        num_channels=8,
        num_heads=2,
        mlp_ratio=2,
        time_embed_dim=8,
        use_gradient_checkpointing=False,
    ).eval()

    for batch, frames in ((1, 1), (2, 5)):
        spec = torch.randn(batch, 2, 3, frames)
        step = torch.arange(batch, dtype=torch.float32)
        cond = torch.randn(batch, 8, frames)
        mask = torch.ones(batch, frames, dtype=torch.bool)
        output = model(spec, step, cond, valid_mask=mask)
        assert output.shape == spec.shape
        assert torch.isfinite(output).all()

    spec = torch.randn(2, 2, 3, 5)
    cond = torch.randn(2, 8, 5)
    step = torch.tensor([[1.0], [2.0]])
    mask = torch.tensor([[True, True, True, False, False], [True] * 5])
    altered_spec = spec.clone()
    altered_cond = cond.clone()
    altered_spec[0, :, :, 3:] = 1000
    altered_cond[0, :, 3:] = -1000
    output = model(spec, step, cond, valid_mask=mask)
    altered_output = model(altered_spec, step, altered_cond, valid_mask=mask)
    torch.testing.assert_close(output[0, :, :, :3], altered_output[0, :, :, :3])
    assert torch.count_nonzero(output[0, :, :, 3:]) == 0

    expect_error(
        ValueError,
        lambda: model(spec, step, cond, valid_mask=torch.zeros(2, 5, dtype=torch.bool)),
        "All-empty masks must be rejected.",
    )


def validate_checkpoint_gradients() -> None:
    hparams.clear()
    hparams.update({"hidden_size": 8})
    model = DiT(
        4,
        1,
        num_layers=1,
        num_channels=8,
        num_heads=2,
        mlp_ratio=2,
        time_embed_dim=8,
        use_gradient_checkpointing=True,
    ).train()
    spec = torch.randn(1, 1, 4, 3)
    cond = torch.randn(1, 8, 3)
    output = model(spec, torch.tensor([1.0]), cond, valid_mask=torch.ones(1, 3, dtype=torch.bool))
    output.sum().backward()
    assert model.blocks[0].attn.qkv.weight.grad is not None


def validate_stability_guards() -> None:
    attention = DiTAttention(8, 2, dropout=0.0).eval()
    with torch.no_grad():
        attention.qkv.bias.zero_()
    inputs = torch.randn(2, 7, 8)
    valid_mask = torch.ones(2, 7, dtype=torch.bool)
    baseline = attention(inputs, valid_mask)
    with torch.no_grad():
        attention.qkv.weight[:16].mul_(100.0)
    scaled = attention(inputs, valid_mask)
    torch.testing.assert_close(scaled, baseline, rtol=1e-4, atol=1e-5)

    modulation = torch.tensor([[-100.0, -2.0, 0.0, 2.0, 100.0]])
    stabilized = DiTBlock._stabilize_modulation(modulation)
    assert torch.all(stabilized >= -1.0)
    assert torch.all(stabilized <= 1.0)
    torch.testing.assert_close(stabilized, torch.tanh(modulation))


def validate_muon_parameter_partition() -> None:
    unmarked_modules = nn.Sequential(nn.Linear(8, 8), nn.Conv1d(8, 8, 3))
    unmarked_muon_param_ids = {
        id(parameter) for parameter in get_params_for_muon(unmarked_modules)
    }
    assert unmarked_muon_param_ids == {
        id(unmarked_modules[0].weight), id(unmarked_modules[1].weight)
    }
    embedding = nn.Embedding(8, 8)
    assert get_params_for_muon(embedding) == []

    hparams.clear()
    hparams.update({"hidden_size": 8})
    model = DiT(
        4,
        1,
        num_layers=2,
        num_channels=8,
        num_heads=2,
        mlp_ratio=2,
        time_embed_dim=8,
        use_gradient_checkpointing=False,
    )
    muon_param_ids = {id(parameter) for parameter in get_params_for_muon(model)}
    expected_muon_param_ids = {
        id(parameter)
        for module in model.modules()
        if not isinstance(module, nn.Embedding)
        for parameter in module.parameters(recurse=False)
        if parameter.requires_grad and parameter.ndim >= 2
    }

    assert muon_param_ids == expected_muon_param_ids

    optimizer = Muon_AdamW(model)
    optimizer_param_ids = [
        {
            id(parameter)
            for group in inner_optimizer.param_groups
            for parameter in group['params']
        }
        for inner_optimizer in optimizer.optimizers
    ]
    assert optimizer_param_ids[0] == expected_muon_param_ids
    assert optimizer_param_ids[1].isdisjoint(expected_muon_param_ids)

    spec = torch.randn(2, 1, 4, 5)
    cond = torch.randn(2, 8, 5)
    target = torch.randn_like(spec)
    output = model(
        spec,
        torch.tensor([100.0, 900.0]),
        cond,
        valid_mask=torch.ones(2, 5, dtype=torch.bool),
    )
    (output - target).square().mean().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    assert torch.count_nonzero(model.output_proj.weight) > 0
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def validate_factory_and_legacy_dispatch() -> None:
    hparams.clear()
    hparams.update({"hidden_size": 256})
    model = build_backbone(
        128,
        1,
        "dit",
        {
            "num_layers": 8,
            "num_channels": 384,
            "num_heads": 6,
            "mlp_ratio": 4,
            "time_embed_dim": 256,
            "patch_size": 1,
            "rope_base": 10000.0,
            "layer_norm_eps": 1e-6,
            "attention_dropout": 0.0,
            "mlp_dropout": 0.0,
            "use_gradient_checkpointing": True,
        },
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 22_019_456

    expect_error(
        ValueError,
        lambda: build_backbone(128, 1, "dit", {"num_layers": 1, "unknown": True}),
        "Unknown DiT arguments must be rejected.",
    )

    class LegacyBackbone(nn.Module):
        def forward(self, spec, diffusion_step, cond):
            return spec + diffusion_step[:, None, None, None] * 0 + cond[:, None, :1, :] * 0

    spec = torch.randn(1, 1, 2, 3)
    output = run_backbone(
        LegacyBackbone(),
        spec,
        torch.zeros(1),
        torch.randn(1, 4, 3),
        valid_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    torch.testing.assert_close(output, spec)


def validate_core_mask_propagation() -> None:
    hparams.clear()
    hparams.update({
        "hidden_size": 8,
        "schedule_type": "linear",
        "use_shallow_diffusion": False,
        "sampling_algorithm": "euler",
        "sampling_steps": 1,
        "diff_speedup": 1,
        "infer": False,
    })
    backbone_args = {
        "num_layers": 1,
        "num_channels": 8,
        "num_heads": 2,
        "mlp_ratio": 2,
        "time_embed_dim": 8,
        "use_gradient_checkpointing": False,
    }
    condition = torch.randn(2, 5, 8)
    target = torch.randn(2, 5, 4)
    mask = torch.tensor([[True, True, True, False, False], [True] * 5])

    ddpm = GaussianDiffusion(
        4,
        timesteps=4,
        k_step=4,
        backbone_type="dit",
        backbone_args=backbone_args,
        spec_min=[-1.0],
        spec_max=[1.0],
    )
    prediction, noise = ddpm(condition, gt_spec=target, infer=False, valid_mask=mask)
    assert prediction.shape == noise.shape == (2, 1, 4, 5)
    assert torch.count_nonzero(prediction[0, :, :, 3:]) == 0

    reflow = RectifiedFlow(
        4,
        time_scale_factor=1000,
        backbone_type="dit",
        backbone_args=backbone_args,
        spec_min=[-1.0],
        spec_max=[1.0],
    )
    velocity, target_velocity, time = reflow(
        condition, gt_spec=target, infer=False, valid_mask=mask
    )
    assert velocity.shape == target_velocity.shape == (2, 1, 4, 5)
    assert time.shape == (2, 1)
    assert torch.count_nonzero(velocity[0, :, :, 3:]) == 0

    x = torch.randn(2, 1, 4, 5)
    t = torch.tensor([0.2, 0.4])
    cond = condition.transpose(1, 2)
    for sampler in (reflow.sample_euler, reflow.sample_rk2, reflow.sample_rk4, reflow.sample_rk5):
        sampled, _ = sampler(x.clone(), t.clone(), 0.1, cond, valid_mask=mask)
        assert sampled.shape == x.shape
        assert torch.isfinite(sampled).all()


def validate_length_policy() -> None:
    validate_sample_lengths([1, 767, 768], 768)
    expect_error(
        ValueError,
        lambda: validate_sample_lengths([768, 769], 768),
        'Samples beyond the DiT frame budget must be rejected before batching.',
    )


def main() -> None:
    torch.manual_seed(1234)
    validate_shapes_and_masks()
    validate_checkpoint_gradients()
    validate_stability_guards()
    validate_muon_parameter_partition()
    validate_factory_and_legacy_dispatch()
    validate_core_mask_propagation()
    validate_length_policy()
    print("DiT P1 validation passed.")


if __name__ == "__main__":
    main()