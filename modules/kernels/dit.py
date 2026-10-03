"""Pickle-safe DiT MLP fusion that preserves module and checkpoint names."""
from modules.backbones.dit import DiTMLP
from modules.kernels.fused_linear_gelu import fused_linear_gelu


class FusedDiTMLP(DiTMLP):
    def forward(self, x):
        if not self.training:
            return super().forward(x)
        x = fused_linear_gelu(x, self.fc1.weight, self.fc1.bias)
        return self.dropout(self.fc2(x))


def patch_dit_model(backbone):
    count = 0
    for block in backbone.blocks:
        if isinstance(block.mlp, DiTMLP):
            block.mlp.__class__ = FusedDiTMLP
            count += 1
    return count
