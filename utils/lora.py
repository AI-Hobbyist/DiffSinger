"""LoRA Linear adapters and lossless checkpoint structure conversion.

Adapter orientation follows dsrx_ref: A=[in, rank], B=[rank, out].
"""
import math
import re

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Linear):
    def __init__(self, in_features, out_features, r=8, alpha=16, bias=True, **kwargs):
        if isinstance(r, bool) or not isinstance(r, int) or r <= 0:
            raise ValueError('LoRA rank must be a positive integer.')
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('LoRA alpha must be a positive finite number.')
        super().__init__(in_features, out_features, bias=bias, **kwargs)
        self.lora_r, self.lora_alpha = r, alpha
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(self.weight.new_empty(in_features, r))
        self.lora_B = nn.Parameter(self.weight.new_zeros(r, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias) + self.scaling * F.linear(
            F.linear(x, self.lora_A.t()), self.lora_B.t())


def inject_lora(model, *, rank=8, alpha=16, target_modules=('linear',), require_match=True):
    if isinstance(target_modules, str) or not target_modules:
        raise ValueError('lora.target_modules must be a nonempty list of regex patterns.')
    patterns = [re.compile(p) for p in target_modules if p not in ('*', 'linear')]
    all_linear = any(p in ('*', 'linear') for p in target_modules)
    matched = []
    for name, child in list(model.named_modules()):
        if not name or not isinstance(child, nn.Linear) or isinstance(child, LoRALinear):
            continue
        if not all_linear and not any(p.search(name) for p in patterns):
            continue
        adapter = LoRALinear(child.in_features, child.out_features, r=rank, alpha=alpha,
                             bias=child.bias is not None, device=child.weight.device, dtype=child.weight.dtype)
        # Preserve parameter identity, sharing, device, and exact base values.
        adapter.weight = child.weight
        adapter.bias = child.bias
        adapter.weight.requires_grad_(False)
        if adapter.bias is not None:
            adapter.bias.requires_grad_(False)
        adapter.train(child.training)
        parent, _, attr = name.rpartition('.')
        setattr(model.get_submodule(parent) if parent else model, attr, adapter)
        matched.append(name)
    if require_match and not matched:
        raise ValueError(f'LoRA target patterns matched no linear modules: {target_modules}')
    return matched


def mark_only_lora_as_trainable(model, train_bias=False):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.endswith(('.lora_A', '.lora_B'))
                                 or (train_bias and name.endswith('.bias')))


def lora_metadata(model):
    return {'format_version': 1, 'modules': {
        name: {'rank': m.lora_r, 'alpha': m.lora_alpha}
        for name, m in model.named_modules() if isinstance(m, LoRALinear)}}


def merge_lora_state_dict(state, metadata=None, config=None, prefix='model'):
    """Merge a full checkpoint before strict loading into a plain model."""
    state = dict(state)
    names = {k[:-len('.lora_A')] for k in state if k.endswith('.lora_A')}
    b_names = {k[:-len('.lora_B')] for k in state if k.endswith('.lora_B')}
    if names != b_names:
        raise ValueError('Incomplete LoRA checkpoint: each adapter requires both A and B.')
    layout = (metadata or {}).get('modules', {})
    if metadata is not None:
        branch = prefix.removeprefix('model').lstrip('.')
        selected = {name[len(branch) + 1:] if branch else name for name in layout
                    if not branch or name.startswith(branch + '.')}
        if selected != names:
            raise ValueError('LoRA checkpoint adapter tensors do not match the saved module metadata.')
    for name in names:
        a, b = state.pop(name + '.lora_A'), state.pop(name + '.lora_B')
        weight = state.get(name + '.weight')
        if weight is None:
            raise ValueError(f'LoRA checkpoint lacks base weight: {name}.weight')
        full_name = '.'.join(p for p in (prefix, name) if p)
        spec = layout.get(full_name.removeprefix('model.'), {})
        rank = spec.get('rank', a.shape[1] if a.ndim == 2 else None)
        alpha = state.pop(name + '.lora_alpha', spec.get('alpha', (config or {}).get('alpha')))
        stored_rank = state.pop(name + '.lora_r', rank)
        if (a.ndim != 2 or b.ndim != 2 or weight.ndim != 2 or rank is None
                or int(stored_rank) != rank or a.shape != (weight.shape[1], rank)
                or b.shape != (rank, weight.shape[0])):
            raise ValueError(f'LoRA tensor shapes do not match: {name}')
        if alpha is None or not math.isfinite(float(alpha)) or float(alpha) <= 0:
            raise ValueError(f'LoRA alpha is missing or invalid: {name}. Use the saved LoRA config.')
        state[name + '.weight'] = (weight.float() + (b.float().t() @ a.float().t())
                                   * (float(alpha) / rank)).to(weight.dtype)
    return state


@torch.no_grad()
def merge_lora_into_model(model):
    for name, child in list(model.named_modules()):
        if not name or not isinstance(child, LoRALinear):
            continue
        merged = nn.Linear(child.in_features, child.out_features, bias=child.bias is not None,
                           device=child.weight.device, dtype=child.weight.dtype)
        merged.weight.copy_((child.weight.float() + child.scaling * (
            child.lora_B.float().t() @ child.lora_A.float().t())).to(child.weight.dtype))
        if child.bias is not None:
            merged.bias.copy_(child.bias)
        merged.train(child.training)
        parent, _, attr = name.rpartition('.')
        setattr(model.get_submodule(parent) if parent else model, attr, merged)


def setup_lora_training(model, config, work_dir):
    from pathlib import Path
    from utils import load_ckpt
    from utils.training_utils import get_latest_checkpoint_path
    base = config.get('base_ckpt')
    # Full LoRA checkpoints are self-contained on resume.
    resume = get_latest_checkpoint_path(Path(work_dir)) if work_dir else None
    if resume is None:
        if not base:
            raise ValueError('LoRA fine-tuning requires lora.base_ckpt.')
        load_ckpt(model, base, strict=True)
    names = inject_lora(model, rank=config.get('rank', 8), alpha=config.get('alpha', 16),
                        target_modules=config.get('target_modules', ['linear']))
    mark_only_lora_as_trainable(model, train_bias=config.get('train_bias', False))
    return names
