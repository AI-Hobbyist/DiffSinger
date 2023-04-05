import glob
import re
import time
import os
import sys
import types
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from basics.base_model import CategorizedModule


def tensors_to_scalars(metrics):
    new_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        if type(v) is dict:
            v = tensors_to_scalars(v)
        new_metrics[k] = v
    return new_metrics


def collate_nd(values, pad_value=0, max_len=None):
    """
    Pad a list of Nd tensors on their first dimension and stack them into a (N+1)d tensor.
    """
    size = ((max(v.size(0) for v in values) if max_len is None else max_len), *values[0].shape[1:])
    res = torch.full((len(values), *size), fill_value=pad_value, dtype=values[0].dtype, device=values[0].device)

    for i, v in enumerate(values):
        res[i, :len(v), ...] = v
    return res


def _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
    if len(batch) == 0:
        return 0
    if len(batch) == max_sentences:
        return 1
    if num_tokens > max_tokens:
        return 1
    return 0


def batch_by_size(
        indices, num_tokens_fn, max_tokens=80000, max_sentences=48,
        required_batch_size_multiple=1
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch
            (default: 80000).
        max_sentences (int, optional): max number of sentences in each
            batch (default: 48).
    """
    bsz_mult = required_batch_size_multiple

    if isinstance(indices, types.GeneratorType):
        indices = np.fromiter(indices, dtype=np.int64, count=-1)

    sample_len = 0
    sample_lens = []
    batch = []
    batches = []
    for i in range(len(indices)):
        idx = indices[i]
        num_tokens = num_tokens_fn(idx)
        sample_lens.append(num_tokens)
        sample_len = max(sample_len, num_tokens)
        assert sample_len <= max_tokens, (
            "sentence at index {} of size {} exceeds max_tokens "
            "limit of {}!".format(idx, sample_len, max_tokens)
        )
        num_tokens = (len(batch) + 1) * sample_len

        if _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            batches.append(batch[:mod_len])
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        batches.append(batch)
    return batches


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (
                   torch.cumsum(mask, dim=1).type_as(mask) * mask
           ).long() + padding_idx


def softmax(x, dim):
    return F.softmax(x, dim=dim, dtype=torch.float32)


def unpack_dict_to_list(samples):
    samples_ = []
    bsz = samples.get('outputs').size(0)
    for i in range(bsz):
        res = {}
        for k, v in samples.items():
            try:
                res[k] = v[i]
            except:
                pass
        samples_.append(res)
    return samples_


def load_ckpt(cur_model, ckpt_base_dir, prefix_in_ckpt='model', required_category=None,
              ckpt_steps=None, strict=True, device='cpu'):
    if os.path.isfile(ckpt_base_dir):
        ckpt_base_dir = os.path.dirname(ckpt_base_dir)
        checkpoint_path = [ckpt_base_dir]
    elif ckpt_steps is not None:
        checkpoint_path = [os.path.join(ckpt_base_dir, f'model_ckpt_steps_{int(ckpt_steps)}.ckpt')]
    else:
        base_dir = ckpt_base_dir
        checkpoint_path = [
            os.path.join(base_dir, ckpt_file)
            for ckpt_file in sorted(
                [
                    os.path.basename(ckpt)
                    for ckpt in glob.glob(f'{base_dir}/model_ckpt_steps_*.ckpt')
                ],
                key=lambda x: int(re.findall(f'model_ckpt_steps_(\d+).ckpt', x.replace('\\', '/'))[0])
            )
        ]
    assert len(checkpoint_path) > 0, f'| ckpt not found in {ckpt_base_dir}.'
    checkpoint_path = checkpoint_path[-1]
    ckpt_loaded = torch.load(checkpoint_path, map_location=device)
    if required_category is not None:
        if not isinstance(cur_model, CategorizedModule):
            raise TypeError(f'The \'{required_category}\' argument can only be used '
                            f'on a \'basics.base_model.CategorizedModule\'.')
        cur_model.check_category(ckpt_loaded.get('category'))
    state_dict = ckpt_loaded['state_dict']
    state_dict = OrderedDict({
        k[len(prefix_in_ckpt) + 1:]: v
        for k, v in state_dict.items() if k.startswith(f'{prefix_in_ckpt}.')
    })
    if not strict:
        cur_model_state_dict = cur_model.state_dict()
        unmatched_keys = []
        for key, param in state_dict.items():
            if key in cur_model_state_dict:
                new_param = cur_model_state_dict[key]
                if new_param.shape != param.shape:
                    unmatched_keys.append(key)
                    print('| Unmatched keys: ', key, new_param.shape, param.shape)
        for key in unmatched_keys:
            del state_dict[key]
    cur_model.load_state_dict(state_dict, strict=strict)
    print(f'| load \'{prefix_in_ckpt}\' from \'{checkpoint_path}\'.')


def remove_padding(x, padding_idx=0):
    if x is None:
        return None
    assert len(x.shape) in [1, 2]
    if len(x.shape) == 2:  # [T, H]
        return x[np.abs(x).sum(-1) != padding_idx]
    elif len(x.shape) == 1:  # [T]
        return x[x != padding_idx]


class Timer:
    timer_map = {}

    def __init__(self, name, print_time=False):
        if name not in Timer.timer_map:
            Timer.timer_map[name] = 0
        self.name = name
        self.print_time = print_time

    def __enter__(self):
        self.t = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        Timer.timer_map[self.name] += time.time() - self.t
        if self.print_time:
            print(self.name, Timer.timer_map[self.name])


def print_arch(model, model_name='model'):
    print(f"| {model_name} Arch: ", model)
    # num_params(model, model_name=model_name)


def num_params(model, print_out=True, model_name="model"):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    if print_out:
        print(f'| {model_name} Trainable Parameters: %.3fM' % parameters)
    return parameters
