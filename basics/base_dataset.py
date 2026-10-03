import os
import pickle

import torch
from torch.utils.data import Dataset

from utils.hparams import hparams
from utils.indexed_datasets import IndexedDataset


def validate_sample_lengths(sizes, max_sample_frames: int) -> None:
    if max_sample_frames <= 0:
        return
    oversized = [
        (index, int(size)) for index, size in enumerate(sizes)
        if int(size) > max_sample_frames
    ]
    if oversized:
        preview = ', '.join(f'{index}:{size}' for index, size in oversized[:8])
        raise ValueError(
            f'{len(oversized)} samples exceed max_sample_frames={max_sample_frames} '
            f'(index:length {preview}). Split them at aligned phrase/word/note '
            f'boundaries before binarization; training-time frame cropping is unsafe.'
        )


class BaseDataset(Dataset):
    """
        Base class for datasets.
        1. *sizes*:
            clipped length if "max_frames" is set;
        2. *num_frames*:
            unclipped length.

        Subclasses should define:
        1. *collate*:
            take the longest data, pad other data to the same length;
        2. *__getitem__*:
            the index function.
    """

    def __init__(self, prefix, size_key='lengths', preload=False, data_dir=None):
        super().__init__()
        self.prefix = prefix
        self.data_dir = data_dir or hparams['binary_data_dir']
        with open(os.path.join(self.data_dir, f'{self.prefix}.meta'), 'rb') as f:
            self.metadata = pickle.load(f)
        self.sizes = self.metadata[size_key]
        validate_sample_lengths(self.sizes, int(hparams.get("max_sample_frames", -1)))
        self._indexed_ds = IndexedDataset(self.data_dir, self.prefix)
        if preload:
            self.indexed_ds = [self._indexed_ds[i] for i in range(len(self._indexed_ds))]
            del self._indexed_ds
        else:
            self.indexed_ds = self._indexed_ds

    def __getitem__(self, index):
        return {'_idx': index, **self.indexed_ds[index]}

    def __len__(self):
        return len(self.sizes)

    def num_frames(self, index):
        return self.sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes[index]

    def collater(self, samples):
        return {
            'size': len(samples),
            'indices': torch.LongTensor([s['_idx'] for s in samples])
        }
