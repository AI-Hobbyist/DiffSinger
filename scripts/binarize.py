import importlib
import os
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.resolve()
os.environ['PYTHONPATH'] = str(root_dir)
sys.path.insert(0, str(root_dir))

from utils.hparams import set_hparams, hparams
from utils.aux_dataset import binarize_aux_dataset, parse_aux_dataset_config, AUX_MODULES

set_hparams()


def binarize():
    binarizer_cls = hparams["binarizer_cls"]
    pkg = ".".join(binarizer_cls.split(".")[:-1])
    cls_name = binarizer_cls.split(".")[-1]
    binarizer_cls = getattr(importlib.import_module(pkg), cls_name)
    print("| Binarizer: ", binarizer_cls)
    aux_config = parse_aux_dataset_config(
        hparams.get('aux_datasets', {}),
        enabled_modules={name for name in AUX_MODULES if hparams.get(f'predict_{name}', False)}
    )
    if aux_config and hparams['binarizer_cls'] not in (
        'preprocessing.variance_binarizer.VarianceBinarizer',
        'preprocessing.all_in_one_binarizer.AllInOneBinarizer'
    ):
        raise ValueError('aux_datasets requires variance or all-in-one binarization.')
    binarizer_cls().process()
    if aux_config:
        binarize_aux_dataset(aux_config, hparams['binary_data_dir'])


if __name__ == '__main__':
    binarize()
