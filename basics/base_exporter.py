import json
import pathlib
import shutil
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

from utils.hparams import hparams


class BaseExporter:
    def __init__(
            self,
            device: Union[str, torch.device] = None,
            cache_dir: Path = None,
            **kwargs
    ):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir: Path = cache_dir.resolve() if cache_dir is not None \
            else Path(__file__).parent.parent / 'deployment' / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # noinspection PyMethodMayBeStatic
    def build_spk_map(self) -> dict:
        if hparams['use_spk_id']:
            with open(Path(hparams['work_dir']) / 'spk_map.json', 'r', encoding='utf8') as f:
                spk_map = json.load(f)
            assert isinstance(spk_map, dict) and len(spk_map) > 0, 'Invalid or empty speaker map!'
            assert len(spk_map) == len(set(spk_map.values())), 'Duplicate speaker id in speaker map!'
            return spk_map
        else:
            return {}

    # noinspection PyMethodMayBeStatic
    def build_lang_map(self) -> dict:
        lang_map_fn = pathlib.Path(hparams['work_dir']) / 'lang_map.json'
        if lang_map_fn.exists():
            with open(lang_map_fn, 'r', encoding='utf8') as f:
                lang_map = json.load(f)
            assert isinstance(lang_map, dict) and len(lang_map) > 0, 'Invalid or empty language map!'
            assert len(lang_map) == len(set(lang_map.values())), 'Duplicate language id in language map!'
            return lang_map
        else:
            return {}

    def build_model(self) -> nn.Module:
        """
        Creates an instance of nn.Module and load its state dict on the target device.
        """
        raise NotImplementedError()

    def export_tensor_graph(self, model, args, path, **kwargs):
        """Use one set of deployment inputs for ONNX and TorchScript writers."""
        from deployment.modules.quantization import prepare_portable_modules, register_integer_onnx_symbolics
        register_integer_onnx_symbolics()
        prepare_portable_modules(model, 'int8', compile_integer=True)
        writer = getattr(self, 'graph_writer', torch.onnx.export)
        if not hasattr(self, 'graph_writer') and getattr(self.model, '_checkpoint_precision', None) in ('int8', 'fp16'):
            from deployment.exporters.torchscript_exporter import compile_stage
            model, args, _ = compile_stage(model, args,
                                         freeze=getattr(self.model, '_checkpoint_precision', None) != 'fp16')
        result = writer(model, args, path, **kwargs)
        if writer is torch.onnx.export:
            self.scope_onnx_graph(path)
        return result

    def scope_onnx_graph(self, path):
        import onnx
        from utils.onnx_helper import prefix_internal_values
        graph = onnx.load(str(path))
        prefix_internal_values(graph, Path(path).stem + '.')
        onnx.save(graph, str(path))

    def export_model(self, path: Path):
        """
        Exports the model to ONNX format.
        :param path: the target model path
        """
        raise NotImplementedError()

    def simplify_graph(self, model, **kwargs):
        import onnx
        import onnxsim
        if getattr(self.model, '_checkpoint_precision', None) is not None:
            # Deployment stages contain dynamic loops and captured parent
            # values. Preserve validated graphs: simplifier folding can either
            # be very expensive or break these references in optimized models.
            onnx.checker.check_model(model)
            return model, True
        return onnxsim.simplify(model, **kwargs)

    # noinspection PyMethodMayBeStatic
    def export_dictionaries(self, path: Path):
        dicts = hparams.get('dictionaries')
        if dicts is not None:
            for lang in dicts.keys():
                fn = f'dictionary-{lang}.txt'
                source = pathlib.Path(hparams['work_dir']) / fn
                if not source.is_file():
                    source = pathlib.Path(dicts[lang])
                shutil.copy(source, path / fn)
                print(f'| export dictionary => {path / fn}')
        else:
            fn = 'dictionary.txt'
            shutil.copy(pathlib.Path(hparams['work_dir']) / fn, path)
            print(f'| export dictionary => {path / fn}')

    def export_attachments(self, path: Path):
        """
        Exports related files and configs (e.g. the dictionary) to the target directory.
        :param path: the target directory
        """
        raise NotImplementedError()

    def export(self, path: Path):
        """
        Exports all the artifacts to the target directory.
        :param path: the target directory
        """
        raise NotImplementedError()
