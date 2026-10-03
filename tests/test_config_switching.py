"""Config inheritance must retain legacy partial overrides and isolate switches."""

import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'utils/hparams.py'
module = ast.parse(source.read_text(encoding='utf-8'))
module.body = [node for node in module.body
               if isinstance(node, ast.FunctionDef) and node.name == 'override_config']
namespace = {}
exec(compile(module, str(source), 'exec'), namespace)
override_config = namespace['override_config']


class ConfigSwitchTests(unittest.TestCase):
    def test_partial_legacy_override(self):
        config = {'backbone_type': 'lynxnet2', 'backbone_args': {'num_layers': 10, 'num_channels': 512},
                  'optimizer_args': {'optimizer_cls': 'Muon', 'lr': 0.001, 'weight_decay': 0.01}}
        override_config(config, {'backbone_args': {'num_channels': 256},
                                 'optimizer_args': {'lr': 0.002}})
        self.assertEqual(config['backbone_args'], {'num_layers': 10, 'num_channels': 256})
        self.assertEqual(config['optimizer_args']['weight_decay'], 0.01)

    def test_nested_backbone_switch_clears_conv_fields(self):
        config = {'pitch_prediction_args': {'repeat_bins': 64, 'backbone_type': 'lynxnet2',
                                           'backbone_args': {'kernel_size': 31, 'glu_type': 'swiglu'}}}
        override_config(config, {'pitch_prediction_args': {'backbone_args': {'num_heads': 6},
                                                          'backbone_type': 'dit'}})
        self.assertEqual(config['pitch_prediction_args']['repeat_bins'], 64)
        self.assertEqual(config['pitch_prediction_args']['backbone_args'], {'num_heads': 6})

    def test_optimizer_and_scheduler_switch(self):
        config = {'optimizer_args': {'optimizer_cls': 'Muon', 'beta1': 0.9},
                  'lr_scheduler_args': {'scheduler_cls': 'StepLR', 'step_size': 10}}
        override_config(config, {'optimizer_args': {'optimizer_cls': 'AdamW', 'lr': 0.001},
                                 'lr_scheduler_args': {'scheduler_cls': 'SequentialLR', 'milestones': [10]}})
        self.assertNotIn('beta1', config['optimizer_args'])
        self.assertNotIn('step_size', config['lr_scheduler_args'])


if __name__ == '__main__':
    unittest.main()
