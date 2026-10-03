import copy
import unittest
from unittest.mock import patch

import torch

from modules.losses.diff_loss import DiffusionLoss
from modules.losses.reflow_loss import RectifiedFlowLoss
from utils.aux_dataset import AUX_MODULES, parse_aux_dataset_config, build_aux_datasets, binarize_aux_dataset
from utils.hparams import set_hparams
from utils.training_schedule import parse_training_interval, training_schedule_options


def aux_config():
    return {'enable': True, 'module': ['pitch', 'voicing'], 'spk_ids': [0],
            'datasets': {'data': [{'zh': 'data/aux'}], 'val': {'zh': ['valid']}}}


class AuxAndScheduleTests(unittest.TestCase):
    def test_curve_loss_isolation_and_legacy_padding(self):
        for loss in (DiffusionLoss('l2'), RectifiedFlowLoss('l2', log_norm=False)):
            pred = torch.tensor([[[[1., 2.]], [[9., 10.]]]], requires_grad=True)
            target = torch.zeros_like(pred)
            kwargs = {'t': torch.tensor([[0.5]])} if isinstance(loss, RectifiedFlowLoss) else {}
            padding = torch.tensor([[[True], [False]]])
            # Existing all-channel mean includes the padded frame in its denominator.
            self.assertEqual(loss(pred, target, non_padding=padding, **kwargs).item(), 20.5)
            selected = loss(pred, target, non_padding=padding,
                            feature_mask=torch.tensor([[True, False]]), **kwargs)
            self.assertEqual(selected.item(), 0.5)
            selected.backward()
            self.assertEqual(pred.grad[0, 1].abs().sum().item(), 0)
            self.assertGreater(pred.grad[0, 0].abs().sum().item(), 0)
            all_features = loss(pred, target, non_padding=padding,
                                feature_mask=torch.ones(1, 2, dtype=torch.bool), **kwargs)
            torch.testing.assert_close(all_features, loss(pred, target, non_padding=padding, **kwargs))

    def test_aux_config_validation_and_binarizer_routing(self):
        config = parse_aux_dataset_config(aux_config(), AUX_MODULES)
        self.assertEqual(config['modules'], {'pitch', 'voicing'})
        self.assertEqual(build_aux_datasets(config)[0]['spk_id'], 0)
        with patch('preprocessing.variance_binarizer.VarianceBinarizer') as binarizer:
            binarize_aux_dataset(config, 'binary')
            self.assertEqual(str(binarizer.call_args.kwargs['binary_data_dir']), str(__import__('pathlib').Path('binary/aux')))
            binarizer.return_value.process.assert_called_once()
        for change in ({'module': ['unknown']}, {'spk_ids': [-1]}, {'spk_ids': [True]},
                       {'datasets': {'data': [{'zh': 'data/aux'}], 'val': {}}}):
            invalid = copy.deepcopy(aux_config())
            invalid.update(change)
            with self.assertRaises(ValueError):
                parse_aux_dataset_config(invalid, AUX_MODULES)
        with self.assertRaisesRegex(ValueError, 'disabled predictors'):
            parse_aux_dataset_config(aux_config(), {'pitch'})

    def test_ep_step_options_and_cli_overrides(self):
        config = dict(max_updates=100, val_check_interval=4, accumulate_grad_batches=8)
        self.assertEqual(training_schedule_options(config), dict(
            max_steps=100, max_epochs=-1, val_check_interval=32, check_val_every_n_epoch=None))
        config.update(max_updates='2ep', val_check_interval='1ep')
        self.assertEqual(training_schedule_options(config), dict(
            max_steps=-1, max_epochs=2, val_check_interval=1.0, check_val_every_n_epoch=1))
        for value in (0, -1, True, 0.5, '0ep', 'bad', '1.5ep'):
            with self.assertRaises(ValueError):
                parse_training_interval(value, 'max_updates')
        result = set_hparams('configs/templates/config_acoustic_dit.yaml',
                             hparams_str='max_updates=2ep,val_check_interval=1step',
                             global_hparams=False, print_hparams=False)
        self.assertEqual(result['max_updates'], '2ep')
        self.assertEqual(result['val_check_interval'], '1step')

    def test_all_in_one_switch_preserves_backbone(self):
        result = set_hparams('configs/templates/config_acoustic_dit.yaml',
                             hparams_str="all_in_one={'enabled':True}",
                             global_hparams=False, print_hparams=False)
        self.assertEqual(result['backbone_type'], 'dit')
        self.assertEqual(result['task_cls'], 'training.all_in_one_task.AllInOneTask')
        self.assertEqual(result['binarizer_cls'], 'preprocessing.all_in_one_binarizer.AllInOneBinarizer')
        self.assertIn('dur_prediction_args', result)


if __name__ == '__main__':
    unittest.main()
