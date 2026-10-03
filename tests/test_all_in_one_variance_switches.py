"""Regression tests for all-in-one predictor selection without model startup."""

import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_task_constructor(hparams):
    """Execute the real constructor with its heavyweight parent stubbed out."""
    source = (ROOT / 'training' / 'all_in_one_task.py').read_text(encoding='utf-8')
    module = ast.parse(source)
    task = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == 'AllInOneTask')
    constructor = next(node for node in task.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    task.body = [constructor]
    module.body = [task]

    class AcousticTask:
        def __init__(self):
            pass

    class VarianceTask:
        @staticmethod
        def _patch_variance_fused_kernels(task):
            pass

    from utils.aux_dataset import AUX_MODULES, parse_aux_dataset_config
    namespace = {
        'AUX_MODULES': AUX_MODULES,
        'parse_aux_dataset_config': parse_aux_dataset_config,
        'AcousticTask': AcousticTask,
        'VarianceTask': VarianceTask,
        'hparams': hparams,
        'validate_variance_validation_config': lambda *args: None,
    }
    exec(compile(module, str(ROOT / 'training' / 'all_in_one_task.py'), 'exec'), namespace)
    return namespace['AllInOneTask']


def load_variance_metric_builder(hparams):
    source_path = ROOT / 'training' / 'variance_task.py'
    module = ast.parse(source_path.read_text(encoding='utf-8'))
    task = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == 'VarianceTask')
    builder = next(node for node in task.body if isinstance(node, ast.FunctionDef) and node.name == 'build_losses_and_metrics')
    namespace = {
        'hparams': hparams,
        'DiffusionLoss': lambda **kwargs: object(),
        'RawCurveR2Score': lambda: object(),
    }
    exec(compile(ast.Module(body=[builder], type_ignores=[]), str(source_path), 'exec'), namespace)
    return namespace['build_losses_and_metrics']


class AllInOneVarianceSwitchTests(unittest.TestCase):
    def test_disabled_energy_stays_disabled_and_other_variances_stay_enabled(self):
        config = {
            'all_in_one': {'enabled': True},
            'val_with_variance': {'enable': False},
            'use_spk_id': False,
            'use_lang_id': False,
            'use_energy_embed': False,
            'predict_dur': True,
            'predict_pitch': True,
            'predict_energy': False,
            'predict_breathiness': True,
            'predict_voicing': True,
            'predict_tension': True,
            'lambda_dur_loss': 1.0,
            'lambda_pitch_loss': 1.0,
            'lambda_var_loss': 1.0,
        }
        task = load_task_constructor(config)()

        self.assertFalse(config['predict_energy'])
        self.assertFalse(config['use_energy_embed'])
        self.assertEqual(task.variance_prediction_list, ['breathiness', 'voicing', 'tension'])
        self.assertTrue(task.predict_variances)

        # AllInOneTask delegates metric creation to this real VarianceTask method.
        task.predict_dur = task.predict_pitch = False
        task.diffusion_type = 'ddpm'
        config['main_loss_type'] = 'mse'
        losses = []
        metrics = []
        task.register_validation_loss = losses.append
        task.register_validation_metric = lambda name, metric: metrics.append(name)
        load_variance_metric_builder(config)(task)
        self.assertEqual(losses, ['var_loss'])
        self.assertEqual(metrics, ['breathiness_r2', 'voicing_r2', 'tension_r2'])

    def test_all_variance_predictors_can_be_disabled(self):
        config = {
            'all_in_one': {'enabled': True},
            'val_with_variance': {'enable': False},
            'use_spk_id': False,
            'use_lang_id': False,
            'predict_dur': True,
            'predict_pitch': True,
            'predict_energy': False,
            'predict_breathiness': False,
            'predict_voicing': False,
            'predict_tension': False,
            'lambda_dur_loss': 1.0,
            'lambda_pitch_loss': 1.0,
            'lambda_var_loss': 1.0,
        }
        task = load_task_constructor(config)()

        self.assertEqual(task.variance_prediction_list, [])
        self.assertFalse(task.predict_variances)
        self.assertFalse(any(config[f'predict_{name}'] for name in ('energy', 'breathiness', 'voicing', 'tension')))

    def test_duration_and_pitch_switches_are_not_overridden(self):
        config = {
            'all_in_one': {'enabled': True},
            'val_with_variance': {'enable': False},
            'use_spk_id': False,
            'use_lang_id': False,
            'predict_dur': False,
            'predict_pitch': False,
            'predict_energy': False,
            'predict_breathiness': True,
            'predict_voicing': False,
            'predict_tension': False,
            'lambda_var_loss': 1.0,
        }
        task = load_task_constructor(config)()

        self.assertFalse(task.predict_dur)
        self.assertFalse(task.predict_pitch)
        self.assertEqual(task.variance_prediction_list, ['breathiness'])
        self.assertNotIn('lambda_dur_loss', task.__dict__)
        self.assertNotIn('lambda_pitch_loss', task.__dict__)

        config['val_with_variance'] = {'enable': True}
        config['dictionaries'] = {}
        with self.assertRaisesRegex(ValueError, 'requires predict_dur and predict_pitch'):
            load_task_constructor(config)()


if __name__ == '__main__':
    unittest.main()
