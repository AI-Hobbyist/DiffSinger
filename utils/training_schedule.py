"""Convert the project's step/ep notation to Lightning trainer options."""

import re


def parse_training_interval(value, name):
    if type(value) is int:
        count, unit = value, 'step'
    elif isinstance(value, str):
        match = re.fullmatch(r'\s*(\d+)\s*(ep|step)?\s*', value, re.IGNORECASE)
        if match is None:
            raise ValueError(f'{name} must be a positive integer, Nstep, or Nep.')
        count, unit = int(match[1]), (match[2] or 'step').lower()
    else:
        raise ValueError(f'{name} must be a positive integer, Nstep, or Nep.')
    if count < 1:
        raise ValueError(f'{name} must be positive.')
    return count, unit


def training_schedule_options(config):
    updates, update_unit = parse_training_interval(config['max_updates'], 'max_updates')
    interval, interval_unit = parse_training_interval(config['val_check_interval'], 'val_check_interval')
    accumulation = config['accumulate_grad_batches']
    if type(accumulation) is not int or accumulation < 1:
        raise ValueError('accumulate_grad_batches must be a positive integer.')
    return {
        'max_steps': updates if update_unit == 'step' else -1,
        'max_epochs': updates if update_unit == 'ep' else -1,
        'val_check_interval': interval * accumulation if interval_unit == 'step' else 1.0,
        'check_val_every_n_epoch': interval if interval_unit == 'ep' else None,
    }
