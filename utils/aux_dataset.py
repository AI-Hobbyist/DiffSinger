import pathlib


AUX_MODULES = {'pitch', 'dur', 'energy', 'breathiness', 'voicing', 'tension'}


def parse_aux_dataset_config(config, enabled_modules=None):
    if not isinstance(config, dict) or not config.get('enable', False):
        return None

    modules = config.get('module', ['pitch', 'voicing'])
    if not isinstance(modules, list) or not modules:
        raise ValueError('aux_datasets.module must be a non-empty list.')
    unknown_modules = set(modules) - AUX_MODULES
    if unknown_modules:
        raise ValueError(f'Unknown aux_datasets modules: {sorted(unknown_modules)}')
    if enabled_modules is not None:
        disabled_modules = set(modules) - set(enabled_modules)
        if disabled_modules:
            raise ValueError(
                f'aux_datasets.module contains disabled predictors: {sorted(disabled_modules)}'
            )

    data = config.get('datasets', {}).get('data', [])
    if isinstance(data, dict):
        data = [{language: path} for language, path in data.items()]
    if not isinstance(data, list) or not data:
        raise ValueError('aux_datasets.datasets.data must contain at least one language-to-path mapping.')
    data_entries = []
    for item in data:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError('Each aux_datasets.datasets.data item must contain exactly one language and path.')
        language, raw_data_dir = next(iter(item.items()))
        if not isinstance(language, str) or not isinstance(raw_data_dir, str):
            raise ValueError('aux_datasets.datasets.data languages and paths must be strings.')
        data_entries.append((language, raw_data_dir))

    val = config.get('datasets', {}).get('val', {})
    if not isinstance(val, dict) or not all(
            isinstance(language, str) and isinstance(prefixes, list)
            and all(isinstance(prefix, str) for prefix in prefixes)
            for language, prefixes in val.items()
    ):
        raise ValueError('aux_datasets.datasets.val must map languages to prefix lists.')
    data_languages = {language for language, _ in data_entries}
    unknown_val_languages = set(val) - data_languages
    if unknown_val_languages:
        raise ValueError(
            f'aux_datasets.datasets.val has no matching data path: {sorted(unknown_val_languages)}'
        )
    if not any(val.values()):
        raise ValueError('aux_datasets.datasets.val must contain at least one validation prefix.')

    spk_ids = config.get('spk_ids', [])
    if not isinstance(spk_ids, list) or not all(type(spk_id) is int and spk_id >= 0 for spk_id in spk_ids):
        raise ValueError('aux_datasets.spk_ids must be a list of integers.')
    if spk_ids and len(spk_ids) != len(data_entries):
        raise ValueError(
            'aux_datasets.spk_ids must be empty or contain one ID for each datasets.data entry.'
        )

    return {
        'modules': set(modules),
        'data': data_entries,
        'val': val,
        'spk_ids': spk_ids
    }


def build_aux_datasets(config):
    datasets = []
    for index, (language, raw_data_dir) in enumerate(config['data']):
        data_path = pathlib.Path(raw_data_dir)
        dataset = {
            'raw_data_dir': raw_data_dir,
            'speaker': data_path.name,
            'language': language,
            'test_prefixes': config['val'].get(language, [])
        }
        if config['spk_ids']:
            dataset['spk_id'] = config['spk_ids'][index]
        datasets.append(dataset)
    return datasets


def get_aux_binary_data_dir(binary_data_dir):
    return pathlib.Path(binary_data_dir) / 'aux'

def binarize_aux_dataset(config, binary_data_dir):
    from preprocessing.variance_binarizer import VarianceBinarizer
    data_dir = get_aux_binary_data_dir(binary_data_dir)
    print('| Auxiliary binary data dir: ', data_dir)
    VarianceBinarizer(datasets=build_aux_datasets(config), binary_data_dir=data_dir).process()
