import json
import pathlib
import random
import warnings


REQUIRED_FIELDS = ('ph_seq', 'note_seq', 'note_dur')


def validate_variance_validation_config(config, dictionaries):
    languages = [key for key in config if key != 'enable']
    if not languages:
        raise ValueError('val_with_variance.enable is true but no language sources are configured.')
    for language in languages:
        if language not in dictionaries:
            raise ValueError(
                f"val_with_variance language '{language}' has no configured dictionary."
            )
        paths = config[language]
        if not isinstance(paths, list) or not paths:
            raise ValueError(
                f"val_with_variance language '{language}' must contain a non-empty path list."
            )
        for path in paths:
            if not pathlib.Path(path).is_file():
                raise FileNotFoundError(f'Variance validation source not found: {path}')


def load_language_dictionary(path):
    entries = {}
    with pathlib.Path(path).open(encoding='utf-8') as dictionary_file:
        for line in dictionary_file:
            line = line.strip()
            if not line:
                continue
            word, phonemes = line.split('\t', maxsplit=1)
            entries[word] = phonemes.split()
    return entries


def text_to_phonemes(text, dictionary):
    words = text.split()
    phoneme_groups = []
    for word in words:
        if word in ('AP', 'SP'):
            phoneme_groups.append([word])
        elif word in dictionary:
            phoneme_groups.append(dictionary[word])
        else:
            raise ValueError(f"Word '{word}' is not present in the configured dictionary.")
    return phoneme_groups


def _partition_size(total, groups):
    if groups <= 0 or total < groups:
        raise ValueError(f'Cannot partition {total} items into {groups} non-empty groups.')
    quotient, remainder = divmod(total, groups)
    return [quotient + (index < remainder) for index in range(groups)]


def prepare_variance_segment(segment, language, dictionary):
    segment = dict(segment)
    if not segment.get('text'):
        raise ValueError('A validation segment requires text.')
    phoneme_groups = text_to_phonemes(segment['text'], dictionary)
    segment['ph_seq'] = ' '.join(phone for group in phoneme_groups for phone in group)
    ph_num = [len(group) for group in phoneme_groups]

    missing = [field for field in REQUIRED_FIELDS if not segment.get(field)]
    if missing:
        raise ValueError(f'Validation segment is missing required fields: {missing}.')

    notes = segment['note_seq'].split()
    note_durations = segment['note_dur'].split()
    if len(notes) != len(note_durations):
        raise ValueError('note_seq and note_dur must have the same number of entries.')
    if len(notes) < len(ph_num):
        raise ValueError('A validation segment cannot contain fewer notes than words.')

    note_groups = _partition_size(len(notes), len(ph_num))
    note_slur = []
    for count in note_groups:
        note_slur.extend([0] + [1] * (count - 1))
    segment['ph_num'] = ' '.join(map(str, ph_num))
    segment['note_slur'] = ' '.join(map(str, note_slur))
    segment['lang'] = language
    return segment


def load_validation_sources(config, dictionaries, previous_language=None, rng=None):
    rng = random if rng is None else rng
    languages = [
        language for language, paths in config.items()
        if language != 'enable' and paths
    ]
    if len(languages) > 1 and previous_language in languages:
        languages.remove(previous_language)
    if not languages:
        raise ValueError('val_with_variance does not contain any language sources.')
    language = rng.choice(languages)
    path = pathlib.Path(rng.choice(config[language]))
    with path.open(encoding='utf-8') as ds_file:
        segments = json.load(ds_file)
    if isinstance(segments, dict):
        segments = [segments]
    if not segments:
        raise ValueError(f'Variance validation source is empty: {path}')
    dictionary = load_language_dictionary(dictionaries[language])
    valid_segments = []
    for segment in segments:
        try:
            valid_segments.append(prepare_variance_segment(segment, language, dictionary))
        except ValueError as error:
            warnings.warn(
                f'Skipping invalid variance validation segment from {path}: {error}',
                UserWarning
            )
    if not valid_segments:
        raise ValueError(f'Variance validation source has no usable text segments: {path}')
    return language, rng.choice(valid_segments)