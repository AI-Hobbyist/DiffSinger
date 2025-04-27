import json
import pathlib

PAD_INDEX = 0


class PhonemeDictionary:
    def __init__(
            self,
            dictionaries: dict[str, pathlib.Path],
            extra_phonemes: list[str] = None,
            merged_groups: list[list[str]] = None
    ):
        # Step 1: Collect all phonemes
        all_phonemes = {"AP", "SP"}
        if extra_phonemes:
            for ph in extra_phonemes:
                if "/" in ph:
                    lang, name = ph.split("/", maxsplit=1)
                    if lang not in dictionaries:
                        raise ValueError(
                            f"Invalid phoneme tag '{ph}' in extra phonemes: "
                            f"unrecognized language name '{lang}'."
                        )
                    if name in all_phonemes:
                        raise ValueError(
                            f"Invalid phoneme tag '{ph}' in extra phonemes: "
                            f"short name conflicts with existing tag."
                        )
                all_phonemes.add(ph)
        for lang, dict_path in dictionaries.items():
            with open(dict_path, "r", encoding="utf8") as dict_file:
                for line in dict_file:
                    _, phonemes = line.strip().split("\t")
                    phonemes = phonemes.split()
                    for phoneme in phonemes:
                        if "/" in phoneme:
                            raise ValueError(
                                f"Invalid phoneme tag '{phoneme}' in dictionary '{dict_path}': "
                                f"should not contain the reserved character '/'."
                            )
                        if phoneme in all_phonemes:
                            continue
                        all_phonemes.add(f"{lang}/{phoneme}")
        # Step 2: Parse merged phoneme groups
        if merged_groups is None:
            merged_groups = []
        else:
            _merged_groups = []
            for group in merged_groups:
                _group = []
                for phoneme in group:
                    if "/" in phoneme:
                        lang, name = phoneme.split("/", maxsplit=1)
                        if lang not in dictionaries:
                            raise ValueError(
                                f"Invalid phoneme tag '{phoneme}' in merged group: "
                                f"unrecognized language name '{lang}'."
                            )
                    if phoneme not in all_phonemes:
                        raise ValueError(
                            f"Invalid phoneme tag '{phoneme}' in merged group: "
                            f"not found in phoneme set."
                        )
                    _group.append(phoneme)
                _merged_groups.append(_group)
            merged_groups = [set(phones) for phones in _merged_groups if len(phones) > 1]
        # Step 3: Build phoneme index
        merged_phonemes_inverted_index = {}
        for idx, group in enumerate(merged_groups):
            other_idx = None
            for phoneme in group:
                if phoneme in merged_phonemes_inverted_index:
                    other_idx = merged_phonemes_inverted_index[phoneme]
                    break
            target_idx = idx if other_idx is None else other_idx
            for phoneme in group:
                merged_phonemes_inverted_index[phoneme] = target_idx
            if other_idx is not None:
                merged_groups[other_idx] |= group
                group.clear()
        phone_to_id = {}
        id_to_phone = []
        cross_lingual_phonemes = set()
        idx = 1
        for phoneme in sorted(all_phonemes):
            if phoneme in merged_phonemes_inverted_index:
                has_assigned = True
                for alias in merged_groups[merged_phonemes_inverted_index[phoneme]]:
                    if alias not in phone_to_id:
                        has_assigned = False
                        phone_to_id[alias] = idx
                if not has_assigned:
                    merged_group = sorted(merged_groups[merged_phonemes_inverted_index[phoneme]])
                    merged_from_langs = {
                        (alias.split("/", maxsplit=1)[0] if "/" in alias else None)
                        for alias in merged_group
                    }
                    id_to_phone.append(tuple(merged_group))
                    idx += 1
                    if len(merged_from_langs) > 1:
                        cross_lingual_phonemes.update(ph for ph in merged_group if "/" in ph)
            else:
                phone_to_id[phoneme] = idx
                id_to_phone.append(phoneme)
                idx += 1
        self._phone_to_id: dict[str, int] = phone_to_id
        self._id_to_phone: list[str | tuple] = id_to_phone
        self._cross_lingual_phonemes = frozenset(cross_lingual_phonemes)

    @property
    def vocab_size(self):
        return len(self._id_to_phone) + 1

    def __len__(self):
        return self.vocab_size

    @property
    def cross_lingual_phonemes(self):
        return self._cross_lingual_phonemes

    def is_cross_lingual(self, phone):
        return phone in self._cross_lingual_phonemes

    def encode_one(self, phone, lang=None):
        if "/" in phone:
            lang, phone = phone.split("/", maxsplit=1)
        if lang is None or phone in self._phone_to_id:
            return self._phone_to_id[phone]
        if "/" not in phone:
            phone = f"{lang}/{phone}"
        return self._phone_to_id[phone]

    def encode(self, sentence, lang=None):
        phones = sentence.strip().split() if isinstance(sentence, str) else sentence
        return [self.encode_one(phone, lang=lang) for phone in phones]

    def decode_one(self, idx, lang=None, scalar=True):
        if idx <= 0:
            return None
        phone = self._id_to_phone[idx - 1]
        if not scalar or isinstance(phone, str):
            return phone
        if lang is None:
            return phone[0]
        for alias in phone:
            if alias.startswith(f"{lang}/"):
                return alias
        return phone[0]

    def decode(self, ids, lang=None, scalar=True):
        ids = list(ids)
        return " ".join([
            self.decode_one(i, lang=lang, scalar=scalar)
            for i in ids
            if i >= 1
        ])

    def dump(self, filename, includes: set[str] = None, excludes: set[str] = None):
        ph_map = self._phone_to_id.copy()
        if includes:
            ph_map = {k: v for k, v in ph_map.items() if k in includes}
        if excludes:
            ph_map = {k: v for k, v in ph_map.items() if k not in excludes}
        with open(filename, "w", encoding="utf8") as fp:
            json.dump(ph_map, fp, ensure_ascii=False, indent=2)
