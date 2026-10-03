"""Preprocessing device/concurrency regressions and CPU/CUDA feature parity."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.fastspeech.tts_modules import LengthRegulator
from utils.binarizer_utils import get_mel_torch, get_mel2ph_torch
from utils.hparams import hparams
from utils.multiprocess_utils import chunked_multiprocess_run
from utils.preprocessing_resources import (preprocessing_device, preprocessing_worker_devices,
                                           release_binarizer_resources)


def worker_probe(value):
    return value * 2


class DeviceProbe:
    def __init__(self):
        self.device = torch.device('cpu')
        self.lr = LengthRegulator()

    def probe(self, value):
        assert torch.device(self.device).index == torch.cuda.current_device()
        return value, str(self.device)


def main():
    torch.set_num_threads(1)
    with patch('torch.cuda.device_count', return_value=4):
        assert preprocessing_worker_devices({}, 20, 'cuda') == (4, [0, 1, 2, 3])
        assert preprocessing_worker_devices({}, 2, 'cuda') == (2, [0, 1])
        assert preprocessing_worker_devices({'num_workers_per_gpu': 2}, 1, 'cuda') == (
            8, [0, 0, 1, 1, 2, 2, 3, 3])
        assert preprocessing_worker_devices({}, 8, 'cuda:2') == (1, [2])
        assert preprocessing_worker_devices({}, 8, 'cpu') == (8, None)
    assert preprocessing_device({'device': 'cpu'}).type == 'cpu'
    assert list(chunked_multiprocess_run(worker_probe, [(i,) for i in range(5)], num_workers=2)) == [0, 2, 4, 6, 8]
    assert list(chunked_multiprocess_run(worker_probe, [], num_workers=2)) == []
    rate = 44100
    waveform = (0.1 * np.sin(2 * np.pi * 220 * np.arange(rate) / rate)).astype(np.float32)
    kwargs = dict(num_mel_bins=16, fmax=16000, hop_size=512, win_size=2048, fft_size=2048)
    with patch('torch.cuda.empty_cache', side_effect=AssertionError('CPU path must not touch CUDA')):
        cpu_mel = get_mel_torch(waveform, rate, device='cpu', **kwargs)
        cpu_alignment = get_mel2ph_torch(LengthRegulator(), torch.tensor([0.5, 0.5]),
                                        len(cpu_mel), 512 / rate, device='cpu')
    # Cached live operators are released between main/joint/auxiliary stages.
    from preprocessing import acoustic_binarizer, variance_binarizer
    acoustic_binarizer.pitch_extractor = object()
    variance_binarizer.pitch_extractor = object()
    release_binarizer_resources()
    assert acoustic_binarizer.pitch_extractor is variance_binarizer.pitch_extractor is None
    if torch.cuda.is_available():
        probe = DeviceProbe()
        assert list(chunked_multiprocess_run(probe.probe, [(1,), (2,)], num_workers=1,
                                             device_ids=[0])) == [(1, 'cuda:0'), (2, 'cuda:0')]
        # Repeated feature extraction must release per-item tensors and preserve numerical results.
        allocations = []
        for _ in range(3):
            gpu_mel = get_mel_torch(waveform, rate, device='cuda:0', **kwargs)
            np.testing.assert_allclose(gpu_mel, cpu_mel, atol=3e-4, rtol=3e-4)
            gpu_alignment = get_mel2ph_torch(LengthRegulator(), torch.tensor([0.5, 0.5]),
                                            len(gpu_mel), 512 / rate, device='cuda:0')
            torch.testing.assert_close(gpu_alignment.cpu(), cpu_alignment)
            del gpu_alignment
            allocations.append(torch.cuda.memory_allocated())
        assert max(allocations) - min(allocations) < 1024 * 1024
        print('CUDA worker affinity and repeated mel/alignment parity passed.')
    else:
        print('CUDA preprocessing checks skipped: no CUDA device.')
    # Exercise the actual item builders and all enabled curve extraction on audio.
    import soundfile as sf
    from scripts.validate_all_in_one import configure_smoke_model
    configure_smoke_model()
    hparams.update(pe='parselmouth', hnsep='world')
    with tempfile.TemporaryDirectory() as directory:
        wave_path = Path(directory) / 'tone.wav'
        sf.write(wave_path, waveform, rate)
        metadata = dict(wav_fn=str(wave_path), spk_id=0, spk_name='opencpop',
                        lang_seq=[0, 0], ph_seq=[3, 3], ph_dur=[0.5, 0.5],
                        ph_text='a a', ds_idx=0, ph_num=[1, 1],
                        note_seq=['C4', 'D4'], note_dur=[0.5, 0.5])
        for device in ('cpu', 'cuda:0') if torch.cuda.is_available() else ('cpu',):
            hparams['binarization_args']['device'] = device
            for cls in (acoustic_binarizer.AcousticBinarizer, variance_binarizer.VarianceBinarizer):
                builder = cls()
                item = builder.process_item('0:tone', metadata, hparams['binarization_args'])
                assert item is not None
                for feature in ('energy', 'breathiness', 'voicing', 'tension'):
                    assert feature in item and np.isfinite(item[feature]).all()
                assert len(item['mel2ph']) == item['length']
                release_binarizer_resources(device)
                print(f'{cls.__name__} full item preprocessing on {device} passed.')
    print('Preprocessing resources validation passed.')


if __name__ == '__main__':
    main()
