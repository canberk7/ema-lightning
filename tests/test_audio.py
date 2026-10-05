import math
import wave

import numpy as np
import pytest
import torch

from ema_lightning.audio import Resampler, write_wav


def run(resampler, signal, sizes):
    parts, i = [], 0
    for n in sizes:
        parts.append(resampler.push(signal[i:i + n]))
        i += n
    parts.append(resampler.push(signal[i:]))
    tail = resampler.flush()
    return torch.cat(parts + ([tail] if tail is not None else []))


def test_48k_passes_through_untouched():
    x = torch.randn(1000)
    assert torch.equal(run(Resampler(48000), x, [300, 300]), x)


@pytest.mark.parametrize("rate", [24000, 16000, 8000])
def test_output_never_depends_on_where_the_input_was_cut(rate):
    torch.manual_seed(0)
    x = torch.randn(48000)
    whole = run(Resampler(rate), x, [])
    cut = run(Resampler(rate), x, [1, 7, 1000, 4096, 13, 20000])
    assert whole.numel() == math.ceil(48000 / (48000 // rate))
    assert torch.allclose(whole, cut, atol=1e-6)


def test_keeps_speech_band_and_removes_aliasing():
    t = torch.arange(48000) / 48000
    low = run(Resampler(8000), torch.sin(2 * math.pi * 1000 * t), [])
    high = run(Resampler(8000), torch.sin(2 * math.pi * 6000 * t), [])
    assert 0.95 < low[400:-400].abs().max() < 1.05
    assert high[400:-400].abs().max() < 0.01


def test_write_wav_round_trip(tmp_path):
    audio = np.sin(np.linspace(0, 100, 1600)).astype(np.float32)
    write_wav(tmp_path / "a.wav", audio, 16000)
    with wave.open(str(tmp_path / "a.wav")) as f:
        assert (f.getframerate(), f.getnchannels(), f.getsampwidth(), f.getnframes()) == (16000, 1, 2, 1600)
