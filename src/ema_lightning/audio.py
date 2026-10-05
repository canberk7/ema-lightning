"""Audio helpers: sample-rate conversion from 48 kHz, window by window, and WAV writing.

The resampler carries its state from window to window, so the output never depends on where the decoder's
windows were cut.
"""
import wave

import numpy as np
import torch

RATE = 48000
RATES = (48000, 24000, 16000, 8000)


class Resampler:
    """48 kHz in, 48 kHz / factor out, through a linear-phase windowed-sinc low-pass."""

    def __init__(self, rate):
        self.factor = RATE // rate
        self.half = 32 * self.factor
        self.taps = None
        self.buffer = None
        self.start = -self.half  # absolute input index of buffer[0]; the signal is preceded by zeros
        self.next = 0  # absolute input index at the centre of the next output sample
        self.end = 0  # absolute input index one past the last real sample

    def push(self, chunk):
        if self.factor == 1:
            return chunk
        if self.buffer is None:
            self.buffer = chunk.new_zeros(self.half)
            n = torch.arange(-self.half, self.half + 1, dtype=torch.float64)
            cutoff = 0.45 / self.factor
            taps = 2 * cutoff * torch.sinc(2 * cutoff * n) * torch.blackman_window(2 * self.half + 1, False,
                                                                                    dtype=torch.float64)
            self.taps = (taps / taps.sum()).to(chunk)[None, None]
        self.buffer = torch.cat([self.buffer, chunk])
        self.end += chunk.numel()
        return self._emit(self.end - 1 - self.half)

    def flush(self):
        if self.factor == 1 or self.buffer is None:
            return None
        self.buffer = torch.cat([self.buffer, self.buffer.new_zeros(self.half + self.factor)])
        return self._emit(self.end - 1)

    def _emit(self, last):
        """Every output whose centre is at or before input index `last`."""
        if last < self.next:
            return self.buffer.new_zeros(0)
        count = (last - self.next) // self.factor + 1
        a = self.next - self.half - self.start
        b = a + (count - 1) * self.factor + 2 * self.half + 1
        out = torch.nn.functional.conv1d(self.buffer[a:b][None, None], self.taps, stride=self.factor)[0, 0]
        self.next += count * self.factor
        drop = self.next - self.half - self.start
        self.buffer, self.start = self.buffer[drop:], self.start + drop
        return out


def write_wav(path, audio, rate):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm.tobytes())
