"""EMA Lightning: Turkish text to speech.

    tts = EMA().lightning()
    speech = tts.say("Merhaba, nasılsınız?", path="merhaba.wav")
    speeches = tts.say(["Birinci cümle.", "İkinci cümle."])
    for chunk in tts.stream("Uzun bir metin..."):
        play(chunk)
"""
import json
import os
import random
import statistics
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from .audio import RATE, RATES, Resampler, write_wav
from .chunker import chunk
from .decoder import load_decoder
from .engine import FIRST_WINDOW, Engine, windows
from .frontend import Frontend
from .graphs import Graphs
from .model import load_acoustic
from .scheduler import DONE, Playhead

REPO = "canberkkkkkk/ema-lightning"
PROBE = "Bugün hava çok güzel, yarın da yağmur yağacakmış; toplantı öğleden sonra başlayacak."
FPS = 25  # frames per second of the acoustic model
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "ema_lightning" / "batch_size_v2.json"


@dataclass(frozen=True)
class Word:
    """One spoken word and when it is heard, in seconds from the start of the audio.

    `text` is the word as it was read aloud, after normalization: "5" comes back as "beş".
    """

    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Speech:
    """One spoken text: float32 audio in [-1, 1] at sample_rate, the seed that made it, and its words' times."""

    audio: np.ndarray = field(repr=False)
    sample_rate: int
    duration: float
    seed: int
    words: tuple = field(default=(), repr=False)


class EMA:
    def __init__(self, device="auto"):
        device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
        try:  # the Hub counts a download each time config.json is requested; nothing here depends on it
            hf_hub_download(REPO, "config.json")
        except Exception:
            pass
        model = load_acoustic(hf_hub_download(REPO, "ema.pt"), device)
        decoder = load_decoder(hf_hub_download(REPO, "decoder.pt"), device)
        self._setup(model, decoder, Frontend(model.vocab), device)

    @classmethod
    def _from_parts(cls, model, decoder, frontend, device):
        self = cls.__new__(cls)
        self._setup(model, decoder, frontend, device)
        return self

    def _setup(self, model, decoder, frontend, device):
        self.device = torch.device(device)
        self._engine = Engine(model, decoder, self.device)
        self._frontend = frontend
        self._batch_size = None
        self._playhead = Playhead(self._engine, self._list_batch)

    def lightning(self, batch_size=None):
        """Compile and record every stage as CUDA graphs, check them against the plain path, report ready."""
        if self.device.type != "cuda":
            warnings.warn("lightning needs a CUDA GPU; EMA keeps working without it", stacklevel=2)
            return self
        if self._engine.graphs is not None:
            return self
        # cuDNN times each convolution shape once and keeps the fastest method. The decoder is all convolutions:
        # on an RTX PRO 6000 this made it about twice as fast. It is process-wide and changes speed, not the audio.
        torch.backends.cudnn.benchmark = True
        start = time.perf_counter()
        size = batch_size or self.best_batch_size()
        try:
            graphs = Graphs(self._engine.model, self._engine.decoder, size)
        except Exception as error:  # e.g. no compiler on this machine: graphs alone still help
            warnings.warn(f"compiling failed, recording graphs without it: {error}", stacklevel=2)
            graphs = Graphs(self._engine.model, self._engine.decoder, size, compile=False)
        built = time.perf_counter() - start
        plain = self._probe()
        self._engine.graphs = graphs
        fast = self._probe()
        if any((a - b).abs().max() > 1e-2 for a, b in zip(plain, fast, strict=True)):
            self._engine.graphs = None
            warnings.warn("recorded graphs disagree with the plain path; lightning stays off", stacklevel=2)
            return self
        self._batch_size = size
        first = statistics.median(self._first_audio() for _ in range(5))
        timings = [self._timed_probe() for _ in range(5)]
        ms, seconds = statistics.median(t for t, _ in timings), timings[0][1]
        print(f"EMA Lightning ready: {graphs.count} graphs built in {built:.1f} s, first audio in {first:.1f} ms, "
              f"a {seconds:.1f} s sentence in {ms:.1f} ms, batch size {size}")
        return self

    def best_batch_size(self):
        """The smallest batch that reaches 90% of this device's best throughput, measured once and cached."""
        if self._batch_size:
            return self._batch_size
        key = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else f"cpu-{os.cpu_count()}"
        cached = json.loads(CACHE.read_text()) if CACHE.exists() else {}
        if key not in cached:
            sizes = (1, 2, 4, 8, 16, 32, 64, 128) if self.device.type == "cuda" else (1, 2, 4, 8)
            rates = {}
            for size in sizes:
                try:
                    rates[size] = self._throughput(size)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    break
            cached[key] = min(s for s, r in rates.items() if r >= 0.9 * max(rates.values()))
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(cached))
        self._batch_size = cached[key]
        return self._batch_size

    def say(self, text, speed=1.0, seed=None, sample_rate=RATE, path=None):
        """Speech for one text, or a list of Speech for a list of texts (batched)."""
        _check(speed, seed, sample_rate)
        if isinstance(text, str):
            seed = _seed(seed)
            request = self._submit(text, speed, seed)
            speech = _speech(list(self._receive(request, sample_rate)), sample_rate, seed, _words(request.pieces))
            if path is not None:
                write_wav(path, speech.audio, sample_rate)
            return speech
        texts = _texts(text)
        seeds = [_seed(seed) for _ in texts]
        requests = [self._submit(t, speed, s) for t, s in zip(texts, seeds, strict=True)]
        out = [_speech(list(self._receive(r, sample_rate)), sample_rate, s, _words(r.pieces))
               for r, s in zip(requests, seeds, strict=True)]
        if path is not None:
            Path(path).mkdir(parents=True, exist_ok=True)
            width = len(str(max(len(out) - 1, 0)))
            for i, speech in enumerate(out):
                write_wav(Path(path) / f"{i:0{width}d}.wav", speech.audio, sample_rate)
        return out

    def stream(self, text, speed=1.0, seed=None, sample_rate=RATE):
        """float32 audio chunks as they are made: one second first, then four seconds at a time.

        Call it from as many threads as you like; every stream shares the GPU through the same queues.
        Stopping early (leaving the loop) drops the rest of that text's work.
        """
        _check(speed, seed, sample_rate)
        if not isinstance(text, str):
            raise TypeError("stream() takes one text; for many, call stream() once per text")
        return self._stream(text, speed, _seed(seed), sample_rate)

    def _stream(self, text, speed, seed, sample_rate):
        """A stream's request, submitted when its first chunk is asked for."""
        yield from self._receive(self._submit(text, speed, seed, first=FIRST_WINDOW), sample_rate)

    def _submit(self, text, speed, seed, first=None):
        pieces = self._pieces(text, speed, seed)
        return self._playhead.submit(pieces, speed) if first is None else self._playhead.submit(pieces, speed, first)

    def _receive(self, request, sample_rate):
        """float32 chunks from a request's outbox, resampled. If the caller stops waiting, its work is dropped."""
        resampler = Resampler(sample_rate)
        try:
            while (item := request.outbox.get()) is not DONE:
                if isinstance(item, BaseException):
                    raise item
                out = resampler.push(item)
                if out.numel():
                    yield out.float().cpu().numpy()
            tail = resampler.flush()
            if tail is not None and tail.numel():
                yield tail.float().cpu().numpy()
        finally:
            request.cancel()

    def _pieces(self, text, speed, seed):
        return [self._engine.piece(piece, pause, (seed * 1_000_003 + i) % 2**63)
                for i, (piece, pause) in enumerate(chunk(self._frontend(text), speed))]

    def _list_batch(self):
        return self._batch_size or self.best_batch_size()

    def _probe(self):
        """Latents and both window sizes of a fixed sentence, for checking graphs against the plain path."""
        (piece,) = self._pieces(PROBE, 1.0, 0)
        with self._engine.lock:
            self._engine.plan([piece], 1.0)
            self._engine.think([piece])
            short = self._engine.decode([(piece, windows(piece.frames, FIRST_WINDOW)[0])])[0]
            full = self._engine.decode([(piece, windows(piece.frames)[0])])[0]
            return piece.latents, short, full

    def _first_audio(self):
        """Milliseconds until a stream of the probe sentence hands over its first chunk."""
        start = time.perf_counter()
        next(iter(self._stream(PROBE, 1.0, 0, RATE)))
        return 1000 * (time.perf_counter() - start)

    def _timed_probe(self):
        """Milliseconds for say() to return the probe sentence, and the seconds of audio it made."""
        start = time.perf_counter()
        speech = self.say(PROBE, seed=0)
        return 1000 * (time.perf_counter() - start), speech.duration

    def _throughput(self, size):
        """Seconds of audio per second for a batch of `size` copies of the probe sentence."""
        (piece,) = self._pieces(PROBE, 1.0, 0)
        pieces = [self._engine.piece(piece.text, 0.0, i) for i in range(size)]
        best = float("inf")
        with self._engine.lock:
            self._engine.plan(pieces, 1.0)
            for _ in range(3):
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                start = time.perf_counter()
                self._engine.think(pieces)
                for span in windows(pieces[0].frames):
                    self._engine.decode([(p, span) for p in pieces])
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                best = min(best, time.perf_counter() - start)
        return size * pieces[0].frames / 25 / best


def _check(speed, seed, sample_rate):
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not 0.25 <= speed <= 4:
        raise ValueError("speed must be a number from 0.25 to 4")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("seed must be a non-negative integer")
    if sample_rate not in RATES:
        raise ValueError(f"sample_rate must be one of {RATES}")


def _texts(text):
    if not isinstance(text, (list, tuple)) or not all(isinstance(t, str) for t in text):
        raise TypeError("text must be a string or a list of strings")
    return list(text)


def _seed(seed):
    return random.SystemRandom().randrange(2**31) if seed is None else seed


def _speech(chunks, sample_rate, seed, words=()):
    audio = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)
    return Speech(audio, sample_rate, len(audio) / sample_rate, seed, words)


def _words(pieces):
    """Every word's start and end, read off the frame plan (one frame = 1/25 s) the audio was made from."""
    words, offset = [], 0.0
    for p in pieces:
        if p.fw is None:  # never planned: nothing was spoken
            continue
        fw = p.fw.tolist()
        first, last = {}, {}
        for frame, w in enumerate(fw):
            first.setdefault(w, frame)
            last[w] = frame
        for w, text in enumerate(p.text.split()):
            if w in first:
                words.append(Word(text, round(offset + first[w] / FPS, 3), round(offset + (last[w] + 1) / FPS, 3)))
        offset += len(fw) / FPS + round(p.pause * RATE) / RATE
    return tuple(words)
