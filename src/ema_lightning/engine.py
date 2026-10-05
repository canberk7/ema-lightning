"""The speech engine: pieces of spoken text in, 48 kHz audio out, one decoded window at a time.

Every piece goes through three stages: plan (the text stage and its frame timeline), think (the
aligner and the four steps) and decode (one window of latents to audio). Each stage takes any batch;
with lightning on, the same stages replay from recorded CUDA graphs instead. Playhead (scheduler.py)
decides which work runs in each batch.
"""
import threading
from dataclasses import dataclass
from itertools import pairwise

import torch

RATE = 48000
FIRST_WINDOW = 25  # a stream's first window is one second, so its first audio comes fast
WINDOW = 100  # four seconds for every later window
CONTEXT = 8  # frames decoded on each side of a window; the decoder reaches 4
# Every batch of windows is padded to one of these lengths before decoding. The GPU picks its convolution
# method by input length, and on an RTX PRO 6000 the windows' own lengths (33, 108, 116 frames) ran about
# twice as slow per frame as 48 and 120. The decoder ignores padding exactly, so the audio does not change.
DECODE_SIZES = (48, 120)
MAX_WORD_FRAMES = 250
MAX_FRAMES = 3000


@dataclass(eq=False)
class Piece:
    text: str
    pause: float
    seed: int
    ids: torch.Tensor
    cw: torch.Tensor  # word of each letter
    wstart: torch.Tensor  # first letter of that word
    h: torch.Tensor = None
    dur: torch.Tensor = None
    fw: torch.Tensor = None  # word of each frame
    fp: torch.Tensor = None  # position of each frame inside its word
    latents: torch.Tensor = None
    spans: list = None  # its decoder windows, fixed when it is planned

    @property
    def letters(self):
        return self.ids.numel()

    @property
    def frames(self):
        return self.fw.numel()


def windows(frames, first=WINDOW):
    """(start, end) of every decoded window: `first` frames first, then four seconds each."""
    spans, s = [], 0
    while s < frames:
        e = min(frames, s + (first if s == 0 else WINDOW))
        spans.append((s, e))
        s = e
    return spans


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


class Engine:
    def __init__(self, model, decoder, device):
        self.model, self.decoder, self.device = model, decoder, torch.device(device)
        self.hop = decoder.hop
        self.graphs = None
        self.lock = threading.RLock()
        self.decode_sizes = DECODE_SIZES if self.device.type == "cuda" else ()  # slow lengths are a GPU problem

    def piece(self, text, pause, seed):
        ids = [self.model.stoi.get(ch, 1) for ch in text]
        starts = [i for i, ch in enumerate(text) if ch != " " and (i == 0 or text[i - 1] == " ")] or [0]
        bounds = [0] + starts[1:] + [len(text)]
        cw, wstart = [], []
        for w, (a, b) in enumerate(pairwise(bounds)):
            cw += [w] * (b - a)
            wstart += [a] * (b - a)
        return Piece(text, pause, seed, torch.tensor(ids), torch.tensor(cw), torch.tensor(wstart))

    @torch.no_grad()
    def plan(self, pieces, speed):
        """Durations for a batch, then each piece's exact frame timeline after a single wait."""
        L = max(p.letters for p in pieces)
        ids = self.stack([p.ids for p in pieces], L, 0)
        mask = ids != 0
        h, dur = self.run("text", ids=ids, mask=mask)
        dur = dur / speed
        word = self.stack([p.cw for p in pieces], L, 0)
        counts = torch.zeros_like(dur).scatter_add_(1, word, dur).round().clamp(1, MAX_WORD_FRAMES).long().cpu()
        for i, p in enumerate(pieces):
            n = counts[i, :int(p.cw[-1]) + 1]
            frames = min(int(n.sum()), MAX_FRAMES)
            fw = torch.repeat_interleave(torch.arange(n.numel()), n)[:frames]
            fp = ((torch.arange(frames) - (n.cumsum(0) - n)[fw]).double() / n[fw].double()).float()
            p.h, p.dur, p.fw, p.fp = h[i, :p.letters].clone(), dur[i, :p.letters].clone(), fw, fp

    @torch.no_grad()
    def think(self, pieces):
        """Latents for a batch, each piece from its own seeded noise."""
        L, T = max(p.letters for p in pieces), max(p.frames for p in pieces)
        latents = self.run(
            "sound",
            h=self.stack([p.h for p in pieces], L, 0.0),
            dur=self.stack([p.dur for p in pieces], L, 0.0),
            mask=self.stack([torch.ones(p.letters, dtype=torch.bool) for p in pieces], L, False),
            cw=self.stack([p.cw for p in pieces], L, -1),
            wstart=self.stack([p.wstart for p in pieces], L, 0),
            fw=self.stack([p.fw for p in pieces], T, -1),
            fp=self.stack([p.fp for p in pieces], T, 0.0),
            fmask=self.stack([torch.ones(p.frames, dtype=torch.bool) for p in pieces], T, False),
            noise=self.stack([self.noise(p) for p in pieces], T, 0.0, dim=1))
        for i, p in enumerate(pieces):
            p.latents = latents[i, :p.frames].clone()

    @torch.no_grad()
    def decode_size(self, piece, span):
        """The length a window is decoded at: itself plus its margins, padded to a fast size on a GPU."""
        s, e = span
        n = min(piece.frames, e + CONTEXT) - max(0, s - CONTEXT)
        return next((size for size in self.decode_sizes if size >= n), n)

    def decode(self, items):
        """Audio for a batch of (piece, window), each cut to its own window."""
        spans = [(max(0, s - CONTEXT), min(p.frames, e + CONTEXT)) for p, (s, e) in items]
        longest = max(b - a for a, b in spans)
        P = next((size for size in self.decode_sizes if size >= longest), longest)
        z = self.stack([p.latents[a:b] for (p, _), (a, b) in zip(items, spans, strict=True)], P, 0.0)
        lengths = torch.tensor([b - a for a, b in spans], device=self.device)
        audio = self.run("decode", z=z, lengths=lengths)
        return [audio[i, (s - a) * self.hop:(e - a) * self.hop].clone()
                for i, ((_, (s, e)), (a, _)) in enumerate(zip(items, spans, strict=True))]

    def run(self, stage, **args):
        out = self.graphs.run(stage, **args) if self.graphs is not None else None
        if out is not None:
            return out
        if stage == "text":
            return self.model.text_stage(**args)
        if stage == "sound":
            return self.model.sound_stage(**args)
        full = bool((args["lengths"] == args["z"].shape[1]).all())  # nothing padded: skip the masks
        return self.decoder(args["z"].transpose(1, 2), None if full else args["lengths"])

    def noise(self, piece):
        generator = torch.Generator(device=self.device).manual_seed(piece.seed)
        shape = (len(self.model.times), piece.frames, self.model.latent_dim)
        return torch.randn(shape, generator=generator, device=self.device)

    def stack(self, rows, size, fill, dim=0):
        shape = list(rows[0].shape)
        shape[dim] = size
        out = torch.full([len(rows)] + shape, fill, dtype=rows[0].dtype, device=self.device)
        for i, row in enumerate(rows):
            out[i].narrow(dim, 0, row.shape[dim]).copy_(row)
        return out
