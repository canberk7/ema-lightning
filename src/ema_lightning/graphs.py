"""Lightning: every stage compiled once for any batch size, then recorded as CUDA graphs at fixed sizes.

Batch sizes are recorded the way vLLM does it: 1, 2, 4, then every multiple of 8 up to the batch size, so a
batch is padded only to the nearest recorded size, never to the largest. Lengths are padded to fixed buckets.
The masks make all padding exact, and each stage then costs a single graph replay. Anything the graphs do not
cover falls back to the plain path.
"""
import torch

from .engine import DECODE_SIZES

LETTERS = (32, 64, 128, 256)
FRAMES = (40, 80, 160, 320)
SPANS = DECODE_SIZES
PADS = {"ids": 0, "mask": False, "h": 0.0, "dur": 0.0, "cw": -1, "wstart": 0, "fw": -1, "fp": 0.0,
        "fmask": False, "noise": 0.0, "z": 0.0}


def capture_sizes(batch_size):
    """Batch sizes to record: 1, 2, 4, then every multiple of 8 up to `batch_size`, and `batch_size` itself."""
    return sorted({s for s in (1, 2, 4) if s <= batch_size} | set(range(8, batch_size + 1, 8)) | {batch_size})


class Recorded:
    """One function recorded at one fixed shape: copy inputs in, replay, read outputs."""

    def __init__(self, fn, inputs, pool):
        self.inputs = inputs
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn(**inputs)
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool):
            self.outputs = fn(**inputs)

    def __call__(self, **values):
        for name, value in values.items():
            self.inputs[name].copy_(value)
        self.graph.replay()
        return self.outputs


def fit(x, shape, fill):
    """x at the start of a tensor of `shape`; missing batch rows repeat the first row."""
    out = x.new_full(shape, fill)
    out[tuple(slice(0, n) for n in x.shape)] = x
    out[x.shape[0]:] = out[:1]
    return out


class Graphs:
    @torch.no_grad()
    def __init__(self, model, decoder, batch_size, compile=True):
        dev = next(model.parameters()).device
        with torch.cuda.device(dev):  # record on the model's own GPU, not the current default
            self._record(model, decoder, batch_size, compile, dev)

    def _record(self, model, decoder, batch_size, compile, dev):
        text, sound = model.text_stage, model.sound_stage

        def decode(z, lengths):
            return decoder(z.transpose(1, 2), lengths)

        if compile:
            text, sound, decode = (torch.compile(f, dynamic=True) for f in (text, sound, decode))
        pool = torch.cuda.graph_pool_handle()
        self.sizes = capture_sizes(batch_size)
        S, Z, D = len(model.times), model.latent_dim, model.d
        L = LETTERS[-1]

        def full(shape, value, dtype=torch.float32):
            return torch.full(shape, value, dtype=dtype, device=dev)

        def words(b, n):
            return (torch.arange(n, device=dev) // 4).repeat(b, 1)

        self.text = {(b, n): Recorded(text, dict(ids=full((b, n), 1, torch.long), mask=full((b, n), True, torch.bool)),
                                      pool) for b in self.sizes for n in LETTERS}
        self.sound = {(b, t): Recorded(sound, dict(
            h=full((b, L, D), 0.0), dur=full((b, L), 1.0), mask=full((b, L), True, torch.bool), cw=words(b, L),
            wstart=words(b, L) * 4, fw=full((b, t), 0, torch.long), fp=full((b, t), 0.0),
            fmask=full((b, t), True, torch.bool), noise=full((b, S, t, Z), 0.0)), pool)
            for b in self.sizes for t in FRAMES}
        self.decode = {(b, p): Recorded(decode, dict(z=full((b, p, Z), 0.0), lengths=full((b,), p, torch.long)), pool)
                       for b in self.sizes for p in SPANS}
        self.count = len(self.text) + len(self.sound) + len(self.decode)

    def batch(self, b):
        return next((s for s in self.sizes if s >= b), None)

    def run(self, stage, **args):
        """The stage's outputs cropped back to the input sizes, or None when no recorded size fits."""
        b = next(iter(args.values())).shape[0]
        B = self.batch(b)
        if stage == "text":
            n = args["ids"].shape[1]
            N = next((x for x in LETTERS if x >= n), None)
            if B is None or N is None:
                return None
            h, dur = self.text[B, N](**{k: fit(v, (B, N), PADS[k]) for k, v in args.items()})
            return h[:b, :n], dur[:b, :n]
        if stage == "sound":
            n, t = args["h"].shape[1], args["fw"].shape[1]
            T = next((x for x in FRAMES if x >= t), None)
            if B is None or T is None or n > LETTERS[-1]:
                return None
            S, Z = args["noise"].shape[1], args["noise"].shape[3]
            shapes = {"h": (B, LETTERS[-1], args["h"].shape[2]), "noise": (B, S, T, Z)}
            for k in ("dur", "mask", "cw", "wstart"):
                shapes[k] = (B, LETTERS[-1])
            for k in ("fw", "fp", "fmask"):
                shapes[k] = (B, T)
            return self.sound[B, T](**{k: fit(v, shapes[k], PADS[k]) for k, v in args.items()})[:b, :t]
        p = args["z"].shape[1]
        P = next((x for x in SPANS if x >= p), None)
        if B is None or P is None:
            return None
        z = fit(args["z"], (B, P, args["z"].shape[2]), 0.0)
        lengths = fit(args["lengths"], (B,), 0)
        return self.decode[B, P](z=z, lengths=lengths)[:b]
