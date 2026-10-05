# API reference

```python
from ema_lightning import EMA, Speech
```

## `EMA(device="auto")`

Loads the acoustic model and the decoder. The weights (`ema.pt`, `decoder.pt`)
are downloaded from `canberkkkkkk/ema-lightning` on Hugging Face on first use and
cached by `huggingface_hub`. `device="auto"` picks CUDA when it is available,
otherwise the CPU; any `torch.device` string works.

Reuse one `EMA` for the whole process.

## `EMA.lightning(batch_size=None)`

The fast path for NVIDIA GPUs. Returns the same `EMA`, so it chains:
`tts = EMA().lightning()`.

1. Turns on cuDNN benchmark mode (`torch.backends.cudnn.benchmark`), which times
   each convolution shape once and keeps the fastest method. The setting is
   process-wide and changes speed, not the audio beyond rounding.
2. Picks the batch size: `batch_size`, or `best_batch_size()` when it is `None`.
3. Compiles the model once with `torch.compile` for any batch size, then records
   CUDA graphs at batch sizes 1, 2, 4 and every multiple of 8 up to the batch
   size; at 32, 64, 128 and 256 letters; at 40, 80, 160 and 320 frames; and at
   the two decoder window sizes, 48 and 120 frames. Each batch is padded up to the
   nearest recorded size. If the compiler is unavailable, graphs are recorded
   without it and a warning says so.
4. Runs a fixed sentence through the plain and the fast path, at both window
   sizes, and compares them. If they disagree, the fast path stays off and a
   warning says so.
5. Prints the number of graphs, the build time, the first audio of a stream, how
   long `say()` takes for a short sentence, and the batch size.

Without CUDA it warns and returns immediately; everything keeps working on the
plain path. Calling it twice is harmless.

## `EMA.best_batch_size()`

The smallest batch that reaches 90% of this device's best throughput. Measured
once per device (1 to 128 on CUDA, 1 to 8 on CPU) and cached in
`$XDG_CACHE_HOME/ema_lightning/batch_size_v2.json` (default `~/.cache`). It is the
most sentences or windows that go into one batch, and `lightning()` records graphs
up to it.

## `EMA.say(text, speed=1.0, seed=None, sample_rate=48000, path=None)`

| Argument | Values |
|---|---|
| `text` | A string, or a list or tuple of strings |
| `speed` | A number from 0.25 to 4; 1.0 is the natural pace |
| `seed` | A non-negative integer, or `None` for a random one |
| `sample_rate` | 48000, 24000, 16000 or 8000 |
| `path` | For one text, a `.wav` file. For a list, a folder: clips are written as `0.wav`, `1.wav`, … (zero-padded) |

One text returns a `Speech` once all of its audio is ready. A list returns a list
of `Speech` in the same order, generated in batches of `best_batch_size()`. With a
seed, every text in a list uses that seed. The texts join the queue in list order.

Safe to call from many threads at once: every call shares the GPU through one
scheduler, which queues sentences first come, first served and runs them in
shared batches. See [many callers](scheduling.md).

## `EMA.stream(text, speed=1.0, seed=None, sample_rate=48000)`

Yields float32 NumPy chunks in [-1, 1] as soon as each is ready: about one second
first, then about four seconds each, with the pauses between sentences as chunks of
silence. Joined, the chunks match `say()` with the same arguments to within
floating-point rounding. Takes one text; the options are those of `say()` without
`path`.

Safe to call from many threads at once; streams and `say()` calls share the same
queues. Leaving the loop early, or closing the iterator, drops the rest of that
text's work.

## `Speech`

A frozen dataclass.

| Field | Meaning |
|---|---|
| `audio` | float32 NumPy array, mono, in [-1, 1] |
| `sample_rate` | Samples per second |
| `duration` | Seconds |
| `seed` | The seed that produced it; pass it back to reproduce the audio (to within floating-point rounding when other work shares the GPU) |

## Errors

| Error | When |
|---|---|
| `ValueError` | `speed`, `seed` or `sample_rate` is out of range, raised before any work |
| `TypeError` | `text` is not a string or a list of strings; `stream()` was given a list |

Text never raises. Text with nothing to read (empty, whitespace, punctuation,
control characters) returns an empty `Speech` and an empty stream.
