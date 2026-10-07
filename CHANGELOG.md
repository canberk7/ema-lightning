# Changelog

## 1.0.4

- The weights load with `torch.load(..., weights_only=True)`, so a checkpoint
  can only hold tensors and plain values and cannot run code when it loads.

## 1.0.3

- `Speech.words`: every spoken word with its `start` and `end` in seconds,
  read off the frame plan the audio is made from. Useful for captions,
  karaoke and alignment. Adds the `Word` dataclass. Thanks to
  [@batuhanozkose](https://github.com/batuhanozkose)
  ([#3](https://github.com/canberk7/ema-lightning/pull/3)).

## 1.0.1

- `EMA()` requests the model's `config.json` before the weights.

## 1.0.0

First release.

- `EMA()` with `say()` for one text or a list, `stream()` and `best_batch_size()`.
- `lightning()`: cuDNN benchmark mode, one compilation for any batch size, and CUDA
  graphs at batch sizes 1, 2, 4 and every multiple of 8 up to the batch size,
  checked against the plain path at startup.
- Text frontend on [normalizer-tr](https://github.com/erdemtuna/normalizer-tr)
  0.4 with the `fallback` policy: numbers, dates, times, money, units,
  abbreviations and symbols are read aloud, and nothing is skipped.
- Playhead, a scheduler inside `EMA`: every `say()` and `stream()` call joins two
  first-come-first-served queues (sentences before the model, windows before the
  decoder) that run in shared GPU batches, with no server or database. Closed
  streams leave both queues, and a failing batch fails only its own callers.
  `benches/callers.py` measures it under load.
- Decoder windows of four seconds; a stream's first window is one second. On a
  GPU, windows are padded to 48 or 120 frames and each decode batch holds one size.
- Requires CPython 3.11 or newer, the minimum of normalizer-tr 0.4.
