# Contributing

Report bugs with a minimal example: the text, the settings (`speed`, `seed`,
`sample_rate`), what you expected to hear, what you heard, the package version,
PyTorch version and device. Attach audio only if you made it with a fixed seed.
Do not include private or customer text.

If a number, date, abbreviation or symbol is read wrongly, check it with
normalizer-tr directly first:

```python
from normalizer_tr import Normalizer
print(Normalizer().normalize("3. kat", ambiguity_policy="fallback").normalized_text)
```

If normalizer-tr gives the same reading, report it
[there](https://github.com/erdemtuna/normalizer-tr/issues).

## Set up

```sh
git clone <this repository>
cd ema-lightning
python -m venv .venv && . .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

The tests use tiny random models with the real architecture, so they need no
weights, no network and no GPU. GPU tests run only when CUDA is available.

## Code layout

| Module | Owns |
|---|---|
| `api.py` | The public `EMA` and `Speech`, settings validation, seeds and WAV output |
| `frontend.py` | Cleaning, normalizer-tr with the fallback policy, the model's alphabet |
| `chunker.py` | Splitting text into pieces with pauses |
| `scheduler.py` | Playhead: the background loop that owns the GPU, with the two first-come-first-served queues (sentences, then windows) |
| `engine.py` | The three stages Playhead runs: plan (durations and the frame timeline), think (sampling) and decode (windows, padded to the decoder sizes on a GPU) |
| `model.py` | The acoustic model: text encoder, durations, aligner and latent generator |
| `decoder.py` | The decoder from latents to 48 kHz audio |
| `graphs.py` | Compilation and CUDA graphs for `lightning()`: the batch-size ladder and the length buckets |
| `audio.py` | Resampling and WAV writing |

Language rules belong in normalizer-tr, not here. `frontend.py` only cleans, calls
it and maps to the alphabet.

## Rules

- Text never raises. A failing normalizer keeps the text as written.
- The same audio made two ways (a list, a stream, many threads, the fast path)
  must match saying the text alone: the same length, with the difference at least
  40 dB quieter than the speech. `same_speech()` in `tests/conftest.py` checks it.
- Only Playhead's loop runs the engine during `say()` and `stream()`. Anything
  else that uses the engine directly (startup checks, measurements) holds
  `engine.lock`.
- Both queues are strictly first come, first served. Whatever does not fit in a
  batch stays at the front for the next turn, and nothing is ever reordered. On a
  GPU, a decode batch holds one window size.
- A failing batch fails only the callers in it; the loop keeps serving.
- The compiled path must match the plain path; `lightning()` checks this at startup.
- Invalid settings raise `ValueError` before any work.
- Add a test with every change in behavior. Do not change expected readings just
  to make a test pass.

## Checks

```sh
python -m ruff check src tests examples benches
python -m pytest -q
```

| Tests | What they check |
|---|---|
| `test_api.py` | The public calls: `say()`, `stream()`, sample rates, seeds, lists, empty text, bad settings, files, leaving a stream early |
| `test_playhead.py` | Both queues' order, overflow, one window size per batch, hang-ups, a failure in each stage, many threads. Runs on a recording stand-in engine, so batch contents are checked exactly |
| `test_engine.py` | Windows, padding to the decoder sizes, the batch-size ladder, the bucketed path against the plain path |
| `test_model.py` | The model and the decoder are exact under padding |
| `test_chunker.py` | Cutting text into sentences and clauses, with pauses |
| `test_frontend.py` | Cleaning, normalizer-tr with the fallback policy, the alphabet |
| `test_audio.py` | Resampling and WAV writing |
| `test_gpu.py` | `lightning()` on a real GPU: graphs against the plain path, streams and lists, benchmark mode. Skipped without CUDA |

On a GPU, run `benches/callers.py` after any change to `scheduler.py`.

CI runs both on CPython 3.11, 3.12 and 3.13 without weights or a GPU.

## Releases

Bump the version in `pyproject.toml` and `src/ema_lightning/__init__.py`, add the
changes to `CHANGELOG.md`, re-measure [PERFORMANCE.md](PERFORMANCE.md) if the model
or frontend changed, then build and publish. Publishing is done by the owner only.

Contributions are Apache-2.0. Keep weights, audio, environments and credentials
out of Git.
