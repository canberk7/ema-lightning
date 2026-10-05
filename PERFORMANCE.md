# Performance and measurement

Measured on the release candidate (5.6M-parameter acoustic model with the
3M-parameter decoder) at speed 1.0. Accuracy was measured with the release
candidate's own text normalizer, before the switch to normalizer-tr.

## Accuracy

Freya-TR-Eval, all 495 sentences, three seeds, raw text in.

| Metric | Value |
|---|---|
| Word error rate | 0.92% (0.79 to 1.05 across seeds) |
| Character error rate | 0.18% |

Transcripts come from Whisper large-v3 (beam size 5) on an 8 kHz band, scored
with Freya's text normalization. Some counted errors are the recognizer's.

## Speed

RTX 4090. Real-time factor (RTF) is generation time divided by audio length:
three passes after a warm-up, middle pass reported.

| | Plain PyTorch | `lightning()` |
|---|---|---|
| One request | RTF 0.0115 (87× real time) | RTF 0.0023 (440× real time) |
| Batch of 64 | 952× real time | 1,316× real time |
| First audio, typical sentence | | 3.86 ms, normalization included |

On a CPU the model runs at RTF 0.166, about 6× real time (measured on a
cloud-container CPU).

At $0.74 an hour for a rented RTX 4090, the batched compiled path reads about
24,000 characters a second: about $0.0085 per million characters.

## The package on an RTX PRO 6000

Measured with this package (`lightning()`, cuDNN benchmark mode on), RTX PRO 6000
Blackwell Server Edition, PyTorch 2.11.

| | |
|---|---|
| First audio from `stream()` | 4.3 ms typical (4.2 to 4.4 ms with graphs recorded up to batch 8, 32, 64 or 128) |
| 1,000 short texts in one `say()` list | about 1,350× real time |
| Plain path, best batch | 1,431× real time at batch 16 |

## Many callers

`benches/callers.py` starts 1, 8, 32, 64 and 128 callers at the same instant, each
calling `say()` with one sentence, and reports how long the calls took (median and
95th percentile) and the total real-time multiple.

```sh
python -m pip install tqdm
python benches/callers.py
```

## Measure on your machine

```python
import time
from ema_lightning import EMA

tts = EMA().lightning()                     # prints first audio, a short sentence's time and the batch size
text = "Bugün hava çok güzel, yarın da yağmur yağacakmış."
start = time.perf_counter()
speech = tts.say(text, seed=0)
print("RTF", (time.perf_counter() - start) / speech.duration)
```
