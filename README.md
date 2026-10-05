# EMA Lightning

Tiny, fast and accurate Turkish text to speech. 8.6M parameters, Apache-2.0.
Runs offline on a GPU or a CPU; no text or audio leaves the machine.

```text
Toplantı 14:30'da, bütçe 1.250.000 TL.
→ toplantı on dört otuzda, bütçe bir milyon iki yüz elli bin türk lirası.  (spoken at 48 kHz)
```

## Install

```sh
python -m pip install ema-lightning
```

CPython 3.11 to 3.13 with PyTorch 2.1 or newer. The weights (`ema.pt` and
`decoder.pt`, about 34 MB) download from
[Hugging Face](https://huggingface.co/canberkkkkkk/ema-lightning) on first use.

## Use

```python
from ema_lightning import EMA

tts = EMA()                                     # uses the GPU if there is one
speech = tts.say("Merhaba! Bugün hava çok güzel.", path="merhaba.wav")
print(speech.duration, speech.sample_rate)      # seconds, 48000
```

On an NVIDIA GPU, chain `.lightning()`. It turns on cuDNN benchmark mode,
compiles the model once, records CUDA graphs for a ladder of batch sizes and
lengths, checks them against the plain path and reports when it is ready:

```python
tts = EMA().lightning()
```

A list is generated in batches and returns one `Speech` per text, in order:

```python
speeches = tts.say(["Birinci cümle.", "İkinci cümle.", "Üçüncü cümle."], path="clips")
```

To play audio while the rest is still being made, stream it. Chunks arrive about
one second first, then four seconds at a time:

```python
for chunk in tts.stream("Uzun bir metin, cümle cümle seslendirilir."):
    play(chunk)                                 # float32 NumPy array
```

`speed` (0.25 to 4), `seed` and `sample_rate` (48000, 24000, 16000, 8000) are
keyword options on `say()` and `stream()`. The same seed gives the same audio. Any
text is accepted; invalid settings raise before any work starts.

One `EMA` serves many callers at once: call `say()` or `stream()` from as many
threads as you need. A built-in scheduler queues every caller's sentences first
come, first served and runs them in shared GPU batches. No server or database to run.

## Text

Written Turkish goes in as it is. [normalizer-tr](https://github.com/erdemtuna/normalizer-tr),
with its `fallback` policy, reads numbers, dates, times, amounts, units,
abbreviations and symbols aloud, and spells out anything it cannot resolve
instead of skipping it. Long text is split at sentence and clause boundaries,
with natural pauses between the pieces.

| Written | Spoken |
|---|---|
| `5 kişi geldi.` | beş kişi geldi. |
| `%15 indirim` | yüzde on beş indirim |
| `12,5 kg un` | on iki virgül beş kilogram un |
| `Dr. Ayşe geldi.` | doktor ayşe geldi. |
| `Kod: 00042` | kod: sıfır sıfır sıfır dört iki |

See [the text guide](docs/text.md) for the full pipeline and its known limits.

## Documentation

| Guide | Contents |
|---|---|
| [API reference](docs/api.md) | Every class, method, option and error |
| [Text](docs/text.md) | Normalization, the alphabet, splitting and pauses |
| [Many callers](docs/scheduling.md) | The two queues behind `say()` and `stream()`, windows and batches |
| [Performance](PERFORMANCE.md) | Accuracy and speed, with how they were measured |
| [Contributing](CONTRIBUTING.md) | Setup, code layout and checks |
| [Changelog](CHANGELOG.md) | Changes by release |

EMA Lightning has one voice. It reads Turkish only; foreign words are read with
Turkish spelling rules.

## License

[Apache-2.0](LICENSE), for the code and the weights. Commercial use included.

Text normalization by [normalizer-tr](https://github.com/erdemtuna/normalizer-tr)
(Apache-2.0). Thanks to [Erdem Tuna](https://github.com/erdemtuna) for building it.
