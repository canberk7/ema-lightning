"""The lightning path, on the tiny model: runs wherever a CUDA GPU is present, skips elsewhere."""
import numpy as np
import pytest
import torch
from conftest import VOCAB, same_speech, tiny_decoder, tiny_model

from ema_lightning import EMA
from ema_lightning.frontend import Frontend

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TEXT = "Bugün hava çok güzel. Yarın da yağmur yağacakmış, toplantı öğleden sonra başlayacak."
LONG = " ".join(["Bu cümle yaklaşık olarak dört saniye sürecek kadar uzun bir cümledir."] * 5)


@pytest.fixture(scope="module")
def pair():
    def build():
        return EMA._from_parts(tiny_model(True).cuda(), tiny_decoder().cuda(), Frontend(VOCAB), "cuda")

    plain, fast = build(), build()
    assert fast.lightning(batch_size=4) is fast
    return plain, fast


def test_lightning_records_graphs(pair):
    _, fast = pair
    assert fast._engine.graphs is not None and fast._engine.graphs.count == 20
    assert fast.lightning() is fast


@pytest.mark.parametrize("text", [TEXT, LONG, "Evet."])
def test_lightning_sounds_like_the_plain_path(pair, text):
    plain, fast = pair
    a, b = plain.say(text, seed=1).audio, fast.say(text, seed=1).audio
    same_speech(a, b)


def test_lightning_stream_and_lists(pair):
    _, fast = pair
    joined = np.concatenate(list(fast.stream(LONG, seed=2, sample_rate=16000)))
    same_speech(joined, fast.say(LONG, seed=2, sample_rate=16000).audio)
    texts = [TEXT, "Evet.", LONG, "", "Kısa bir cümle.", TEXT, TEXT]
    for text, s in zip(texts, fast.say(texts, seed=3), strict=True):
        same_speech(s.audio, fast.say(text, seed=3).audio)


def test_pieces_beyond_the_largest_bucket_fall_back_to_plain(pair):
    text = ("bu çok uzun bir parça olacak. " * 8).strip()
    latents = []
    for tts in pair:
        piece = tts._engine.piece(text, 0.0, 4)
        tts._engine.plan([piece], 0.5)
        tts._engine.think([piece])
        latents.append(piece.latents)
    assert latents[0].shape[0] > 320 and torch.allclose(latents[0], latents[1], atol=1e-3)


def test_lightning_turns_on_cudnn_benchmark(pair):
    assert torch.backends.cudnn.benchmark
