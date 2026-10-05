"""Shared fixtures: a tiny random model and decoder with the real structure, and a stand-in normalizer."""
import os
import sys
import tempfile
import types

import numpy as np
import pytest
import torch

os.environ["XDG_CACHE_HOME"] = tempfile.mkdtemp()

try:
    import normalizer_tr  # noqa: F401
except ImportError:
    class _Result:
        def __init__(self, text):
            self.normalized_text = text

    class _Normalizer:
        calls = []

        def normalize(self, text, *, ambiguity_policy="preserve", **_):
            self.calls.append(ambiguity_policy)
            return _Result(text.replace("5", "beş").replace("2", "iki"))

    sys.modules["normalizer_tr"] = types.SimpleNamespace(Normalizer=_Normalizer)

from ema_lightning import EMA  # noqa: E402
from ema_lightning.decoder import Decoder  # noqa: E402
from ema_lightning.frontend import Frontend  # noqa: E402
from ema_lightning.model import Acoustic  # noqa: E402

VOCAB = ["<pad>", "<unk>"] + list(" !\"%&'(),-./:;?abcdefghijklmnopqrstuvwxyzçöüğış")


def tiny_model(letter_pos):
    torch.manual_seed(0)
    cfg = dict(d=32, n_heads=2, n_layers=2, text_conv=1, text_attn=1, dur_hidden=16, align_heads=2, ff_mult=2,
               letter_pos=letter_pos, latent_dim=64, distilled=dict(times=[0.0, 0.25, 0.5, 0.75]))
    model = Acoustic(cfg, VOCAB, shared=True)
    model.chardur.out.bias.data.fill_(1.0)  # about two frames per letter, like the real model
    for p in model.aligner.parameters():
        p.data.normal_(0, 0.1)
    return model.eval().requires_grad_(False)


def tiny_decoder():
    torch.manual_seed(1)
    return Decoder(ch=64, rb_kernels=(3,), rb_dilations=((1, 3),)).eval().requires_grad_(False)


@pytest.fixture(params=[True, False], ids=["letter_pos", "word_pos"])
def model(request):
    return tiny_model(request.param)


@pytest.fixture
def decoder():
    return tiny_decoder()


@pytest.fixture
def tts(model, decoder):
    return EMA._from_parts(model, decoder, Frontend(VOCAB), "cpu")


def same_speech(a, b, db=40.0):
    """The same audio made two ways: the same length, and the difference at least `db` dB quieter than the speech.

    Batching, threads and compiled graphs change the last digits of the arithmetic; measured on real weights the
    difference is about 60 dB quieter than the speech, so 40 dB leaves a 20 dB margin.
    """
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert a.shape == b.shape, f"lengths differ: {a.shape} vs {b.shape}"
    diff = float(np.sum((a - b) ** 2))
    if diff == 0.0:
        return
    speech = float(np.sum(a**2))
    quieter = 10 * np.log10(speech / diff) if speech > 0 else float("-inf")
    assert quieter >= db, f"the difference is only {quieter:.1f} dB quieter than the speech"
