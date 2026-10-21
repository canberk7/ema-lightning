"""English grapheme-to-phoneme: a pronunciation lexicon with a small neural fallback.

The network is g2p_en's GRU encoder-decoder (Apache-2.0, Kyubyong Park and Jongseok Kim), stored as int8
and run with numpy. The lexicon is every CMUdict (BSD-2-Clause) word, with phonemes stored only for the
words the network gets wrong, so a lookup followed by the network gives the same answer as the full
dictionary in a fraction of the space. Both files are produced by ``tools/build_phonetics.py``.
"""
import lzma
from functools import lru_cache
from importlib import resources

import numpy as np

GRAPHEMES = ["<pad>", "<unk>", "</s>"] + list("abcdefghijklmnopqrstuvwxyz")
PHONEMES = ["<pad>", "<unk>", "<s>", "</s>"] + [
    "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0", "AO1", "AO2", "AW0", "AW1", "AW2",
    "AY0", "AY1", "AY2", "B", "CH", "D", "DH", "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2",
    "F", "G", "HH", "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG", "OW0", "OW1",
    "OW2", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "T", "TH", "UH0", "UH1", "UH2", "UW", "UW0", "UW1",
    "UW2", "V", "W", "Y", "Z", "ZH",
]
G2IDX = {g: i for i, g in enumerate(GRAPHEMES)}
P2IDX = {p: i for i, p in enumerate(PHONEMES)}
START, END, MAX_PHONES = 2, 3, 20
CODE_BASE = 0x30  # the lexicon writes phoneme i as chr(CODE_BASE + i)

DATA = resources.files("ema_lightning") / "data"
LEXICON_FILE = "en_lexicon.xz"
NETWORK_FILE = "en_g2p.npz"
LAYERS = ("enc_emb", "enc_w_ih", "enc_w_hh", "enc_b_ih", "enc_b_hh",
          "dec_emb", "dec_w_ih", "dec_w_hh", "dec_b_ih", "dec_b_hh", "fc_w", "fc_b")


def encode_phones(phones):
    return "".join(chr(CODE_BASE + P2IDX[p]) for p in phones)


def decode_phones(code):
    return tuple(PHONEMES[ord(c) - CODE_BASE] for c in code)


def quantize(weights):
    """Per-row symmetric int8. Returns arrays to store under ``name`` and ``name.scale``."""
    weights = np.asarray(weights, np.float32)
    if weights.ndim == 1:
        return weights.astype(np.float16), None
    scale = np.abs(weights).max(axis=-1, keepdims=True) / 127.0
    scale[scale == 0] = 1.0
    return np.round(weights / scale).astype(np.int8), scale.astype(np.float32)


def dequantize(store, name):
    values = store[name].astype(np.float32)
    scale_key = name + ".scale"
    return values * store[scale_key] if scale_key in store else values


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _gru_cell(x, h, w_ih, w_hh, b_ih, b_hh):
    gi = x @ w_ih.T + b_ih
    gh = h @ w_hh.T + b_hh
    split = gi.shape[-1] * 2 // 3
    r, z = np.split(_sigmoid(gi[:, :split] + gh[:, :split]), 2, axis=-1)
    n = np.tanh(gi[:, split:] + r * gh[:, split:])
    return (1.0 - z) * n + z * h


class Network:
    """Greedy GRU encoder-decoder, batched over words."""

    def __init__(self, store):
        for name in LAYERS:
            setattr(self, name, dequantize(store, name))

    def predict(self, words):
        """Phonemes for each lowercase word."""
        if not words:
            return []
        lengths = np.array([len(w) + 1 for w in words])
        ids = np.zeros((len(words), lengths.max()), np.int64)
        for row, word in enumerate(words):
            ids[row, :len(word) + 1] = [G2IDX.get(c, 1) for c in word] + [G2IDX["</s>"]]
        x = self.enc_emb[ids]
        h = np.zeros((len(words), self.enc_w_hh.shape[1]), np.float32)
        for t in range(ids.shape[1]):
            step = _gru_cell(x[:, t], h, self.enc_w_ih, self.enc_w_hh, self.enc_b_ih, self.enc_b_hh)
            h = np.where((t < lengths)[:, None], step, h)

        token = np.full(len(words), START)
        done = np.zeros(len(words), bool)
        out = [[] for _ in words]
        for _ in range(MAX_PHONES):
            h = _gru_cell(self.dec_emb[token], h, self.dec_w_ih, self.dec_w_hh, self.dec_b_ih, self.dec_b_hh)
            token = (h @ self.fc_w.T + self.fc_b).argmax(axis=-1)
            done |= token == END
            for row in np.flatnonzero(~done):
                out[row].append(PHONEMES[token[row]])
            if done.all():
                break
        return [tuple(p) for p in out]


@lru_cache(maxsize=1)
def network():
    with (DATA / NETWORK_FILE).open("rb") as f:
        return Network(dict(np.load(f)))


@lru_cache(maxsize=1)
def lexicon():
    """Every CMUdict word, mapped to its encoded phonemes, or to "" where the network already agrees."""
    text = lzma.decompress((DATA / LEXICON_FILE).read_bytes()).decode("ascii")
    return dict(line.split("\t") for line in text.splitlines() if line)


def known(word):
    return word in lexicon()


@lru_cache(maxsize=4096)
def pronounce(word):
    """ARPAbet phonemes (with stress digits) for a lowercase a-z word."""
    code = lexicon().get(word)
    if code:
        return decode_phones(code)
    return network().predict([word])[0]
