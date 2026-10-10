import random
import types

import normalizer_tr
import pytest
from conftest import VOCAB

from ema_lightning.frontend import BLOCK_BYTES, POLICY, Frontend, blocks

ALPHABET = set(VOCAB[2:])
real_normalizer = pytest.mark.skipif(not hasattr(normalizer_tr, "NORMALIZER_ID"), reason="normalizer-tr not installed")


class Recorder:
    def __init__(self):
        self.calls = []

    def normalize(self, text, *, ambiguity_policy="preserve", **_):
        self.calls.append(ambiguity_policy)
        return types.SimpleNamespace(normalized_text=text)


def test_turkish_lowercasing_and_folding():
    f = Frontend(VOCAB)
    assert f.alphabet("İSTANBUL'da IŞIK") == "istanbul'da ışık"
    assert f.alphabet("Café “déjà vu” — tamam…") == 'cafe "deja vu" - tamam...'
    assert f.alphabet("Şöyle 🙂 güzel") == "şöyle güzel"


def test_asks_the_normalizer_for_the_fallback_policy():
    f = Frontend(VOCAB)
    f.normalizer = Recorder()
    f("5 kişi")
    assert POLICY == "fallback" and f.normalizer.calls == ["fallback"]


def test_never_raises_and_only_emits_the_alphabet():
    f = Frontend(VOCAB)
    rng = random.Random(0)
    for _ in range(300):
        text = "".join(chr(rng.randrange(0, 0x2FFFF)) for _ in range(rng.randrange(0, 60)))
        out = f(text)
        assert isinstance(out, str) and set(out) <= ALPHABET


def test_normalizer_failure_keeps_the_text():
    class Broken:
        def normalize(self, *_, **__):
            raise RuntimeError("result limit")

    f = Frontend(VOCAB)
    f.normalizer = Broken()
    assert f("Merhaba dünya") == "merhaba dünya"


def test_long_text_is_normalized_in_safe_blocks():
    text = " ".join(["kelime"] * 5000)
    parts = list(blocks(text))
    assert len(parts) > 1 and all(len(p.encode()) <= BLOCK_BYTES for p in parts)
    assert " ".join(parts) == text


@real_normalizer
@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("5 kişi geldi.", "beş kişi geldi."),
        ("Toplantı 14:30'da.", "toplantı on dört otuzda."),
        ("Bütçe 1.250.000 TL.", "bütçe bir milyon iki yüz elli bin türk lirası."),
        ("%15 indirim", "yüzde on beş indirim"),
        ("Kod: 00042", "kod: sıfır sıfır sıfır dört iki"),
        ("ChatGPT'ye sordum.", "çet ci pi tiye sordum."),
        ("Instagram'da gördüm, WhatsApp'tan yazdım.", "instagramda gördüm, vatsaptan yazdım."),
        ("TRT'de yayınlandı.", "te re tede yayınlandı."),
        ("SON DAKİKA", "son dakika"),
    ],
)
def test_reads_numbers_and_notation_aloud(text, spoken):
    assert Frontend(VOCAB)(text) == spoken


@real_normalizer
def test_fallback_leaves_no_digit_unread():
    out = Frontend(VOCAB)("Saat 25:70, 40.03.2026, IV. Murat, 10-15 kişi, 3. kat, x² ve 🫠")
    assert out and not any(ch.isdigit() for ch in out)
