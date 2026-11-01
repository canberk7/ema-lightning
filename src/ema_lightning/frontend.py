"""Text frontend: any written Turkish in, text in the model's own alphabet out.

English words are first respelled the Turkish way (developer -> divelopır), except those normalizer-tr
already reads itself, such as brand names. normalizer-tr then reads numbers, dates, times, money, units,
symbols and abbreviations aloud. Its "fallback" policy leaves nothing unread: notation it cannot resolve
is spoken literally instead of being kept as written. The alphabet step then lowercases the Turkish way
and drops whatever the model cannot read. Nothing here ever raises on text.
"""
import re
import unicodedata
from functools import lru_cache

from normalizer_tr import Normalizer

from ema_lightning.phonetics import phoneticize_tr

POLICY = "fallback"
BLOCK_BYTES = 8 * 1024
TURKISH = frozenset("çğıöşüÇĞİÖŞÜ")
TYPOGRAPHY = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "´": "'", "`": "'", "“": '"', "”": '"', "„": '"',
                            "«": '"', "»": '"', "–": "-", "—": "-", "−": "-", "…": "..."})
UNSAFE = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]")


class Frontend:
    def __init__(self, vocab):
        self.vocab = frozenset(vocab)
        self.normalizer = Normalizer()
        self.reads = lru_cache(maxsize=4096)(self._reads)

    def __call__(self, text):
        text = UNSAFE.sub(" ", text.encode("utf-8", "ignore").decode("utf-8"))
        if not text.strip():
            return ""
        return self.alphabet(" ".join(self.spoken(block) for block in blocks(text)))

    def spoken(self, text):
        try:
            text = phoneticize_tr(text, keep=self.reads)
            return self.normalizer.normalize(text, ambiguity_policy=POLICY).normalized_text
        except Exception:  # invalid input or a resource limit: keep the words rather than lose the sentence
            return text

    def _reads(self, word):
        """Whether normalizer-tr has its own reading for a word as written (Google -> gugıl)."""
        try:
            return self.normalizer.normalize(word, ambiguity_policy=POLICY).normalized_text.lower() != word.lower()
        except Exception:
            return False

    def alphabet(self, text):
        text = text.translate(TYPOGRAPHY).replace("İ", "i").replace("I", "ı").lower()
        out = []
        for ch in text:
            if ch not in TURKISH:
                ch = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
            out.append(ch if ch and all(c in self.vocab for c in ch) else " ")
        return re.sub(r"\s+", " ", "".join(out)).strip()


def blocks(text):
    """Split at whitespace into pieces the normalizer accepts in one call."""
    words, block, size = text.split(), [], 0
    for word in words:
        n = len(word.encode("utf-8")) + 1
        if block and size + n > BLOCK_BYTES:
            yield " ".join(block)
            block, size = [], 0
        block.append(word)
        size += n
    if block:
        yield " ".join(block)
