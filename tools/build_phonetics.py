"""Build the data files behind ema_lightning.english and ema_lightning.phonetics.

Inputs (none of them ship with the package):
  --cmudict      cmudict.dict from https://github.com/cmusphinx/cmudict (BSD-2-Clause)
  --checkpoint   checkpoint20.npz from the g2p_en wheel (Apache-2.0)
  --tr-freq      Turkish word frequency list, "word count" per line, most frequent first
                 (tr_50k.txt from https://github.com/hermitdave/FrequencyWords, content/2018)
  --en-freq      English word frequency list in the same format (en_50k.txt from the same place),
                 only used to filter the Turkish list

Outputs in src/ema_lightning/data/:
  en_g2p.npz       int8 GRU weights
  en_lexicon.xz    every CMUdict word; the phonemes are kept only where the network disagrees
  tr_words.xz      frequent Turkish words written with ASCII letters, as is or typed without Turkish letters
  lang_ngrams.xz   character trigram log-odds, English versus Turkish

Usage:
  python tools/build_phonetics.py --cmudict cmudict.dict --checkpoint checkpoint20.npz \
      --tr-freq tr_50k.txt --en-freq en_50k.txt
"""
import argparse
import io
import lzma
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ema_lightning.english import LAYERS, Network, encode_phones, quantize  # noqa: E402
from ema_lightning.phonetics import ENGLISH_SPELLING, NGRAM, NGRAM_STEP, grams  # noqa: E402

OUT = ROOT / "src" / "ema_lightning" / "data"
WORD = re.compile(r"[a-z]+")
TURKISH_WORD = re.compile(r"[a-zçğıöşü]+")
ASCII = str.maketrans("çğıöşü", "cgiosu")


def xz(text):
    return lzma.compress(text.encode("utf-8"), preset=9 | lzma.PRESET_EXTREME)


def read_freq(path, limit):
    out = []
    for line in open(path, encoding="utf-8"):
        parts = line.split()
        if len(parts) == 2:
            out.append((parts[0], int(parts[1])))
        if len(out) >= limit:
            break
    return out


def read_cmudict(path):
    lexicon = {}
    for line in open(path, encoding="utf-8"):
        line = line.split("#")[0].split()
        if len(line) < 2 or "(" in line[0] or not WORD.fullmatch(line[0]) or len(line[0]) < 2:
            continue
        lexicon.setdefault(line[0], tuple(line[1:]))
    return lexicon


def build_network(checkpoint):
    raw = np.load(checkpoint)
    store = {}
    for name in LAYERS:
        values, scale = quantize(raw[name])
        store[name] = values
        if scale is not None:
            store[name + ".scale"] = scale
    buf = io.BytesIO()
    np.savez_compressed(buf, **store)
    (OUT / "en_g2p.npz").write_bytes(buf.getvalue())
    return Network(store)


def build_lexicon(lexicon, net, batch=512):
    words = sorted(lexicon)
    lines, kept = [], 0
    for start in range(0, len(words), batch):
        chunk = words[start:start + batch]
        for word, guess in zip(chunk, net.predict(chunk), strict=True):
            if guess == lexicon[word]:
                lines.append(word + "\t")
            else:
                lines.append(word + "\t" + encode_phones(lexicon[word]))
                kept += 1
    (OUT / "en_lexicon.xz").write_bytes(xz("\n".join(lines)))
    return kept, len(words)


def build_turkish(tr, en):
    en_freq = dict(en)
    en_total, tr_total = sum(en_freq.values()), sum(c for _, c in tr)
    words = set()
    for word, count in tr:
        # Words with Turkish letters are recognised without a list, but people often type them without
        # (değer -> deger, çay -> cay), so their ASCII form goes in.
        word = word.translate(ASCII)
        if not WORD.fullmatch(word):
            continue
        # Subtitles contain English too ("okay", "yeah"). Frequency ratios misjudge short shared words
        # (can, say, at), so only drop a word when it also has English-only spelling and is more English.
        if ENGLISH_SPELLING.search(word) and en_freq.get(word, 0) / en_total >= 0.5 * count / tr_total:
            continue
        words.add(word)
    (OUT / "tr_words.xz").write_bytes(xz("\n".join(sorted(words))))
    return words


def build_ngrams(tr, english_words):
    tr_counts, en_counts = Counter(), Counter()
    for word, _ in tr:
        if TURKISH_WORD.fullmatch(word):
            tr_counts.update(grams(word))
    for word in english_words:
        en_counts.update(grams(word))
    tr_total, en_total = sum(tr_counts.values()), sum(en_counts.values())
    vocab = set(tr_counts) | set(en_counts)
    lines = []
    for g in sorted(vocab):
        odds = (math.log((en_counts[g] + 0.5) / (en_total + 0.5 * len(vocab)))
                - math.log((tr_counts[g] + 0.5) / (tr_total + 0.5 * len(vocab))))
        q = max(-127, min(127, round(odds / NGRAM_STEP)))
        if q:
            lines.append(f"{g}\t{q}")
    (OUT / "lang_ngrams.xz").write_bytes(xz("\n".join(lines)))
    return len(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cmudict", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tr-freq", required=True)
    parser.add_argument("--en-freq", required=True)
    parser.add_argument("--tr-words", type=int, default=50000)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    lexicon = read_cmudict(args.cmudict)
    net = build_network(args.checkpoint)
    kept, total = build_lexicon(lexicon, net)
    tr = read_freq(args.tr_freq, args.tr_words)
    en = read_freq(args.en_freq, 10 ** 9)
    turkish = build_turkish(tr, en)
    ngrams = build_ngrams(tr, lexicon)
    print(f"lexicon: {total} words, {kept} need stored phonemes (n={NGRAM} grams: {ngrams}, tr words: {len(turkish)})")
    for path in sorted(OUT.iterdir()):
        print(f"  {path.name:16s} {path.stat().st_size / 1024:8.1f} KB")


if __name__ == "__main__":
    main()
