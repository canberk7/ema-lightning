"""Turkish-friendly spellings for English words found in Turkish text.

Turkish is read as written, so Turkish words need no dictionary. Only foreign words do, and those are
handled by rules instead of a hand-written list:

1. Detect. A word written in Turkish letters, a frequent Turkish word (also as typed without Turkish
   letters: deger, cay), or a known Turkish word plus a suffix stays as is. Otherwise English-only spelling
   (w, q, x, sh, th, oo, ...), or a CMUdict word with an English-looking letter pattern, marks it as foreign.
2. Pronounce. ``ema_lightning.english`` gives its English phonemes (CMUdict, then a small network).
3. Spell. Each phoneme becomes Turkish letters; unstressed vowels follow the written vowel the way Turkish
   speakers say them (developer -> divelopır, docker -> dokır).

Suffixes written after an apostrophe (developer'ın) are already Turkish and are left alone.
"""
import lzma
import math
import re
from functools import lru_cache

from ema_lightning import english

# Words whose usual Turkish reading is not the English one, or compound tech terms.
EXCEPTIONS = {
    "croissant": "kruvasan",
    "latte": "latte",
    "github": "githab",
    "unity": "yuniti",
    "coffee": "kafi",
    "javascript": "javasikript",
    "typescript": "taypsikript",
    "kubernetes": "kubırnetis",
    "backend": "bekend",
    "deploy": "diploy",
    "meeting": "miting",
    "deadline": "didlayn",
    "commit": "komit",
    "push": "puş",
    "imac": "aymek",
    "software": "softveyır",
    "react": "reakt",
}

NGRAM = 3
NGRAM_STEP = 1 / 16  # the n-gram table stores log-odds as int8 in these steps
TURKISH_LETTERS = frozenset("çğıöşüÇĞİÖŞÜ")
# Spellings Turkish never uses: an English word for certain.
ENGLISH_SPELLING = re.compile(
    r"[qwx]|sh|th|ph|ck|gh|oo|ee|ea|ou|oa|ie|ue|ui|eu|tion|ing$|(?<![aeiou])y(?![aeiou])"
)
# First letters of common Turkish suffixes, to recognise "<known word><suffix>" without an apostrophe.
TURKISH_SUFFIX = re.compile(
    r"(l[ae]r|d[ae]|t[ae]|[ıiuüae]|n[ıiuüae]|y[ıiuüae]|s[ıiuü]|m|k|c[ıiuü]|ç[ıiuü]|l[ıiuü]|k[ıiuü]|m[ae]|ş)"
)
STRONG_MIN, LEXICON_MIN = -1.0, 1.0
# The network garbles long words it has not seen (hyperparameterization): a quarter of the phonemes go
# wrong from about this length even on its training words, so Turkish rules read them better.
GUESS_MAX = 13

# A word on its own: not glued to an address, a file name or code (www.google.com, ali@gmail.com, next.js).
TOKEN = re.compile(r"(?<![\w@/])(?<!\w\.)[^\W\d_]+(?![\w@])(?![./:]\w)")
CAMEL = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])")
VOWEL_GROUP = re.compile(r"[aeiou]+y?|(?<=[^aeiou])y(?![aeiou])|y$")

CONSONANTS = {
    "B": "b", "CH": "ç", "D": "d", "DH": "d", "F": "f", "G": "g", "HH": "h", "JH": "c", "K": "k", "L": "l",
    "M": "m", "N": "n", "NG": "ng", "P": "p", "R": "r", "S": "s", "SH": "ş", "T": "t", "TH": "t", "V": "v",
    "W": "v", "Y": "y", "Z": "z", "ZH": "j",
}
VOWELS = {
    "AA": "a", "AE": "e", "AH": "a", "AO": "o", "AW": "au", "AY": "ay", "EH": "e", "ER": "ör", "EY": "ey",
    "IH": "i", "IY": "i", "OW": "o", "OY": "oy", "UH": "u", "UW": "u",
}


def grams(word, n=NGRAM):
    word = "^" + word + "$"
    return [word[i:i + n] for i in range(len(word) - n + 1)]


def _xz_lines(name):
    return lzma.decompress((english.DATA / name).read_bytes()).decode("utf-8").splitlines()


@lru_cache(maxsize=1)
def turkish_words():
    return frozenset(_xz_lines("tr_words.xz"))


@lru_cache(maxsize=1)
def ngram_table():
    return {g: int(q) * NGRAM_STEP for g, q in (line.split("\t") for line in _xz_lines("lang_ngrams.xz"))}


def englishness(word):
    """Length-normalised log-odds that the letters are English rather than Turkish."""
    table, gs = ngram_table(), grams(word)
    return sum(table.get(g, 0.0) for g in gs) / math.sqrt(len(gs))


def is_turkish(word):
    words = turkish_words()
    if word in words:
        return True
    return any(word[:i] in words and TURKISH_SUFFIX.match(word, i) for i in range(len(word) - 1, 3, -1))


def is_foreign(word):
    """Decide for a lowercase a-z word. Misreading Turkish is worse than missing English, so lean Turkish."""
    if word in EXCEPTIONS:
        return True
    if len(word) < 2 or is_turkish(word):
        return False
    # Letter patterns alone misread unlisted Turkish (agac, gunes, nilufer), so a word needs English spelling
    # or a CMUdict entry before its letters are weighed.
    if ENGLISH_SPELLING.search(word):
        return englishness(word) > STRONG_MIN and (len(word) <= GUESS_MAX or english.known(word))
    return english.known(word) and englishness(word) > LEXICON_MIN


def _written_vowels(word, count):
    """The written vowel group behind each spoken vowel, or None when they cannot be paired up."""
    groups = [m.group() for m in VOWEL_GROUP.finditer(word)]
    if len(groups) == count + 1 and re.search(r"[^aeiou]e$", word):
        groups.pop()  # silent final e: software, code
    return groups if len(groups) == count else None


def spell(word, phones):
    """Turkish letters for an English pronunciation; ``word`` guides unstressed vowels."""
    vowels = [p for p in phones if p[0] in "AEIOU"]
    written = _written_vowels(word, len(vowels)) or [""] * len(vowels)
    out, v = [], 0
    for phone in phones:
        base, stress = phone.rstrip("012"), phone[len(phone.rstrip("012")):]
        if base not in VOWELS:
            out.append(CONSONANTS.get(base, ""))
            continue
        letter = written[v][:1]
        v += 1
        sound = VOWELS[base]
        if base == "AA" and letter == "o":
            sound = "o"  # docker, body: Turkish speakers follow the spelling
        elif base == "AH" and stress == "0":
            sound = {"a": "a", "e": "e", "i": "i", "o": "o", "u": "a"}.get(letter, "ı")
        elif base == "ER" and stress == "0":
            sound = "ır"
        out.append(sound)
    text = "".join(out)
    return text.replace("ngk", "nk").replace("ngg", "ng").replace("yy", "y")


@lru_cache(maxsize=4096)
def turkish_spelling(word):
    """Turkish spelling for a lowercase English word."""
    if word in EXCEPTIONS:
        return EXCEPTIONS[word]
    return spell(word, english.pronounce(word))


def _convert(token, keep):
    if any(c in TURKISH_LETTERS for c in token) or not token.isascii():
        return token
    lower = token.lower()
    if lower in EXCEPTIONS:
        return EXCEPTIONS[lower]
    if token.isupper():
        return token  # an acronym (API, UNICEF, UEFA) is read the Turkish way or spelled by the normalizer
    parts = CAMEL.findall(token)
    camel = len(parts) > 1 and not english.known(lower)
    if not (any(is_foreign(p.lower()) for p in parts) if camel else is_foreign(lower)):
        return token
    if keep is not None and keep(token):
        return token  # the normalizer has its own reading and joins Turkish suffixes to it (Google'da -> gugılda)
    return "".join(turkish_spelling(p.lower()) for p in parts) if camel else turkish_spelling(lower)


def phoneticize_tr(text, keep=None):
    """Rewrite foreign words in Turkish text with Turkish-friendly spellings.

    ``keep``, when given, is called with a foreign word as written and returns True to leave it alone. The
    frontend uses it for words normalizer-tr already reads, such as brand names.
    """
    if not text:
        return text
    return TOKEN.sub(lambda match: _convert(match.group(), keep), text)
