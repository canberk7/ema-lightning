"""Greedy chunking: spoken text in, pieces short enough for one pass of the model out.

Text that fits in about ten seconds of speech stays one piece; the model reads its punctuation itself.
Longer text is cut at the last good spot inside each ten-second window: a sentence end, then a clause
mark, then a space, and only when there is none of those, exactly at the limit.
"""
import re

LETTERS_PER_SECOND = 18.0
MAX_SECONDS = 10.0
MAX_LETTERS = 250
SENTENCE_PAUSE = 0.25
CLAUSE_PAUSE = 0.12
CUTS = ((re.compile(r"[.!?]+[\"')]*(?= )"), SENTENCE_PAUSE), (re.compile(r"[,;:](?= )"), CLAUSE_PAUSE),
        (re.compile(r"\S(?= )"), CLAUSE_PAUSE))
LETTER = re.compile(r"[^\W\d_]")


def chunk(text, speed):
    """[(piece, seconds of silence after it)], each piece ending in terminal punctuation."""
    limit = int(min(MAX_LETTERS, LETTERS_PER_SECOND * MAX_SECONDS * speed))
    pieces, rest = [], text.strip()
    while rest:
        cut, pause = len(rest), 0.0
        if len(rest) > limit:
            cut = limit
            for pattern, gap in CUTS:
                ends = [m.end() for m in pattern.finditer(rest, 0, limit + 1)]
                if ends:
                    cut, pause = ends[-1], gap
                    break
        piece, rest = rest[:cut].strip(), rest[cut:].strip()
        if LETTER.search(piece):
            pieces.append((finish(piece), pause))
    if pieces:
        pieces[-1] = (pieces[-1][0], 0.0)
    return pieces


def finish(piece):
    """End every piece the way the model was trained: on a sentence end."""
    if piece.rstrip("\"')")[-1:] in (".", "!", "?"):
        return piece
    return piece.rstrip(",;:- ") + "."
