from ema_lightning.chunker import CLAUSE_PAUSE, MAX_LETTERS, SENTENCE_PAUSE, chunk, finish

SENTENCE = "bu cümle yaklaşık olarak dört saniye sürecek kadar uzun bir cümledir."


def test_short_text_stays_one_piece():
    assert chunk("merhaba, nasılsınız", 1.0) == [("merhaba, nasılsınız.", 0.0)]


def test_long_text_is_cut_at_sentence_ends():
    pieces = chunk(" ".join([SENTENCE] * 6), 1.0)
    assert len(pieces) > 1
    assert all(len(p) <= 180 for p, _ in pieces)
    assert [pause for _, pause in pieces[:-1]] == [SENTENCE_PAUSE] * (len(pieces) - 1)
    assert pieces[-1][1] == 0.0
    assert all(p.endswith(".") for p, _ in pieces)


def test_without_punctuation_it_cuts_at_spaces():
    pieces = chunk(" ".join(["kelime"] * 100), 1.0)
    assert all(pause == CLAUSE_PAUSE for _, pause in pieces[:-1])
    assert all(not p[:-1].endswith(" ") and "kelim." not in p for p, _ in pieces)


def test_one_giant_token_is_cut_by_force():
    pieces = chunk("a" * 1000, 1.0)
    assert all(len(p) <= 181 for p, _ in pieces)
    assert all(pause == 0.0 for _, pause in pieces)
    assert sum(len(p) - 1 for p, _ in pieces) == 1000


def test_slower_speech_means_shorter_pieces():
    text = " ".join([SENTENCE] * 6)
    assert len(chunk(text, 0.5)) > len(chunk(text, 1.0)) > len(chunk(text, 2.0))
    assert all(len(p) <= MAX_LETTERS + 1 for p, _ in chunk(text, 4.0))


def test_nothing_speakable_gives_no_pieces():
    assert chunk("", 1.0) == []
    assert chunk("   ", 1.0) == []
    assert chunk("... ? !", 1.0) == []


def test_every_piece_ends_like_a_sentence():
    assert finish("evet,") == "evet."
    assert finish("tamam") == "tamam."
    assert finish('dedi."') == 'dedi."'
    assert finish("gerçekten mi?") == "gerçekten mi?"
