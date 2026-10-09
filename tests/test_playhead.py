"""Playhead's two queues, checked on a recording stand-in engine, then end to end on the tiny model."""
import threading
import time

import numpy as np
import pytest
import torch
from conftest import same_speech

from ema_lightning.engine import Piece, windows
from ema_lightning.scheduler import DONE, Playhead

HOP = 1920


class Recorder:
    """The engine's three stages, recording every batch. Every word lasts 25 frames."""

    def __init__(self, fail=()):
        self.lock = threading.RLock()
        self.calls = []
        self.fail = list(fail)  # stages whose next call raises, in order

    def _maybe_fail(self, stage):
        if self.fail and self.fail[0] == stage:
            self.fail.pop(0)
            raise RuntimeError(f"{stage} failed")

    def plan(self, pieces, speed):
        self.calls.append(("plan", list(pieces)))
        self._maybe_fail("plan")
        for p in pieces:
            p.fw = torch.zeros(25 * len(p.text.split()), dtype=torch.long)
            p.h = p.dur = torch.zeros(1)

    def think(self, pieces):
        self.calls.append(("think", list(pieces)))
        self._maybe_fail("think")
        for p in pieces:
            p.latents = torch.zeros(p.frames, 4)

    def decode(self, items):
        self.calls.append(("decode", list(items)))
        self._maybe_fail("decode")
        return [torch.full(((e - s) * HOP,), float(p.seed)) for p, (s, e) in items]

    def batches(self, stage):
        return [batch for kind, batch in self.calls if kind == stage]


def sentences(caller, count, words=1, pause=0.0):
    """`count` sentences of `words` words each; a sentence's seed is caller * 1000 + its number."""
    text = " ".join(["kelime"] * words)
    n = len(text)
    return [Piece(text, pause, caller * 1000 + i, torch.zeros(n, dtype=torch.long),
                  torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long)) for i in range(count)]


def submit_together(playhead, callers):
    with playhead.cond:  # the loop cannot take the inbox until every caller is in
        return [playhead.submit(pieces, 1.0) for pieces in callers]


def drain(request, timeout=10):
    chunks = []
    while (item := request.outbox.get(timeout=timeout)) is not DONE:
        if isinstance(item, BaseException):
            raise item
        chunks.append(item)
    return chunks


def test_everything_that_arrives_together_is_planned_together():
    engine = Recorder()
    playhead = Playhead(engine, lambda: 8)
    for r in submit_together(playhead, [sentences(c, 1, words=9) for c in range(4)]):
        drain(r)
    plans = engine.batches("plan")
    assert len(plans) == 1 and len(plans[0]) == 4


def test_queue_1_is_first_come_first_served():
    engine = Recorder()
    playhead = Playhead(engine, lambda: 2)
    for r in submit_together(playhead, [sentences(1, 3), sentences(2, 2), sentences(3, 1)]):
        drain(r)
    thought = [[p.seed for p in batch] for batch in engine.batches("think")]
    assert thought == [[1000, 1001], [1002, 2000], [2001, 3000]]


def test_queue_2_is_first_come_first_served():
    engine = Recorder()
    playhead = Playhead(engine, lambda: 2)
    for r in submit_together(playhead, [sentences(1, 2, words=9), sentences(2, 1, words=9)]):
        drain(r)
    decoded = [(p.seed, span) for batch in engine.batches("decode") for p, span in batch]
    expected = [(seed, span) for seed in (1000, 1001, 2000) for span in windows(9 * 25)]
    assert decoded == expected
    assert all(len(batch) <= 2 for batch in engine.batches("decode"))


def test_a_streams_first_sentence_starts_with_one_second():
    engine = Recorder()
    playhead = Playhead(engine, lambda: 8)
    with playhead.cond:
        request = playhead.submit(sentences(1, 2, words=9), 1.0, first=25)
    drain(request)
    decoded = [(p.seed, span) for batch in engine.batches("decode") for p, span in batch]
    assert decoded == [(1000, s) for s in windows(225, 25)] + [(1001, s) for s in windows(225)]


class SizedRecorder(Recorder):
    """The recorder with a GPU's two decode sizes: a window's size is its length plus margins, padded up."""

    decode_sizes = (48, 120)

    def decode_size(self, piece, span):
        s, e = span
        n = min(piece.frames, e + 8) - max(0, s - 8)
        return next((size for size in self.decode_sizes if size >= n), n)


def test_on_a_gpu_a_decode_batch_holds_one_window_size_in_arrival_order():
    engine = SizedRecorder()
    playhead = Playhead(engine, lambda: 8)
    with playhead.cond:
        stream = playhead.submit(sentences(1, 2, words=9), 1.0, first=25)
        said = playhead.submit(sentences(2, 2, words=3), 1.0)
    drain(stream)
    drain(said)
    batches = engine.batches("decode")
    for batch in batches:
        assert len({engine.decode_size(p, span) for p, span in batch}) == 1, batch
    assert [(p.seed, span) for p, span in batches[0]] == [(1000, (0, 25))]  # the one-second window, alone
    decoded = [(p.seed, span) for batch in batches for p, span in batch]
    expected = ([(1000, s) for s in windows(225, 25)] + [(1001, s) for s in windows(225)]
                + [(2000, s) for s in windows(75)] + [(2001, s) for s in windows(75)])
    assert decoded == expected  # nothing reordered: batches only stop early at a change of size


def test_on_a_cpu_one_second_windows_are_not_batched_with_longer_ones():
    engine = Recorder()
    playhead = Playhead(engine, lambda: 8)
    with playhead.cond:
        streams = [playhead.submit(sentences(c, 1, words=9), 1.0, first=25) for c in (1, 2)]
    for stream in streams:
        drain(stream)
    batches = engine.batches("decode")
    for batch in batches:
        assert len({e - s <= 25 for _, (s, e) in batch}) == 1, batch
    assert [(p.seed, span) for p, span in batches[0]] == [(1000, (0, 25))]  # the one-second window, alone
    decoded = [(p.seed, span) for batch in batches for p, span in batch]
    assert decoded == [(1000, s) for s in windows(225, 25)] + [(2000, s) for s in windows(225, 25)]


def test_overflow_waits_at_the_front_for_the_next_turn():
    engine = Recorder()
    playhead = Playhead(engine, lambda: 64)
    for r in submit_together(playhead, [sentences(1, 70), sentences(2, 1)]):
        drain(r)
    first, second = engine.batches("think")[:2]
    assert [p.seed for p in first] == [1000 + i for i in range(64)]
    assert [p.seed for p in second] == [1000 + i for i in range(64, 70)] + [2000]


def test_every_caller_gets_their_own_audio_in_order_with_pauses():
    engine = Recorder()
    playhead = Playhead(engine, lambda: 4)
    requests = submit_together(playhead, [sentences(c, 3, words=9, pause=0.25) for c in range(3)])
    for c, r in enumerate(requests):
        chunks = drain(r)
        spoken = [int(x[0]) for x in chunks if x.any()]
        assert spoken == sorted(spoken) and set(spoken) <= {c * 1000 + i for i in range(3)}
        assert sum(1 for x in chunks if x.numel() == 12000 and not x.any()) == 3
        assert sum(x.numel() for x in chunks) == 3 * (9 * 25 * HOP + 12000)


def test_a_caller_who_hangs_up_leaves_both_queues():
    engine = Recorder()
    playhead = Playhead(engine, lambda: 1)
    (request,) = submit_together(playhead, [sentences(1, 30, words=9)])
    assert request.outbox.get(timeout=10) is not DONE
    request.cancel()
    time.sleep(0.3)
    calls = len(engine.calls)
    time.sleep(0.3)
    assert len(engine.calls) == calls
    assert not playhead.sentences and not playhead.windows


@pytest.mark.parametrize("stage", ["plan", "think", "decode"])
def test_a_failing_batch_only_fails_its_own_callers(stage):
    engine = Recorder(fail=[stage])
    playhead = Playhead(engine, lambda: 1)
    first, second = submit_together(playhead, [sentences(1, 2), sentences(2, 2)])
    with pytest.raises(RuntimeError, match=f"{stage} failed"):
        drain(first)
    assert sum(x.numel() for x in drain(second)) == 2 * 25 * HOP
    (later,) = submit_together(playhead, [sentences(3, 1)])
    assert sum(x.numel() for x in drain(later)) == 25 * HOP


def test_empty_text_finishes_without_touching_the_queues():
    engine = Recorder()
    playhead = Playhead(engine, lambda: 4)
    request = playhead.submit([], 1.0)
    assert request.outbox.get(timeout=1) is DONE
    assert engine.calls == [] and playhead.thread is None


def test_callers_on_many_threads_sound_like_each_text_alone(tts):
    texts = ["Merhaba, bugün hava çok güzel.", "Toplantı yarın sabah başlıyor, herkes hazır olsun.",
             "Evet.", "Kargonuz yola çıktı; yarın öğlene kadar teslim edilecek."]
    alone = [tts.say(t, seed=i).audio for i, t in enumerate(texts)]
    results = [None] * len(texts)
    start = threading.Barrier(len(texts))

    def call(i):
        start.wait()
        results[i] = tts.say(texts[i], seed=i).audio

    threads = [threading.Thread(target=call, args=(i,)) for i in range(len(texts))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    for got, want in zip(results, alone, strict=True):
        assert got is not None
        same_speech(got, want)


def test_streams_on_many_threads_sound_like_each_text_alone(tts):
    texts = ["Merhaba, bugün hava çok güzel.", "Toplantı yarın sabah başlıyor, herkes hazır olsun.",
             "Evet.", "Kargonuz yola çıktı; yarın öğlene kadar teslim edilecek."]
    alone = [tts.say(t, seed=i).audio for i, t in enumerate(texts)]
    results = [None] * len(texts)
    start = threading.Barrier(len(texts))

    def listen(i):
        start.wait()
        results[i] = np.concatenate(list(tts.stream(texts[i], seed=i)))

    threads = [threading.Thread(target=listen, args=(i,)) for i in range(len(texts))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    for got, want in zip(results, alone, strict=True):
        assert got is not None
        same_speech(got, want)
