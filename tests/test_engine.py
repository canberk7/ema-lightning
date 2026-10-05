import numpy as np
import torch
from conftest import same_speech

from ema_lightning import graphs as graph_module
from ema_lightning.engine import CONTEXT, DECODE_SIZES, FIRST_WINDOW, windows

TEXT = "bugün hava çok güzel, yarın da yağmur yağacakmış. toplantı öğleden sonra başlayacak."


def test_windows_layout():
    assert windows(0) == []
    assert windows(10) == [(0, 10)]
    assert windows(126) == [(0, 100), (100, 126)]  # say(): four seconds each
    assert windows(126, FIRST_WINDOW) == [(0, 25), (25, 125), (125, 126)]  # a stream's first sentence


def test_piece_marks_words_like_the_release_model(tts):
    p = tts._engine.piece("merhaba dünya.", 0.0, 0)
    assert p.cw.tolist() == [0] * 8 + [1] * 6
    assert p.wstart.tolist() == [0] * 8 + [8] * 6


def test_plan_builds_an_exact_timeline(tts):
    engine = tts._engine
    p = engine.piece(TEXT, 0.0, 0)
    engine.plan([p], 1.0)
    assert p.frames == p.fp.numel() > 0
    assert (p.fw[1:] >= p.fw[:-1]).all()
    assert ((p.fp >= 0) & (p.fp < 1)).all()


def run(engine, pieces, speed=1.0):
    engine.plan(pieces, speed)
    engine.think(pieces)
    return [torch.cat(engine.decode([(p, span) for span in windows(p.frames)])) for p in pieces]


def test_a_batch_sounds_like_each_piece_alone(tts):
    engine = tts._engine
    texts = [TEXT, "merhaba.", "kısa bir cümle daha, biraz uzun olsun diye."]
    together = run(engine, [engine.piece(t, 0.0, i) for i, t in enumerate(texts)])
    for i, t in enumerate(texts):
        (alone,) = run(engine, [engine.piece(t, 0.0, i)])
        assert torch.allclose(together[i], alone, atol=1e-4)


def test_windowed_audio_matches_whole_decode(tts):
    engine = tts._engine
    p = engine.piece(TEXT + " " + TEXT, 0.0, 3)
    (windowed,) = run(engine, [p])
    whole = engine.decoder(p.latents.T[None])[0]
    assert windowed.shape == whole.shape and torch.allclose(windowed, whole, atol=1e-5)
    assert CONTEXT >= 4


class Direct:
    """A recorded stage without CUDA: same inputs, same padding, called directly."""

    def __init__(self, fn, inputs):
        self.fn, self.inputs = fn, inputs

    def __call__(self, **values):
        return self.fn(**values)


def cpu_graphs(model, decoder, batch_size):
    g = graph_module.Graphs.__new__(graph_module.Graphs)
    g.sizes = graph_module.capture_sizes(batch_size)

    def decode(z, lengths):
        return decoder(z.transpose(1, 2), lengths)

    g.text = {(b, n): Direct(model.text_stage, None) for b in g.sizes for n in graph_module.LETTERS}
    g.sound = {(b, t): Direct(model.sound_stage, None) for b in g.sizes for t in graph_module.FRAMES}
    g.decode = {(b, p): Direct(decode, None) for b in g.sizes for p in graph_module.SPANS}
    return g


def test_bucket_padding_gives_the_plain_audio(tts):
    engine = tts._engine
    texts = [TEXT, "evet.", "bu da üçüncü cümle olsun."]
    plain = run(engine, [engine.piece(t, 0.0, i) for i, t in enumerate(texts)])
    engine.graphs = cpu_graphs(engine.model, engine.decoder, 4)
    try:
        padded = run(engine, [engine.piece(t, 0.0, i) for i, t in enumerate(texts)])
        (single,) = run(engine, [engine.piece(TEXT, 0.0, 0)])
    finally:
        engine.graphs = None
    for a, b in zip(plain, padded, strict=True):
        assert a.shape == b.shape and torch.allclose(a, b, atol=1e-4)
    assert torch.allclose(single, plain[0], atol=1e-4)


def test_sizes_beyond_the_buckets_fall_back_to_plain(tts):
    engine = tts._engine
    engine.graphs = cpu_graphs(engine.model, engine.decoder, 2)
    try:
        ids = torch.ones(1, graph_module.LETTERS[-1] + 1, dtype=torch.long)
        assert engine.graphs.run("text", ids=ids, mask=ids != 0) is None
        assert engine.graphs.batch(3) is None
    finally:
        engine.graphs = None


def test_decode_pads_every_window_to_a_fast_size_without_changing_the_audio(tts):
    text = " ".join(["Bu cümle yaklaşık olarak dört saniye sürecek kadar uzun bir cümledir."] * 4)
    plain_say, plain_stream = tts.say(text, seed=4).audio, np.concatenate(list(tts.stream(text, seed=4)))
    engine, run, sizes = tts._engine, tts._engine.run, []

    def spy(stage, **args):
        if stage == "decode":
            sizes.append(args["z"].shape[1])
        return run(stage, **args)

    engine.decode_sizes, engine.run = DECODE_SIZES, spy
    try:
        padded_say, padded_stream = tts.say(text, seed=4).audio, np.concatenate(list(tts.stream(text, seed=4)))
    finally:
        engine.decode_sizes, engine.run = (), run
    assert set(sizes) <= set(DECODE_SIZES) and set(sizes) == set(DECODE_SIZES)
    same_speech(padded_say, plain_say, db=80)
    same_speech(padded_stream, plain_stream, db=80)


def test_capture_sizes_follow_vllm():
    assert graph_module.capture_sizes(1) == [1]
    assert graph_module.capture_sizes(4) == [1, 2, 4]
    assert graph_module.capture_sizes(8) == [1, 2, 4, 8]
    assert graph_module.capture_sizes(20) == [1, 2, 4, 8, 16, 20]
    assert graph_module.capture_sizes(32) == [1, 2, 4, 8, 16, 24, 32]


def test_a_batch_is_padded_only_to_the_nearest_recorded_size():
    g = graph_module.Graphs.__new__(graph_module.Graphs)
    g.sizes = graph_module.capture_sizes(32)
    assert [g.batch(b) for b in (1, 2, 3, 5, 8, 9, 13, 25, 32)] == [1, 2, 4, 8, 8, 16, 16, 32, 32]
    assert g.batch(33) is None  # beyond the largest size: the plain path


def test_a_streams_first_decode_is_its_one_second_window_alone(tts):
    engine, run, calls = tts._engine, tts._engine.run, []

    def spy(stage, **args):
        if stage == "decode":
            calls.append(tuple(args["z"].shape[:2]))
        return run(stage, **args)

    engine.decode_sizes, engine.run = DECODE_SIZES, spy
    try:
        chunks = tts.stream(" ".join(["Bu cümle yaklaşık olarak dört saniye sürecek kadar uzun bir cümledir."] * 2),
                            seed=0)
        first = next(chunks)
        chunks.close()
    finally:
        engine.decode_sizes, engine.run = (), run
    assert first.size == 25 * 1920
    assert calls[0] == (1, 48)
