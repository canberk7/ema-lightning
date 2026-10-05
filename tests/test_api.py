import time
import wave

import numpy as np
import pytest
from conftest import same_speech

from ema_lightning import Speech
from ema_lightning.scheduler import DONE

TEXT = "Bugün hava çok güzel. Yarın da yağmur yağacakmış, toplantı öğleden sonra başlayacak."
LONG = " ".join(["Bu cümle yaklaşık olarak dört saniye sürecek kadar uzun bir cümledir."] * 5)


def test_say_returns_speech(tts):
    s = tts.say(TEXT, seed=7)
    assert isinstance(s, Speech) and s.seed == 7 and s.sample_rate == 48000
    assert s.audio.dtype == np.float32 and s.audio.ndim == 1 and s.audio.size > 0
    assert s.duration == pytest.approx(s.audio.size / 48000)
    assert np.abs(s.audio).max() <= 1.0


@pytest.mark.parametrize("rate", [48000, 24000, 16000, 8000])
def test_every_sample_rate_gives_the_same_speech(tts, rate):
    full = tts.say(LONG, seed=3)
    s = tts.say(LONG, seed=3, sample_rate=rate)
    assert s.sample_rate == rate and s.audio.dtype == np.float32
    assert s.duration == pytest.approx(full.duration, abs=0.01)


@pytest.mark.parametrize("rate", [48000, 24000, 16000, 8000])
def test_stream_joined_matches_say(tts, rate):
    chunks = list(tts.stream(LONG, seed=3, sample_rate=rate))
    assert len(chunks) > 2 and all(c.dtype == np.float32 for c in chunks)
    same_speech(np.concatenate(chunks), tts.say(LONG, seed=3, sample_rate=rate).audio)


def test_first_chunk_of_a_stream_is_one_second(tts):
    first = next(iter(tts.stream(LONG, seed=0)))
    assert first.size == 25 * 1920


def _raw(request):
    chunks = []
    while (item := request.outbox.get(timeout=30)) is not DONE:
        chunks.append(item)
    return chunks


def test_say_decodes_four_second_windows_from_the_start(tts):
    request = tts._submit(LONG, 1.0, 0)
    chunks = _raw(request)
    assert chunks[0].numel() == min(request.pieces[0].frames, 100) * 1920


def test_only_a_streams_first_sentence_starts_with_one_second(tts):
    request = tts._submit(LONG, 1.0, 0, first=25)
    chunks = _raw(request)
    first, second = request.pieces[:2]
    assert chunks[0].numel() == min(first.frames, 25) * 1920
    after = len(first.spans) + (1 if first.pause else 0)  # the first sentence's windows, then its pause
    assert chunks[after].numel() == min(second.frames, 100) * 1920


def test_leaving_a_stream_early_drops_its_work(tts):
    chunks = tts.stream(" ".join([LONG] * 4), seed=0)
    assert next(chunks).size > 0
    chunks.close()
    deadline = time.time() + 10
    while (tts._playhead.sentences or tts._playhead.windows) and time.time() < deadline:
        time.sleep(0.05)
    assert not tts._playhead.sentences and not tts._playhead.windows
    assert tts.say(TEXT, seed=0).audio.size > 0


def test_seed_decides_the_audio(tts):
    a, b, c = tts.say(TEXT, seed=1), tts.say(TEXT, seed=1), tts.say(TEXT, seed=2)
    assert np.array_equal(a.audio, b.audio)
    assert not np.array_equal(a.audio, c.audio)
    assert isinstance(tts.say(TEXT).seed, int)


def test_list_matches_saying_each_text(tts):
    texts = [TEXT, "Evet.", LONG, "", "Kısa bir cümle daha."]
    many = tts.say(texts, seed=5, sample_rate=16000)
    assert [s.seed for s in many] == [5] * len(texts)
    for text, s in zip(texts, many, strict=True):
        one = tts.say(text, seed=5, sample_rate=16000)
        same_speech(s.audio, one.audio)
    assert tts.say([]) == []


def test_unspeakable_text_is_silent_not_an_error(tts):
    for text in ["", "   ", "...", "?!", "\x00\x01"]:
        assert tts.say(text).audio.size == 0
        assert list(tts.stream(text)) == []
    assert [s.audio.size for s in tts.say(["", "..."])] == [0, 0]


def test_symbols_are_read_by_name(tts):
    import normalizer_tr

    if not hasattr(normalizer_tr, "NORMALIZER_ID"):
        pytest.skip("normalizer-tr not installed")
    assert tts.say("🙂").audio.size > 0  # fallback reads it as "gülümseyen yüz"


@pytest.mark.parametrize("kwargs", [dict(speed=0), dict(speed=5), dict(speed="fast"), dict(speed=True),
                                    dict(seed=-1), dict(seed=1.5), dict(sample_rate=44100)])
def test_bad_settings_raise_before_any_work(tts, kwargs):
    with pytest.raises(ValueError):
        tts.say(TEXT, **kwargs)
    with pytest.raises(ValueError):
        tts.say([TEXT], **kwargs)
    with pytest.raises(ValueError):
        tts.stream(TEXT, **kwargs)


def test_wrong_text_types_raise(tts):
    with pytest.raises(TypeError):
        tts.stream([TEXT])
    with pytest.raises(TypeError):
        tts.say(123)
    with pytest.raises(TypeError):
        tts.say([TEXT, 5])


def test_path_writes_wav_files(tts, tmp_path):
    s = tts.say(TEXT, seed=0, sample_rate=24000, path=tmp_path / "one.wav")
    with wave.open(str(tmp_path / "one.wav")) as f:
        assert f.getframerate() == 24000 and f.getnframes() == s.audio.size
    tts.say([TEXT] * 11, seed=0, path=tmp_path / "many")
    assert sorted(p.name for p in (tmp_path / "many").iterdir()) == [f"{i:02d}.wav" for i in range(11)]


def test_sentence_cuts_leave_a_pause(tts):
    request = tts._submit(LONG, 1.0, 0)
    chunks = []
    while (item := request.outbox.get()) is not DONE:
        chunks.append(item)
    assert any(c.numel() == 12000 and not c.any() for c in chunks)


def test_lightning_without_a_gpu_warns_and_keeps_working(tts):
    with pytest.warns(UserWarning):
        assert tts.lightning() is tts
    assert tts.say(TEXT).audio.size > 0


def test_best_batch_size_is_measured_once(tts, monkeypatch):
    size = tts.best_batch_size()
    assert size in (1, 2, 4, 8)
    tts._batch_size = None
    monkeypatch.setattr(type(tts), "_throughput", lambda *_: (_ for _ in ()).throw(AssertionError("measured twice")))
    assert tts.best_batch_size() == size
