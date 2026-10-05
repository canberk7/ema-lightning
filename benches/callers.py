"""Load test for Playhead: N callers call say() at the same instant, each with one sentence.

For every N it reports how long each call took to return (median and 95th percentile) and the total
real-time multiple across all callers.
Needs tqdm: python -m pip install tqdm
"""
import statistics
import threading
import time

from tqdm import tqdm

from ema_lightning import EMA

CALLERS = [1, 8, 32, 64, 128]
TEXTS = [
    "Merhaba, size nasıl yardımcı olabilirim?",
    "Kargonuz bugün yola çıktı; yarın öğlene kadar adresinize teslim edilecek.",
    "Randevunuz on dört Ekim Salı günü saat 14:30'da.",
    "Bu görüşme kalite standartları gereği kayıt altına alınmaktadır.",
]
SEED = 0


def call(tts, text, out, start):
    start.wait()
    t0 = time.perf_counter()
    speech = tts.say(text, seed=SEED)
    out.append(((time.perf_counter() - t0) * 1000, speech.duration))


def main():
    tts = EMA().lightning()
    for n in tqdm(CALLERS, desc="callers"):
        results, start = [], threading.Barrier(n + 1)
        threads = [threading.Thread(target=call, args=(tts, TEXTS[i % len(TEXTS)], results, start)) for i in range(n)]
        for t in threads:
            t.start()
        start.wait()
        wall = time.perf_counter()
        for t in threads:
            t.join()
        wall = time.perf_counter() - wall
        took = sorted(ms for ms, _ in results)
        p95 = took[min(len(took) - 1, int(0.95 * len(took)))]
        audio = sum(seconds for _, seconds in results)
        tqdm.write(f"{n:4d} callers | say() took p50 {statistics.median(took):7.1f} ms, p95 {p95:7.1f} ms"
                   f" | {audio / wall:7.1f}x real time")


if __name__ == "__main__":
    main()
