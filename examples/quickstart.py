"""Say one sentence and save it."""
from ema_lightning import EMA

tts = EMA()
speech = tts.say("Merhaba! Bugün hava çok güzel.", path="merhaba.wav")
print(f"{speech.duration:.2f} s at {speech.sample_rate} Hz, seed {speech.seed}")
