"""Say many texts in batches and write one WAV per text into a folder."""
from ema_lightning import EMA

texts = [
    "Toplantı 14:30'da başlıyor.",
    "Bütçe 1.250.000 TL olarak onaylandı.",
    "Kargonuz yarın teslim edilecek.",
]

tts = EMA().lightning()  # on a CPU this warns once and carries on
for speech in tts.say(texts, path="clips"):
    print(f"{speech.duration:.2f} s")
