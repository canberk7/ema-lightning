import pytest

from ema_lightning.phonetics import phoneticize_tr


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Software developer olarak çalışıyorum.", "softveyır divelopır olarak çalışıyorum."),
        ("Frontend ve backend kodunu deploy ettim.", "Frontend ve bekend kodunu diploy ettim."),
        ("Kubernetes cluster kurduk.", "kubırnetis klastır kurduk."),
        ("Yarın meeting var, deadline cuma.", "Yarın miting var, didlayn cuma."),
        ("GitHub", "githab"),
        ("UNITY", "yuniti"),
        ("JavaScript", "javasikript"),
        ("TypeScript", "taypsikript"),
        ("UNESCO, UNICEF, UEFA ve IKEA", "UNESCO, UNICEF, UEFA ve IKEA"),
        ("croissant!", "kruvasan!"),
        ("coffee, please.", "kafi, pliz."),
        ("developer'ın commit'i", "divelopır'ın komit'i"),
        ("GitHub'a push ettik.", "githab'a puş ettik."),
        ("", ""),
        ("iMac aldım.", "aymek aldım."),
        ("Next.js ve Node.js", "Next.js ve Node.js"),
        ("ali@example.com ve www.example.com", "ali@example.com ve www.example.com"),
    ],
)
def test_phoneticize_tr(source, expected):
    assert phoneticize_tr(source) == expected


@pytest.mark.parametrize(
    "turkish_sentence",
    [
        "Bugün kahve içip yazılım geliştireceğim.",
        "Model ve problem çözüldü, program hazır.",
        "Ali ile Ankara'ya gittik, kalem masa kitap.",
        "Polis macera filmi izledik, bence harika.",
        "Toplantı saat üçte başlayacak.",
        "Çocuklar bahçede oyun oynuyorlar.",
        "Bugun cay icip gunes altinda oturduk, deger mi?",  # typed without Turkish letters
        "Agac altinda universite sinavina calistik.",
        "Datca, Nilufer ve Trabzonspor.",
    ],
)
def test_pure_turkish_sentences_are_preserved(turkish_sentence):
    assert phoneticize_tr(turkish_sentence) == turkish_sentence
