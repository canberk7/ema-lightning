# Text

Every call goes through the same steps before the model sees the text.

1. **Clean.** Control and bidirectional formatting characters become spaces.
   Invalid UTF-8 is dropped.
2. **Normalize.** [normalizer-tr](https://github.com/erdemtuna/normalizer-tr)
   0.6 or newer with `ambiguity_policy="fallback"` writes numbers, dates, times,
   money, units, abbreviations, brand names and symbols as spoken Turkish. Text is
   sent in blocks of up to 8 KiB, split at whitespace.
3. **Alphabet.** Turkish lowercasing (`İ` → `i`, `I` → `ı`), typographic quotes,
   dashes and ellipses mapped to plain ones, accents removed from non-Turkish
   letters (`é` → `e`), and anything the model cannot read dropped.
4. **Split.** Pieces of at most about 10 seconds (250 letters at speed 1.0) are cut
   at sentence ends first, then at commas, semicolons and colons, then at spaces.
   A sentence end adds a 0.25 s pause, any other cut 0.12 s.

## Fallback

`preserve`, normalizer-tr's default, keeps anything it cannot resolve as written,
and the model cannot read digits. `fallback` leaves nothing unread: it prefers
clear formats, then literal readings and named symbols, then Unicode codes.

| Written | Spoken |
|---|---|
| `5 kişi geldi.` | beş kişi geldi. |
| `Toplantı 14:30'da.` | toplantı on dört otuzda. |
| `Bütçe 1.250.000 TL.` | bütçe bir milyon iki yüz elli bin türk lirası. |
| `%15 indirim` | yüzde on beş indirim |
| `12,5 kg un` | on iki virgül beş kilogram un |
| `Dr. Ayşe geldi.` | doktor ayşe geldi. |
| `Kod: 00042` | kod: sıfır sıfır sıfır dört iki |
| `ChatGPT'ye sordum.` | çet ci pi tiye sordum. |
| `iPhone'umdan aradım.` | ayfonumdan aradım. |
| `TRT'de yayınlandı.` | te re tede yayınlandı. |
| `SON DAKİKA` | son dakika |
| `🙂` | gülümseyen yüz |

### Known limits

These come from fallback's rules, which are documented in
[normalizer-tr](https://github.com/erdemtuna/normalizer-tr/blob/main/docs/fallback.md):

- **A brand name's suffix has to fit how the name is said.** `Instagram'a` is
  read as "instagrama", but `Instagram'e` is spelled letter by letter.
- **Initialisms it does not know are read as words.** `KKTC` is read as "kktc".
  Write the letters as said (`ke ke te ce`) when it matters.
- **Ordinals are not inferred from a period.** `3. kat` is read as "üç. kat".
  Write `üçüncü kat` when the ordinal matters.
- **Characters with no Turkish name are read as Unicode codes.** `🫠` is read as
  "unikod u artı bir fe a e sıfır".

If normalization ever fails (invalid input or a resource limit), the block is
kept as written rather than dropped, and synthesis continues.
