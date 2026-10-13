# Przykłady

Dziesięć zdań, każde trzema głosami modelu `pl_multi`. Każde zdanie sprawdza jedną
rzecz, którą poprawia frontend.

| nr | zdanie | co sprawdza | glos3 | design | lektor |
|---|---|---|---|---|---|
| 01 | Trzy drzewa rosną przy ulicy Długiej, tuż obok starego dworca. | trz / drz | [wav](01-glos3.wav) | [wav](01-design.wav) | [wav](01-lektor.wav) |
| 02 | Czy trzydzieści trzy złote to dobra cena za ten dżem? | trz kontra cz, dż kontra drz | [wav](02-glos3.wav) | [wav](02-design.wav) | [wav](02-lektor.wav) |
| 03 | Naciśnij 3, aby połączyć się z konsultantem. | cyfra „3” | [wav](03-glos3.wav) | [wav](03-design.wav) | [wav](03-lektor.wav) |
| 04 | Na koncie zostało jeszcze 1250 złotych. | liczba 1250 | [wav](04-glos3.wav) | [wav](04-design.wav) | [wav](04-lektor.wav) |
| 05 | W sklepie AGD zapłaciłem kartą, a resztę przelałem przez BLIK. | akronimy: literowany i czytany jak słowo | [wav](05-glos3.wav) | [wav](05-design.wav) | [wav](05-lektor.wav) |
| 06 | Zamek zbudowano w XIV wieku, a przebudowano w XIX wieku. | liczby rzymskie z odmianą | [wav](06-glos3.wav) | [wav](06-design.wav) | [wav](06-lektor.wav) |
| 07 | Po spotkaniu zrobimy backup i update na laptopie. | angielskie słowa | [wav](07-glos3.wav) | [wav](07-design.wav) | [wav](07-lektor.wav) |
| 08 | Dział wsparcia oddzwoni do pana jutro przed południem. | błąd eSpeaka na „-rcia” | [wav](08-glos3.wav) | [wav](08-design.wav) | [wav](08-lektor.wav) |
| 09 | Przy drzwiach szeptał: trzydzieści złotych, sześć groszy, żadnych zmian, dziękuję bardzo. | tekst nagrań wzorcowych | [wav](09-glos3.wav) | [wav](09-design.wav) | [wav](09-lektor.wav) |
| 10 | Dzień dobry, z tej strony biuro obsługi klienta. W czym mogę pomóc? | dwa zdania, pytanie | [wav](10-glos3.wav) | [wav](10-design.wav) | [wav](10-lektor.wav) |

WAV, 24 kHz, mono, 16 bit. Wygenerowane 13 października 2026 przez `pltts` 0.1.1
z modelem `pl_multi` e15, z domyślnymi parametrami.

## Sprawdzenie

Wszystkie 30 nagrań przepuszczono przez Whisper large-v3-turbo. Każde wróciło
z tymi samymi słowami, tylko liczby Whisper zapisał cyframi.

Pierwsze nagranie zdania 05 głosem `lektor` wyszło źle („przerałem przez BIK”).
Z trzech kolejnych prób jedna też była zła („przez byjk”), dwie dobre, i w
katalogu jest jedna z dobrych. `lektor` czasem połyka „BLIK” na końcu zdania.

## Odtworzenie

```sh
python generate.py              # pltts zainstalowany z modelem w środku
python generate.py -m pl_multi.onnx
```

Synteza nie jest deterministyczna (`noise_scale`, `noise_w`), więc kolejne
uruchomienie da trochę inną prozodię, ale te same fonemy.
