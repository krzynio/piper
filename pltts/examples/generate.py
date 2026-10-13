#!/usr/bin/env python
"""Regenerate the example recordings: every sentence below, in every voice.

    python generate.py                  # pltts installed with the model inside
    python generate.py -m pl_multi.onnx # or point at the model

Each sentence exercises one thing the frontend fixes. Synthesis is not
deterministic (noise_scale, noise_w), so a rerun gives slightly different
prosody, never different phonemes.
"""
from __future__ import annotations

import argparse
import pathlib

import pltts

HERE = pathlib.Path(__file__).parent

SENTENCES = [
    ("trz / drz", "Trzy drzewa rosną przy ulicy Długiej, tuż obok starego dworca."),
    ("trz kontra cz, dż kontra drz",
     "Czy trzydzieści trzy złote to dobra cena za ten dżem?"),
    ("cyfra „3”", "Naciśnij 3, aby połączyć się z konsultantem."),
    ("liczba 1250", "Na koncie zostało jeszcze 1250 złotych."),
    ("akronimy: literowany i czytany jak słowo",
     "W sklepie AGD zapłaciłem kartą, a resztę przelałem przez BLIK."),
    ("liczby rzymskie z odmianą",
     "Zamek zbudowano w XIV wieku, a przebudowano w XIX wieku."),
    ("angielskie słowa", "Po spotkaniu zrobimy backup i update na laptopie."),
    ("błąd eSpeaka na „-rcia”", "Dział wsparcia oddzwoni do pana jutro przed południem."),
    ("tekst nagrań wzorcowych",
     "Przy drzwiach szeptał: trzydzieści złotych, sześć groszy, żadnych zmian, "
     "dziękuję bardzo."),
    ("dwa zdania, pytanie", "Dzień dobry, z tej strony biuro obsługi klienta. W czym mogę pomóc?"),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-m", "--model", default=str(pltts.DEFAULT_MODEL))
    a = p.parse_args()

    voices = sorted(pltts.voices(a.model).items(), key=lambda kv: kv[1])
    for i, (_, text) in enumerate(SENTENCES, 1):
        for name, _ in voices:
            audio, rate = pltts.synth(text, name, a.model)
            out = HERE / f"{i:02d}-{name}.wav"
            pltts.save(out, audio, rate)
            print(f"{out.name}  {len(audio) / rate:5.2f}s")


if __name__ == "__main__":
    main()
