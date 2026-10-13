"""Command line: pltts "tekst" -v lektor -o out.wav"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from .frontend import keep_polish, pronounce_as, pronounce_phonemes
from .tts import DEFAULT_MODEL, phoneme_ids, save, synth, voices


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="pltts", description="Polish TTS with a corrected phoneme frontend.")
    p.add_argument("text", nargs="?", help="text to read; omit to read stdin")
    p.add_argument("-v", "--voice", default=None,
                   help="voice name; repeat to render several, 'all' for every one")
    p.add_argument("-o", "--out", default=None,
                   help="output WAV; default probe_<voice>.wav, '-' for stdout")
    p.add_argument("-m", "--model", default=str(DEFAULT_MODEL))
    p.add_argument("-s", "--speed", type=float, default=1.0,
                   help="1.0 is normal, above 1.0 is slower")
    p.add_argument("--noise", type=float, default=0.667)
    p.add_argument("--noise-w", type=float, default=0.8)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--no-english", action="store_true",
                   help="do not read detected English words as English")
    p.add_argument("--no-acronyms", action="store_true",
                   help="do not expand acronyms or Roman numerals")
    p.add_argument("--keep-polish", metavar="SŁOWA", default=None,
                   help="czytaj te słowa po polsku, np. nazwy własne: Romet,Budimex")
    p.add_argument("--pronounce", metavar="AKR=ZAPIS", action="append", default=[],
                   help="przepisz akronim, np. SMS=esemes; powtarzalne")
    p.add_argument("--pronounce-phonemes", metavar="AKR=FONEMY", action="append",
                   default=[], help="przypnij fonemy, np. CRM=tsɛɛrˈɛm; powtarzalne")
    p.add_argument("--phonemes", action="store_true",
                   help="print the phonemes and exit without synthesising")
    p.add_argument("--list-voices", action="store_true")
    a = p.parse_args(argv)

    if a.keep_polish:
        keep_polish(*[w for w in re.split(r"[,\s]+", a.keep_polish) if w])
    for spec, fn in ((a.pronounce, pronounce_as),
                     (a.pronounce_phonemes, pronounce_phonemes)):
        for item in spec:
            if "=" not in item:
                p.error(f"oczekiwano AKRONIM=WARTOŚĆ, dostałem {item!r}")
            k, v = item.split("=", 1)
            fn(**{k: v})

    known = voices(a.model)
    if a.list_voices:
        for name, sid in sorted(known.items(), key=lambda kv: kv[1]):
            print(f"{sid}  {name}")
        return 0

    text = a.text if a.text is not None else sys.stdin.read()
    text = text.strip()
    if not text:
        p.error("no text given")

    kw = dict(english=not a.no_english, acronyms=not a.no_acronyms)
    if a.phonemes:
        _, ph = phoneme_ids(text, a.model, **kw)
        print(f"/{ph}/")
        return 0

    if a.voice in (None, "all"):
        chosen = sorted(known, key=known.get) if a.voice == "all" else \
            [sorted(known, key=known.get)[0]]
    else:
        chosen = [a.voice]
    for name in chosen:
        if name not in known:
            p.error(f"unknown voice {name!r}; have {', '.join(sorted(known))}")

    for name in chosen:
        t0 = time.perf_counter()
        audio, rate = synth(text, name, a.model, length_scale=a.speed,
                            noise_scale=a.noise, noise_w=a.noise_w,
                            threads=a.threads, **kw)
        took = time.perf_counter() - t0
        dur = len(audio) / rate
        if a.out == "-":
            sys.stdout.buffer.write(audio.tobytes())
            continue
        out = Path(a.out) if a.out and len(chosen) == 1 else Path(f"probe_{name}.wav")
        save(out, audio, rate)
        print(f"{name:<8} {dur:5.2f}s audio in {took:5.2f}s "
              f"(RTF {took/dur:.2f})  -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
