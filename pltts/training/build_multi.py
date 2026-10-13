#!/usr/bin/env python
"""Merge the three single-speaker corpora into one multi-speaker dataset.

Piper reads `filename|speaker|text` and assigns speaker ids itself, by utterance
count descending. The only real work here is the filenames: all three corpora
number their clips from 000000.wav, so they are put under per-speaker directories
and the metadata references those, otherwise two thirds of the audio would
silently resolve to the wrong voice.

Every row is verified against the filesystem before it is written, and the result
is re-parsed with csv.reader — the same parser preprocess uses — because an
unmatched quote once merged 3,582 rows into one utterance and asked for 947 GiB.
"""
from __future__ import annotations

import csv
import pathlib
import sys

OUT = pathlib.Path("/workspace/piper-multi")
# The cache key is sha256 of the *absolute audio path*, not of the audio, so the
# merged metadata must reference the exact paths the three single-speaker runs
# used. Anything else re-derives 232 GB of spectrograms for no reason.
SOURCES = [
    ("glos3", pathlib.Path("/workspace/piper-v3")),
    ("lektor", pathlib.Path("/workspace/piper-lektor")),
    ("design", pathlib.Path("/workspace/piper-design")),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str, str]] = []
    for speaker, root in SOURCES:
        audio = root / "audio"
        if not audio.exists():
            sys.exit(f"brak katalogu audio: {audio}")
        meta = root / "metadata.csv"
        if meta.exists():
            src_rows = [r for r in csv.reader(open(meta, encoding="utf-8"), delimiter="|")
                        if len(r) >= 2]
            pairs = [(pathlib.Path(r[0]).name, r[-1]) for r in src_rows]
        else:
            # design_v1 was never turned into a Piper corpus; read its manifest.
            import json
            pairs = []
            seen: set[str] = set()
            for line in open(root / "manifest.jsonl", encoding="utf-8"):
                d = json.loads(line)
                name = pathlib.Path(d.get("audio") or d["audio_path"]).name
                if name in seen:
                    continue
                seen.add(name)
                txt = " ".join(d["text"].replace('"', "").replace("|", ",").split())
                if txt:
                    pairs.append((name, txt))

        kept = 0
        for name, text in pairs:
            if not (audio / name).exists():
                continue
            rows.append((str(audio / name), speaker, text))
            kept += 1
        print(f"  {speaker:<8} {kept:>6} wypowiedzi  <- {root}")

    path = OUT / "metadata.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        for a, s, t in rows:
            f.write(f"{a}|{s}|{t}\n")

    # Same parser preprocess will use, same delimiter.
    parsed = list(csv.reader(path.open(encoding="utf-8"), delimiter="|"))
    bad = [r for r in parsed if len(r) != 3]
    print(f"\nzapisano {len(rows)} wierszy, csv.reader widzi {len(parsed)}"
          f"  ({'OK' if len(parsed) == len(rows) and not bad else 'NIEZGODNOSC'})")
    if bad:
        print(f"  wierszy o zlej liczbie kolumn: {len(bad)}, np. {bad[0][:2]}")
        sys.exit(1)
    # Spot-check that a path from each speaker really resolves.
    for speaker, _ in SOURCES:
        r = next(r for r in parsed if r[1] == speaker)
        p = OUT / r[0]
        print(f"  {speaker}: {r[0]} -> {'istnieje' if p.exists() else 'BRAK'}")


if __name__ == "__main__":
    main()
