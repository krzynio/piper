#!/usr/bin/env python
"""Reference implementation of the per-word detector, reading the packed bytes.

Decision order, cheapest and most certain first:

  1. a Polish diacritic anywhere -> Polish. No English word has one.
  2. present in the Polish exception Bloom filter -> Polish. This is what saves
     "hotel", "start", "bank", "test": strings identical in both languages, which
     no orthographic model can separate.
  3. n-gram log-ratio above threshold -> English.
  4. otherwise Polish.

Foreign names get Polish endings ("Trumpem", "Muska", "Netfliksie"), and the
inflected form is in no wordlist. So when a word fails the test as a whole, its
Polish suffixes are peeled off and the stem is retested; a stem that reads as
English gives an English stem plus a Polish ending, which is exactly how the
word is said.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import struct

import numpy as np

HERE = pathlib.Path(__file__).parent
META = json.loads((HERE / "lid_meta.json").read_text())
ORDER = META["order"]
BOUND = "\x02"
VOCAB = {c: i for i, c in enumerate(META["vocab"])}
UNK = len(META["vocab"])
CTXS = {c: i for i, c in enumerate(
    (HERE / "lid_ctxs.txt").read_text(encoding="utf-8").split("\n"))}
_Q = np.frombuffer((HERE / "lid_ngram.bin").read_bytes(), dtype=np.uint8)
DELTA = _Q.reshape(META["rows"], META["cols"]).astype(np.float32) * META["scale"] + META["lo"]

def _load_bloom(name: str):
    b = (HERE / name).read_bytes()
    m, k = struct.unpack("<QI", b[:12])
    return m, k, b[12:]


BLOOM_M, BLOOM_K, BLOOM = _load_bloom("lid_plfull.bloom")
EN_M, EN_K, EN_BLOOM = _load_bloom("lid_en.bloom")

DIAC = set("ąćęłńóśźż")
# Ordered longest first so "-ami" is tried before "-i".
SUFFIXES = ("owskiego", "owskim", "owski", "owie", "iego", "iemu", "ach",
            "ami", "owi", "iem", "om", "ów", "em", "ie", "ia", "y", "u",
            "a", "e", "i", "o", "m",
            "'a", "'em", "'owi", "'ach", "'ami", "'u", "'e", "'ie")
# Stems are short and noisy, so the suffix path needs a firmer margin than a
# whole word does before it may override the Polish frontend.
STEM_THR = 0.4
THR = 0.0


def _member(word: str, m: int, k: int, bits: bytes) -> bool:
    h = hashlib.blake2b(word.encode("utf-8"), digest_size=16).digest()
    h1 = int.from_bytes(h[:8], "big")
    h2 = int.from_bytes(h[8:], "big") | 1
    for i in range(k):
        idx = (h1 + i * h2) % m
        if not (bits[idx >> 3] >> (idx & 7)) & 1:
            return False
    return True


def in_bloom(word: str) -> bool:
    return _member(word, BLOOM_M, BLOOM_K, BLOOM)


def in_english(word: str) -> bool:
    return len(word) >= 3 and _member(word, EN_M, EN_K, EN_BLOOM)


def score(word: str) -> float:
    s = BOUND * (ORDER - 1) + word.lower() + BOUND
    tot = 0.0
    n = 0
    for i in range(len(s) - ORDER + 1):
        r = CTXS.get(s[i:i + ORDER - 1])
        if r is None:
            continue
        tot += float(DELTA[r, VOCAB.get(s[i + ORDER - 1], UNK)])
        n += 1
    return tot / max(n, 1)


# Loanwords that ARE in the Polish dictionary but are not said the way
# eSpeak-pl reads them. This is a lexical fact no orthographic model can
# recover, and the set is closed and small, so it is simply listed.
# Only words where eSpeak-pl is measurably wrong. eSpeak-ng ships a Polish
# loanword dictionary and already says "weekend" as łikent, "backup" as bekap,
# "Microsoft" as majkrosoft, "laptop" as laptop. Routing those through en-us
# would make them worse, so they are deliberately absent.
FORCE_EN = set("""software voucher router password callback checkout
smartphone google facebook netflix youtube spotify airbnb playstation
deadline lockdown streaming cash design display feature update upgrade
download upload online offline outsourcing crowdfunding""".split())


def _plain_english(w: str) -> bool:
    lw = w.lower()
    if len(lw) < 3 or (set(lw) & DIAC):
        return False
    if lw in FORCE_EN:
        return True
    if in_bloom(lw):
        return False          # known Polish form, read by the Polish frontend
    return score(lw) > THR


def analyse(word: str, sentence_initial: bool = False) -> tuple[str, str] | None:
    """Return (english_stem, polish_ending) if the word should be read as English.

    The ending may be empty. None means: leave it to the Polish frontend.

    Every Polish ending that fits is treated as a candidate and they are ranked,
    rather than taking the first match. Taking the first match split "deadlinem"
    into "deadlin"+"em" because -em was tried before -m, losing the English word
    that was actually there.

    sentence_initial matters because Polish capitalises the first word of every
    sentence, so a capital elsewhere is evidence of a proper noun. Without it
    "Muska" cannot be told from the verb form "muska", which the dictionary
    contains and would otherwise always win.
    """
    lw = word.lower()
    whole = _plain_english(word)
    if whole and (in_english(lw) or lw in FORCE_EN):
        return word, ""          # a real English word, not a stem + ending
    if set(lw) & DIAC or len(lw) < 5:
        return (word, "") if whole else None

    proper = (word[:1].isupper() and not sentence_initial and score(lw) > 0.0)
    # A word the Polish dictionary knows is Polish, full stop — unless it is a
    # capitalised proper noun, where the dictionary hit is usually a homograph.
    if in_bloom(lw) and not proper and not whole:
        return None

    base = score(lw)
    best = None                  # (rank, stem_len, stem, suffix)
    for suf in SUFFIXES:
        if not lw.endswith(suf) or len(lw) - len(suf) < 3:
            continue
        stem = word[:len(word) - len(suf)]
        sl = stem.lower()
        if len(sl) < 3 or (set(sl) & DIAC):
            continue
        if sl in FORCE_EN or in_english(sl):
            rank = 2             # the stem is a real English word — strongest
        elif in_bloom(sl):
            continue             # the stem is a Polish word; not a foreign stem
        else:
            # Splitting has to buy something: the stem must read as English more
            # strongly than the unsplit word did, or there is no reason to split.
            margin = 0.15 if whole else 0.0
            if score(sl) <= max(STEM_THR, base + margin if whole else STEM_THR):
                continue
            rank = 1
        cand = (rank, len(sl), stem, lw[len(lw) - len(suf):])
        if best is None or cand[:2] > best[:2]:
            best = cand
    if best:
        return best[2], best[3]
    return (word, "") if whole else None


PL_TRAPS = """komputer system start hotel film bank test plan student sport radio
wideo problem moment program projekt kontakt telefon adres numer serwis internet
temperatura moderator agent lider partner sektor transport import eksport
kredyt debet limit rabat termin kontrakt standard format monitor procesor
dokument prezent koncert instrument profesor doktor magister absolwent
premier minister parlament senat prezydent gabinet komisja delegacja
restauracja apartament basen park stadion trener mecz sezon""".split()

EN_TARGETS = """Trump Musk Netflix Facebook Amazon Google Microsoft iPhone Windows
Android YouTube TikTok Instagram WhatsApp Uber Airbnb Spotify PlayStation Xbox
software hardware meeting deadline weekend backup laptop marketing briefing
coaching lockdown streaming shopping outsourcing crowdfunding
Biden Obama Johnson Shakespeare Liverpool Manchester Chelsea Arsenal
cash voucher router password callback checkout refund invoice delivery
smartphone update upgrade download upload email online offline""".split()

INFLECTED = """Trumpem Trumpa Trumpowi Muskiem Muska Netflixa Netflixie
Facebooka Google'a Amazonu iPhonem laptopa laptopem weekendu weekendem
marketingu deadlinem backupem streamingu Bidena Obamy Johnsonem""".split()


def main() -> None:
    import json as _j
    import random
    import re
    words = set()
    for line in open("/workspace/voices/eliza/textpool/prompts_all.jsonl",
                     encoding="utf-8"):
        words.update(re.findall(r"[^\W\d_]+", _j.loads(line)["text"], re.UNICODE))
    random.seed(0)
    pl_real = random.sample(sorted(words), 20000)

    import re as _re
    roman = _re.compile(r"^[IVXLCDM]+$")
    pl_real = [w for w in pl_real if not roman.match(w)]
    fp = [w for w in pl_real if analyse(w)]
    tp = [w for w in EN_TARGETS if analyse(w)]
    tr = [w for w in PL_TRAPS if analyse(w)]
    inf = [(w, analyse(w, sentence_initial=False)) for w in INFLECTED]

    print(f"fałszywe alarmy na polskim tekście: {len(fp)}/{len(pl_real)} "
          f"= {len(fp)/len(pl_real)*100:.2f}%")
    print(f"trafione angielskie:                {len(tp)}/{len(EN_TARGETS)} "
          f"= {len(tp)/len(EN_TARGETS)*100:.1f}%")
    print(f"pułapki polskie źle:                {len(tr)}/{len(PL_TRAPS)} "
          f"= {len(tr)/len(PL_TRAPS)*100:.1f}%  {tr}")
    print(f"\nprzeoczone angielskie: {[w for w in EN_TARGETS if not analyse(w)]}")
    print(f"\npierwsze fałszywe alarmy: {fp[:30]}")
    print("\nodmienione obce (temat + końcówka):")
    for w, r in inf:
        print(f"  {w:<14} {r if r else '— nie wykryto'}")


if __name__ == "__main__":
    main()
