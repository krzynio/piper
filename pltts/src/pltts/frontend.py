#!/usr/bin/env python
"""Full Polish frontend: espeak-pl, then the trz/drz fix, then English words.

Order matters. The trz/drz fix reasons about Polish spelling, so it has to see
the sentence before any English word has been replaced; the English pass then
overwrites whole word groups, which cannot disturb a retroflex that was placed
inside a Polish word.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from piper_phonemize import phonemize_espeak

from . import pl_trz_fix
from .lid import detect
from .pl_en_fix import _fold

# Apostrophes belong to the token: Polish declines foreign names as "Google'a",
# and splitting that into "Google" + "a" breaks the word-count alignment.
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)

# Polish letter names. eSpeak reads an all-caps token as one blob — "AGD" becomes
# /ˌaɡjˌɛdˈɛ/ in most contexts and the word "akt", /ˈakt/, in some ("Sklep AGD
# i RTV"). At best the letters are right, but there is never a word boundary
# between them, so the model has nothing to segment on and slurs them together.
# Expanding the acronym into separate words before phonemization gives eSpeak a
# reason to emit the ' ' separators.
_LETTER = {
    "A": "a", "Ą": "ą", "B": "be", "C": "ce", "Ć": "cie", "D": "de", "E": "e",
    "Ę": "ę", "F": "ef", "G": "gie", "H": "ha", "I": "i", "J": "jot", "K": "ka",
    "L": "el", "Ł": "eł", "M": "em", "N": "en", "Ń": "eń", "O": "o", "Ó": "ó",
    "P": "pe", "Q": "ku", "R": "er", "S": "es", "Ś": "eś", "T": "te", "U": "u",
    "V": "fau", "W": "wu", "X": "iks", "Y": "igrek", "Z": "zet", "Ź": "ziet",
    "Ż": "żet",
}
# Some acronyms Polish reads as ordinary words (NATO, ZUS, BLIK), most it spells
# out (PKP, BMW, AGD). A hand-kept list of the first kind is unmaintainable —
# every new brand is a miss, and a miss is audible ("be el i ka" for BLIK). The
# distinguishing property is phonotactic, not lexical: an acronym is read as a
# word exactly when its letters happen to spell a pronounceable Polish word. So
# the test is whether the token syllabifies.
_VOWELS = set("aąeęioóuy")
# Legal Polish word-initial consonant clusters. The list need not be exhaustive:
# a cluster wrongly judged illegal only means the token gets spelled out, which
# is the correct default for an acronym anyway.
_ONSETS = {
    "bl", "br", "bz", "ch", "cw", "cz", "dl", "dr", "dw", "dz", "dź", "dż",
    "fl", "fr", "gd", "gł", "gn", "gr", "gw", "gz", "kl", "kn", "kr", "ks",
    "kt", "kw", "ml", "mn", "mr", "pl", "pn", "pr", "ps", "pt", "sk", "sl",
    "sm", "sn", "sp", "sr", "st", "sw", "sz", "śc", "śl", "śm", "śn", "śp",
    "św", "tl", "tr", "tw", "wl", "wr", "ws", "wz", "zb", "zd", "zg", "zl",
    "zm", "zn", "zr", "zw", "źd", "żb", "żd", "żm", "żr",
    "chł", "chr", "chw", "czł", "czw", "prz", "prze", "szk", "szl", "szp",
    "szt", "szw", "str", "skr", "spr", "wsz", "zdr", "zgr", "zbr", "wzr",
}
# Legal word-final clusters, same reasoning.
_CODAS = {
    "ch", "cz", "sz", "rz", "st", "ść", "śc", "sk", "šć", "nt", "nk", "ng",
    "ns", "nc", "mp", "mb", "ms", "rt", "rk", "rs", "rd", "rc", "rn", "lt",
    "lk", "ls", "łt", "łd", "łk", "js", "jn", "jd", "kt", "pt", "ps", "ks",
    "ft", "zd", "zm", "sm", "tr", "br", "dr", "gl", "bl",
}
# Two acronyms the rule gets wrong, and why. USA syllabifies cleanly as "u-sa",
# yet Polish spells it out; AIDS ends in an illegal coda, yet is read as a word.
_FORCE_SPELL = {"USA", "UE", "UK", "EU"}
_FORCE_WORD = {"AIDS", "EURO", "WIFI", "COVID"}


def _syllabic(a: str) -> bool:
    """True if the token could be read as a Polish word rather than spelled."""
    w = a.lower()
    if not any(c in _VOWELS for c in w):
        return False
    first = next(i for i, c in enumerate(w) if c in _VOWELS)
    last = max(i for i, c in enumerate(w) if c in _VOWELS)
    onset, coda = w[:first], w[last + 1:]
    if len(onset) > 1 and onset not in _ONSETS:
        return False
    if len(coda) > 1 and coda not in _CODAS:
        return False
    # An interior cluster splits across the syllable break, so two consonants are
    # always fine; three or more is not a shape Polish acronyms take.
    for run in re.findall(r"[^aąeęioóuy]+", w[first:last + 1]):
        if len(run) > 2:
            return False
    return True


def reads_as_word(a: str) -> bool:
    if a in _FORCE_SPELL:
        return False
    if a in _FORCE_WORD:
        return True
    return _syllabic(a)
_ROMAN_CHARS = set("IVXLCDM")

# Some acronyms are said as one run-together word rather than separate letter
# names: "CRM" is "ceerem", not "ce · er · em" with a gap between each. Spelling
# the letters apart is right for AGD but puts audible pauses inside these, so
# they get an explicit respelling instead.
PRONOUNCE = {"SMS": "esemes", "MMS": "ememes"}

# Respelling can only go so far: Polish stress falls on the penultimate syllable,
# so "ceerem" is read "ce-E-rem", while a spelled-out acronym takes the stress on
# its last letter name — "ce-er-EM". That cannot be written orthographically, so
# these entries give the phonemes outright and are spliced in after eSpeak, the
# same way English words are.
PHONEMES = {"CRM": "tsɛɛrˈɛm"}


def pronounce_phonemes(**pairs: str) -> None:
    """Pin an exact phoneme string: pronounce_phonemes(CRM="tsɛɛrˈɛm")."""
    PHONEMES.update({k.upper(): v for k, v in pairs.items()})


def pronounce_as(**pairs: str) -> None:
    """Respell acronyms that are said as one word: pronounce_as(CRM="ceerem")."""
    PRONOUNCE.update({k.upper(): v for k, v in pairs.items()})
    # A respelling is by construction not a Polish word, so the English detector
    # would claim it — "ceerem" came back as /sˈirɛm/ — and has to be told off.
    keep_polish(*PRONOUNCE.values())

# eSpeak-pl mispronounces every word ending in "-rcia": "wsparcia" comes back as
# /fsparsˈija/, as if a Latin "-tia" rule had fired. The class is a common one —
# wsparcia, otwarcia, starcia, poparcia — and "dział wsparcia" is unavoidable in
# call-centre copy. "-rciach" and "-rciu" are fine, so the repair is narrow:
# respell just the word-final case, which yields the phonemes eSpeak produces
# correctly everywhere else.
_RESPELL = [(re.compile(r"rcia\b"), "rća")]

# Brand names are the one thing a per-word language detector cannot get right:
# "Romet" is not a Polish word and looks English, so it comes out /rˈɔmit/ —
# "romit". eSpeak-pl reads it correctly on its own, so the fix is only to keep
# the English pass away from it.
KEEP_POLISH = set()


def keep_polish(*words: str) -> None:
    """Read these words with Polish rules, whatever the detector thinks."""
    KEEP_POLISH.update(w.lower() for w in words)


keep_polish(*PRONOUNCE.values())


# Cardinals, nominative, so that a digit standing where "trzy" belongs is spelled
# out before eSpeak sees it. eSpeak reads digits in the nominative too, and it
# reads them correctly — except that its "trzy" merges into "czy", which the
# trz/drz fix then cannot repair, because the source text holds a digit and not
# the letters it reasons about. So "3 minuty" is heard as "czy minuty", and the
# same goes for 13, 23, 30, 300 and every number containing them.
_U = ["zero", "jeden", "dwa", "trzy", "cztery", "pięć", "sześć", "siedem",
      "osiem", "dziewięć"]
_T = ["dziesięć", "jedenaście", "dwanaście", "trzynaście", "czternaście",
      "piętnaście", "szesnaście", "siedemnaście", "osiemnaście", "dziewiętnaście"]
_D = ["", "", "dwadzieścia", "trzydzieści", "czterdzieści", "pięćdziesiąt",
      "sześćdziesiąt", "siedemdziesiąt", "osiemdziesiąt", "dziewięćdziesiąt"]
_H = ["", "sto", "dwieście", "trzysta", "czterysta", "pięćset", "sześćset",
      "siedemset", "osiemset", "dziewięćset"]
# (singular, 2-4, 5+) — "dwa tysiące" but "pięć tysięcy" and "dwanaście tysięcy".
_SCALE = [("", "", ""), ("tysiąc", "tysiące", "tysięcy"),
          ("milion", "miliony", "milionów"), ("miliard", "miliardy", "miliardów")]


def _under_thousand(n: int) -> List[str]:
    out = []
    if n >= 100:
        out.append(_H[n // 100])
        n %= 100
    if n >= 20:
        out.append(_D[n // 10])
        n %= 10
    elif n >= 10:
        return out + [_T[n - 10]]
    if n:
        out.append(_U[n])
    return out


def cardinal(n: int) -> str:
    if n == 0:
        return _U[0]
    groups = []
    while n:
        groups.append(n % 1000)
        n //= 1000
    out: List[str] = []
    for i in reversed(range(len(groups))):
        g = groups[i]
        if not g:
            continue
        if i and g == 1:
            out.append(_SCALE[i][0])
            continue
        out += _under_thousand(g)
        if i:
            # The plural form follows the last digit, not the last two: 144 takes
            # "miliony" like 4 does. Twelve to fourteen are the exception.
            if 2 <= g % 10 <= 4 and not 12 <= g % 100 <= 14:
                out.append(_SCALE[i][1])
            else:
                out.append(_SCALE[i][2])
    return " ".join(out)


# A standalone integer, not part of a decimal or a date. The trailing separator
# has to be allowed through — "Naciśnij 3," and "numer 4821." are the normal
# cases — so only a separator that is itself followed by a digit disqualifies.
_INT = re.compile(r"(?<![\d.,])(\d{1,12})(?!\d)(?![.,]\d)")


def spell_numbers(text: str) -> str:
    """Spell integers out, because eSpeak's own digit path is worse than its
    letter path for the same words: it glues them together with no boundary
    ("183" -> /stˈɔɔɕɛmdʑˈɛɕɔnttʃˈɨ/), drops phonemes ("603" -> /ʃˈɛɕsɛt/,
    losing the tɕ of "sześćset"), picks non-standard variants (/ɕcʲi/ for the
    "ści" of "trzydzieści"), and merges trzy into czy where the trz/drz fix
    cannot reach it, the source text holding a digit and not the letters the fix
    reasons about. The case is unchanged — eSpeak reads digits in the nominative
    and so does `cardinal` — so this only ever swaps how the words are voiced,
    never which grammatical form is chosen. Getting the form right for the
    surrounding sentence is the text normaliser's job, not this layer's.
    """
    return _INT.sub(lambda m: cardinal(int(m.group(1))), text)
_ACRONYM = re.compile(r"(?<![\w])[A-ZĄĆĘŁŃÓŚŹŻ]{2,7}(?![\w])")


_ROMAN_VAL = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
# Left alone, eSpeak-pl reads "XIX" as the English words "roman nineteen"
# (/rˌɔwmɛnnˈajntin/). So Roman numerals must be turned into Polish words here or
# they never get a chance to be right.
_AFTER_ROMAN = re.compile(r"\s*(wiek\w*|w\.)", re.IGNORECASE)
# The century noun carries the case, so the ordinal has to agree with it:
# "XIX wiek" is nominative, "XIX wieku" genitive, "XIX wiekiem" instrumental.
_CENTURY_CASE = {"wiek": "nom", "wieki": "nom", "wiekiem": "ins",
                 "wiekach": "ins", "wiekami": "ins"}
# "w XIX wieku" is locative, not genitive, and the masculine locative adjective
# ending coincides with the instrumental one.
_LOCATIVE = re.compile(r"(?:^|\s)(?:w|we|na|po|przy)\s+$", re.IGNORECASE)
# "XX w." is the century abbreviation; spelling the noun out first lets the rule
# below see an ordinary "XX wieku" and pick the case from it.
_CENTURY_ABBR = re.compile(r"\b([IVXLCDM]+)\s+w\.")
_SENT_END = re.compile(r"\s*$|\s+[A-ZĄĆĘŁŃÓŚŹŻ]")


def _expand_century_abbr(text: str) -> str:
    def sub(m):
        # At the end of a sentence the one period is both the abbreviation's and
        # the sentence's; consuming it would cost the phrase-final contour.
        end = "." if _SENT_END.match(text[m.end():]) else ""
        return f"{m.group(1)} wieku{end}"

    return _CENTURY_ABBR.sub(sub, text)
# "CD" and "MC" are valid Roman numerals and also ordinary acronyms, so being
# well-formed is not enough — a Roman reading needs a context that asks for one.
_BEFORE_ROMAN = re.compile(
    r"(?:rozdzia\w+|cz[eę]\w+|tom|ksi[eę]\w+|klas\w+|punkt\w*|ustęp\w*|"
    r"artyku\w+|paragraf\w*|za[wł]\w+|etap\w*|grup\w+|typ\w*|"
    r"stopni\w*|liceum|LO|wojn\w+|Jan\w*|Piotr\w*|Ludwik\w*|Karol\w*)\s+$",
    re.IGNORECASE)
_CANON_ROMAN = re.compile(
    r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")


# Ordinals, masculine nominative, 1-30 — enough for centuries and chapter numbers.
# Spelled out rather than computed so the frontend keeps zero dependencies and can
# be ported to WASM as is.
_ORD = ["", "pierwszy", "drugi", "trzeci", "czwarty", "piąty", "szósty",
        "siódmy", "ósmy", "dziewiąty", "dziesiąty", "jedenasty", "dwunasty",
        "trzynasty", "czternasty", "piętnasty", "szesnasty", "siedemnasty",
        "osiemnasty", "dziewiętnasty", "dwudziesty"] + [
    "dwudziesty " + t for t in ("pierwszy", "drugi", "trzeci", "czwarty",
                                "piąty", "szósty", "siódmy", "ósmy", "dziewiąty")
] + ["trzydziesty"]


def _inflect(w: str, case: str) -> str:
    if case == "nom":
        return w
    suf = "ego" if case == "gen" else "ym"
    # -y drops before the ending, -i keeps: pierwszy -> pierwszego / pierwszym,
    # trzeci -> trzeciego / trzecim.
    return " ".join(x[:-1] + suf if x.endswith("y") else
                    x + (suf if case == "gen" else "m") for x in w.split())


def _roman_to_int(a: str) -> int:
    t = 0
    for i, c in enumerate(a):
        v = _ROMAN_VAL[c]
        t += -v if i + 1 < len(a) and v < _ROMAN_VAL[a[i + 1]] else v
    return t


def expand_acronyms(text: str) -> str:
    def sub(m):
        a = m.group(0)
        if a in PHONEMES:
            # Left as written so eSpeak still emits one word group here for the
            # splice to replace, and so the word count keeps lining up.
            return a
        if a in PRONOUNCE:
            return PRONOUNCE[a]
        # Roman numerals are checked first and never fall through to the
        # word-reading test: "XIX" happens to be syllabic ("xix"), and left to
        # eSpeak it comes out as the English "roman nineteen".
        if _CANON_ROMAN.match(a):
            century = _AFTER_ROMAN.match(text[m.end():])
            if century or _BEFORE_ROMAN.search(text[:m.start()]):
                v = _roman_to_int(a)
                if 1 <= v <= 30:
                    case = "nom"
                    if century:
                        word = century.group(1).lower()
                        case = _CENTURY_CASE.get(word, "gen")
                        if case == "gen" and _LOCATIVE.search(text[:m.start()]):
                            case = "ins"
                    return _inflect(_ORD[v], case)
            # No numeral context, so it is an acronym like "CD" — spell it, but
            # never hand it back to eSpeak intact.
            return " ".join(_LETTER[c] for c in a)
        # Read as an ordinary word: lowercase it, because eSpeak spells out
        # anything all-caps regardless of how Polish actually says it.
        if reads_as_word(a):
            return a.lower()
        try:
            return " ".join(_LETTER[c] for c in a)
        except KeyError:
            return a

    text = _expand_century_abbr(text)
    for pat, rep in _RESPELL:
        text = pat.sub(rep, text)
    text = spell_numbers(text)
    return _ACRONYM.sub(sub, text)
# Polish inflectional endings, written out. Deriving them from eSpeak by
# phonemizing a dummy stem produced junk — "k'a" came back with a doubled vowel,
# "km" with an epenthetic one — and there are only twenty of them.
_ENDINGS = {
    "a": "a", "e": "ɛ", "i": "i", "o": "ɔ", "u": "u", "y": "ɨ", "m": "m",
    "em": "ɛm", "ie": "ʲɛ", "ia": "ʲa", "iem": "ʲɛm", "owi": "ɔvʲi",
    "om": "ɔm", "ów": "uf", "ach": "ax", "ami": "amʲi", "owie": "ɔvʲɛ",
    "iego": "ʲɛɡɔ", "iemu": "ʲɛmu", "owski": "ɔfskʲi", "owskim": "ɔfskʲim",
    "owskiego": "ɔfskʲɛɡɔ",
}


def _ending_phonemes(ending: str) -> List[str]:
    return list(_ENDINGS.get(ending.lstrip("'"), ""))


def english_words(text: str) -> List[Tuple[str, List[str]]]:
    """(surface, phonemes) for every word that should be read as English."""
    out = []
    for i, m in enumerate(_WORD.finditer(text)):
        w = m.group(0)
        if w.lower() in KEEP_POLISH:
            continue
        res = detect.analyse(w, sentence_initial=(m.start() == 0))
        if not res:
            continue
        stem, ending = res
        en = [p for s in phonemize_espeak(stem, "en-us") for p in s]
        ph = [p for p in _fold(en) if p != " "]
        if not ph:
            continue
        if ending:
            end = _ending_phonemes(ending)
            # "deadline" + "-m": the English silent e vanishes in phonemization,
            # so the stem ends in a consonant and Polish attaches -em, not -m.
            if end == ["m"] and ph and ph[-1] not in "aɛiɔuɨ":
                end = ["ɛ", "m"]
            ph = ph + end
        out.append((w, ph))
    return out


def phonemize(text: str, *, english: bool = True, acronyms: bool = True) -> List[str]:
    if acronyms:
        text = expand_acronyms(text)
    sentences = phonemize_espeak(text, "pl")
    sentences = pl_trz_fix.fix_all(sentences, text)
    subs = dict(english_words(text)) if english else {}
    # Pinned phonemes win over the detector, which claims "CRM" as English and
    # would otherwise read it "si-or-em".
    for w in _WORD.findall(text):
        if w in PHONEMES:
            subs[w] = list(PHONEMES[w])
    if subs:
        sentences = [_splice(s, text, subs) for s in sentences]
    flat: List[str] = []
    for s in sentences:
        if flat:
            flat.append(" ")
        flat.extend(s)
    return flat


def _keep_punct(group: List[str], new: List[str]) -> List[str]:
    """Swap a word group's phonemes but keep the punctuation attached to it.

    eSpeak puts sentence-final punctuation inside the last word's group, so
    replacing the group wholesale silently deletes the full stop and the model
    loses the phrase-final contour.
    """
    tail = []
    for p in reversed(group):
        if p.isalpha() or p in ("ˈ", "ˌ", "ː", "ʲ"):
            break
        tail.append(p)
    return list(new) + list(reversed(tail))


def _splice(phonemes: List[str], text: str, subs: dict) -> List[str]:
    groups: List[List[str]] = [[]]
    for ph in phonemes:
        if ph == " ":
            groups.append([])
        else:
            groups[-1].append(ph)
    words = _WORD.findall(text)
    idx = [i for i, g in enumerate(groups) if any(p.isalpha() for p in g)]
    if len(idx) == len(words):
        for gi, w in zip(idx, words):
            if w in subs:
                groups[gi] = _keep_punct(groups[gi], subs[w])
    else:
        # Counts disagree (eSpeak merged something). Locate each replaced word by
        # its own Polish phonemization rather than abandoning the sentence.
        # Compare on letters only: the group carries the sentence's punctuation
        # and stress marks, the isolated phonemization does not.
        def core(seq):
            return [p for p in seq if p.isalpha() or p == "ʲ"]

        for w, ph in subs.items():
            alone = [p for s in phonemize_espeak(w, "pl") for p in s]
            needle = core(alone)
            for gi, g in enumerate(groups):
                if core(g) == needle:
                    groups[gi] = _keep_punct(g, ph)
                    break
    out: List[str] = []
    for i, g in enumerate(groups):
        if i:
            out.append(" ")
        out.extend(g)
    return out


if __name__ == "__main__":
    for t in ["Prezydent USA Donald Trump ogłosił decyzję.",
              "Rozmawiałem z Trumpem i Muskiem o Netflixie.",
              "Nowy laptop i weekend w hotelu za trzysta złotych.",
              "Zainstaluj software i zrób backup."]:
        print(f"\n{t}")
        print(f"  bez : /{''.join(phonemize(t, english=False))}/")
        print(f"  z   : /{''.join(phonemize(t, english=True))}/")
