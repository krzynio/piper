"""Pronounce English words inside Polish text.

eSpeak-pl applies Polish letter-to-sound rules to everything, so English words
come out as nonsense: "cash" -> tsˈaʃ ("tsasz"), "software" -> sɔftfˈarɛ
("softfare"), "update" -> ubdˈatɛ ("ubdate").

Phonemizing them with en-us instead is only half a fix. The model was trained on
48 distinct phonemes; the other 109 ids in phoneme_id_map never received a
gradient, so feeding it ɹ, æ, ɪ or ə produces noise rather than a sound. Of the
36 phonemes en-us needs for typical loanwords, 13 are in that dead set.

So English is phonemized with en-us and then folded onto the trained Polish
inventory. The result is a Polish-accented rendering — which is what a Polish
listener expects from a Polish voice anyway. "cash" becomes "kesz", not /kæʃ/.

Two layers:
  LEXICON  — hand-written phoneme strings for words that matter enough to get
             right. Always wins, because the automatic path cannot know that
             Polish says "gugiel" rather than "gugel".
  fallback — en-us phonemes mapped through EN2PL for anything unlisted.

Words are marked either by appearing in LEXICON/AUTO, or explicitly in the text
as [[en:word]] for one-off cases the caller knows about.
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence

# Phonemes the model actually saw during training (>=100 occurrences in the v3
# corpus). Everything this module emits is asserted against this set.
TRAINED = set("abdefiklmnoprstuvwxzɔɛɡɨɟʃʒʑʐʂɕʲɲŋtɡ") | {
    "a", "b", "d", "dz", "e", "f", "g", "i", "j", "k", "l", "m", "n", "o", "p",
    "r", "s", "t", "u", "v", "w", "x", "z", "ɔ", "ɛ", "ɡ", "ɨ", "ɲ", "ŋ", "ɕ",
    "ʑ", "ʃ", "ʒ", "ʂ", "ʐ", "ʲ", "ˈ", "ˌ", "ː", " ",
}

# English -> Polish phoneme folding. Multi-character values are expanded into
# separate phonemes. Diphthongs are written as vowel+glide, the way Polish
# spells them ("aj", "ej", "ow").
EN2PL: Dict[str, str] = {
    # vowels
    "iː": "i", "ɪ": "i", "i": "i",
    "ɛ": "ɛ", "e": "ɛ", "æ": "ɛ",
    "ɑː": "ɔ", "ɑ": "ɔ", "ʌ": "a", "ɐ": "a", "a": "a",
    "ɔː": "ɔ", "ɔ": "ɔ", "ɒ": "ɔ",
    "uː": "u", "ʊ": "u", "u": "u",
    "ɜː": "ɛr", "ɜ": "ɛr", "ɚ": "ɛr", "ɝ": "ɛr",
    "ə": "ɛ",
    "oʊ": "ɔw", "o": "ɔ",
    "aɪ": "aj", "eɪ": "ɛj", "aʊ": "aw", "ɔɪ": "ɔj", "ɪə": "iɛ", "ɛə": "ɛr",
    # consonants
    "ɹ": "r", "r": "r", "ɾ": "t",
    "θ": "t", "ð": "d",
    "ŋ": "ŋ", "j": "j", "w": "w", "h": "x",
    "tʃ": "tʃ", "dʒ": "dʒ", "ʃ": "ʃ", "ʒ": "ʒ",
    "b": "b", "d": "d", "f": "f", "ɡ": "ɡ", "g": "ɡ", "k": "k", "l": "l",
    "m": "m", "n": "n", "p": "p", "s": "s", "t": "t", "v": "v", "z": "z",
    # suprasegmentals pass through
    "ˈ": "ˈ", "ˌ": "ˌ", "ː": "", " ": " ",
}

# Hand-tuned pronunciations. Written as the phoneme string a Polish speaker
# produces, not as a transliteration of the English one.
LEXICON: Dict[str, str] = {
    "cash":        "kˈɛʃ",
    "software":    "sˈɔftwɛr",
    "hardware":    "xˈardwɛr",
    "update":      "ˈapdɛjt",
    "upgrade":     "ˈapɡrɛjt",
    "backup":      "bˈɛkap",
    "email":       "ˈimɛjl",
    "e-mail":      "ˈimɛjl",
    "meeting":     "mˈitiŋk",
    "online":      "ˈɔnlajn",
    "offline":     "ˈɔflajn",
    "weekend":     "wˈikɛnt",
    "deadline":    "dˈɛdlajn",
    "voucher":     "vˈawtʃɛr",
    "router":      "rˈutɛr",
    "password":    "pˈaswɔrt",
    "smartphone":  "smˈartfɔn",
    "callback":    "kˈɔlbɛk",
    "google":      "ɡˈuɡɛl",
    "microsoft":   "mˈajkrɔsɔft",
    "windows":     "wˈindɔws",
    "android":     "ˈandrɔjt",
    "iphone":      "ˈajfɔn",
    "internet":    "intˈɛrnɛt",
    "wifi":        "wˈifi",
    "wi-fi":       "wˈifi",
    "sms":         "ɛsɛmˈɛs",
    "ok":          "ɔkˈɛj",
    "okay":        "ɔkˈɛj",
    "premium":     "prˈɛmjum",
    "standard":    "stˈandart",
    "support":     "sapˈɔrt",
    "service":     "sˈɛrvis",
    "manager":     "mˈɛnɛdʒɛr",
    "leasing":     "lˈisiŋk",
    "marketing":   "mˈarkɛtiŋk",
    "startup":     "stˈartap",
    "laptop":      "lˈaptɔp",
    "tablet":      "tˈablɛt",
    "chat":        "tʃˈat",
    "check":       "tʃˈɛk",
    "checkout":    "tʃˈɛkawt",
    "cancel":      "kˈansɛl",
    "refund":      "rifˈant",
    "invoice":     "ˈinvɔjs",
    "delivery":    "dɛlˈivɛri",
    "tracking":    "trˈɛkiŋk",
}

# Explicit per-call escape hatch: [[en:whatever]] in the source text.
_TAG = re.compile(r"\[\[en:([^\]]+)\]\]", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def _split_phonemes(s: str) -> List[str]:
    """Split a hand-written phoneme string into single-phoneme tokens.

    Affricates are two symbols in Piper's inventory (t + ʃ), matching what
    eSpeak emits, so nothing here needs to be kept together.
    """
    return list(s)


def _fold(en: Sequence[str]) -> List[str]:
    """Map an en-us phoneme sequence onto the trained Polish inventory."""
    out: List[str] = []
    i = 0
    seq = list(en)
    while i < len(seq):
        # Longest match first so diphthongs and long vowels win over their
        # first character (aɪ before a, iː before i).
        for n in (2, 1):
            chunk = "".join(seq[i:i + n])
            if chunk in EN2PL:
                out.extend(_split_phonemes(EN2PL[chunk]))
                i += n
                break
        else:
            i += 1  # unmappable: drop rather than emit a dead embedding
    return out


def pronounce(word: str) -> List[str] | None:
    """Phonemes for one English word, or None if it should be left to Polish."""
    key = word.lower().strip("'-")
    if key in LEXICON:
        return _split_phonemes(LEXICON[key])
    try:
        from piper_phonemize import phonemize_espeak
        en = [p for s in phonemize_espeak(word, "en-us") for p in s]
    except Exception:
        return None
    if not en:
        return None
    folded = [p for p in _fold(en) if p not in (" ",)]
    bad = [p for p in folded if p not in TRAINED]
    # Refusing is the right failure mode: a Polish mispronunciation is worse
    # than nothing only if it is unintelligible, and a dead embedding is.
    return None if bad or not folded else folded


def extract_tags(text: str) -> tuple[str, Dict[str, List[str]]]:
    """Strip [[en:...]] markers, returning clean text and forced pronunciations."""
    forced: Dict[str, List[str]] = {}

    def sub(m):
        w = m.group(1)
        ph = pronounce(w)
        if ph:
            forced[w.lower()] = ph
        return w

    return _TAG.sub(sub, text), forced


def fix_sentence(phonemes: List[str], text: str,
                 forced: Dict[str, List[str]] | None = None) -> List[str]:
    """Replace the phonemes of English words in one already-phonemized sentence.

    Alignment is the same word-group splice used for the trz/drz fix: eSpeak
    separates words with a ' ' phoneme, so orthographic and phoneme words line
    up as long as the counts agree. When they do not — eSpeak sometimes merges
    short words — each English word is located by its own isolated phonemization
    instead of abandoning the sentence.
    """
    forced = forced or {}
    words = _WORD.findall(text)
    targets = {w.lower(): (forced.get(w.lower()) or pronounce(w)) for w in words}
    targets = {w: p for w, p in targets.items() if p}
    if not targets:
        return phonemes

    groups: List[List[str]] = [[]]
    for ph in phonemes:
        if ph == " ":
            groups.append([])
        else:
            groups[-1].append(ph)

    def is_word(g: List[str]) -> bool:
        return any(p.isalpha() for p in g)

    idx = [i for i, g in enumerate(groups) if is_word(g)]
    if len(idx) == len(words):
        for gi, w in zip(idx, words):
            if w.lower() in targets:
                groups[gi] = list(targets[w.lower()])
    else:
        from piper_phonemize import phonemize_espeak
        supra = ("ˈ", "ˌ", "ː", "ˑ")
        for w, ph in targets.items():
            try:
                alone = [p for s in phonemize_espeak(w, "pl") for p in s]
            except Exception:
                continue
            needle = [p for p in alone if p not in supra and p != " "]
            for gi, g in enumerate(groups):
                if [p for p in g if p not in supra] == needle:
                    groups[gi] = list(ph)
                    break

    out: List[str] = []
    for i, g in enumerate(groups):
        if i:
            out.append(" ")
        out.extend(g)
    return out


def fix_all(all_phonemes: List[List[str]], text: str,
            forced: Dict[str, List[str]] | None = None) -> List[List[str]]:
    if len(all_phonemes) == 1:
        return [fix_sentence(all_phonemes[0], text, forced)]
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(parts) != len(all_phonemes):
        return all_phonemes
    return [fix_sentence(ph, part, forced) for ph, part in zip(all_phonemes, parts)]
