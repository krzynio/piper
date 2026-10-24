"""
Undo eSpeak's Polish <trz>/<cz> and <drz>/<dż> merger.

eSpeak-pl emits the same phonemes for orthographically distinct sequences:

    trzydzieści -> tʃɨdʑˈɛɕtɕi        czydzieści -> tʃɨdʑˈɛɕtɕi   (identical)
    drzewo      -> dʒˈɛvɔ             dżewo      -> dʒˈɛvɔ        (identical)

Piper's model sees identical phoneme ids for two different sounds and learns
their average, which lands on the more frequent <cz>. "trzydzieści" then comes
out as "czydzieści". No amount of training fixes it — the distinction is gone
before the model sees anything.

Fix: where the *spelling* has <trz>/<drz>, emit retroflex ʂ/ʐ instead of
postalveolar ʃ/ʒ. Both are already in Piper's phoneme_id_map (ʂ=95, ʐ=106) and
eSpeak-pl never emits them, so they are free to carry the distinction and the
symbol table does not change.

Alignment is by word: phonemize_espeak separates words with a ' ' phoneme, so
orthographic words and phoneme words line up. If the counts disagree — eSpeak
occasionally merges or splits — the utterance is left untouched rather than
patched at the wrong position.
"""
from __future__ import annotations

import re
from typing import List

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
# Every spelling that yields tʃ, in one alternation so their order in the word
# is known. Only <trz> and <cz> produce tʃ — <ć>, <ci>+vowel and friends give
# the alveolo-palatal tɕ, which does not compete. An earlier version treated
# them as competitors and silently skipped 14% of trz words ("patrzcie" ->
# pˈatʃtɕɛ has one tʃ from trz and one tɕ from cie).
_T_SOURCES = re.compile(r"trz|cz", re.IGNORECASE)
_D_SOURCES = re.compile(r"drz|dż", re.IGNORECASE)

# Suprasegmentals to ignore when matching a word phonemized in isolation
# against the same word inside a sentence: the isolated form carries no length
# mark, the sentence one may ("drzewo" -> dʒˈɛvɔː).
_SUPRA = ("ˈ", "ˌ", "ː", "ˑ", " ")


def _strip(seq: List[str]) -> List[str]:
    return [p for p in seq if p not in _SUPRA]


def _retroflexify_word(phonemes: List[str], word: str) -> List[str]:
    """Convert only those tʃ/dʒ that came from <trz>/<drz>.

    Matching is positional: the Nth tʃ in the phoneme string corresponds to the
    Nth tʃ-producing spelling in the word, so a word containing both <trz> and
    <cz> is handled correctly instead of being skipped.
    """
    low = word.lower()
    t_src = [m.group(0).lower().startswith("trz") for m in _T_SOURCES.finditer(low)]
    d_src = [m.group(0).lower().startswith("drz") for m in _D_SOURCES.finditer(low)]
    if not any(t_src) and not any(d_src):
        return phonemes

    out = list(phonemes)
    t_seen = d_seen = 0
    for i in range(len(out) - 1):
        if out[i] == "t" and out[i + 1] == "ʃ":
            if t_seen < len(t_src) and t_src[t_seen]:
                out[i + 1] = "ʂ"
            t_seen += 1
        elif out[i] == "d" and out[i + 1] == "ʒ":
            if d_seen < len(d_src) and d_src[d_seen]:
                out[i + 1] = "ʐ"
            d_seen += 1
    # If the counts disagree, eSpeak did something unexpected with this word
    # (elision, a foreign-word switch). Leave it alone rather than guess.
    if t_seen != len(t_src) or d_seen != len(d_src):
        return phonemes
    return out


def fix_sentence(phonemes: List[str], text: str) -> List[str]:
    """Retroflexify one sentence's phoneme list using its source text."""
    words = _WORD.findall(text)
    if not words:
        return phonemes
    if not re.search(r"trz|drz", text, re.IGNORECASE):
        return phonemes

    # Split the phoneme stream into word groups on the ' ' separator, keeping
    # every token so the sequence can be rebuilt exactly.
    groups: List[List[str]] = [[]]
    for ph in phonemes:
        if ph == " ":
            groups.append([])
        else:
            groups[-1].append(ph)

    # Groups carrying no letter phonemes (pure punctuation) are not words.
    def is_word(g: List[str]) -> bool:
        return any(p.isalpha() for p in g)

    word_idx = [i for i, g in enumerate(groups) if is_word(g)]
    if len(word_idx) == len(words):
        for gi, w in zip(word_idx, words):
            groups[gi] = _retroflexify_word(groups[gi], w)
    else:
        # eSpeak merged or split words ("na te" -> nˈatɛ is one group), so
        # positional alignment is unreliable. Abandoning the whole sentence
        # loses the fix for every other word in it, so instead locate each
        # trz/drz word by phonemizing it on its own and matching that phoneme
        # sequence inside the sentence, ignoring stress marks.
        targets = [w for w in words if re.search(r"trz|drz", w, re.IGNORECASE)]
        if not targets:
            return phonemes
        for w in targets:
            try:
                from piper_phonemize import phonemize_espeak
                alone = [p for s in phonemize_espeak(w, "pl") for p in s]
            except Exception:
                continue
            fixed_alone = _retroflexify_word(alone, w)
            if fixed_alone == alone:
                continue
            needle, patched = _strip(alone), _strip(fixed_alone)
            if not needle:
                continue
            for gi, g in enumerate(groups):
                hay = _strip(g)
                if hay != needle:
                    continue
                # Rebuild the group, substituting the retroflex phonemes while
                # keeping the sentence's own stress marks in place.
                it = iter(patched)
                groups[gi] = [p if p in _SUPRA else next(it, p) for p in g]
                break

    out: List[str] = []
    for i, g in enumerate(groups):
        if i:
            out.append(" ")
        out.extend(g)
    return out


def fix_all(all_phonemes: List[List[str]], text: str) -> List[List[str]]:
    """Apply to every sentence eSpeak produced for one utterance.

    eSpeak may split an utterance into several sentences. Rather than guess how
    the source text divides, the same text is offered to each sentence; the
    per-word alignment guard inside fix_sentence rejects any sentence whose word
    count does not match, so a mis-split utterance is simply left unpatched.
    """
    if len(all_phonemes) == 1:
        return [fix_sentence(all_phonemes[0], text)]
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(parts) != len(all_phonemes):
        return all_phonemes
    return [fix_sentence(ph, part) for ph, part in zip(all_phonemes, parts)]
