"""Polish TTS with a corrected phoneme frontend.

    import pltts
    audio, rate = pltts.synth("Trzy drzewa przy ulicy Długiej.", "lektor")
    pltts.save("out.wav", audio, rate)

What the frontend fixes on top of plain eSpeak-pl:

  trz / drz     eSpeak merges "trz" into "cz" and "drz" into "dż", so "trzy
                drzewa" and "czy dżewa" get identical phonemes. The retroflex
                pair ʂ/ʐ is repurposed to keep them apart, and the model was
                trained on that convention.
  acronyms      "AGD" comes out of eSpeak as one unsegmentable blob; it is
                expanded to letter names, unless it reads as a word (ZUS, BLIK).
  Roman numals  eSpeak-pl reads "XIX" as the English "roman nineteen".
  English words detected per word, no list, and folded onto Polish phonemes so
                "Trump" is said "Tramp" and "weekend" "łikend".
"""
from .frontend import (expand_acronyms, keep_polish, phonemize,
                       pronounce_as, pronounce_phonemes, reads_as_word,
                       spell_numbers)
from .tts import DEFAULT_MODEL, phoneme_ids, save, synth, voices

__all__ = ["synth", "save", "voices", "phonemize", "phoneme_ids",
           "expand_acronyms", "reads_as_word", "spell_numbers",
           "keep_polish", "pronounce_as", "pronounce_phonemes",
           "DEFAULT_MODEL", "__version__"]
__version__ = "0.1.1"
