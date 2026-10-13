"""Inference: text in, waveform out.

The model bundled with the package is multi-speaker, so a voice is chosen with
the fourth ONNX input `sid`. Everything the text passes through — eSpeak-pl, the
trz/drz retroflex fix, acronym expansion, English words — happens in `frontend`
and is shared by all voices, because the speaker vector conditions the acoustic
modules only and never reaches the text encoder.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime

from .frontend import phonemize

MODELS = Path(__file__).parent / "models"
DEFAULT_MODEL = MODELS / "pl_multi.onnx"

_SESSIONS: Dict[Tuple[str, int], onnxruntime.InferenceSession] = {}


def voices(model: Path | str = DEFAULT_MODEL) -> Dict[str, int]:
    """Voice name -> speaker id."""
    cfg = _config(model)
    return dict(cfg.get("speaker_id_map") or {"default": 0})


def _config(model: Path | str) -> dict:
    return json.loads(Path(str(model) + ".json").read_text(encoding="utf-8"))


def _session(model: Path | str, threads: int) -> onnxruntime.InferenceSession:
    key = (str(model), threads)
    if key not in _SESSIONS:
        opts = onnxruntime.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        _SESSIONS[key] = onnxruntime.InferenceSession(
            str(model), sess_options=opts, providers=["CPUExecutionProvider"])
    return _SESSIONS[key]


def phoneme_ids(text: str, model: Path | str = DEFAULT_MODEL, **kw) -> Tuple[List[int], str]:
    """Model input ids, plus the phoneme string they came from (for debugging)."""
    pm = _config(model)["phoneme_id_map"]
    flat = phonemize(text, **kw)
    ids = list(pm["^"])
    for p in flat:
        if p in pm:
            ids += pm[p] + pm["_"]
    ids += pm["$"]
    return ids, "".join(flat)


def synth(text: str, voice: str | int = 0, model: Path | str = DEFAULT_MODEL, *,
          noise_scale: float = 0.667, length_scale: float = 1.0,
          noise_w: float = 0.8, english: bool = True, acronyms: bool = True,
          threads: int = 0) -> Tuple[np.ndarray, int]:
    """Synthesise `text`. Returns (float32 mono samples in [-1, 1], sample rate).

    `voice` is a name from `voices()` or a raw speaker id. `length_scale` above
    1.0 slows the speech down; `noise_scale` and `noise_w` control how much the
    prosody varies between calls, so two calls on the same text differ slightly.
    """
    cfg = _config(model)
    sid = voice if isinstance(voice, int) else voices(model)[voice]
    ids, _ = phoneme_ids(text, model, english=english, acronyms=acronyms)
    feed = {
        "input": np.array([ids], dtype=np.int64),
        "input_lengths": np.array([len(ids)], dtype=np.int64),
        "scales": np.array([noise_scale, length_scale, noise_w], dtype=np.float32),
    }
    if cfg.get("num_speakers", 1) > 1:
        feed["sid"] = np.array([sid], dtype=np.int64)
    audio = _session(model, threads).run(None, feed)[0].squeeze()
    audio = audio / (np.abs(audio).max() + 1e-9) * 0.9
    return audio.astype(np.float32), cfg["audio"]["sample_rate"]


def save(path: Path | str, audio: np.ndarray, rate: int) -> None:
    """Write a WAV without pulling in soundfile."""
    import wave

    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
