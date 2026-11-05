"""HTTP + WebSocket server for pltts: OpenAI- and ElevenLabs-compatible HTTP plus Opus streaming.

    pltts-server --host 0.0.0.0 --port 8080

Endpoints (OpenAI / ElevenLabs paths and auth, so existing clients work):

    GET  /app                         demo page (voice, parameters, Opus playback)
    GET  /health                      GET /voices            GET /v1/models
    POST /v1/audio/speech             OpenAI style, wav | pcm | opus
    POST /v1/text-to-speech/{voice}   ElevenLabs style, ?output_format=wav|pcm_24000|opus
    WS   /v1/audio/ws                 Folx protocol, binary Opus frames

Opus is 24 kHz mono, 20 ms frames, every packet prefixed with its length as u16
big-endian. Text is cut into sentences and each one is sent as soon as it is
synthesised, so the first audio arrives after the first sentence, not the last.

Model parameters (all optional, per request, query or body):
    noise_scale      prosody/timbre variation   (default 0.667)
    noise_w_scale    rhythm variation           (default 0.8; alias noise_w)
    length_scale     above 1.0 is slower        (default 1.0)
    speed            OpenAI/ElevenLabs-style tempo, above 1.0 is faster; used only when
                     length_scale is absent (length_scale = 1 / speed)

Environment: API_TOKEN (if set, required as Bearer / xi-api-key / ?token=),
PLTTS_MODEL, PLTTS_LANES (parallel syntheses, default 2), PLTTS_THREADS (ONNX
threads per lane, default cpu_count // lanes), PLTTS_DEFAULT_VOICE.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import re
import struct
import time
import wave
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Iterator, Optional

import numpy as np
from fastapi import (Depends, FastAPI, Header, HTTPException, Query, WebSocket,
                     WebSocketDisconnect)
from fastapi.responses import HTMLResponse, RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel

from . import __version__
from .tts import DEFAULT_MODEL, synth, voices

RATE = 24000
FRAME = RATE * 20 // 1000          # 480 samples
MODEL = os.environ.get("PLTTS_MODEL", str(DEFAULT_MODEL))
LANES = max(1, int(os.environ.get("PLTTS_LANES", "2")))
THREADS = int(os.environ.get("PLTTS_THREADS", "0")) or max(1, (os.cpu_count() or 2) // LANES)
TOKEN = os.environ.get("API_TOKEN") or os.environ.get("AUTH_TOKEN") or ""
DEFAULTS = {"noise_scale": 0.667, "noise_w_scale": 0.8, "length_scale": 1.0}

app = FastAPI(title="pltts TTS", version=__version__, description=__doc__)
_pool = ThreadPoolExecutor(LANES, thread_name_prefix="synth")


# ---------------------------------------------------------------- helpers

def _default_voice() -> str:
    known = voices(MODEL)
    want = os.environ.get("PLTTS_DEFAULT_VOICE", "F3")
    return want if want in known else sorted(known, key=known.get)[0]


def check_token(authorization: Optional[str] = Header(None),
                xi_api_key: Optional[str] = Header(None)) -> None:
    if not TOKEN:
        return
    given = xi_api_key or (authorization or "").removeprefix("Bearer ").strip()
    if given != TOKEN:
        raise HTTPException(401, "Unauthorized")


class Params(BaseModel):
    noise_scale: Optional[float] = None
    noise_w_scale: Optional[float] = None
    noise_w: Optional[float] = None
    length_scale: Optional[float] = None
    speed: Optional[float] = None

    def resolved(self, base: dict | None = None) -> dict:
        out = dict(base or DEFAULTS)
        if self.noise_scale is not None:
            out["noise_scale"] = self.noise_scale
        w = self.noise_w_scale if self.noise_w_scale is not None else self.noise_w
        if w is not None:
            out["noise_w_scale"] = w
        if self.length_scale is not None:
            out["length_scale"] = self.length_scale
        elif self.speed:
            out["length_scale"] = 1.0 / self.speed
        for k, lo, hi in (("noise_scale", 0.0, 2.0), ("noise_w_scale", 0.0, 2.0),
                          ("length_scale", 0.3, 3.0)):
            if not lo <= out[k] <= hi:
                raise HTTPException(422, f"{k} must be in [{lo}, {hi}]")
        return out


class SpeechRequest(Params):
    input: str
    voice: Optional[str] = None
    model: Optional[str] = None
    response_format: str = "wav"
    language: Optional[str] = None
    seed: Optional[int] = None


class ElevenBody(Params):
    text: str
    model_id: Optional[str] = None
    language: Optional[str] = None
    seed: Optional[int] = None


_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")


def sentences(text: str, soft: int = 220) -> list[str]:
    """Sentence-sized chunks: short ones are merged, long ones cut at commas."""
    out: list[str] = []
    for part in _SPLIT.split(text.strip()):
        part = part.strip()
        while len(part) > soft:
            cut = max(part.rfind(", ", 0, soft), part.rfind("; ", 0, soft))
            if cut < 40:
                cut = part.rfind(" ", 0, soft)
            if cut < 1:
                break
            out.append(part[:cut + 1].strip())
            part = part[cut + 1:].strip()
        if part:
            out.append(part)
    merged: list[str] = []
    for s in out:
        if merged and len(merged[-1]) < 25:
            merged[-1] += " " + s
        else:
            merged.append(s)
    return merged


def _synth_chunk(text: str, voice: str, p: dict) -> np.ndarray:
    audio, _ = synth(text, voice, MODEL, noise_scale=p["noise_scale"],
                     length_scale=p["length_scale"], noise_w=p["noise_w_scale"],
                     threads=THREADS, normalize=False)
    return audio


async def pcm_chunks(text: str, voice: str, p: dict,
                     cancel: asyncio.Event | None = None) -> AsyncIterator[np.ndarray]:
    """int16 audio per sentence. Raw model level is kept (one gain for the whole
    text); only a peak above full scale is pulled down."""
    loop = asyncio.get_running_loop()
    for s in sentences(text):
        if cancel is not None and cancel.is_set():
            return
        audio = await loop.run_in_executor(_pool, _synth_chunk, s, voice, p)
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if peak > 0.98:
            audio = audio * (0.98 / peak)
        yield (np.clip(audio, -1, 1) * 32767).astype("<i2")


class OpusStream:
    """int16 PCM in, length-prefixed 20 ms Opus packets out."""

    def __init__(self) -> None:
        import opuslib
        self.enc = opuslib.Encoder(RATE, 1, opuslib.APPLICATION_AUDIO)
        self.buf = np.zeros(0, "<i2")
        self.packets = 0

    def _drain(self, final: bool) -> bytes:
        out = bytearray()
        n = len(self.buf)
        if final and n % FRAME:
            self.buf = np.concatenate([self.buf, np.zeros(FRAME - n % FRAME, "<i2")])
        for i in range(0, len(self.buf) - FRAME + 1, FRAME):
            pkt = self.enc.encode(self.buf[i:i + FRAME].tobytes(), FRAME)
            out += struct.pack(">H", len(pkt)) + pkt
            self.packets += 1
        self.buf = self.buf[(len(self.buf) // FRAME) * FRAME:]
        return bytes(out)

    def feed(self, pcm: np.ndarray) -> bytes:
        self.buf = np.concatenate([self.buf, pcm])
        return self._drain(False)

    def flush(self) -> bytes:
        return self._drain(True)


def _wav(pcm: np.ndarray) -> bytes:
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())
    return bio.getvalue()


async def _respond(text: str, voice: Optional[str], fmt: str, p: dict) -> Response:
    known = voices(MODEL)
    voice = voice or _default_voice()
    if voice not in known:
        raise HTTPException(404, f"unknown voice {voice!r}; have {', '.join(sorted(known))}")
    if not text.strip():
        raise HTTPException(422, "empty text")
    fmt = fmt.lower()
    if fmt.startswith("pcm"):
        fmt = "pcm"
    if fmt not in ("wav", "pcm", "opus"):
        raise HTTPException(400, "response_format must be wav, pcm or opus (no mp3 in this server)")

    if fmt == "wav":
        parts = [c async for c in pcm_chunks(text, voice, p)]
        return Response(_wav(np.concatenate(parts)) if parts else _wav(np.zeros(0, "<i2")),
                        media_type="audio/wav")

    async def body() -> AsyncIterator[bytes]:
        enc = OpusStream() if fmt == "opus" else None
        async for c in pcm_chunks(text, voice, p):
            data = enc.feed(c) if enc else c.tobytes()
            if data:
                yield data
        if enc:
            tail = enc.flush()
            if tail:
                yield tail

    return StreamingResponse(body(), media_type="audio/pcm" if fmt == "pcm" else "application/octet-stream")


# ---------------------------------------------------------------- HTTP

@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "pltts", "version": __version__, "device": "cpu",
            "lanes": LANES, "threads_per_lane": THREADS, "opus_ws": "/v1/audio/ws",
            "opus_sr": RATE, "opus_frame_ms": 20, "sample_rate": RATE, **DEFAULTS}


@app.get("/app", response_class=HTMLResponse, include_in_schema=False)
def demo_page() -> str:
    return (Path(__file__).parent / "app.html").read_text(encoding="utf-8")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/app")


@app.get("/voices")
def list_voices() -> dict:
    known = voices(MODEL)
    return {"default_voice": _default_voice(),
            "voices": [{"voice": n, "name": n, "category": "female" if n.startswith("F") or n == "weronika" else "male"}
                       for n in sorted(known, key=known.get)]}


@app.get("/v1/models", dependencies=[Depends(check_token)])
def models() -> dict:
    return {"object": "list", "data": [{"id": i, "object": "model", "owned_by": "pltts"}
                                       for i in ("tts-1", "pltts-piper11")]}


def _q(noise_scale: Optional[float], noise_w_scale: Optional[float], noise_w: Optional[float],
       length_scale: Optional[float], speed: Optional[float]) -> Params:
    return Params(noise_scale=noise_scale, noise_w_scale=noise_w_scale, noise_w=noise_w,
                  length_scale=length_scale, speed=speed)


@app.post("/v1/audio/speech", dependencies=[Depends(check_token)])
async def speech(req: SpeechRequest,
                 noise_scale: Optional[float] = Query(None), noise_w_scale: Optional[float] = Query(None),
                 noise_w: Optional[float] = Query(None), length_scale: Optional[float] = Query(None),
                 speed: Optional[float] = Query(None)) -> Response:
    q = _q(noise_scale, noise_w_scale, noise_w, length_scale, speed)
    merged = req.model_copy(update={k: v for k, v in q.model_dump().items() if v is not None})
    return await _respond(req.input, req.voice, req.response_format, merged.resolved())


@app.post("/v1/text-to-speech/{voice_id}", dependencies=[Depends(check_token)])
async def eleven(voice_id: str, body: ElevenBody, output_format: str = "wav",
                 noise_scale: Optional[float] = Query(None), noise_w_scale: Optional[float] = Query(None),
                 noise_w: Optional[float] = Query(None), length_scale: Optional[float] = Query(None),
                 speed: Optional[float] = Query(None)) -> Response:
    q = _q(noise_scale, noise_w_scale, noise_w, length_scale, speed)
    merged = body.model_copy(update={k: v for k, v in q.model_dump().items() if v is not None})
    return await _respond(body.text, voice_id, output_format, merged.resolved())


# ---------------------------------------------------------------- WebSocket

@app.websocket("/v1/audio/ws")
async def audio_ws(ws: WebSocket, token: str = "",
                   noise_scale: Optional[float] = None, noise_w_scale: Optional[float] = None,
                   noise_w: Optional[float] = None, length_scale: Optional[float] = None,
                   speed: Optional[float] = None) -> None:
    if TOKEN and token != TOKEN:
        await ws.close(code=4401)
        return
    await ws.accept()
    try:
        state = _q(noise_scale, noise_w_scale, noise_w, length_scale, speed).resolved()
    except HTTPException as e:
        await ws.send_json({"type": "error", "message": e.detail})
        await ws.close()
        return
    await ws.send_json({"type": "session", **state, "sample_rate": RATE, "frame_ms": 20})
    cancel = asyncio.Event()
    task: asyncio.Task | None = None
    speech_id = 0

    async def speak(msg: dict, sid: int, p: dict) -> None:
        voice = msg.get("voice") or _default_voice()
        text = (msg.get("input") or msg.get("text") or "").strip()
        if voice not in voices(MODEL):
            await ws.send_json({"type": "error", "message": f"unknown voice {voice!r}"})
            return
        if not text:
            await ws.send_json({"type": "error", "message": "empty input"})
            return
        t0 = time.perf_counter()
        await ws.send_json({"type": "speech_started", "speech_id": sid, **p})
        enc, first, samples = OpusStream(), True, 0
        async for c in pcm_chunks(text, voice, p, cancel):
            data = enc.feed(c)
            samples += len(c)
            if first:
                first = False
                await ws.send_json({"type": "ttfa", "speech_id": sid,
                                    "ms": int((time.perf_counter() - t0) * 1000)})
            if data:
                await ws.send_bytes(data)
        if cancel.is_set():
            return
        tail = enc.flush()
        if tail:
            await ws.send_bytes(tail)
        await ws.send_json({"type": "generation_complete", "speech_id": sid,
                            "chunks": enc.packets, "duration_seconds": round(samples / RATE, 3)})

    try:
        while True:
            msg = json.loads(await ws.receive_text())
            kind = msg.get("type")
            if kind == "stop":
                cancel.set()
                if task:
                    await task
                await ws.send_json({"type": "stopped"})
            elif kind == "configure":
                try:
                    state = Params(**{k: msg[k] for k in Params.model_fields if k in msg}).resolved(state)
                    await ws.send_json({"type": "session", **state})
                except HTTPException as e:
                    await ws.send_json({"type": "error", "message": e.detail})
            elif kind == "speak":
                if task and not task.done():
                    cancel.set()
                    await task
                cancel.clear()
                speech_id += 1
                try:
                    p = Params(**{k: msg[k] for k in Params.model_fields if k in msg}).resolved(state)
                except HTTPException as e:
                    await ws.send_json({"type": "error", "message": e.detail})
                    continue
                task = asyncio.create_task(speak(msg, speech_id, p))
            else:
                await ws.send_json({"type": "error", "message": f"unknown type {kind!r}"})
    except WebSocketDisconnect:
        cancel.set()
    except Exception as e:  # noqa: BLE001 — report and drop the connection
        cancel.set()
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    import uvicorn
    p = argparse.ArgumentParser(prog="pltts-server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    a = p.parse_args(argv)
    if not TOKEN:
        print("pltts-server: API_TOKEN is not set, the server is open")
    for v in voices(MODEL):                      # load the model before the first request
        break
    _synth_chunk("Start.", _default_voice(), DEFAULTS)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
