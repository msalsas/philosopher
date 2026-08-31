"""Server-side TTS using Kokoro (kokoro-onnx) -- local, offline, natural.

Kokoro is an 82M-param neural TTS (Apache-2.0) that runs on onnxruntime (no
torch). Spanish voices carry a soft Latin accent. The model (~310MB onnx +
~27MB voices) loads once at startup; each sentence synthesizes in ~1s on CPU.
An optional pitch factor deepens the voice (lowers pitch AND formants) for a
plush-character timbre.
"""
from __future__ import annotations

import asyncio
import io
import wave
from pathlib import Path

import numpy as np


class KokoroTTS:
    """Synthesizes speech with a resident Kokoro (onnx) model."""

    def __init__(self, model_dir: str = "./data/kokoro", voice: str = "em_santa",
                 lang: str = "es", speed: float = 1.0, pitch: float = 1.0,
                 mock: bool = False):
        self.voice = voice
        self.lang = lang
        self.speed = speed
        self._pitch = pitch
        self.mock = mock
        self._model_dir = model_dir
        self._k = None
        self._load_error: str | None = None
        if not mock:
            self._load()

    def _load(self) -> None:
        onnx = Path(self._model_dir) / "kokoro-v1.0.onnx"
        voices = Path(self._model_dir) / "voices-v1.0.bin"
        try:
            from kokoro_onnx import Kokoro
            self._k = Kokoro(str(onnx), str(voices))
        except Exception as exc:  # noqa: BLE001
            self._k = None
            self._load_error = repr(exc)

    def status(self) -> dict:
        if self.mock:
            return {"status": "mock", "voice": self.voice}
        if self._k is None:
            return {"status": "degraded", "voice": self.voice, "error": self._load_error}
        return {"status": "ok", "voice": self.voice}

    async def synthesize(self, text: str) -> bytes:
        if self.mock:
            return b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 40
        text = " ".join(text.split())
        if not text or self._k is None:
            return b""
        try:
            samples, sr = await asyncio.to_thread(
                self._k.create, text, self.voice, self.speed, self.lang
            )
            return self._to_wav(np.asarray(samples, dtype=np.float32), int(sr))
        except Exception:  # noqa: BLE001
            return b""

    def _to_wav(self, samples: np.ndarray, sr: int) -> bytes:
        data = samples * 32767.0
        if self._pitch != 1.0 and len(data):
            # Resample: pitch<1 lowers pitch AND formants (bigger/plush voice).
            idx = np.linspace(0, len(data) - 1, int(len(data) / self._pitch))
            data = np.interp(idx, np.arange(len(data)), data)
        pcm = np.clip(data, -32768, 32767).astype(np.int16)
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())
        return out.getvalue()

    async def close(self) -> None:
        return None
