"""Server-side TTS using Microsoft Edge neural voices (edge-tts).

Much more natural than Piper, free, no API key -- but it calls Microsoft's
online service, so it needs internet (Piper stays the offline fallback). Edge
returns MP3; we transcode to WAV with ffmpeg so the toy can play it as-is.
"""
from __future__ import annotations

import asyncio
import io
import wave

import numpy as np


class EdgeTTS:
    """Synthesizes speech with edge-tts (online Microsoft neural voices)."""

    def __init__(self, voice: str = "es-ES-AlvaroNeural", rate: str = "+0%",
                 pitch_hz: str = "+0Hz", pitch: float = 1.0, mock: bool = False):
        self.voice = voice
        self.rate = rate          # edge prosody rate, e.g. "-5%"
        self.pitch_hz = pitch_hz  # edge native pitch, e.g. "-20Hz"
        self._pitch = pitch       # extra post resample (1.0 = none; <1 deeper)
        self.mock = mock

    def status(self) -> dict:
        if self.mock:
            return {"status": "mock", "voice": self.voice}
        import shutil
        try:
            import edge_tts  # noqa: F401
        except Exception:
            return {"status": "degraded", "voice": self.voice, "edge_tts": False}
        if not shutil.which("ffmpeg"):
            return {"status": "degraded", "voice": self.voice, "ffmpeg": False}
        return {"status": "ok", "voice": self.voice}

    async def synthesize(self, text: str) -> bytes:
        if self.mock:
            return b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 40
        text = " ".join(text.split())
        if not text:
            return b""
        try:
            import edge_tts
            comm = edge_tts.Communicate(text, self.voice, rate=self.rate, pitch=self.pitch_hz)
            mp3 = bytearray()
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    mp3.extend(chunk["data"])
            if not mp3:
                return b""
            wav = await self._mp3_to_wav(bytes(mp3))
            return self._apply_pitch(wav)
        except Exception:
            # No network / edge_tts missing / ffmpeg missing -> text only.
            return b""

    async def _mp3_to_wav(self, mp3: bytes) -> bytes:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0", "-ar", "22050", "-ac", "1", "-f", "wav", "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate(input=mp3)
        return out if proc.returncode == 0 else b""

    def _apply_pitch(self, wav_bytes: bytes) -> bytes:
        """Optional deepen (pitch<1 lowers pitch AND formants). Mono 16-bit."""
        if self._pitch == 1.0 or not wav_bytes:
            return wav_bytes
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as w:
                nch, sr, n = w.getnchannels(), w.getframerate(), w.getnframes()
                data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)
            idx = np.linspace(0, len(data) - 1, int(len(data) / self._pitch))
            res = np.interp(idx, np.arange(len(data)), data).astype(np.int16)
            out = io.BytesIO()
            with wave.open(out, "wb") as w:
                w.setnchannels(nch)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(res.tobytes())
            return out.getvalue()
        except (wave.Error, ValueError):
            return wav_bytes

    async def close(self) -> None:
        return None
