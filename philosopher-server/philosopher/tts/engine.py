"""Server-side TTS using a persistent Piper process.

Piper is spawned once in ``--json-input`` mode and left running: the ~1.3s
voice-model load is paid a single time, then each sentence synthesizes in
~1.2s instead of re-loading the ~28MB model on every call (the old
subprocess-per-sentence path cost ~3-4s each on the RPi4). Piper prints the
finished WAV's path on stdout, one line per input line — that line is the
completion signal we wait on.
"""
from __future__ import annotations

import asyncio
import io
import json
import tempfile
import urllib.request
import wave
from pathlib import Path

import numpy as np

# Piper voices are hosted on HuggingFace (rhasspy/piper-voices). A voice id like
# "es_ES-carlfm-x_low" maps to the path "es/es_ES/carlfm/x_low/<voice>.onnx".
_PIPER_VOICES_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# Cap on how long a single synthesis may take before we give up and restart the
# resident process (a wedged piper must never hang the reply path forever).
_SYNTH_TIMEOUT = 30.0


class PiperTTS:
    """Synthesizes speech with a resident Piper process (model loaded once)."""

    def __init__(self, model_path: str = "", voice: str = "es_ES-carlfm-x_low",
                 length_scale: float = 1.0, noise_scale: float = 0.667,
                 noise_w: float = 0.8, pitch: float = 1.0, mock: bool = False):
        self.voice = voice
        self.mock = mock
        self._length_scale = length_scale
        self._noise_scale = noise_scale
        self._noise_w = noise_w
        self._pitch = pitch
        # Always resolve a deterministic model path; only fetch when needed.
        self.model_path = model_path or self._default_model_path(voice)
        if not model_path and not mock:
            self._download_model(voice)
        # Resident piper process + its output dir. Started lazily on first use;
        # a lock serialises access (one stdin/stdout, one synth at a time).
        self._proc: asyncio.subprocess.Process | None = None
        self._outdir: str | None = None
        self._lock = asyncio.Lock()

    @staticmethod
    def _default_model_path(voice: str) -> str:
        model_dir = Path("./data/piper_models")
        model_dir.mkdir(parents=True, exist_ok=True)
        return str(model_dir / f"{voice}.onnx")

    @staticmethod
    def _voice_repo_path(voice: str) -> str:
        """'es_ES-carlfm-x_low' -> 'es/es_ES/carlfm/x_low/es_ES-carlfm-x_low'."""
        lang_region, name, quality = voice.split("-", 2)
        lang = lang_region.split("_")[0]
        return f"{lang}/{lang_region}/{name}/{quality}/{voice}"

    def _download_model(self, voice: str) -> None:
        """Download the Piper .onnx + .onnx.json if missing. Idempotent.

        Any failure is swallowed; ``synthesize`` then degrades to empty bytes
        (text-only) rather than crashing.
        """
        model_file = Path(self.model_path)
        config_file = Path(self.model_path + ".json")  # <voice>.onnx.json
        if model_file.exists() and config_file.exists():
            return
        try:
            base = f"{_PIPER_VOICES_BASE}/{self._voice_repo_path(voice)}"
            if not model_file.exists():
                urllib.request.urlretrieve(f"{base}.onnx", str(model_file))  # noqa: S310
            if not config_file.exists():
                urllib.request.urlretrieve(f"{base}.onnx.json", str(config_file))  # noqa: S310
        except Exception:
            pass

    @property
    def _config_path(self) -> str:
        return self.model_path.replace(".onnx", ".onnx.json")

    def status(self) -> dict:
        """Engine readiness for /health: is piper on PATH and the voice present?"""
        if self.mock:
            return {"status": "mock", "voice": self.voice}
        import shutil
        binary = shutil.which("piper")
        model_ok = Path(self.model_path).exists()
        if binary and model_ok:
            return {"status": "ok", "voice": self.voice}
        # Not fatal (toy still gets text), but the user hears no speech — surface why.
        return {
            "status": "degraded", "voice": self.voice,
            "piper_binary": bool(binary), "voice_model": model_ok,
        }

    async def _ensure_proc(self) -> bool:
        """Ensure the resident piper is running. Returns False if it can't start."""
        if self._proc is not None and self._proc.returncode is None:
            return True
        if self._outdir is None:
            self._outdir = tempfile.mkdtemp(prefix="piper_")
        try:
            self._proc = await asyncio.create_subprocess_exec(
                "piper",
                "--model", self.model_path,
                "--config", self._config_path,
                "--json-input",
                "--output_dir", self._outdir,
                "--length_scale", str(self._length_scale),
                "--noise_scale", str(self._noise_scale),
                "--noise_w", str(self._noise_w),
                "--quiet",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            return True
        except (FileNotFoundError, OSError):
            self._proc = None
            return False

    async def _kill_proc(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
                await self._proc.wait()
            except (ProcessLookupError, OSError):
                pass
            self._proc = None

    async def synthesize(self, text: str) -> bytes:
        if self.mock:
            return b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 40
        # Piper reads one JSON object per line, so the text must be single-line.
        text = " ".join(text.split())
        if not text:
            return b""
        async with self._lock:
            for _attempt in (1, 2):  # one retry if the resident process died
                if not await self._ensure_proc():
                    return b""
                try:
                    self._proc.stdin.write((json.dumps({"text": text}) + "\n").encode())
                    await self._proc.stdin.drain()
                    # One input line -> one WAV; piper prints its path when done.
                    out = await asyncio.wait_for(
                        self._proc.stdout.readline(), timeout=_SYNTH_TIMEOUT
                    )
                    path = out.decode().strip()
                    if not path:  # EOF: the process died — restart and retry once
                        await self._kill_proc()
                        continue
                    data = Path(path).read_bytes()
                    Path(path).unlink(missing_ok=True)
                    return self._apply_pitch(data)
                except (asyncio.TimeoutError, BrokenPipeError, ConnectionResetError, OSError):
                    await self._kill_proc()
                    continue
            return b""

    def _apply_pitch(self, wav_bytes: bytes) -> bytes:
        """Deepen (or raise) the voice by resampling: pitch<1 lowers pitch AND
        formants for a bigger, plush-animal timbre. length_scale on synthesis
        compensates the pace, so this only shifts pitch, not speed. Piper output
        is mono 16-bit; returns the input unchanged when pitch == 1.0."""
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
            return wav_bytes  # on any parse issue, ship the original audio

    async def close(self) -> None:
        """Stop the resident piper process (call on shutdown)."""
        await self._kill_proc()
