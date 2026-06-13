"""USB microphone capture with energy-based VAD.

Audio input goes through `arecord` (a subprocess), not PyAudio: on the 512 MB
Banana Pi, PyAudio's device enumeration probes every ALSA/JACK PCM and takes
minutes — and intermittently pegs the CPU until the board hangs. `arecord`
opens the device instantly and, via the ALSA `plug` plugin, resamples to the
target rate for us, so no software downsampling is needed either.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import time

import numpy as np


class _ArecordStream:
    """Minimal blocking reader over an `arecord` subprocess (raw S16_LE mono).

    Exposes a PyAudio-like ``read(num_frames)`` so the VAD loop — and the tests
    that inject a fake stream — stay unchanged.
    """

    def __init__(self, device: str, rate: int):
        self._proc = subprocess.Popen(
            ["arecord", "-q", "-D", device, "-f", "S16_LE",
             "-r", str(rate), "-c", "1", "-t", "raw"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

    def read(self, num_frames: int, exception_on_overflow: bool = False) -> bytes:
        # 2 bytes/frame (S16_LE mono). read() on the buffered pipe returns this
        # many bytes unless arecord died (then fewer / empty).
        return self._proc.stdout.read(num_frames * 2)

    def close(self) -> None:
        self._proc.terminate()
        try:
            self._proc.wait(timeout=1)
        except Exception:
            self._proc.kill()


class MicrophoneCapture:
    """Captures audio from a USB microphone (via arecord) with energy VAD."""

    def __init__(self, rate=16000, chunk=1024, threshold=300, silence=2.0, gate=None,
                 mock=False):
        self.rate = rate
        self.chunk = chunk
        # VAD energy threshold and a software gain multiplier are env-tunable:
        # cheap USB mics (e.g. PCM2902) have a high noise floor, so the threshold
        # must sit above it; gain lifts a quiet capture (note: gain scales noise
        # too, so it doesn't improve SNR — the threshold is what separates them).
        thr_env = os.getenv("PHILOSOPHER_VAD_THRESHOLD")
        self.threshold = float(thr_env) if thr_env else threshold
        self.gain = float(os.getenv("PHILOSOPHER_MIC_GAIN", "1.0"))
        self.silence = silence
        # gate() -> True means "suppress the mic" (e.g. while the toy is speaking),
        # so the toy never transcribes its own TTS (no echo cancellation needed).
        self.gate = gate
        self.mock = mock
        self._stream = None
        # Kept for the test interface; arecord resamples for us so always 1.
        self._decim = 1
        # PHILOSOPHER_MIC_DEBUG=1 prints periodic energy vs threshold for tuning.
        self._debug = os.getenv("PHILOSOPHER_MIC_DEBUG", "").lower() in ("1", "true")
        self._dbg_count = 0

    async def init(self):
        if self.mock:
            print("[Mic] Mock mode: audio capture disabled")
            return
        if shutil.which("arecord") is None:
            print("[WARNING] arecord (alsa-utils) not found, mic disabled")
            self.mock = True
            return
        # ALSA device; 'default' follows ~/.asoundrc (capture -> USB mic, with
        # plug resampling to self.rate). Override with PHILOSOPHER_MIC_ALSA_DEVICE.
        device = os.getenv("PHILOSOPHER_MIC_ALSA_DEVICE", "default")
        try:
            self._stream = _ArecordStream(device, self.rate)
            print(f"[Mic] Capturing via arecord ({device}) at {self.rate} Hz")
        except Exception as exc:
            # No usable input device: degrade to a silent mic instead of crashing
            # the toy — camera and servos still work without audio.
            print(f"[WARNING] Audio input unavailable ({exc}), mic disabled")
            self._stream = None
            self.mock = True

    def _read_chunk(self) -> bytes:
        """Read one chunk and apply software gain."""
        raw = self._stream.read(self.chunk * self._decim, exception_on_overflow=False)
        samples = np.frombuffer(raw, dtype=np.int16)
        if self._decim > 1:
            n = (len(samples) // self._decim) * self._decim
            samples = samples[:n].reshape(-1, self._decim).mean(axis=1)
        else:
            samples = samples.astype(np.float32)
        if self.gain != 1.0:
            samples = samples * self.gain
        return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()

    def _energy(self, pcm_bytes: bytes) -> float:
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))

    async def capture_stream(self):
        """Yield audio events when speech is detected and ended."""
        if self._stream is None:  # mock / degraded: a silent mic, never yields
            while True:
                await asyncio.sleep(3600)
        buffer = bytearray()
        is_speaking = False
        silence_frames = 0
        silence_threshold_frames = int(self.rate / self.chunk * self.silence)
        gated_until = 0.0

        while True:
            pcm_bytes = self._read_chunk()
            if not pcm_bytes:  # arecord stopped delivering; back off, don't spin
                await asyncio.sleep(0.1)
                continue
            now = time.monotonic()
            # Half-duplex: while the toy is speaking (+ a short tail), read & drop
            # audio so it never hears its own voice, and reset any partial utterance.
            if self.gate and self.gate():
                gated_until = now + 0.3
            if now < gated_until:
                is_speaking = False
                buffer = bytearray()
                silence_frames = 0
                await asyncio.sleep(0)
                continue

            energy = self._energy(pcm_bytes)
            if self._debug:
                self._dbg_count += 1
                if self._dbg_count % 8 == 0:  # ~0.5 s at 16 kHz / 1024
                    peak = "SPEECH" if energy > self.threshold else "silence"
                    print(f"[Mic] energy={energy:7.0f}  threshold={self.threshold:.0f}  {peak}")

            if energy > self.threshold:
                if not is_speaking:
                    is_speaking = True
                    silence_frames = 0
                    yield {"type": "speech_started", "audio": b""}
                buffer.extend(pcm_bytes)
            else:
                if is_speaking:
                    buffer.extend(pcm_bytes)
                    silence_frames += 1
                    if silence_frames >= silence_threshold_frames:
                        is_speaking = False
                        audio_data = bytes(buffer)
                        buffer = bytearray()
                        yield {"type": "speech_ended", "audio": audio_data}
                else:
                    pass

            await asyncio.sleep(0)

    def close(self):
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
