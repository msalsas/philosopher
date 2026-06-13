"""USB microphone capture with energy-based VAD."""
from __future__ import annotations

import asyncio
import os
import time

import numpy as np


class MicrophoneCapture:
    """Captures audio from USB microphone with VAD."""

    def __init__(self, rate=16000, chunk=1024, threshold=300, silence=2.0, gate=None,
                 mock=False):
        self.rate = rate
        self.chunk = chunk
        # VAD energy threshold and a software gain multiplier are env-tunable:
        # cheap USB mics (e.g. PCM2902) capture quietly, so we may need to
        # amplify in software and/or lower the speech-detection threshold.
        thr_env = os.getenv("PHILOSOPHER_VAD_THRESHOLD")
        self.threshold = float(thr_env) if thr_env else threshold
        self.gain = float(os.getenv("PHILOSOPHER_MIC_GAIN", "1.0"))
        self.silence = silence
        # gate() -> True means "suppress the mic" (e.g. while the toy is speaking),
        # so the toy never transcribes its own TTS (no echo cancellation needed).
        self.gate = gate
        self.mock = mock
        self._pa = None
        self._stream = None
        self.input_device_index = None
        # Rate the device is actually opened at, and the integer factor by which
        # we downsample it to self.rate (1 = no resampling).
        self._capture_rate = rate
        self._decim = 1
        # PHILOSOPHER_MIC_DEBUG=1 prints periodic energy vs threshold for tuning.
        self._debug = os.getenv("PHILOSOPHER_MIC_DEBUG", "").lower() in ("1", "true")
        self._dbg_count = 0

    async def init(self):
        if self.mock:
            print("[Mic] Mock mode: audio capture disabled")
            return
        import pyaudio
        self._pa = pyaudio.PyAudio()
        # Optional explicit device index (PHILOSOPHER_MIC_DEVICE); else pick the
        # first input device whose name looks like a USB sound card.
        dev_env = os.getenv("PHILOSOPHER_MIC_DEVICE")
        if dev_env not in (None, ""):
            self.input_device_index = int(dev_env)
        else:
            for i in range(self._pa.get_device_count()):
                info = self._pa.get_device_info_by_index(i)
                name = info.get("name", "").lower()
                if ("usb" in name or "audio" in name) and info.get("maxInputChannels", 0) > 0:
                    self.input_device_index = i
                    break
            if self.input_device_index is None:
                print("[WARNING] USB microphone not found, using default device")
        # Many cheap USB mics only support 48 kHz, not 16 kHz. Try the target
        # rate first; on 'invalid sample rate' fall back to 48 kHz and downsample
        # in software (48000/16000 = 3, an exact integer factor).
        last_exc: Exception | None = None
        for cap_rate in (self.rate, 48000):
            decim = max(1, round(cap_rate / self.rate))
            try:
                self._stream = self._pa.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=cap_rate,
                    input=True,
                    input_device_index=self.input_device_index,
                    frames_per_buffer=self.chunk * decim,
                )
                self._capture_rate = cap_rate
                self._decim = decim
                if decim > 1:
                    print(f"[Mic] Capturing at {cap_rate} Hz, downsampling to {self.rate} Hz")
                break
            except OSError as exc:
                last_exc = exc
                continue
        else:
            # No usable input device: degrade to a silent mic instead of crashing
            # the toy — camera and servos still work without audio.
            print(f"[WARNING] Audio input unavailable ({last_exc}), mic disabled")
            self._pa.terminate()
            self._pa = None
            self.mock = True

    def _read_chunk(self) -> bytes:
        """Read one chunk, downsampling to self.rate and applying gain."""
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
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa:
            self._pa.terminate()
