"""USB microphone capture with energy-based VAD."""
from __future__ import annotations

import asyncio
import time

import numpy as np


class MicrophoneCapture:
    """Captures audio from USB microphone with VAD."""

    def __init__(self, rate=16000, chunk=1024, threshold=300, silence=2.0, gate=None):
        self.rate = rate
        self.chunk = chunk
        self.threshold = threshold
        self.silence = silence
        # gate() -> True means "suppress the mic" (e.g. while the toy is speaking),
        # so the toy never transcribes its own TTS (no echo cancellation needed).
        self.gate = gate
        self._pa = None
        self._stream = None
        self.input_device_index = None

    async def init(self):
        import pyaudio
        self._pa = pyaudio.PyAudio()
        for i in range(self._pa.get_device_count()):
            info = self._pa.get_device_info_by_index(i)
            name = info.get("name", "").lower()
            if ("usb" in name or "audio" in name) and info.get("maxInputChannels", 0) > 0:
                self.input_device_index = i
                break
        if self.input_device_index is None:
            print("[WARNING] USB microphone not found, using default device")
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            input_device_index=self.input_device_index,
            frames_per_buffer=self.chunk,
        )

    def _energy(self, pcm_bytes: bytes) -> float:
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))

    async def capture_stream(self):
        """Yield audio events when speech is detected and ended."""
        buffer = bytearray()
        is_speaking = False
        silence_frames = 0
        silence_threshold_frames = int(self.rate / self.chunk * self.silence)
        gated_until = 0.0

        while True:
            pcm_bytes = self._stream.read(self.chunk, exception_on_overflow=False)
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
