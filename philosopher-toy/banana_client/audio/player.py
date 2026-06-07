"""WAV audio player using PyAudio."""
from __future__ import annotations

import asyncio
import io
import wave


class AudioPlayer:
    """Plays WAV audio received from server."""

    def __init__(self, rate: int = 22050, channels: int = 1):
        self.rate = rate
        self.channels = channels
        self._pa = None
        self._stream = None
        # Current open-stream format, so we can reopen when a WAV differs.
        self._rate = None
        self._channels = None
        self._width = None
        self._play_queue = asyncio.Queue()
        self._playing = False

    def init(self):
        import pyaudio
        self._pa = pyaudio.PyAudio()
        # Open a default stream; playback reopens it to match each WAV's format.
        self._open(self.rate, self.channels, 2)

    def _open(self, rate: int, channels: int, width: int) -> None:
        fmt = self._pa.get_format_from_width(width)
        self._stream = self._pa.open(
            format=fmt, channels=channels, rate=rate,
            output=True, frames_per_buffer=1024,
        )
        self._rate, self._channels, self._width = rate, channels, width

    def _ensure_stream(self, rate: int, channels: int, width: int) -> None:
        """Reopen the output stream if the WAV format differs from the current one."""
        if (self._stream is not None and self._rate == rate
                and self._channels == channels and self._width == width):
            return
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
        self._open(rate, channels, width)

    @property
    def is_playing(self) -> bool:
        return self._playing

    async def play_wav(self, wav_bytes: bytes):
        """Parse WAV and queue for playback."""
        await self._play_queue.put(wav_bytes)
        if not self._playing:
            self._playing = True
            asyncio.create_task(self._playback_loop())

    async def _playback_loop(self):
        while True:
            try:
                wav_bytes = await asyncio.wait_for(self._play_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if self._play_queue.empty():
                    self._playing = False
                    break
                continue
            try:
                with io.BytesIO(wav_bytes) as f, wave.open(f, "rb") as wav:
                    # Match the output stream to THIS clip's real format — Piper
                    # voices vary (e.g. x_low is 16 kHz, not the 22050 default).
                    self._ensure_stream(
                        wav.getframerate(), wav.getnchannels(), wav.getsampwidth(),
                    )
                    frames = wav.readframes(wav.getnframes())
                    loop = asyncio.get_event_loop()
                    for i in range(0, len(frames), 1024):
                        await loop.run_in_executor(None, self._stream.write, frames[i:i + 1024])
            except Exception:
                # Not a parseable WAV: play as raw PCM if a stream is open.
                if self._stream is not None:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self._stream.write, wav_bytes)

    def close(self):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa:
            self._pa.terminate()
