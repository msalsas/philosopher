"""WAV audio player via `aplay` (subprocess).

Like capture, playback avoids PyAudio (too heavy / fragile on the 512 MB Banana
Pi). `aplay` reads a WAV from stdin and auto-detects its format, so a 16 kHz
Piper clip plays at the right speed with no output-stream format juggling.
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess


class AudioPlayer:
    """Plays WAV audio received from the server through aplay."""

    def __init__(self, rate: int = 22050, channels: int = 1, mock: bool = False):
        self.rate = rate
        self.channels = channels
        self.mock = mock
        # ALSA output device; 'default' follows ~/.asoundrc (playback -> USB
        # speaker). Override with PHILOSOPHER_SPEAKER_ALSA_DEVICE.
        self._device = os.getenv("PHILOSOPHER_SPEAKER_ALSA_DEVICE", "default")
        self._ready = False
        self._play_queue: asyncio.Queue = asyncio.Queue()
        self._playing = False

    def init(self):
        if self.mock:
            print("[Player] Mock mode: audio playback disabled")
            return
        if shutil.which("aplay") is None:
            print("[WARNING] aplay (alsa-utils) not found, playback disabled")
            self.mock = True
            return
        self._ready = True
        self._apply_volume()
        print(f"[Player] Playback via aplay ({self._device})")

    def _apply_volume(self) -> None:
        """Set the speaker volume from PHILOSOPHER_SPEAKER_VOLUME (percent) via
        amixer, so it survives reboots without alsactl. Best-effort: a missing
        amixer/control just leaves the current volume. Control name defaults to
        'PCM' (this USB speaker); override with PHILOSOPHER_SPEAKER_MIXER."""
        vol = os.getenv("PHILOSOPHER_SPEAKER_VOLUME")
        if not vol or shutil.which("amixer") is None:
            return
        mixer = os.getenv("PHILOSOPHER_SPEAKER_MIXER", "PCM")
        # Card as a number ("plughw:1,0") or a name ("plughw:CARD=Foo,DEV=0").
        m = re.search(r"hw:(?:CARD=)?([^,]+)", self._device)
        cmd = ["amixer"]
        if m:
            cmd += ["-c", m.group(1)]
        cmd += ["sset", mixer, f"{vol}%"]
        try:
            subprocess.run(cmd, capture_output=True, timeout=5, check=False)
            print(f"[Player] Speaker volume set to {vol}% ({mixer})")
        except (OSError, subprocess.SubprocessError):
            pass

    @property
    def is_playing(self) -> bool:
        return self._playing

    async def play_wav(self, wav_bytes: bytes):
        """Queue a WAV clip for playback."""
        if not self._ready:  # mock / degraded: drop audio silently
            print(f"[Player] (mock) dropping {len(wav_bytes)} bytes of audio")
            return
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
                # aplay reads the WAV (header + data) from stdin and picks the
                # right rate/format itself — one process per clip.
                proc = await asyncio.create_subprocess_exec(
                    "aplay", "-q", "-D", self._device,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.communicate(wav_bytes)
            except Exception as exc:
                print(f"[Player] playback error: {exc}")

    def close(self):
        self._ready = False
