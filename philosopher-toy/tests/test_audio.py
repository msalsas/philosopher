"""Tests for audio capture and player."""
from __future__ import annotations

import pytest

from banana_client.audio.capture import MicrophoneCapture
from banana_client.audio.player import AudioPlayer


class TestMicrophoneCapture:
    def test_energy_silence(self):
        cap = MicrophoneCapture()
        energy = cap._energy(b"\x00" * 1024)
        assert energy == 0.0

    def test_energy_signal(self):
        cap = MicrophoneCapture()
        import numpy as np
        samples = (np.sin(np.linspace(0, 2 * np.pi, 512)) * 1000).astype(np.int16)
        energy = cap._energy(samples.tobytes())
        assert energy > 0

    @pytest.mark.asyncio
    async def test_mic_gating_suppresses_while_speaking(self):
        import asyncio

        import numpy as np

        speaking = {"on": True}
        cap = MicrophoneCapture(threshold=10, silence=0.05,
                                gate=lambda: speaking["on"])
        loud = (np.ones(1024) * 1000).astype(np.int16).tobytes()

        class FakeStream:
            def read(self, n, exception_on_overflow=False):
                return loud

        cap._stream = FakeStream()
        gen = cap.capture_stream()
        nxt = asyncio.ensure_future(gen.__anext__())
        await asyncio.sleep(0.05)
        assert not nxt.done()          # gated: nothing emitted despite loud audio

        speaking["on"] = False         # toy stopped speaking
        event = await asyncio.wait_for(nxt, timeout=1.0)
        assert event["type"] == "speech_started"
        await gen.aclose()


class TestAudioPlayer:
    def test_init(self, monkeypatch):
        # Inject a fake `pyaudio` so init() opens a stream without real hardware.
        import sys
        import types

        class FakeStream:
            def stop_stream(self):
                pass

            def close(self):
                pass

        class FakePA:
            def get_format_from_width(self, w):
                return w

            def open(self, **kw):
                return FakeStream()

            def terminate(self):
                pass

        fake = types.ModuleType("pyaudio")
        fake.PyAudio = FakePA
        monkeypatch.setitem(sys.modules, "pyaudio", fake)

        player = AudioPlayer()
        player.init()
        assert player._stream is not None
        player.close()

    def test_reopens_stream_for_wav_rate(self):
        # No pyaudio needed: inject a fake PyAudio and verify the player reopens
        # the output stream at the WAV's real rate (16 kHz) instead of the 22050
        # default — otherwise a 16 kHz Piper voice plays ~37% too fast.
        class FakeStream:
            def stop_stream(self):
                pass

            def close(self):
                pass

        class FakePA:
            def __init__(self):
                self.opened = []

            def get_format_from_width(self, w):
                return w

            def open(self, **kw):
                self.opened.append(kw)
                return FakeStream()

        player = AudioPlayer()
        player._pa = FakePA()
        player._open(22050, 1, 2)            # default
        player._ensure_stream(16000, 1, 2)   # 16 kHz WAV -> must reopen
        assert player._pa.opened[-1]["rate"] == 16000
        player._ensure_stream(16000, 1, 2)   # same format -> no reopen
        assert len(player._pa.opened) == 2
