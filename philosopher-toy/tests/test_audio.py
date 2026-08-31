"""Tests for audio capture and player."""
from __future__ import annotations

import pytest

from toy_client.audio.capture import MicrophoneCapture
from toy_client.audio.player import AudioPlayer


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
    def test_init_ready_when_aplay_present(self, monkeypatch):
        import toy_client.audio.player as player_mod
        monkeypatch.setattr(player_mod.shutil, "which", lambda _: "/usr/bin/aplay")
        player = AudioPlayer()
        player.init()
        assert player._ready and not player.mock

    def test_init_degrades_without_aplay(self, monkeypatch):
        import toy_client.audio.player as player_mod
        monkeypatch.setattr(player_mod.shutil, "which", lambda _: None)
        player = AudioPlayer()
        player.init()
        assert player.mock and not player._ready

    @pytest.mark.asyncio
    async def test_play_wav_dropped_when_not_ready(self, capsys):
        # mock / no aplay: play_wav must not raise and must not spawn anything.
        player = AudioPlayer(mock=True)
        player.init()
        await player.play_wav(b"RIFF....")
        assert "mock" in capsys.readouterr().out.lower()


class TestMicrophoneDegrade:
    @pytest.mark.asyncio
    async def test_mic_degrades_without_arecord(self, monkeypatch):
        import toy_client.audio.capture as cap_mod
        monkeypatch.setattr(cap_mod.shutil, "which", lambda _: None)
        cap = MicrophoneCapture()
        await cap.init()
        assert cap.mock
