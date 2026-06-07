"""Tests for TTS engine."""
from __future__ import annotations

import pytest

from philosopher.tts.engine import PiperTTS


class TestPiperTTS:
    @pytest.mark.asyncio
    async def test_mock_synthesize(self):
        tts = PiperTTS(mock=True)
        audio = await tts.synthesize("hello")
        assert audio.startswith(b"RIFF")

    def test_model_path_default(self):
        tts = PiperTTS(mock=True, voice="test_voice")
        assert "test_voice.onnx" in tts.model_path
