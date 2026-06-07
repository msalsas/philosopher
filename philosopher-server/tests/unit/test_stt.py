"""Tests for STT engine."""
from __future__ import annotations

import pytest

from philosopher.stt.engine import StreamingSTT


class TestStreamingSTT:
    @pytest.mark.asyncio
    async def test_mock_transcribe(self):
        stt = StreamingSTT(mock=True)
        stt.add_audio_chunk("t1", b"some audio")
        result = await stt.transcribe("t1")
        assert result["text"] == "Hello world"
        assert result["low_confidence"] is False

    @pytest.mark.asyncio
    async def test_clear_buffer(self):
        stt = StreamingSTT(mock=True)
        stt.add_audio_chunk("t1", b"audio")
        stt.clear_buffer("t1")
        result = await stt.transcribe("t1")
        assert result["text"] == "Hello world"

    @pytest.mark.asyncio
    async def test_short_audio_returns_empty(self):
        # Pretend a model is loaded so we exercise the real (non-mock) path;
        # with no buffered audio the input is too short and must return empty
        # without ever invoking the model.
        stt = StreamingSTT(mock=False)
        stt.model = object()
        result = await stt.transcribe("t1")
        assert result["text"] == ""
        assert result["is_final"] is True

    @pytest.mark.asyncio
    async def test_model_load_failure_falls_back_to_mock(self):
        stt = StreamingSTT(mock=False)
        stt.model = None  # simulate faster-whisper unavailable / load failure
        result = await stt.transcribe("t1")
        assert result["text"] == "Hello world"
