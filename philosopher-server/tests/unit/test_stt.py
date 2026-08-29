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

    def test_compute_type_forwarded_to_whisper(self, monkeypatch):
        # device/compute_type must reach faster-whisper (so a GPU laptop can use
        # float16 instead of the CPU-default int8).
        import faster_whisper
        seen = {}

        class FakeModel:
            def __init__(self, model_size, device="cpu", compute_type="int8", cpu_threads=0):
                seen.update(model_size=model_size, device=device,
                            compute_type=compute_type, cpu_threads=cpu_threads)

        monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
        stt = StreamingSTT(model_size="base", device="cuda",
                           compute_type="float16", cpu_threads=3, mock=False)
        assert stt.model is not None
        assert seen == {"model_size": "base", "device": "cuda",
                        "compute_type": "float16", "cpu_threads": 3}

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
    async def test_model_load_failure_drops_speech(self):
        # Real mode with no model must NOT fabricate "Hello world" (that would
        # make the toy answer phantom speech forever). It drops the utterance:
        # empty text + an explicit unavailable flag, so the caller is a no-op.
        stt = StreamingSTT(mock=False)
        stt.model = None  # simulate faster-whisper unavailable / load failure
        stt.add_audio_chunk("t1", b"\x00" * 32000)
        result = await stt.transcribe("t1")
        assert result["text"] == ""
        assert result["unavailable"] is True

    def test_status_reports_engine_state(self):
        assert StreamingSTT(mock=True).status()["status"] == "mock"
        broken = StreamingSTT(mock=False)
        broken.model = None
        assert broken.status()["status"] == "unavailable"
        loaded = StreamingSTT(mock=False)
        loaded.model = object()
        assert loaded.status()["status"] == "ok"
