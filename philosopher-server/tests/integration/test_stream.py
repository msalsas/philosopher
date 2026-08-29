"""Integration test for the WebSocket streaming path (orchestrator._stream_response).

This exercises the real production flow end-to-end with the LLM mocked:
nodes 1-3 (perception/memory/prompt) -> token stream -> per-sentence format +
send (text/servo/WAV) -> node_store. STT/vision/TTS run in mock mode.
"""
from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

from philosopher.core.orchestrator import Orchestrator
from philosopher.stt.engine import StreamingSTT
from philosopher.tts.engine import PiperTTS
from philosopher.vision.engine import VisionProcessor


class FakeWS:
    def __init__(self) -> None:
        self.json: list[dict] = []
        self.binary: list[tuple[int, bytes]] = []

    async def send_json(self, toy_id: str, data: dict) -> None:
        self.json.append(data)

    async def send_binary(self, toy_id: str, frame_type: int, payload: bytes) -> None:
        self.binary.append((frame_type, payload))


@pytest.mark.asyncio
async def test_stream_response_end_to_end(tmp_path, monkeypatch):
    # Isolate the SQLite DB so the test never touches the real ./data DB
    # (and never contends on its WAL lock with a parallel run).
    from philosopher.config.settings import get_settings
    monkeypatch.setenv("PHILOSOPHER_MEMORY_DB_PATH", str(tmp_path / "t.db"))
    get_settings.cache_clear()
    try:
        orch = Orchestrator()
        await orch.init()
        orch.stt = StreamingSTT(mock=True)
        orch.vision = VisionProcessor(mock=True)
        orch.tts = PiperTTS(mock=True)
        ws = FakeWS()
        orch.ws_manager = ws

        # Terminators are at the END of each token so the sentence splitter fires.
        async def fake_stream(messages, system_prompt=None, max_tokens=250,
                              temperature=0.7) -> AsyncGenerator[str, None]:
            for tok in ["Hola, todo bien.", " Hasta luego."]:
                yield tok

        orch.llm.chat_stream = fake_stream

        await orch._stream_response("toy1", "hola, como estas")

        texts = [m for m in ws.json if m.get("type") == "text"]
        servos = [m for m in ws.json if m.get("type") == "servo"]

        assert len(texts) == 2
        assert all(m["content"] for m in texts)
        assert len(servos) == 2
        # One-shot TTS: text/servo stream per sentence, but the whole reply is
        # synthesized in a single Piper call -> exactly one WAV blob.
        assert len(ws.binary) == 1
        assert all(p.startswith(b"RIFF") for _ft, p in ws.binary)

        # The full interaction must have been persisted.
        stats = await orch.memory.long.stats()
        assert stats["total_memories"] >= 1

        await orch.stop()
    finally:
        get_settings.cache_clear()


class _StubVision:
    """Vision stub that always reports one happy face."""

    def __init__(self, face: dict) -> None:
        self._face = face

    async def process_frame(self, jpeg_bytes: bytes) -> dict:
        return {"faces": [self._face], "primary_face": self._face}


@pytest.mark.asyncio
async def test_vision_face_emotion_reaches_stream(tmp_path, monkeypatch):
    from philosopher.config.settings import get_settings
    monkeypatch.setenv("PHILOSOPHER_MEMORY_DB_PATH", str(tmp_path / "t.db"))
    get_settings.cache_clear()
    try:
        orch = Orchestrator()
        await orch.init()
        orch.stt = StreamingSTT(mock=True)
        orch.tts = PiperTTS(mock=True)
        orch.vision = _StubVision({"face_id": "abc123", "name": "Juan",
                                   "emotion": "happy", "confidence": 0.9})
        ws = FakeWS()
        orch.ws_manager = ws

        # A frame arrives first -> the dominant face is remembered.
        await orch.handle_video_frame("toy1", b"jpeg")
        assert orch.last_vision["toy1"]["emotion"] == "happy"

        async def fake_stream(messages, system_prompt=None, max_tokens=250,
                              temperature=0.7) -> AsyncGenerator[str, None]:
            yield "Hola."

        orch.llm.chat_stream = fake_stream
        await orch._stream_response("toy1", "hola")

        # The detected emotion must reach the toy's servo command.
        servos = [m for m in ws.json if m.get("type") == "servo"]
        assert servos and servos[0]["emotion"] == "happy"

        await orch.stop()
    finally:
        get_settings.cache_clear()
