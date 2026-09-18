"""Background-noise vs unclear-speech handling (orchestrator.process_speech).

Whisper can't always turn audio into words. Two very different reasons:
  - NOISE (nobody talking): the toy must stay SILENT, not nag "didn't catch that".
  - unclear SPEECH (someone spoke, unintelligible): the toy DOES ask to repeat.
These are told apart by the empty transcript / high `no_speech_prob` (noise) vs
a present transcript with low confidence (unclear speech).
"""
from __future__ import annotations

import pytest

from philosopher.core.orchestrator import Orchestrator


class FakeWS:
    def __init__(self) -> None:
        self.json: list[dict] = []
        self.binary: list[tuple[int, bytes]] = []

    async def send_json(self, toy_id: str, data: dict) -> None:
        self.json.append(data)

    async def send_binary(self, toy_id: str, frame_type: int, payload: bytes) -> None:
        self.binary.append((frame_type, payload))


class StubSTT:
    """Returns a fixed transcribe() result so we can drive each branch."""
    def __init__(self, result: dict) -> None:
        self.result = result

    async def transcribe(self, toy_id, language="es", confidence_threshold=-0.5):
        return self.result


async def _make_orch(tmp_path, monkeypatch, stt_result):
    from philosopher.config.settings import get_settings
    monkeypatch.setenv("PHILOSOPHER_MEMORY_DB_PATH", str(tmp_path / "t.db"))
    get_settings.cache_clear()
    orch = Orchestrator()
    await orch.init()
    orch.ws_manager = FakeWS()
    orch.stt = StubSTT(stt_result)
    orch.settings.personality.wake_enabled = False  # exercise the plain path
    return orch


@pytest.mark.asyncio
async def test_empty_transcript_is_silent(tmp_path, monkeypatch):
    """Nothing said -> ignore silently (idle frame, no spoken 'didn't catch that')."""
    from philosopher.config.settings import get_settings
    orch = await _make_orch(tmp_path, monkeypatch,
                            {"text": "", "low_confidence": False, "no_speech_prob": 0.1})
    try:
        await orch.process_speech("toy1")
        assert orch.ws_manager.json == [{"type": "idle"}]
        assert orch.ws_manager.binary == []  # no TTS
    finally:
        await orch.stop()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_high_no_speech_prob_is_silent(tmp_path, monkeypatch):
    """Whisper hallucinated words on noise (no_speech_prob high) -> still silent."""
    from philosopher.config.settings import get_settings
    orch = await _make_orch(
        tmp_path, monkeypatch,
        {"text": "gracias por ver el video", "low_confidence": False,
         "no_speech_prob": 0.95})
    try:
        await orch.process_speech("toy1")
        assert orch.ws_manager.json == [{"type": "idle"}]
        assert orch.ws_manager.binary == []
    finally:
        await orch.stop()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_unclear_speech_asks_to_repeat(tmp_path, monkeypatch):
    """Real (low-confidence) speech -> the toy DOES reply 'please repeat'."""
    from philosopher.config.settings import get_settings
    orch = await _make_orch(
        tmp_path, monkeypatch,
        {"text": "mmm no sé qué", "low_confidence": True, "no_speech_prob": 0.1})
    try:
        await orch.process_speech("toy1")
        texts = [m for m in orch.ws_manager.json if m.get("type") == "text"]
        assert texts, "unclear speech should get a spoken 'didn't catch that'"
        assert {"type": "idle"} not in orch.ws_manager.json[:1]
    finally:
        await orch.stop()
        get_settings.cache_clear()
