"""Wake-word gate state machine (orchestrator._apply_wake_gate).

The toy starts asleep and only engages after the activation phrase; a
deactivation phrase or `sleep_timeout` of inactivity sends it back to sleep.
All matching is on the (server-side) transcript, so no toy-side ML is needed.
"""
from __future__ import annotations

import pytest

from philosopher.core.orchestrator import Orchestrator
from philosopher.tts.engine import PiperTTS


class FakeWS:
    def __init__(self) -> None:
        self.json: list[dict] = []
        self.binary: list[tuple[int, bytes]] = []

    async def send_json(self, toy_id: str, data: dict) -> None:
        self.json.append(data)

    async def send_binary(self, toy_id: str, frame_type: int, payload: bytes) -> None:
        self.binary.append((frame_type, payload))


async def _make_orch(tmp_path, monkeypatch):
    from philosopher.config.settings import get_settings
    monkeypatch.setenv("PHILOSOPHER_MEMORY_DB_PATH", str(tmp_path / "t.db"))
    get_settings.cache_clear()
    orch = Orchestrator()
    await orch.init()
    orch.tts = PiperTTS(mock=True)
    orch.ws_manager = FakeWS()
    orch.settings.personality.wake_enabled = True
    orch.settings.personality.sleep_timeout = 1800.0
    return orch


@pytest.mark.asyncio
async def test_asleep_ignores_and_releases_gate(tmp_path, monkeypatch):
    from philosopher.config.settings import get_settings
    orch = await _make_orch(tmp_path, monkeypatch)
    try:
        # Asleep by default: a normal sentence is ignored (returns None) but the
        # mic gate is released with a silent {type:"idle"}.
        out = await orch._apply_wake_gate("toy1", "cuentame algo interesante")
        assert out is None
        assert orch.awake.get("toy1") is not True
        assert {"type": "idle"} in orch.ws_manager.json
    finally:
        await orch.stop()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_wake_word_greets_and_wakes(tmp_path, monkeypatch):
    from philosopher.config.settings import get_settings
    orch = await _make_orch(tmp_path, monkeypatch)
    try:
        # Accent/case survive normalization ("Despiérta" -> "despierta").
        out = await orch._apply_wake_gate("toy1", "Despiérta.")
        assert out is None  # just the wake word -> greet only, nothing to answer
        assert orch.awake["toy1"] is True
        assert any(m.get("type") == "text" for m in orch.ws_manager.json)
    finally:
        await orch.stop()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_one_shot_returns_remainder(tmp_path, monkeypatch):
    from philosopher.config.settings import get_settings
    orch = await _make_orch(tmp_path, monkeypatch)
    try:
        out = await orch._apply_wake_gate("toy1", "despierta cuentame un chiste")
        assert out is not None
        assert "chiste" in out
        assert orch.awake["toy1"] is True
    finally:
        await orch.stop()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_awake_passes_through_then_sleep_word(tmp_path, monkeypatch):
    from philosopher.config.settings import get_settings
    orch = await _make_orch(tmp_path, monkeypatch)
    try:
        import time
        orch.awake["toy1"] = True
        orch.last_active["toy1"] = time.monotonic()
        # Normal speech while awake passes through unchanged.
        out = await orch._apply_wake_gate("toy1", "que tal estas hoy")
        assert out == "que tal estas hoy"
        # Deactivation phrase -> farewell + back to sleep.
        out = await orch._apply_wake_gate("toy1", "hasta luego")
        assert out is None
        assert orch.awake["toy1"] is False
    finally:
        await orch.stop()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_inactivity_auto_sleep(tmp_path, monkeypatch):
    from philosopher.config.settings import get_settings
    orch = await _make_orch(tmp_path, monkeypatch)
    try:
        import time
        orch.awake["toy1"] = True
        orch.settings.personality.sleep_timeout = 1.0
        orch.last_active["toy1"] = time.monotonic() - 5.0  # 5s idle > 1s timeout
        # Auto-slept: a normal sentence is now ignored (needs the wake word).
        out = await orch._apply_wake_gate("toy1", "sigues ahi")
        assert out is None
        assert orch.awake["toy1"] is False
    finally:
        await orch.stop()
        get_settings.cache_clear()
