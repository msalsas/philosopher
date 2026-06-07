"""Tests for /health engine-readiness surfacing (orchestrator.health)."""
from __future__ import annotations

import pytest

from philosopher.core.orchestrator import Orchestrator
from philosopher.stt.engine import StreamingSTT
from philosopher.tts.engine import PiperTTS


@pytest.mark.asyncio
async def test_health_reports_ok_when_engines_mocked():
    orch = Orchestrator()
    orch.stt = StreamingSTT(mock=True)
    orch.tts = PiperTTS(mock=True)
    h = await orch.health()
    assert h["stt"]["status"] == "mock"
    assert h["tts"]["status"] == "mock"
    assert h["status"] == "ok"


@pytest.mark.asyncio
async def test_health_rolls_up_unavailable_stt_to_degraded():
    # A failed STT load (the dangerous "answers phantom speech" case) must be
    # visible at the top level, not buried.
    orch = Orchestrator()
    broken = StreamingSTT(mock=False)
    broken.model = None
    orch.stt = broken
    h = await orch.health()
    assert h["stt"]["status"] == "unavailable"
    assert h["status"] == "degraded"
