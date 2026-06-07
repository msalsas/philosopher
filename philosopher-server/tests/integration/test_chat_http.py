"""Integration tests for the HTTP /chat path (orchestrator.process -> graph).

This path was previously broken (the LangGraph sync wrapper raised "no current
event loop in thread" and silently produced empty replies) and had no test.
"""
from __future__ import annotations

import pytest

from philosopher.config.settings import get_settings
from philosopher.core.orchestrator import Orchestrator
from philosopher.llm.client import LLMResponse


@pytest.mark.asyncio
async def test_http_chat_produces_reply(tmp_path, monkeypatch):
    monkeypatch.setenv("PHILOSOPHER_MEMORY_DB_PATH", str(tmp_path / "chat.db"))
    get_settings.cache_clear()
    try:
        orch = Orchestrator()
        await orch.init()

        async def fake_chat(messages, system_prompt=None, temperature=None, max_tokens=None):
            return LLMResponse(content="Hola, soy el filosofo.", finish_reason="stop")

        orch.llm.chat = fake_chat

        # face_id forces the DB-touching nodes (perception/memory) to run, which
        # is exactly where the old cross-loop/event-loop bug surfaced.
        result = await orch.process("hola, en que piensas?", face_id="abc", emotion="happy")
        assert result.formatted == "Hola, soy el filosofo."
        assert result.turn == 1
        assert result.error is None
        await orch.stop()
    finally:
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_http_chat_llm_error_localized_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("PHILOSOPHER_MEMORY_DB_PATH", str(tmp_path / "chat2.db"))
    get_settings.cache_clear()
    try:
        orch = Orchestrator()
        await orch.init()

        async def fail_chat(messages, system_prompt=None, temperature=None, max_tokens=None):
            return LLMResponse(content="", finish_reason="connection_error")

        orch.llm.chat = fail_chat

        result = await orch.process("hola")
        # node_think flags the error; node_format must substitute the LOCALIZED
        # fallback (Spanish), never the old hardcoded English string.
        assert result.formatted == orch.personality.fallback("error")
        assert "could you repeat" not in result.formatted.lower()
        await orch.stop()
    finally:
        get_settings.cache_clear()
