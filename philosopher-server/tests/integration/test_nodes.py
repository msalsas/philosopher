"""Integration tests for LangGraph nodes."""
from __future__ import annotations

import pytest
import pytest_asyncio

from philosopher.config.settings import (
    LLMSettings,
    MemorySettings,
    PersonalitySettings,
    Settings,
)
from philosopher.core.nodes import NodeCtx, node_format, node_memory, node_perception, node_prompt
from philosopher.core.state import AgentState
from philosopher.llm.client import LLMClient
from philosopher.memory.manager import MemoryManager
from philosopher.personality.engine import PersonalityEngine


@pytest_asyncio.fixture
async def ctx(tmp_dir):
    s = Settings(
        llm=LLMSettings(provider="local", base_url="http://test:1234/v1",
                       model="test-model", max_tokens=100),
        memory=MemorySettings(db_path=str(tmp_dir / "test.db"), short_term_limit=5),
        personality=PersonalitySettings(personality="filosofo", language="es"),
    )
    llm = LLMClient(s)
    pe = PersonalityEngine(s.personality)
    mem = MemoryManager(s.memory, llm)
    await mem.init()
    return NodeCtx(llm, mem, pe)


@pytest.mark.asyncio
async def test_perception_no_face(ctx):
    s = AgentState(user_message="hello")
    r = await node_perception(s, ctx)
    assert r.face_name is None


@pytest.mark.asyncio
async def test_perception_with_face(ctx):
    await ctx.memory.long.store("hi", face_id="f1", face_name="Alice")
    s = AgentState(user_message="hello", face_id="f1")
    r = await node_perception(s, ctx)
    assert r.face_name == "Alice"


@pytest.mark.asyncio
async def test_memory(ctx):
    s = AgentState(user_message="test message")
    r = await node_memory(s, ctx)
    assert isinstance(r.short_context, list)


@pytest.mark.asyncio
async def test_prompt(ctx):
    s = AgentState(user_message="hi", emotion="happy")
    s.short_context = []
    s.long_memories = []
    r = await node_prompt(s, ctx)
    assert len(r.system_prompt) > 0


@pytest.mark.asyncio
async def test_format_with_error(ctx):
    s = AgentState(error="some error")
    r = await node_format(s, ctx)
    assert len(r.formatted) > 0


@pytest.mark.asyncio
async def test_format_normal(ctx):
    s = AgentState(llm_response="hello world")
    r = await node_format(s, ctx)
    assert r.formatted == "hello world"
