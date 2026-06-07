"""Tests for face registration background extraction."""
from __future__ import annotations

import pytest
import pytest_asyncio

from philosopher.config.settings import (
    LLMSettings,
    MemorySettings,
    PersonalitySettings,
    Settings,
)
from philosopher.core.nodes import NodeCtx, background_extract_name
from philosopher.core.state import AgentState
from philosopher.llm.client import LLMClient
from philosopher.memory.manager import MemoryManager
from philosopher.personality.engine import PersonalityEngine


@pytest_asyncio.fixture
async def ctx(tmp_dir):
    s = Settings(
        llm=LLMSettings(provider="local", base_url="http://test:1234/v1",
                       api_key="test", model="test-model", max_tokens=100),
        memory=MemorySettings(db_path=str(tmp_dir / "test.db"), short_term_limit=5),
        personality=PersonalitySettings(personality="filosofo", language="es"),
    )
    llm = LLMClient(s)
    pe = PersonalityEngine(s.personality)
    mem = MemoryManager(s.memory, llm)
    await mem.init()
    return NodeCtx(llm, mem, pe)


@pytest.mark.asyncio
async def test_extract_name_short_reply(ctx):
    # Simulate face being registered during conversation
    await ctx.memory.long._update_face("f1", None)
    state = AgentState(user_message="Soy Juan", face_id="f1", is_new_face=True)
    await background_extract_name(state, ctx)
    face = await ctx.memory.long.get_face("f1")
    assert face["name"] == "Juan"


@pytest.mark.asyncio
async def test_extract_name_too_long(ctx):
    await ctx.memory.long._update_face("f2", None)
    state = AgentState(
        user_message="Me llamo Juan Carlos Rodriguez", face_id="f2", is_new_face=True,
    )
    await background_extract_name(state, ctx)
    face = await ctx.memory.long.get_face("f2")
    assert face["name"] is None


@pytest.mark.asyncio
async def test_extract_name_not_new_face(ctx):
    state = AgentState(user_message="Juan", face_id="f3", is_new_face=False)
    await background_extract_name(state, ctx)
    face = await ctx.memory.long.get_face("f3")
    assert face is None
