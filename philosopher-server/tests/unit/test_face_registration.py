"""Tests for face registration background extraction."""
from __future__ import annotations

import asyncio

import numpy as np
import pytest
import pytest_asyncio

from philosopher.config.settings import (
    LLMSettings,
    MemorySettings,
    PersonalitySettings,
    Settings,
)
from philosopher.core.nodes import NodeCtx, background_extract_name, node_store
from philosopher.core.state import AgentState
from philosopher.llm.client import LLMClient
from philosopher.memory.manager import MemoryManager
from philosopher.personality.engine import PersonalityEngine
from philosopher.vision.engine import VisionProcessor


@pytest_asyncio.fixture
async def ctx(tmp_dir):
    s = Settings(
        llm=LLMSettings(provider="local", base_url="http://test:1234/v1",
                       api_key="test", model="test-model", max_tokens=100),
        memory=MemorySettings(db_path=str(tmp_dir / "test.db"), short_term_limit=5),
        personality=PersonalitySettings(personality="philosopher", language="es"),
    )
    llm = LLMClient(s)
    pe = PersonalityEngine(s.personality)
    mem = MemoryManager(s.memory, llm)
    await mem.init()
    yield NodeCtx(llm, mem, pe)
    # Close the aiosqlite connection so its worker thread doesn't raise during
    # teardown (these tests schedule real background DB tasks).
    await mem.close()


@pytest.mark.asyncio
async def test_extract_name_short_reply(ctx):
    # Simulate face being registered during conversation
    await ctx.memory.long._update_face("f1", None)
    state = AgentState(user_message="Soy Juan", face_id="f1", expect_name=True)
    await background_extract_name(state, ctx)
    face = await ctx.memory.long.get_face("f1")
    assert face["name"] == "Juan"


@pytest.mark.asyncio
async def test_extract_name_too_long(ctx):
    await ctx.memory.long._update_face("f2", None)
    state = AgentState(
        user_message="Me llamo Juan Carlos Rodriguez", face_id="f2", expect_name=True,
    )
    await background_extract_name(state, ctx)
    face = await ctx.memory.long.get_face("f2")
    assert face["name"] is None


@pytest.mark.asyncio
async def test_extract_name_not_expected(ctx):
    # Name wasn't asked last turn (expect_name False) → don't store noise as a name.
    state = AgentState(user_message="Juan", face_id="f3", expect_name=False)
    await background_extract_name(state, ctx)
    face = await ctx.memory.long.get_face("f3")
    assert face is None


async def _seed_two_faces(ctx, fresh_enc, known_enc):
    """A known 'Pedro' plus a freshly-minted face, wired into DB + vision."""
    ref = np.asarray(known_enc, dtype=np.float64)
    new = np.asarray(fresh_enc, dtype=np.float64)
    await ctx.memory.long.add_face("known", ref.tobytes(), name="Pedro")
    await ctx.memory.long.add_face("fresh", new.tobytes())
    await ctx.memory.long.store("hola", face_id="fresh", face_name="Pedro")
    vp = VisionProcessor(mock=False)
    vp.known_faces = {
        "known": {"name": "Pedro", "encoding": ref},
        "fresh": {"name": None, "encoding": new},
    }
    ctx.vision = vp
    return vp


@pytest.mark.asyncio
async def test_dedup_merges_same_person(ctx):
    # Same person, face drifted: encodings within merge_band → fold fresh into known.
    ref = np.linspace(0, 1, 128)
    vp = await _seed_two_faces(ctx, fresh_enc=ref + 0.001, known_enc=ref)
    state = AgentState(user_message="Soy Pedro", face_id="fresh", expect_name=True)
    await background_extract_name(state, ctx)

    assert await ctx.memory.long.get_face("fresh") is None      # duplicate removed
    assert (await ctx.memory.long.get_face("known"))["name"] == "Pedro"
    assert "fresh" not in vp.known_faces                         # vision repointed


@pytest.mark.asyncio
async def test_dedup_keeps_namesake_separate(ctx):
    # Different people who share the name: encodings far apart → keep both.
    ref = np.linspace(0, 1, 128)
    vp = await _seed_two_faces(ctx, fresh_enc=ref + 5.0, known_enc=ref)
    state = AgentState(user_message="Soy Pedro", face_id="fresh", expect_name=True)
    await background_extract_name(state, ctx)

    assert (await ctx.memory.long.get_face("fresh"))["name"] == "Pedro"  # kept + labeled
    assert await ctx.memory.long.get_face("known") is not None
    assert "fresh" in vp.known_faces                                     # not merged


@pytest.mark.asyncio
async def test_node_store_logs_background_failure(ctx, caplog, monkeypatch):
    # The fire-and-forget name task must surface its errors, not swallow them.
    async def boom(*a, **k):
        raise RuntimeError("db exploded")

    monkeypatch.setattr(ctx.memory.long, "get_face", boom)
    state = AgentState(user_message="Soy Ana", face_id="fX", expect_name=True,
                       formatted="hola")

    with caplog.at_level("ERROR", logger="philosopher.core.nodes"):
        await node_store(state, ctx)
        await asyncio.sleep(0.05)

    assert any("db exploded" in str(r.exc_info) or "db exploded" in r.getMessage()
               for r in caplog.records), "the background task failure should be logged"
