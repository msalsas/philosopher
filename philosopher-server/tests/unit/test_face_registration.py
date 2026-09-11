"""Tests for the name-learning node (LLM-based) + biometric dedup."""
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
from philosopher.core.nodes import NodeCtx, _log_task_error, node_learn_name
from philosopher.core.state import AgentState
from philosopher.llm.client import LLMClient, LLMResponse
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
    await mem.close()


def _mock_llm(ctx, name: str) -> None:
    """Make the LLM name-extractor return `name` (e.g. "Ana" or "NONE")."""
    async def chat(messages, system_prompt=None, temperature=None, max_tokens=None):
        return LLMResponse(content=name)
    ctx.llm.chat = chat


@pytest.mark.asyncio
async def test_learn_name_stores(ctx):
    await ctx.memory.long._update_face("f1", None)
    _mock_llm(ctx, "Ana")
    state = AgentState(user_message="Yo me llamo Ana.", face_id="f1")
    await node_learn_name(state, ctx)
    assert (await ctx.memory.long.get_face("f1"))["name"] == "Ana"


@pytest.mark.asyncio
async def test_learn_name_none_not_stored(ctx):
    await ctx.memory.long._update_face("f2", None)
    _mock_llm(ctx, "NONE")  # message states no name
    state = AgentState(user_message="que tal todo", face_id="f2")
    await node_learn_name(state, ctx)
    assert (await ctx.memory.long.get_face("f2"))["name"] is None


@pytest.mark.asyncio
async def test_learn_name_skips_already_named(ctx):
    await ctx.memory.long._update_face("f3", None)
    await ctx.memory.long.update_face_name("f3", "Pedro")
    _mock_llm(ctx, "Otro")  # must not overwrite an existing name
    state = AgentState(user_message="me llamo Otro", face_id="f3")
    await node_learn_name(state, ctx)
    assert (await ctx.memory.long.get_face("f3"))["name"] == "Pedro"


@pytest.mark.asyncio
async def test_learn_name_no_face_noop(ctx):
    _mock_llm(ctx, "Ana")
    state = AgentState(user_message="me llamo Ana", face_id=None)
    await node_learn_name(state, ctx)  # no face -> nothing to name, no error


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
    _mock_llm(ctx, "Pedro")
    state = AgentState(user_message="Soy Pedro", face_id="fresh")
    await node_learn_name(state, ctx)

    assert await ctx.memory.long.get_face("fresh") is None      # duplicate removed
    assert (await ctx.memory.long.get_face("known"))["name"] == "Pedro"
    assert "fresh" not in vp.known_faces                         # vision repointed


@pytest.mark.asyncio
async def test_dedup_keeps_namesake_separate(ctx):
    # Different people who share the name: encodings far apart → keep both.
    ref = np.linspace(0, 1, 128)
    vp = await _seed_two_faces(ctx, fresh_enc=ref + 5.0, known_enc=ref)
    _mock_llm(ctx, "Pedro")
    state = AgentState(user_message="Soy Pedro", face_id="fresh")
    await node_learn_name(state, ctx)

    assert (await ctx.memory.long.get_face("fresh"))["name"] == "Pedro"  # kept + labeled
    assert await ctx.memory.long.get_face("known") is not None
    assert "fresh" in vp.known_faces                                     # not merged


@pytest.mark.asyncio
async def test_learn_name_error_logged(ctx, caplog, monkeypatch):
    # Fire-and-forget (as the orchestrator does): errors must surface in the log.
    async def boom(*a, **k):
        raise RuntimeError("db exploded")

    monkeypatch.setattr(ctx.memory.long, "get_face", boom)
    _mock_llm(ctx, "Ana")
    state = AgentState(user_message="Soy Ana", face_id="fX")

    with caplog.at_level("ERROR", logger="philosopher.core.nodes"):
        task = asyncio.create_task(node_learn_name(state, ctx))
        task.add_done_callback(_log_task_error)
        await asyncio.sleep(0.05)

    assert any("db exploded" in str(r.exc_info) or "db exploded" in r.getMessage()
               for r in caplog.records), "the background task failure should be logged"
