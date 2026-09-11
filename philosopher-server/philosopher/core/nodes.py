"""LangGraph nodes for the conversational agent pipeline."""
from __future__ import annotations

import asyncio
import logging

from philosopher.core.state import AgentState
from philosopher.llm.client import LLMMessage

logger = logging.getLogger(__name__)


def _log_task_error(task: asyncio.Task) -> None:
    """Surface a fire-and-forget task's exception instead of swallowing it."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("background task failed: %r", exc, exc_info=exc)


class NodeCtx:
    """Shared context passed to all graph nodes."""
    def __init__(self, llm, memory, personality, events=None, vision=None):
        self.llm = llm
        self.memory = memory
        self.personality = personality
        self.events = events
        # Set after wiring (vision is built after the orchestrator). Lets the
        # name-registration task confirm identity biometrically before merging.
        self.vision = vision


async def node_perception(state: AgentState, ctx: NodeCtx) -> AgentState:
    """Load face context from long-term memory."""
    if state.face_id:
        face = await ctx.memory.long.get_face(state.face_id)
        if face:
            state.face_name = face["name"]
            state.is_new_face = face["encounters"] <= 1
    return state


async def node_memory(state: AgentState, ctx: NodeCtx) -> AgentState:
    """Retrieve relevant memories for the current message."""
    if not state.message:
        return state
    mem = await ctx.memory.get_context(state.message, state.face_id)
    state.short_context = mem["short"]
    state.long_memories = mem["long"]
    return state


async def node_prompt(state: AgentState, ctx: NodeCtx) -> AgentState:
    """Build the system prompt with personality + context."""
    state.system_prompt = ctx.personality.build_prompt(
        emotion=state.emotion,
        face_name=state.face_name,
        memories=state.long_memories,
    )
    return state


def build_messages(state: AgentState) -> list[LLMMessage]:
    """Assemble the LLM message list from short context + the current message,
    plus a new-face introduction/name-ask instruction.

    Shared by the HTTP graph (`node_think`) and the WebSocket streaming path
    (`Orchestrator._stream_response`) so both behave identically.
    """
    msgs = [LLMMessage(role=m["role"], content=m["content"]) for m in state.short_context[-5:]]
    msgs.append(LLMMessage(role="user", content=state.message))
    if state.is_new_face and state.face_name:
        msgs.append(LLMMessage(
            role="user", content=f"[Introduce yourself to {state.face_name}]"))
    elif not state.face_name:
        # New face, or a known one still unnamed — keep asking so it can be learned.
        msgs.append(LLMMessage(
            role="user",
            content="[You don't know this person's name yet. "
                    "Greet them warmly and ask their name.]"))
    return msgs


async def node_think(state: AgentState, ctx: NodeCtx) -> AgentState:
    """Call the LLM to generate a response."""
    if not state.message:
        state.error = "No user message"
        return state
    msgs = build_messages(state)
    resp = await ctx.llm.chat(msgs, system_prompt=state.system_prompt, max_tokens=250)
    state.llm_response = resp.content
    if resp.finish_reason not in ("stop", ""):
        state.error = f"LLM error: {resp.finish_reason}"
    return state


async def node_format(state: AgentState, ctx: NodeCtx) -> AgentState:
    """Format the LLM response with personality styling."""
    if state.error:
        state.formatted = ctx.personality.fallback("error")
    else:
        state.formatted = ctx.personality.format_response(state.llm_response)
    return state


_NAME_SYS = (
    "You extract the speaker's OWN first name from their message. Reply with ONLY "
    "that name, capitalized, and nothing else — or the single word NONE if the "
    "message does not state the speaker's own name."
)


async def _llm_extract_name(llm, message: str) -> str | None:
    """Ask the LLM for the speaker's own first name (robust + multilingual). The
    LLM client never raises (empty content on error), so a bad call yields None."""
    resp = await llm.chat(
        [LLMMessage(role="user", content=message)],
        system_prompt=_NAME_SYS, max_tokens=8,
    )
    tokens = (resp.content or "").strip().split()
    if not tokens:
        return None
    name = tokens[0].strip(".,!?¡¿").title()
    if name.upper() == "NONE" or len(name) < 2 or not name.isalpha():
        return None
    return name


async def node_learn_name(state: AgentState, ctx: NodeCtx) -> AgentState:
    """Pipeline node: learn the speaker's name for a face that has none yet, via
    an LLM (any language). Runs last, after the reply, so it adds no latency; it
    does nothing when there is no face or the message states no name."""
    if not state.face_id:
        return state
    face = await ctx.memory.long.get_face(state.face_id)
    if face and face.get("name"):
        return state  # already named

    name = await _llm_extract_name(ctx.llm, state.user_message)
    if not name:
        return state

    # Name is just a label, not a unique key — two people can be "Pedro". If a
    # face with this name already exists, only merge into it when the biometrics
    # agree (same person whose face wasn't recognized); otherwise keep them apart.
    existing = await ctx.memory.long.find_face_by_name(name, exclude_id=state.face_id)
    vision = getattr(ctx, "vision", None)
    enc = (vision.maybe_merge(state.face_id, existing["face_id"], name)
           if existing and vision else None)
    if existing and enc is not None:
        await ctx.memory.long.merge_face(state.face_id, existing["face_id"], encoding=enc)
    else:
        await ctx.memory.long.update_face_name(state.face_id, name)
    return state


async def node_store(state: AgentState, ctx: NodeCtx) -> AgentState:
    """Store the interaction in memory."""
    if state.message and state.formatted:
        await ctx.memory.add(state.message, state.formatted, state.emotion,
                             state.face_id, state.face_name)
    state.turn += 1
    return state
