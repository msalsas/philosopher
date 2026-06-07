"""LangGraph definition for the Philosopher agent pipeline.

The six nodes are registered as **async** functions and run via ``ainvoke``, so
LangGraph awaits them on the caller's event loop. (The earlier version wrapped
them in a *sync* shim that called ``asyncio.get_event_loop().run_until_complete()``
inside a worker thread — which raised "no current event loop in thread" and made
every HTTP /chat reply silently empty. LangGraph supports async nodes natively,
so no such wrapper is needed.)

Keeping LangGraph buys declarative edges (for future conditional branches),
state tracing, and LangSmith integration.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from philosopher.core.nodes import (
    NodeCtx,
    node_format,
    node_memory,
    node_perception,
    node_prompt,
    node_store,
    node_think,
)
from philosopher.core.state import AgentState


class AgentGraph:
    """Compiled LangGraph for processing conversations."""

    def __init__(self, ctx: NodeCtx) -> None:
        self.ctx = ctx
        self._app = self._build().compile()

    def _build(self) -> StateGraph:
        g = StateGraph(dict)

        def wrap(fn):
            """Adapt an async node (AgentState->AgentState) to a LangGraph node."""
            async def anode(d: dict) -> dict:
                s = AgentState.from_dict(d)
                try:
                    s = await fn(s, self.ctx)
                except Exception as e:  # noqa: BLE001 - surface as state.error
                    s.error = str(e)
                return s.to_dict()
            return anode

        g.add_node("perception", wrap(node_perception))
        g.add_node("memory", wrap(node_memory))
        g.add_node("prompt", wrap(node_prompt))
        g.add_node("think", wrap(node_think))
        g.add_node("format", wrap(node_format))
        g.add_node("store", wrap(node_store))

        g.add_edge(START, "perception")
        g.add_edge("perception", "memory")
        g.add_edge("memory", "prompt")
        g.add_edge("prompt", "think")
        g.add_edge("think", "format")
        g.add_edge("format", "store")
        g.add_edge("store", END)
        return g

    async def process(self, state: AgentState) -> AgentState:
        result = await self._app.ainvoke(state.to_dict())
        return AgentState.from_dict(result)


def create_graph(ctx: NodeCtx) -> AgentGraph:
    return AgentGraph(ctx)
