"""Tests for agent state."""
from __future__ import annotations

from philosopher.core.state import AgentState


def test_defaults():
    s = AgentState()
    assert s.user_message == ""
    assert s.turn == 0
    assert s.error is None


def test_message():
    s = AgentState(user_message="hello")
    assert s.message == "hello"


def test_to_dict():
    s = AgentState(user_message="hi", turn=5, emotion="happy")
    d = s.to_dict()
    assert d["user_message"] == "hi"
    assert d["turn"] == 5
    assert d["emotion"] == "happy"
