"""Tests for agent state from_dict."""
from __future__ import annotations

from philosopher.core.state import AgentState


def test_from_dict():
    d = {
        "user_message": "hello",
        "face_id": "abc",
        "face_name": "Alice",
        "emotion": "happy",
        "is_new_face": True,
        "system_prompt": "You are a philosopher",
        "short_context": [{"role": "user", "content": "hi"}],
        "long_memories": [],
        "llm_response": "world",
        "formatted": "world",
        "error": None,
        "turn": 3,
        "pending_audio": True,
        "servo_emotion": "happy",
    }
    s = AgentState.from_dict(d)
    assert s.user_message == "hello"
    assert s.face_id == "abc"
    assert s.pending_audio is True
    assert s.servo_emotion == "happy"


def test_from_dict_ignores_extra_keys():
    d = {"user_message": "hi", "extra_key": "should be ignored"}
    s = AgentState.from_dict(d)
    assert s.user_message == "hi"
    assert s.turn == 0
