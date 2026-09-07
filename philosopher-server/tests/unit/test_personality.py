"""Tests for personality engine."""
from __future__ import annotations

from philosopher.config.settings import PersonalitySettings
from philosopher.personality.engine import PersonalityEngine


def test_load():
    ps = PersonalitySettings(personality="philosopher", language="es")
    pe = PersonalityEngine(ps)
    assert pe.system_prompt != ""
    assert len(pe.name) > 0


def test_greeting():
    ps = PersonalitySettings(language="es")
    pe = PersonalityEngine(ps)
    g = pe.greeting()
    assert len(g) > 0


def test_emotion_responses():
    ps = PersonalitySettings(language="es")
    pe = PersonalityEngine(ps)
    for emotion in ["happy", "sad", "angry", "surprised", "neutral"]:
        r = pe.greeting(emotion)
        assert len(r) > 0


def test_build_prompt():
    ps = PersonalitySettings(language="es")
    pe = PersonalityEngine(ps)
    p = pe.build_prompt(emotion="happy", face_name="Alice",
                       memories=[{"summary": "likes coffee"}])
    assert "Alice" in p
    assert "coffee" in p


def test_format():
    ps = PersonalitySettings(language="es")
    pe = PersonalityEngine(ps)
    assert pe.format_response("hello") == "hello"


def test_list():
    ps = PersonalitySettings(language="es")
    pe = PersonalityEngine(ps)
    p = pe.list_all()
    assert "philosopher" in p
    assert "curious" in p
    assert "friend" in p


def test_different_personalities():
    for name in ["philosopher", "curious", "poetic", "friend", "wise"]:
        ps = PersonalitySettings(personality=name, language="es")
        pe = PersonalityEngine(ps)
        assert pe.system_prompt != ""
