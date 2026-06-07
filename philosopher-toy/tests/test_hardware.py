"""Tests for hardware controllers."""
from __future__ import annotations

import pytest

from banana_client.hardware.servos import ServoController


@pytest.mark.asyncio
async def test_servo_mock():
    s = await ServoController(mock=True).init()
    assert s.mock is True
    await s.move("head", 45)
    assert s.pos["head"] == 45


@pytest.mark.asyncio
async def test_servo_animate():
    s = await ServoController(mock=True).init()
    calls = _spy_moves(s)
    await s.animate("happy")
    assert "right_arm" in calls        # gestured with the arm
    assert "head" not in calls          # head not used for emotion (gaze only)
    assert s.pos["right_arm"] == 0      # arm returned to rest, not left raised


@pytest.mark.asyncio
async def test_look_sets_head_base():
    s = await ServoController(mock=True).init()
    await s.look(1.0)                      # full right -> MAX_PAN
    assert s.head_base == s.MAX_PAN
    assert s.pos["head"] == s.MAX_PAN
    await s.look(-1.0)                     # full left
    assert s.head_base == -s.MAX_PAN


@pytest.mark.asyncio
async def test_look_deadzone_ignores_small_change():
    s = await ServoController(mock=True).init()
    await s.look(1.0)
    base = s.head_base
    await s.look(1.0)                      # identical -> within deadzone, no move
    assert s.head_base == base


@pytest.mark.asyncio
async def test_animate_keeps_head_on_gaze_base():
    s = await ServoController(mock=True).init()
    await s.look(1.0)                      # gaze base = MAX_PAN
    calls = _spy_moves(s)
    await s.animate("happy")              # emotion via arm; head stays put
    assert "head" not in calls             # head not moved for emotion
    assert s.pos["head"] == s.MAX_PAN       # head still on gaze base
    assert s.pos["right_arm"] == 0          # arm returned to rest


@pytest.mark.asyncio
async def test_animate_returns_arms_to_rest():
    s = await ServoController(mock=True).init()
    await s.animate("surprised")          # raises both arms, holds, lowers both
    assert s.pos["right_arm"] == 0
    assert s.pos["left_arm"] == 0


def _spy_moves(s):
    """Wrap s.move to record which servos get moved."""
    calls = []
    orig = s.move

    async def spy(name, angle, duration=0.5):
        calls.append(name)
        await orig(name, angle, duration)

    s.move = spy
    return calls


@pytest.mark.asyncio
async def test_react_positive_uses_arm_not_head():
    s = await ServoController(mock=True).init()
    await s.look(1.0)                      # gaze base = MAX_PAN
    calls = _spy_moves(s)
    await s.react("happy")
    assert "right_arm" in calls
    assert "head" not in calls             # positive must NOT wiggle the head ("no")
    assert s.pos["head"] == s.MAX_PAN       # head stays on gaze base
    assert s.pos["right_arm"] == 0          # arm returns to rest


@pytest.mark.asyncio
async def test_react_negative_shakes_head():
    s = await ServoController(mock=True).init()
    calls = _spy_moves(s)
    await s.react("sad")                   # head shake = "no", apt for negative emotion
    assert "head" in calls
    assert "right_arm" not in calls and "left_arm" not in calls
    assert s.pos["head"] == s.head_base     # settles back on the gaze base


@pytest.mark.asyncio
async def test_react_neutral_is_noop():
    s = await ServoController(mock=True).init()
    calls = _spy_moves(s)
    await s.react("neutral")
    assert calls == []


@pytest.mark.asyncio
async def test_idle_tick_stays_near_base():
    s = await ServoController(mock=True).init()
    await s.look(1.0)                      # base = MAX_PAN
    await s.idle_tick()
    assert abs(s.pos["head"] - s.head_base) <= s.IDLE_DRIFT
