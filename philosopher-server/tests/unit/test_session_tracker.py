"""Tests for the event-bus-driven SessionTracker."""
from __future__ import annotations

import asyncio

import pytest

from philosopher.events.bus import Event, EventBus, EventType
from philosopher.events.session_tracker import SessionTracker


async def _drain():
    # The bus dispatches asynchronously; give handlers time to run.
    for _ in range(20):
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_tracker_aggregates_per_face():
    EventBus._instance = None
    bus = EventBus()
    await bus.start()
    tracker = SessionTracker()
    tracker.register(bus)

    await bus.emit(Event(EventType.FACE_RECOGNIZED, {"face_id": "f1", "name": "Ana"}))
    await bus.emit(Event(EventType.EMOTION_DETECTED, {"face_id": "f1", "emotion": "happy"}))
    await bus.emit(Event(EventType.USER_TEXT, {"face_id": "f1", "text": "hola"}))
    await bus.emit(Event(EventType.RESPONSE_READY, {"face_id": "f1", "text": "hola Ana"}))
    await bus.emit(Event(EventType.USER_TEXT, {"face_id": "f2", "text": "hey"}))
    await _drain()
    await bus.stop()

    snap = tracker.snapshot()
    assert snap["f1"]["name"] == "Ana"
    assert snap["f1"]["messages"] == 1
    assert snap["f1"]["responses"] == 1
    assert snap["f1"]["emotions"].get("happy") == 1
    # A different person is tracked separately.
    assert snap["f2"]["messages"] == 1
    assert snap["f2"]["name"] is None
