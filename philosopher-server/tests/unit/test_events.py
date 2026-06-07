"""Tests for event bus."""
from __future__ import annotations

import asyncio

import pytest

from philosopher.events.bus import Event, EventBus, EventType


def test_event():
    e = Event(type=EventType.USER_TEXT, data={"text": "hi"})
    assert e.get("text") == "hi"
    assert e.get("missing") is None


class TestEventBus:
    @pytest.fixture(autouse=True)
    def reset(self):
        EventBus._instance = None
        yield
        EventBus._instance = None

    def test_subscribe(self):
        EventBus._instance = None
        bus = EventBus()
        async def h(e): pass
        bus.subscribe(EventType.USER_TEXT, h)
        assert h in bus._subs[EventType.USER_TEXT]

    @pytest.mark.asyncio
    async def test_emit_and_receive(self, event_bus):
        received = []
        async def h(ev):
            received.append(ev)
        event_bus.subscribe(EventType.USER_TEXT, h)
        await event_bus.emit(Event(type=EventType.USER_TEXT, data={"text": "hello"}))
        await asyncio.sleep(0.1)
        assert len(received) == 1
        assert received[0].get("text") == "hello"

    @pytest.mark.asyncio
    async def test_global(self, event_bus):
        received = []
        async def h(ev):
            received.append(ev.type.value)
        event_bus.subscribe_all(h)
        await event_bus.emit(Event(type=EventType.BUTTON_PRESSED, data={}))
        await asyncio.sleep(0.1)
        assert len(received) == 1
