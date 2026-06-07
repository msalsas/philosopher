"""Tests for the dashboard SSE fan-out hub."""
from __future__ import annotations

import asyncio

import pytest

from philosopher.events.bus import Event, EventBus, EventType
from philosopher.events.dashboard import DashboardHub


@pytest.mark.asyncio
async def test_hub_fans_events_to_clients():
    EventBus._instance = None
    bus = EventBus()
    await bus.start()
    hub = DashboardHub()
    hub.register(bus)
    client = hub.connect()

    await bus.emit(Event(EventType.USER_TEXT, {"face_id": "f1", "text": "hola"}))
    item = await asyncio.wait_for(client.get(), timeout=1.0)
    await bus.stop()

    assert item["type"] == "user_text"
    assert item["data"]["text"] == "hola"
    # backlog keeps recent events for newly opened dashboards.
    assert hub.recent()[-1]["type"] == "user_text"


@pytest.mark.asyncio
async def test_hub_disconnect_stops_delivery():
    EventBus._instance = None
    bus = EventBus()
    await bus.start()
    hub = DashboardHub()
    hub.register(bus)
    client = hub.connect()
    hub.disconnect(client)

    await bus.emit(Event(EventType.RESPONSE_READY, {"face_id": "f1"}))
    await asyncio.sleep(0.05)
    await bus.stop()

    assert client.empty()  # nothing delivered after disconnect
