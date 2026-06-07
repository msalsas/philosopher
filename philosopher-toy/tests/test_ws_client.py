"""Tests for WebSocket client."""
from __future__ import annotations

import pytest

from banana_client.protocol.ws_client import ToyWebSocketClient


def test_creation():
    c = ToyWebSocketClient("ws://test:8080", "toy_01")
    assert c.url == "ws://test:8080"
    assert c.toy_id == "toy_01"


def test_reconnect_limits():
    c = ToyWebSocketClient("ws://test:8080", max_reconnect=3)
    assert c.max_reconnect == 3


@pytest.mark.asyncio
async def test_reconnect_succeeds_after_failures():
    c = ToyWebSocketClient("ws://test:8080", reconnect_interval=0, max_reconnect=5)
    calls = {"n": 0}

    async def fake_connect():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("server down")
        c._reconnect_count = 0  # real connect() resets this on success

    c.connect = fake_connect
    assert await c.reconnect() is True
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_reconnect_gives_up():
    c = ToyWebSocketClient("ws://test:8080", reconnect_interval=0, max_reconnect=2)
    calls = {"n": 0}

    async def always_fail():
        calls["n"] += 1
        raise ConnectionError("server down")

    c.connect = always_fail
    assert await c.reconnect() is False
    assert calls["n"] == 2  # exactly max_reconnect attempts
