"""WebSocket client for toy-server communication."""
from __future__ import annotations

import asyncio
import json

import aiohttp


class ToyWebSocketClient:
    """WebSocket client for toy-server communication."""

    def __init__(self, server_url: str, toy_id: str = "toy_01",
                 reconnect_interval=5.0, max_reconnect=10, send_timeout=4.0):
        self.url = server_url.rstrip("/")
        self.toy_id = toy_id
        self.reconnect_interval = reconnect_interval
        self.max_reconnect = max_reconnect
        # Cap on how long a single send may block. On a flaky Wi-Fi link
        # ws.send_bytes() can hang for minutes draining TCP on a half-dead
        # socket; without this the audio task stalls there and the mic loop
        # stops feeding forever (only reconnect via the receive loop survives).
        self.send_timeout = send_timeout
        self.session: aiohttp.ClientSession | None = None
        self.ws: aiohttp.ClientWebSocketResponse | None = None
        self._reconnect_count = 0
        self._heartbeat_task: asyncio.Task | None = None

    async def connect(self):
        # Close any stale session so reconnects don't leak connections.
        if self.session and not self.session.closed:
            await self.session.close()
        self.session = aiohttp.ClientSession()
        self.ws = await self.session.ws_connect(
            f"{self.url}/ws?toy_id={self.toy_id}",
        )
        self._reconnect_count = 0
        # Replace any prior heartbeat loop so reconnects don't leak one each.
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self):
        while self.ws and not self.ws.closed:
            await self.send_json({
                "type": "ping",
                "timestamp": asyncio.get_running_loop().time(),
            })
            await asyncio.sleep(5)

    async def reconnect(self) -> bool:
        """Re-establish the connection with capped backoff.

        Returns True once reconnected, or False after max_reconnect attempts.
        """
        while self._reconnect_count < self.max_reconnect:
            self._reconnect_count += 1
            await asyncio.sleep(min(self.reconnect_interval * self._reconnect_count, 60))
            try:
                await self.connect()  # resets _reconnect_count to 0 on success
                return True
            except Exception:
                continue
        return False

    async def send_audio(self, pcm_bytes: bytes):
        await self._send_bytes(b"\x01" + pcm_bytes)

    async def send_frame(self, jpeg_bytes: bytes):
        await self._send_bytes(b"\x02" + jpeg_bytes)

    async def _send_bytes(self, data: bytes):
        # Drop the chunk if the socket is down; the receive loop drives reconnect.
        # wait_for bounds a hang on a half-dead socket so the caller (the mic
        # loop) can never be blocked here indefinitely — drop the chunk instead.
        if self.ws and not self.ws.closed:
            try:
                await asyncio.wait_for(self.ws.send_bytes(data), self.send_timeout)
            except Exception:
                pass

    async def send_json(self, data: dict):
        if self.ws and not self.ws.closed:
            try:
                await asyncio.wait_for(self.ws.send_str(json.dumps(data)), self.send_timeout)
            except Exception:
                pass

    async def receive(self):
        if self.ws:
            msg = await self.ws.receive()
            if msg.type == aiohttp.WSMsgType.TEXT:
                return json.loads(msg.data)
            elif msg.type == aiohttp.WSMsgType.BINARY:
                return {"type": "binary", "frame_type": msg.data[0], "payload": msg.data[1:]}
            elif msg.type == aiohttp.WSMsgType.CLOSED:
                return {"type": "closed"}
        return {"type": "none"}

    async def close(self):
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
        if self.ws:
            await self.ws.close()
        if self.session:
            await self.session.close()
