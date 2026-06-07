"""WebSocket endpoint for toy communication."""
from __future__ import annotations

import json
import time

from fastapi import WebSocket, WebSocketDisconnect


class WebSocketManager:
    """Manages WebSocket connections from toys."""

    # Seconds without a face before the head slowly recenters.
    RECENTER_AFTER = 4.0

    def __init__(self, orchestrator):
        self.orch = orchestrator
        self.connections: dict[str, WebSocket] = {}
        self._last_face_at: dict[str, float] = {}   # toy_id -> last time a face was seen
        self._last_emotion: dict[str, str] = {}      # toy_id -> last emotion (for change detection)

    async def handle_client(self, websocket: WebSocket, toy_id: str):
        await websocket.accept()
        self.connections[toy_id] = websocket
        try:
            while True:
                message = await websocket.receive()
                # Starlette's raw receive() returns a disconnect message rather
                # than raising; ignoring it and calling receive() again raises
                # RuntimeError, so handle it explicitly.
                if message["type"] == "websocket.disconnect":
                    break
                await self._handle_message(toy_id, message)
        except WebSocketDisconnect:
            pass
        finally:
            self.connections.pop(toy_id, None)

    async def _handle_message(self, toy_id: str, message: dict):
        if message.get("type") == "websocket.receive":
            if "bytes" in message:
                await self._handle_binary(toy_id, message["bytes"])
            elif "text" in message:
                await self._handle_json(toy_id, message["text"])

    async def _handle_binary(self, toy_id: str, data: bytes):
        if not data:
            return
        frame_type = data[0]
        payload = data[1:]
        if frame_type == 0x01:
            await self.orch.handle_audio_chunk(toy_id, payload)
        elif frame_type == 0x02:
            result = await self.orch.handle_video_frame(toy_id, payload)
            face = result.get("primary_face") if result else None
            now = time.monotonic()
            if face:
                self._last_face_at[toy_id] = now
                # Per-frame: track the face by setting the head's base position.
                await self.send_json(toy_id, {
                    "type": "look",
                    "offset_x": face.get("offset_x", 0.0),
                    "emotion": face.get("emotion", "neutral"),
                })
                # React only when the emotion *changes* (not every frame, not neutral).
                emotion = face.get("emotion", "neutral")
                if emotion != "neutral" and emotion != self._last_emotion.get(toy_id):
                    await self.send_json(toy_id, {"type": "react", "emotion": emotion})
                self._last_emotion[toy_id] = emotion
            else:
                # No face: once the grace period passes, recenter the head (once).
                last = self._last_face_at.get(toy_id)
                if last is not None and now - last > self.RECENTER_AFTER:
                    await self.send_json(toy_id, {"type": "look", "offset_x": 0.0})
                    self._last_face_at[toy_id] = None
                    self._last_emotion.pop(toy_id, None)

    async def _handle_json(self, toy_id: str, text: str):
        try:
            msg = json.loads(text)
        except json.JSONDecodeError:
            return
        if msg.get("type") == "speech_started":
            await self.orch.handle_speech_started(toy_id)
        elif msg.get("type") == "speech_ended":
            await self.orch.process_speech(toy_id)
        elif msg.get("type") == "ping":
            await self.send_json(toy_id, {"type": "pong", "timestamp": msg.get("timestamp")})

    async def send_json(self, toy_id: str, data: dict):
        ws = self.connections.get(toy_id)
        if ws:
            await ws.send_json(data)

    async def send_binary(self, toy_id: str, frame_type: int, payload: bytes):
        ws = self.connections.get(toy_id)
        if ws:
            await ws.send_bytes(bytes([frame_type]) + payload)
