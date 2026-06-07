"""Per-person session/presence tracker driven by the event bus.

Subscribes to face/emotion/text/response events and keeps a small per-face
summary (who's around, how often they interact, their emotion mix). This is the
first real consumer of the event bus and the seed for a future live dashboard
(another subscriber can push these over SSE/WebSocket).
"""
from __future__ import annotations

import time
from typing import Any

from philosopher.events.bus import Event, EventBus, EventType


class SessionTracker:
    """Aggregates interaction stats per recognized face."""

    _ANON = "_anon"

    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, Any]] = {}

    def register(self, bus: EventBus) -> None:
        bus.subscribe(EventType.FACE_RECOGNIZED, self._on_face)
        bus.subscribe(EventType.EMOTION_DETECTED, self._on_emotion)
        bus.subscribe(EventType.USER_TEXT, self._on_text)
        bus.subscribe(EventType.RESPONSE_READY, self._on_response)

    def _s(self, face_id: str | None) -> dict[str, Any]:
        fid = face_id or self._ANON
        s = self.sessions.get(fid)
        if s is None:
            now = time.time()
            s = {"name": None, "messages": 0, "responses": 0,
                 "emotions": {}, "first_seen": now, "last_seen": now}
            self.sessions[fid] = s
        return s

    async def _on_face(self, event: Event) -> None:
        s = self._s(event.get("face_id"))
        s["last_seen"] = time.time()
        if event.get("name"):
            s["name"] = event.get("name")

    async def _on_emotion(self, event: Event) -> None:
        s = self._s(event.get("face_id"))
        emo = event.get("emotion") or "neutral"
        s["emotions"][emo] = s["emotions"].get(emo, 0) + 1

    async def _on_text(self, event: Event) -> None:
        s = self._s(event.get("face_id"))
        s["messages"] += 1
        s["last_seen"] = time.time()

    async def _on_response(self, event: Event) -> None:
        self._s(event.get("face_id"))["responses"] += 1

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {fid: dict(s) for fid, s in self.sessions.items()}
