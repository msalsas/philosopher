"""Async event bus for decoupled module communication."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(Enum):
    USER_TEXT = "user_text"
    USER_VOICE = "user_voice"
    FACE_DETECTED = "face_detected"
    FACE_RECOGNIZED = "face_recognized"
    EMOTION_DETECTED = "emotion_detected"
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"
    TTS_STARTED = "tts_started"
    TTS_ENDED = "tts_ended"
    BUTTON_PRESSED = "button_pressed"
    TOUCH_DETECTED = "touch_detected"
    RESPONSE_READY = "response_ready"
    THINKING_STARTED = "thinking_started"
    THINKING_ENDED = "thinking_ended"
    WAKE_UP = "wake_up"
    SLEEP = "sleep"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class Event:
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


Handler = Callable[[Event], Awaitable[None]]


class EventBus:
    """Singleton async event bus with pub/sub pattern."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._subs: dict[EventType, list[Handler]] = {t: [] for t in EventType}
        self._globals: list[Handler] = []
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._initialized = True
        self._running = False

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._dispatch())

    async def stop(self):
        self._running = False
        if hasattr(self, "_task"):
            self._task.cancel()

    async def _dispatch(self):
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                for h in self._subs.get(event.type, []):
                    asyncio.create_task(h(event))
                for h in self._globals:
                    asyncio.create_task(h(event))
            except asyncio.TimeoutError:
                continue

    def subscribe(self, event_type: EventType, handler: Handler):
        if handler not in self._subs[event_type]:
            self._subs[event_type].append(handler)

    def subscribe_all(self, handler: Handler):
        if handler not in self._globals:
            self._globals.append(handler)

    async def emit(self, event: Event):
        await self._queue.put(event)
