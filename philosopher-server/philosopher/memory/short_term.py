"""Short-term memory: circular conversation buffer."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class MessageEntry:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


class ShortTermMemory:
    """Recent conversation buffer, kept **per person** (face_id).

    Multiple people talk to the same toy, so each face gets its own circular
    buffer — otherwise one person's recent turns would leak into another's
    context. Unknown faces share an anonymous bucket.
    """

    _ANON = "_anon"

    def __init__(self, max_messages: int = 10) -> None:
        self.max_messages = max_messages
        self._buffers: dict[str, deque[MessageEntry]] = {}

    def _buf(self, key: str | None) -> deque[MessageEntry]:
        k = key or self._ANON
        buf = self._buffers.get(k)
        if buf is None:
            buf = deque(maxlen=self.max_messages)
            self._buffers[k] = buf
        return buf

    async def add(self, role: str, content: str, metadata: dict | None = None,
                  key: str | None = None) -> None:
        self._buf(key).append(MessageEntry(role=role, content=content, metadata=metadata or {}))

    async def get_context(self, key: str | None = None, limit: int | None = None) -> list[dict]:
        msgs = list(self._buf(key))
        if limit:
            msgs = msgs[-limit:]
        return [
            {"role": m.role, "content": m.content}
            for m in msgs if m.role in ("user", "assistant")
        ]

    async def clear(self, key: str | None = None) -> None:
        if key is None:
            self._buffers.clear()
        else:
            self._buffers.pop(key or self._ANON, None)

    @property
    def count(self) -> int:
        return sum(len(b) for b in self._buffers.values())

    @property
    def is_empty(self) -> bool:
        return self.count == 0
