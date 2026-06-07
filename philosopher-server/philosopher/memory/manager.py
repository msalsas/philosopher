"""Coordinates short-term and long-term memory systems."""
from __future__ import annotations

from philosopher.memory.long_term import LongTermMemory
from philosopher.memory.short_term import ShortTermMemory


class MemoryManager:
    def __init__(self, settings, llm_client) -> None:
        self.short = ShortTermMemory(settings.short_term_limit)
        self.long = LongTermMemory(settings.db_path, settings.similarity_threshold)
        self.llm = llm_client

    async def init(self) -> None:
        await self.long._get_db()

    async def add(
        self, user_msg: str, assistant_msg: str,
        emotion: str | None = None, face_id: str | None = None, face_name: str | None = None,
    ) -> None:
        await self.short.add("user", user_msg, {"emotion": emotion, "face": face_id}, key=face_id)
        await self.short.add("assistant", assistant_msg, key=face_id)
        summary = f"U: {user_msg[:100]} -> A: {assistant_msg[:100]}"
        await self.long.store(
            f"U: {user_msg}\nA: {assistant_msg}", summary,
            emotion=emotion, face_id=face_id, face_name=face_name,
        )

    async def get_context(self, message: str, face_id: str | None = None) -> dict:
        return {
            "short": await self.short.get_context(key=face_id),
            "long": await self.long.retrieve(message, face_id=face_id),
            "face": await self.long.get_face(face_id) if face_id else None,
        }

    async def close(self) -> None:
        await self.long.close()
