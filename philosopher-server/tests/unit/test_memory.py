"""Tests for memory subsystem."""
from __future__ import annotations

import pytest

from philosopher.memory.short_term import ShortTermMemory


class TestShortTermMemory:
    @pytest.mark.asyncio
    async def test_add(self):
        stm = ShortTermMemory(5)
        await stm.add("user", "hello")
        assert stm.count == 1

    @pytest.mark.asyncio
    async def test_limit(self):
        stm = ShortTermMemory(3)
        for i in range(5):
            await stm.add("user", f"msg {i}")
        assert stm.count == 3

    @pytest.mark.asyncio
    async def test_context(self):
        stm = ShortTermMemory()
        await stm.add("user", "hi")
        await stm.add("assistant", "hello")
        ctx = await stm.get_context()
        assert len(ctx) == 2
        assert ctx[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_clear(self):
        stm = ShortTermMemory()
        await stm.add("user", "hi")
        await stm.clear()
        assert stm.is_empty

    @pytest.mark.asyncio
    async def test_per_face_isolation(self):
        # Two people must not see each other's recent context.
        stm = ShortTermMemory()
        await stm.add("user", "soy Ana", key="ana")
        await stm.add("user", "soy Bob", key="bob")
        ana = await stm.get_context(key="ana")
        bob = await stm.get_context(key="bob")
        assert [m["content"] for m in ana] == ["soy Ana"]
        assert [m["content"] for m in bob] == ["soy Bob"]
        assert stm.count == 2


class TestLongTermMemory:
    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, tmp_dir):
        from philosopher.memory.long_term import LongTermMemory
        ltm = LongTermMemory(str(tmp_dir / "test.db"))
        await ltm._get_db()
        mid = await ltm.store("hello world", "test", emotion="happy",
                              face_id="f1", face_name="Alice")
        assert mid is not None
        results = await ltm.retrieve("hello", top_k=5)
        assert len(results) > 0
        face = await ltm.get_face("f1")
        assert face["name"] == "Alice"
        assert face["encounters"] == 1
        stats = await ltm.stats()
        assert stats["total_memories"] == 1
        await ltm.close()

    @pytest.mark.asyncio
    async def test_face_encounters(self, tmp_dir):
        from philosopher.memory.long_term import LongTermMemory
        ltm = LongTermMemory(str(tmp_dir / "test.db"))
        await ltm._get_db()
        for i in range(3):
            await ltm.store(f"msg {i}", face_id="repeat", face_name="Bob")
        face = await ltm.get_face("repeat")
        assert face["encounters"] == 3
        await ltm.close()

    @pytest.mark.asyncio
    async def test_face_encoding_roundtrip(self, tmp_dir):
        import numpy as np

        from philosopher.memory.long_term import LongTermMemory
        ltm = LongTermMemory(str(tmp_dir / "test.db"))
        await ltm._get_db()
        enc = np.linspace(0, 1, 128, dtype=np.float64)
        await ltm.add_face("face1", enc.tobytes(), name="Ana")
        await ltm.add_face("face1", enc.tobytes(), name="Ana")  # idempotent
        faces = await ltm.all_faces()
        assert len(faces) == 1
        assert faces[0]["face_id"] == "face1"
        back = np.frombuffer(faces[0]["encoding"], dtype=np.float64)
        assert np.allclose(back, enc)
        await ltm.close()
