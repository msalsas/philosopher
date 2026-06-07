"""Long-term memory: persistent SQLite storage with keyword search."""
from __future__ import annotations

import contextlib
import hashlib
import time
from typing import Any

import aiosqlite


class LongTermMemory:
    """SQLite-based persistent memory with keyword similarity retrieval."""

    def __init__(self, db_path: str, threshold: float = 0.6) -> None:
        self.db_path = db_path
        self.threshold = threshold
        self._db: aiosqlite.Connection | None = None

    async def _get_db(self) -> aiosqlite.Connection:
        if self._db is None:
            self._db = await aiosqlite.connect(self.db_path)
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._create_tables()
        return self._db

    async def _create_tables(self) -> None:
        db = await self._get_db()
        await db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY, timestamp REAL, type TEXT, content TEXT,
                summary TEXT, emotion TEXT, face_id TEXT, face_name TEXT,
                metadata TEXT, keywords TEXT
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_ts ON memories(timestamp)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_face ON memories(face_id)")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS known_faces (
                face_id TEXT PRIMARY KEY, name TEXT, first_seen REAL,
                last_seen REAL, encounters INTEGER DEFAULT 1, encoding BLOB
            )
        """)
        # Migrate older DBs created before the encoding column existed.
        with contextlib.suppress(Exception):
            await db.execute("ALTER TABLE known_faces ADD COLUMN encoding BLOB")
        await db.commit()

    async def store(
        self, content: str, summary: str = "", mtype: str = "conversation",
        emotion: str | None = None, face_id: str | None = None, face_name: str | None = None,
    ) -> str:
        db = await self._get_db()
        mid = hashlib.sha256(f"{content}{time.time()}".encode()).hexdigest()[:16]
        kw = " ".join([w for w in content.lower().split() if len(w) > 3][:20])
        await db.execute(
            "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?)",
            (mid, time.time(), mtype, content, summary, emotion, face_id, face_name, "{}", kw),
        )
        if face_id:
            await self._update_face(face_id, face_name)
        await db.commit()
        return mid

    async def _update_face(self, fid: str, name: str | None) -> None:
        db = await self._get_db()
        cur = await db.execute("SELECT encounters FROM known_faces WHERE face_id=?", (fid,))
        row = await cur.fetchone()
        if row:
            await db.execute(
                "UPDATE known_faces SET last_seen=?, encounters=encounters+1 WHERE face_id=?",
                (time.time(), fid),
            )
        else:
            t = time.time()
            await db.execute(
                "INSERT INTO known_faces (face_id, name, first_seen, last_seen, encounters) "
                "VALUES (?,?,?,?,?)",
                (fid, name, t, t, 1),
            )
        await db.commit()

    async def add_face(self, face_id: str, encoding: bytes, name: str | None = None) -> None:
        """Register a face with its recognition encoding (no-op if it exists)."""
        db = await self._get_db()
        t = time.time()
        await db.execute(
            "INSERT OR IGNORE INTO known_faces "
            "(face_id, name, first_seen, last_seen, encounters, encoding) "
            "VALUES (?,?,?,?,?,?)",
            (face_id, name, t, t, 1, encoding),
        )
        await db.commit()

    async def retrieve(self, query: str, top_k: int = 5, face_id: str | None = None) -> list[dict]:
        db = await self._get_db()
        qwords = set(query.lower().split())
        if face_id:
            cur = await db.execute(
                "SELECT * FROM memories WHERE face_id=? ORDER BY timestamp DESC LIMIT 100",
                (face_id,),
            )
        else:
            cur = await db.execute("SELECT * FROM memories ORDER BY timestamp DESC LIMIT 100")
        rows = await cur.fetchall()
        results = []
        for row in rows:
            rkw = set((row[9] or "").split())
            score = len(qwords & rkw) / max(len(qwords), 1) if qwords else 0
            age = time.time() - row[1]
            score += max(0, 1 - age / 604800) * 0.3
            if score >= self.threshold:
                results.append((score, {
                    "content": row[3], "summary": row[4],
                    "emotion": row[5], "face_name": row[7],
                }))
        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results[:top_k]]

    async def get_face(self, fid: str) -> dict[str, Any] | None:
        db = await self._get_db()
        cur = await db.execute("SELECT * FROM known_faces WHERE face_id=?", (fid,))
        row = await cur.fetchone()
        if row:
            return {"face_id": row[0], "name": row[1], "first_seen": row[2],
                    "last_seen": row[3], "encounters": row[4]}
        return None

    async def update_face_name(self, face_id: str, name: str) -> None:
        db = await self._get_db()
        await db.execute(
            "UPDATE known_faces SET name=? WHERE face_id=?",
            (name, face_id),
        )
        await db.commit()

    async def all_faces(self) -> list[dict]:
        db = await self._get_db()
        cur = await db.execute("SELECT * FROM known_faces ORDER BY last_seen DESC")
        return [
            {"face_id": r[0], "name": r[1], "encounters": r[4], "encoding": r[5]}
            for r in await cur.fetchall()
        ]

    async def stats(self) -> dict[str, Any]:
        db = await self._get_db()
        c = await db.execute("SELECT COUNT(*) FROM memories")
        total = (await c.fetchone())[0]
        c = await db.execute("SELECT COUNT(*) FROM known_faces")
        faces = (await c.fetchone())[0]
        c = await db.execute("SELECT type, COUNT(*) FROM memories GROUP BY type")
        by_type = {r[0]: r[1] for r in await c.fetchall()}
        return {"total_memories": total, "known_faces": faces, "by_type": by_type}

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
