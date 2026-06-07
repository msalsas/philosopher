"""End-to-end test over a real in-process WebSocket.

Drives the whole production path: a WebSocket client (FastAPI TestClient) ->
`ws_server.WebSocketManager` -> `Orchestrator` -> LangGraph nodes + memory ->
streamed text/servo/WAV back to the client. The LLM is mocked and STT/vision/TTS
run in mock mode, so no models or hardware are needed.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi.testclient import TestClient

from philosopher.api.app import create_app
from philosopher.config.settings import get_settings
from philosopher.core.orchestrator import Orchestrator
from philosopher.stt.engine import StreamingSTT
from philosopher.tts.engine import PiperTTS


class _StubVision:
    """Vision stub that always reports one happy face (no dlib needed)."""

    async def process_frame(self, jpeg_bytes: bytes) -> dict:
        face = {"face_id": "abc123", "name": "Juan", "emotion": "happy",
                "offset_x": 0.5, "confidence": 0.9}
        return {"faces": [face], "primary_face": face}


def _build_client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("PHILOSOPHER_MEMORY_DB_PATH", str(tmp_path / "e2e.db"))
    get_settings.cache_clear()

    orch = Orchestrator()
    app = create_app(orch)  # registers ws_manager on orch

    @asynccontextmanager
    async def lifespan(_app):
        # init runs on the app's event loop so the aiosqlite connection binds to it
        await orch.init()
        orch.stt = StreamingSTT(mock=True)        # transcribe() -> "Hello world"
        orch.vision = _StubVision()
        orch.tts = PiperTTS(mock=True)            # synthesize() -> RIFF/WAV bytes

        async def fake_stream(messages, system_prompt=None, max_tokens=250,
                              temperature=0.7):
            yield "Hola Juan."

        orch.llm.chat_stream = fake_stream
        yield

    app.router.lifespan_context = lifespan
    return TestClient(app)


def _drain(ws, n):
    out = []
    for _ in range(n):
        m = ws.receive()
        if m.get("text") is not None:
            out.append(("json", json.loads(m["text"])))
        elif m.get("bytes") is not None:
            out.append(("bytes", m["bytes"]))
    return out


def test_e2e_full_conversation(tmp_path, monkeypatch):
    client = _build_client(tmp_path, monkeypatch)
    try:
        with client, client.websocket_connect("/ws?toy_id=t1") as ws:
            # 1) A camera frame arrives -> server detects a face -> "look" command
            #    that tracks the face (sets the head base position).
            ws.send_bytes(b"\x02" + b"jpegdata")
            look = json.loads(ws.receive()["text"])
            assert look["type"] == "look"
            assert look["offset_x"] == 0.5
            assert look["emotion"] == "happy"
            # Emotion went none -> happy, so a brief "react" gesture follows.
            react = json.loads(ws.receive()["text"])
            assert react["type"] == "react"
            assert react["emotion"] == "happy"

            # 2) The user speaks: speech_started -> audio -> speech_ended.
            ws.send_json({"type": "speech_started"})
            ws.send_bytes(b"\x01" + b"audiodata")
            ws.send_json({"type": "speech_ended"})

            # One sentence => text + servo (JSON) + WAV (binary 0x03).
            got = _drain(ws, 3)
            kinds = [k for k, _ in got]
            assert kinds.count("json") == 2
            assert kinds.count("bytes") == 1

            texts = [v for k, v in got if k == "json" and v.get("type") == "text"]
            assert texts and texts[0]["content"]

            wav = next(v for k, v in got if k == "bytes")
            assert wav[0] == 0x03           # WAV frame type
            assert wav[1:5] == b"RIFF"      # real WAV payload from (mock) Piper
    finally:
        get_settings.cache_clear()


def test_e2e_ping_pong(tmp_path, monkeypatch):
    client = _build_client(tmp_path, monkeypatch)
    try:
        with client, client.websocket_connect("/ws?toy_id=t2") as ws:
            ws.send_json({"type": "ping", "timestamp": 123.0})
            pong = ws.receive_json()
            assert pong["type"] == "pong"
            assert pong["timestamp"] == 123.0
    finally:
        get_settings.cache_clear()


def test_e2e_dashboard_page_and_sessions(tmp_path, monkeypatch):
    client = _build_client(tmp_path, monkeypatch)
    try:
        with client:
            page = client.get("/dashboard")
            assert page.status_code == 200
            assert "Philosopher Dashboard" in page.text
            assert client.get("/sessions").status_code == 200
    finally:
        get_settings.cache_clear()
