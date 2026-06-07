"""Tests for WebSocket server."""
from __future__ import annotations

import pytest
from fastapi import WebSocketDisconnect

from philosopher.api.ws_server import WebSocketManager


class MockOrch:
    def __init__(self):
        self.audio_chunks = []
        self.video_frames = []
        self.speech_events = []
        self.ws_manager = None

    async def handle_audio_chunk(self, toy_id, pcm):
        self.audio_chunks.append((toy_id, pcm))

    async def handle_video_frame(self, toy_id, jpeg):
        self.video_frames.append((toy_id, jpeg))
        return {"primary_face": {"emotion": "happy", "offset_x": 0.5}}

    async def handle_speech_started(self, toy_id):
        self.speech_events.append(("started", toy_id))

    async def process_speech(self, toy_id):
        self.speech_events.append(("ended", toy_id))


class MockWebSocket:
    def __init__(self, max_receives=1):
        self.sent = []
        self.closed = False
        self._receive_count = 0
        self._max_receives = max_receives

    async def accept(self):
        pass

    async def receive(self):
        self._receive_count += 1
        if self._receive_count > self._max_receives:
            raise WebSocketDisconnect()
        return {"type": "websocket.receive", "text": '{"type": "ping"}'}

    async def send_json(self, data):
        self.sent.append(("json", data))

    async def send_bytes(self, data):
        self.sent.append(("bytes", data))

    async def close(self):
        self.closed = True


class TestWebSocketManager:
    @pytest.mark.asyncio
    async def test_handle_json_ping(self):
        orch = MockOrch()
        ws_mgr = WebSocketManager(orch)
        ws = MockWebSocket(max_receives=1)
        await ws_mgr.handle_client(ws, "t1")
        assert any(s[1].get("type") == "pong" for s in ws.sent)

    @pytest.mark.asyncio
    async def test_handle_binary_audio(self):
        orch = MockOrch()
        ws_mgr = WebSocketManager(orch)
        await ws_mgr._handle_binary("t1", b"\x01audio_data")
        assert orch.audio_chunks == [("t1", b"audio_data")]

    @pytest.mark.asyncio
    async def test_handle_binary_video(self):
        orch = MockOrch()
        ws_mgr = WebSocketManager(orch)
        ws = MockWebSocket(max_receives=1)
        ws_mgr.connections["t1"] = ws
        await ws_mgr._handle_binary("t1", b"\x02jpeg_data")
        assert orch.video_frames == [("t1", b"jpeg_data")]
        looks = [s[1] for s in ws.sent if s[1].get("type") == "look"]
        assert looks and looks[0]["offset_x"] == 0.5

    @pytest.mark.asyncio
    async def test_send_json(self):
        orch = MockOrch()
        ws_mgr = WebSocketManager(orch)
        ws = MockWebSocket()
        ws_mgr.connections["t1"] = ws
        await ws_mgr.send_json("t1", {"type": "text", "content": "hello"})
        assert any(s[1]["type"] == "text" for s in ws.sent)

    @pytest.mark.asyncio
    async def test_disconnect_message_cleans_up(self):
        # Starlette's raw receive() RETURNS a disconnect message (it does not
        # raise WebSocketDisconnect). The manager must break and clean up.
        class DisconnectWS:
            def __init__(self):
                self.sent = []
                self._msgs = [
                    {"type": "websocket.receive", "text": '{"type": "ping"}'},
                    {"type": "websocket.disconnect", "code": 1000},
                ]

            async def accept(self):
                pass

            async def receive(self):
                return self._msgs.pop(0)

            async def send_json(self, data):
                self.sent.append(data)

        orch = MockOrch()
        mgr = WebSocketManager(orch)
        ws = DisconnectWS()
        await mgr.handle_client(ws, "t1")
        assert "t1" not in mgr.connections                       # connection cleaned up
        assert any(s.get("type") == "pong" for s in ws.sent)     # ping handled before disconnect

    @pytest.mark.asyncio
    async def test_video_reacts_only_on_emotion_change(self):
        orch = MockOrch()  # always returns the same happy face
        mgr = WebSocketManager(orch)
        ws = MockWebSocket()
        mgr.connections["t1"] = ws
        await mgr._handle_binary("t1", b"\x02f1")
        await mgr._handle_binary("t1", b"\x02f2")
        reacts = [s[1] for s in ws.sent if s[1].get("type") == "react"]
        assert len(reacts) == 1            # happy never changes -> reacts once
        assert reacts[0]["emotion"] == "happy"

    @pytest.mark.asyncio
    async def test_video_recenters_after_face_lost(self):
        import time

        class NoFaceOrch(MockOrch):
            async def handle_video_frame(self, toy_id, jpeg):
                return {"primary_face": None}

        mgr = WebSocketManager(NoFaceOrch())
        ws = MockWebSocket()
        mgr.connections["t1"] = ws
        mgr._last_face_at["t1"] = time.monotonic() - 10  # face seen long ago
        await mgr._handle_binary("t1", b"\x02frame")
        looks = [s[1] for s in ws.sent if s[1].get("type") == "look"]
        assert looks and looks[-1]["offset_x"] == 0.0   # head recentered
