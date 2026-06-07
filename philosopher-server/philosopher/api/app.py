"""FastAPI app for the Philosopher HTTP API."""
from __future__ import annotations

import json

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from philosopher.api.ws_server import WebSocketManager
from philosopher.events.dashboard import DASHBOARD_HTML


class ChatReq(BaseModel):
    message: str
    face_id: str | None = None
    face_name: str | None = None
    emotion: str | None = None


class ChatResp(BaseModel):
    response: str
    emotion: str | None = None
    turn: int = 0


def create_app(orch) -> FastAPI:
    app = FastAPI(title="Philosopher API", version="0.2.0")

    ws_manager = WebSocketManager(orch)
    orch.ws_manager = ws_manager

    @app.get("/")
    def root():
        return {"name": "Philosopher", "version": "0.2.0"}

    @app.get("/health")
    async def health():
        return await orch.health()

    @app.post("/chat")
    async def chat(req: ChatReq):
        result = await orch.process(req.message, req.face_id, req.face_name, req.emotion)
        return {"response": result.formatted, "emotion": result.emotion, "turn": result.turn}

    @app.get("/personalities")
    def personalities():
        return orch.personality.list_all()

    @app.post("/personality")
    def set_personality(name: str):
        orch.personality.load(name)
        return {"personality": name}

    @app.get("/memory/stats")
    async def mem_stats():
        return await orch.memory.long.stats()

    @app.get("/sessions")
    def sessions():
        return orch.tracker.snapshot() if orch.tracker else {}

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard_page():
        return DASHBOARD_HTML

    @app.get("/dashboard/stream")
    async def dashboard_stream():
        hub = orch.dashboard
        if not hub:
            return StreamingResponse(iter(()), media_type="text/event-stream")
        queue = hub.connect()

        async def gen():
            try:
                for ev in hub.recent():
                    yield f"data: {json.dumps(ev)}\n\n"
                while True:
                    ev = await queue.get()
                    yield f"data: {json.dumps(ev)}\n\n"
            finally:
                hub.disconnect(queue)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket, toy_id: str = "default"):
        await ws_manager.handle_client(websocket, toy_id)

    return app
