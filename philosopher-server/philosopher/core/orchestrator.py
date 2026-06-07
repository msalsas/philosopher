"""Main orchestrator that coordinates all server modules."""
from __future__ import annotations

import signal
from typing import Any

from philosopher.config.settings import Settings, get_settings
from philosopher.core.graph import AgentGraph, NodeCtx
from philosopher.core.nodes import (
    build_messages,
    node_format,
    node_memory,
    node_perception,
    node_prompt,
    node_store,
)
from philosopher.core.state import AgentState
from philosopher.events.bus import Event, EventBus, EventType
from philosopher.events.dashboard import DashboardHub
from philosopher.events.session_tracker import SessionTracker
from philosopher.llm.client import LLMClient
from philosopher.memory.manager import MemoryManager
from philosopher.personality.engine import PersonalityEngine


class Orchestrator:
    """Coordinates LLM, memory, personality, and the LangGraph pipeline."""

    def __init__(self) -> None:
        self.settings: Settings | None = None
        self.llm: LLMClient | None = None
        self.memory: MemoryManager | None = None
        self.personality: PersonalityEngine | None = None
        self.graph: AgentGraph | None = None
        self.stt = None
        self.vision = None
        self.tts = None
        self.ws_manager = None
        self.events: EventBus | None = None
        self.tracker: SessionTracker | None = None
        self.dashboard: DashboardHub | None = None
        # Last detected face/emotion per toy, fed into the next streamed reply.
        self.last_vision: dict[str, dict] = {}
        self._running = False

    async def init(self) -> None:
        self.settings = get_settings()
        self.llm = LLMClient(self.settings)
        self.personality = PersonalityEngine(self.settings.personality)
        self.memory = MemoryManager(self.settings.memory, self.llm)
        await self.memory.init()
        # Event bus + first consumer (per-person session stats). Reset the
        # singleton so each Orchestrator gets an isolated bus.
        EventBus._instance = None
        self.events = EventBus()
        await self.events.start()
        self.tracker = SessionTracker()
        self.tracker.register(self.events)
        self.dashboard = DashboardHub()
        self.dashboard.register(self.events)
        ctx = NodeCtx(self.llm, self.memory, self.personality, self.events)
        self.graph = AgentGraph(ctx)
        self._running = True

    async def _emit(self, etype: EventType, **data) -> None:
        if self.events:
            await self.events.emit(Event(etype, data, source="orchestrator"))

    async def process(
        self, message: str, face_id: str | None = None,
        face_name: str | None = None, emotion: str | None = None,
    ) -> AgentState:
        if not self.graph:
            raise RuntimeError("Orchestrator not initialized")
        state = AgentState(user_message=message, face_id=face_id,
                           face_name=face_name, emotion=emotion)
        return await self.graph.process(state)

    async def handle_speech_started(self, toy_id: str):
        """Toy detected speech start. Clear audio buffer."""
        if self.stt:
            self.stt.clear_buffer(toy_id)

    async def handle_audio_chunk(self, toy_id: str, pcm_bytes: bytes):
        """Receive audio chunk from toy via WebSocket."""
        if self.stt:
            self.stt.add_audio_chunk(toy_id, pcm_bytes)

    async def handle_video_frame(self, toy_id: str, jpeg_bytes: bytes):
        """Receive video frame from toy via WebSocket."""
        if not self.vision:
            return None
        result = await self.vision.process_frame(jpeg_bytes)
        # Remember the dominant face so the next streamed reply knows who it is
        # talking to and their emotion.
        if result and result.get("primary_face"):
            face = result["primary_face"]
            self.last_vision[toy_id] = face
            await self._emit(EventType.FACE_RECOGNIZED,
                             face_id=face.get("face_id"), name=face.get("name"))
            await self._emit(EventType.EMOTION_DETECTED,
                             face_id=face.get("face_id"), emotion=face.get("emotion"))
        return result

    async def process_speech(self, toy_id: str):
        """Called when toy signals speech ended."""
        if not self.stt or not self.ws_manager:
            return

        result = await self.stt.transcribe(
            toy_id, confidence_threshold=self.settings.stt.confidence_threshold,
        )

        if result.get("low_confidence"):
            await self.ws_manager.send_json(toy_id, {
                "type": "text",
                "content": self.personality.fallback("not_understood"),
                "emotion": "neutral",
            })
            return

        text = result.get("text", "")
        if not text:
            return

        face = self.last_vision.get(toy_id) or {}
        await self._emit(EventType.USER_TEXT, face_id=face.get("face_id"), text=text)
        await self._stream_response(toy_id, text)

    async def _send_sentence(self, toy_id: str, state: AgentState, sentence: str) -> None:
        """Format one sentence and stream it to the toy (text + servo + WAV)."""
        state.llm_response = sentence
        await node_format(state, self.graph.ctx)
        formatted = state.formatted
        if not formatted:
            return
        await self.ws_manager.send_json(toy_id, {
            "type": "text", "content": formatted, "emotion": state.emotion or "neutral",
        })
        await self.ws_manager.send_json(toy_id, {
            "type": "servo", "emotion": state.emotion or "neutral",
        })
        if self.tts and self.settings.tts.enabled:
            audio_bytes = await self.tts.synthesize(formatted)
            if audio_bytes:
                await self.ws_manager.send_binary(toy_id, 0x03, audio_bytes)

    async def _send_fallback(self, toy_id: str) -> None:
        """Send a single localized fallback (text + WAV) when generation fails."""
        text = self.personality.fallback("error")
        await self.ws_manager.send_json(toy_id, {
            "type": "text", "content": text, "emotion": "neutral",
        })
        if self.tts and self.settings.tts.enabled:
            audio_bytes = await self.tts.synthesize(text)
            if audio_bytes:
                await self.ws_manager.send_binary(toy_id, 0x03, audio_bytes)

    async def _stream_response(self, toy_id: str, user_message: str):
        """Split-graph streaming: nodes 1-3, then external LLM stream, then store."""
        if not self.graph or not self.ws_manager:
            return

        # Inject the most recent face/emotion seen on this toy so perception can
        # resolve a name and the prompt reflects the user's emotion.
        face = self.last_vision.get(toy_id) or {}
        state = AgentState(
            user_message=user_message,
            face_id=face.get("face_id"),
            face_name=face.get("name"),
            emotion=face.get("emotion"),
        )
        state = await node_perception(state, self.graph.ctx)
        state = await node_memory(state, self.graph.ctx)
        state = await node_prompt(state, self.graph.ctx)

        # Shared message assembly (incl. new-face introduction / name-ask).
        messages = build_messages(state)

        buffer = ""
        full_response = ""
        try:
            async for token in self.llm.chat_stream(
                messages,
                system_prompt=state.system_prompt,
                max_tokens=250,
                temperature=self.settings.llm.temperature,
            ):
                buffer += token
                full_response += token
                if any(buffer.endswith(t) for t in [".", "!", "?", "\n"]):
                    sentence = buffer.strip()
                    if sentence:
                        await self._send_sentence(toy_id, state, sentence)
                    buffer = ""
            if buffer.strip():
                await self._send_sentence(toy_id, state, buffer.strip())
        except Exception:
            pass  # stream failed mid-way; fall through to the empty/fallback check

        if not full_response.strip():
            # Total failure (or empty generation): apologize once, don't store.
            await self._send_fallback(toy_id)
            return

        # Store the full interaction once at the end.
        state.llm_response = full_response
        await node_format(state, self.graph.ctx)
        await node_store(state, self.graph.ctx)
        await self._emit(EventType.RESPONSE_READY,
                         face_id=state.face_id, text=full_response)

    async def run(self) -> None:
        if not self.settings or not self.settings.api.enabled:
            return
        import uvicorn

        from philosopher.api.app import create_app
        app = create_app(self)
        config = uvicorn.Config(
            app, host=self.settings.api.host,
            port=self.settings.api.port, log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def stop(self) -> None:
        self._running = False
        if self.events:
            await self.events.stop()
        if self.memory:
            await self.memory.close()

    async def health(self) -> dict[str, Any]:
        h = {"status": "ok", "running": self._running}
        if self.llm:
            h["llm"] = await self.llm.health_check()
        if self.memory:
            h["memory"] = await self.memory.long.stats()
        return h


async def main():
    orch = Orchestrator()
    await orch.init()
    loop = __import__("asyncio").get_event_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(s, lambda: loop.create_task(orch.stop()))
    await orch.run()
