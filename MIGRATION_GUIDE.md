# MIGRATION GUIDE: Philosopher v2 — From HTTP to WebSocket Streaming

> Incremental refactor of the existing Philosopher project.
> Hardware: Raspberry Pi Zero WH (512MB) + Raspberry Pi 4 (4GB).
> Goal: WebSocket streaming, server-side AI, configurable quality.
> **This guide is authoritative. Where REWRITE_PROMPT.md conflicts, this guide wins.**

---

## 0. PRINCIPLE: Incremental Migration, Not Rewrite

**Do NOT start from scratch.** The existing project has:
- 28 passing server tests (config, memory, personality, state, events, LangGraph nodes)
- 3 passing toy tests
- Working SQLite memory with WAL mode
- Working personality YAML engine
- Working event bus with 17 event types
- Clean async architecture

**Strategy**: Keep what works. Add WebSocket alongside HTTP. Migrate piece by piece. Validate at every step.

**Critical decisions already made (do not change):**
1. **Incremental refactor** — Modify existing `philosopher-server/` and `philosopher-toy/`, no new folders
2. **Piper TTS on server** — Invoked via subprocess (not a Python package). Server sends full WAV per sentence to toy as binary frame `0x03`
3. **Split-graph streaming** — Run nodes 1-3 via LangGraph for context, then `llm.chat_stream()` externally, buffer sentences, format per sentence, run `node_store` once at end
4. **Batch STT** — Toy VAD sends `speech_started`/`speech_ended`. Server accumulates chunks and transcribes once at `speech_ended`. No server-side VAD
5. **Zero-latency face registration** — Background regex extraction after `node_store`. Extracts last word from 1-3 word replies. No LLM call on critical path
6. **Camera FPS** — Accept any positive float, round to nearest 0.5 step, clamp 0.5-5.0
7. **Servo commands on both** vision frames AND LLM responses
8. **Auto-language detection deferred to v3** — Static language per personality for now

---

## 1. WHAT STAYS COMPLETELY UNCHANGED

### philosopher-server

| Module | Status | Reason |
|--------|--------|--------|
| `config/settings.py` | Modify | Add new env vars only (STT, vision, TTS settings) |
| `config/personalities/*.yaml` | Modify | Add `face_registration` section |
| `core/state.py` | Modify | Add `from_dict()` and new fields |
| `core/nodes.py` | Modify | Add `background_extract_name()`, keep 6 nodes |
| `core/graph.py` | Keep | `wrap()` pattern works for sync LangGraph |
| `memory/*` | Modify | Add `update_face_name()` to long_term.py |
| `personality/engine.py` | Modify | Add `get_name_question()` |
| `events/bus.py` | Keep | 17 event types, async pub/sub — perfect |
| `utils/logger.py` | Keep | structlog + rich logging works |
| `api/app.py` | Modify | Keep HTTP endpoints, add WebSocket route |

### philosopher-toy

| Module | Status | Reason |
|--------|--------|--------|
| `hardware/servos.py` | Modify | Add sequential movement with 300ms delays, update poses |
| `audio/tts.py` | Remove | Replace with `audio/player.py` for streamed WAV |
| `audio/stt.py` | Remove | STT moves to server. Create `audio/capture.py` for mic + VAD |
| `vision/camera.py` | Simplify | Remove face detection. Keep capture + encode only. Configurable FPS with rounding |
| `protocol/client.py` | Replace | HTTP client → `protocol/ws_client.py` |
| `main.py` | Rewrite loop | WebSocket tasks instead of HTTP POST loop |

---

## 2. ARCHITECTURE AFTER MIGRATION

```
┌──────────────────────────────────────────────────────────────────────┐
│  BPi-M2 Zero (512MB) — Thin Client                                   │
│                                                                      │
│  [USB Mic] ──▶ [VAD] ──▶ [WebSocket] ───────────────────────────────┤
│  [OV5647 CSI] ──▶ [Capture] ──▶ [WebSocket] ────────────────────────┤
│  [Servos] ◀── [WebSocket] ◀── commands (sequential)                 │
│  [USB Speaker] ◀── [PyAudio Player] ◀── WAV audio (from server)     │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ WebSocket
┌──────────────────────────────────────────────────────────────────────┐
│  Raspberry Pi 4 (4GB) — Server (Brain)                               │
│                                                                      │
│  [Whisper STT] ◀── audio chunks                                     │
│  [Face Detection + DeepFace] ◀── JPEG frames                        │
│  [LangGraph Pipeline] ──▶ [Streaming LLM]                            │
│  [Memory + Personality + Events] — unchanged                          │
│  [Piper TTS] ──▶ WAV file ──▶ WebSocket ──▶ toy                     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. SERVER — ADD WEBSOCKET & AI MODULES

### 3.1 Add Dependencies

Edit `philosopher-server/pyproject.toml`:
```toml
dependencies = [
    # Keep all existing dependencies
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "websockets>=12.0",           # Already listed, now actually used
    "faster-whisper>=1.0.0",      # NEW: server-side STT
    "deepface>=0.0.89",           # NEW: server-side vision
    "face-recognition>=1.3.0",    # NEW: server-side face matching
    "opencv-python-headless>=4.9.0",  # NEW: server-side JPEG decode (headless: no libGL/display dep)
    "Pillow>=10.0.0",             # NEW: JPEG decode from toy
    # ... keep everything else (langgraph, openai, httpx, aiosqlite, etc.)
]
```

**Note:** Piper TTS is NOT a Python package. It is invoked via `subprocess.run(["piper", ...])`. Requires `piper-tts` binary installed on the system.

### 3.2 Add Settings

Edit `philosopher/config/settings.py`. Add to `Settings` class:
```python
class STTSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PHILOSOPHER_STT_")
    model: str = "tiny"              # "tiny" | "base" | "small"
    device: str = "cpu"
    confidence_threshold: float = -0.5

class VisionSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PHILOSOPHER_CAMERA_")
    fps: float = 0.5                 # Any positive float, rounded to 0.5 steps, clamped 0.5-5.0
    quality: int = 60                # JPEG quality: 30-90

class TTSSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PHILOSOPHER_TTS_")
    model_path: str = ""             # Path to Piper .onnx file. Empty = auto-download default
    voice: str = "es_ES-carlfm-x_low"
    enabled: bool = True
```

### 3.3 Add `AgentState.from_dict()`

Edit `philosopher/core/state.py`:
```python
@classmethod
def from_dict(cls, d: dict) -> AgentState:
    return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
```

Also add new fields:
```python
pending_audio: bool = False
servo_emotion: str | None = None
```

### 3.4 Add Face Registration Background Task

Edit `philosopher/core/nodes.py`. Add after `node_store`:
```python
async def background_extract_name(state: AgentState, ctx: NodeCtx):
    """Extract name from new faces — runs OFF the critical path."""
    if not state.is_new_face or not state.face_id:
        return
    
    face = await ctx.memory.long.get_face(state.face_id)
    if face and face.get("name"):
        return  # Already has a name
    
    words = state.user_message.strip().split()
    if 1 <= len(words) <= 3:
        name = words[-1].strip(".,!?").title()
        if len(name) >= 2 and name.isalpha():
            await ctx.memory.long.update_face_name(state.face_id, name)
```

Spawn this as a background task from `node_store`:
```python
async def node_store(state: AgentState, ctx: NodeCtx) -> AgentState:
    """Store the interaction in memory."""
    if state.message and state.formatted:
        await ctx.memory.add(state.message, state.formatted, state.emotion,
                             state.face_id, state.face_name)
    state.turn += 1
    
    # Background face name extraction (zero latency)
    asyncio.create_task(background_extract_name(state, ctx))
    
    return state
```

### 3.5 Add `update_face_name()` to Memory

Edit `philosopher/memory/long_term.py`:
```python
async def update_face_name(self, face_id: str, name: str) -> None:
    db = await self._get_db()
    await db.execute(
        "UPDATE known_faces SET name=? WHERE face_id=?",
        (name, face_id),
    )
    await db.commit()
```

### 3.6 Add WebSocket Module

Create `philosopher/api/ws_server.py`:
```python
"""WebSocket endpoint for toy communication."""
from fastapi import WebSocket, WebSocketDisconnect

class WebSocketManager:
    def __init__(self, orchestrator):
        self.orch = orchestrator
        self.connections: dict[str, WebSocket] = {}

    async def handle_client(self, websocket: WebSocket, toy_id: str):
        await websocket.accept()
        self.connections[toy_id] = websocket
        try:
            while True:
                message = await websocket.receive()
                await self._handle_message(toy_id, message)
        except WebSocketDisconnect:
            del self.connections[toy_id]

    async def _handle_message(self, toy_id: str, message: dict):
        if message["type"] == "websocket.receive":
            if "bytes" in message:
                await self._handle_binary(toy_id, message["bytes"])
            elif "text" in message:
                await self._handle_json(toy_id, message["text"])

    async def _handle_binary(self, toy_id: str, data: bytes):
        frame_type = data[0]
        payload = data[1:]
        if frame_type == 0x01:  # Audio
            await self.orch.handle_audio_chunk(toy_id, payload)
        elif frame_type == 0x02:  # Video
            result = await self.orch.handle_video_frame(toy_id, payload)
            # Send servo command immediately on face detection
            if result and result.get("primary_face"):
                await self.send_json(toy_id, {
                    "type": "servo",
                    "emotion": result["primary_face"]["emotion"],
                })

    async def _handle_json(self, toy_id: str, text: str):
        import json
        msg = json.loads(text)
        if msg.get("type") == "speech_started":
            await self.orch.handle_speech_started(toy_id)
        elif msg.get("type") == "speech_ended":
            await self.orch.process_speech(toy_id)
        elif msg.get("type") == "ping":
            await self.send_json(toy_id, {"type": "pong", "timestamp": msg["timestamp"]})

    async def send_json(self, toy_id: str, data: dict):
        if toy_id in self.connections:
            await self.connections[toy_id].send_json(data)

    async def send_binary(self, toy_id: str, frame_type: int, payload: bytes):
        if toy_id in self.connections:
            await self.connections[toy_id].send_bytes(bytes([frame_type]) + payload)
```

### 3.7 Add STT Module

Create `philosopher/stt/engine.py`:
```python
"""Server-side batch STT using faster-whisper."""
from faster_whisper import WhisperModel
import numpy as np
import asyncio

class StreamingSTT:
    def __init__(self, model_size: str = "tiny", device: str = "cpu", mock: bool = False):
        self.mock = mock
        if not mock:
            self.model = WhisperModel(model_size, device=device, compute_type="int8")
        self.audio_buffers: dict[str, bytearray] = {}  # toy_id -> accumulated audio

    def add_audio_chunk(self, toy_id: str, pcm_bytes: bytes):
        if toy_id not in self.audio_buffers:
            self.audio_buffers[toy_id] = bytearray()
        self.audio_buffers[toy_id].extend(pcm_bytes)

    def clear_buffer(self, toy_id: str):
        self.audio_buffers[toy_id] = bytearray()

    async def transcribe(self, toy_id: str, language: str = "es", confidence_threshold: float = -0.5) -> dict:
        if self.mock:
            return {"text": "Hello world", "is_final": True, "confidence": 0.9, "low_confidence": False}
        
        buffer = self.audio_buffers.get(toy_id, bytearray())
        if len(buffer) < 16000:  # Less than 0.5s of audio
            return {"text": "", "is_final": True, "confidence": 0.0, "low_confidence": False}

        audio_np = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
        segments, info = self.model.transcribe(audio_np, language=language, condition_on_previous_text=False)

        text = " ".join([segment.text for segment in segments])
        confidence = min([segment.avg_logprob for segment in segments]) if segments else -1.0

        # Clear buffer after transcription
        self.audio_buffers[toy_id] = bytearray()

        return {
            "text": text.strip(),
            "is_final": True,
            "confidence": confidence,
            "low_confidence": confidence < confidence_threshold,
        }
```

### 3.8 Add Vision Module

Create `philosopher/vision/engine.py`:
```python
"""Server-side face detection and emotion recognition."""
import cv2
import numpy as np
import face_recognition
from deepface import DeepFace
import hashlib

class VisionProcessor:
    def __init__(self, memory_manager=None, mock: bool = False):
        self.memory = memory_manager
        self.mock = mock
        self.known_faces: dict[str, dict] = {}  # face_id -> {name, encoding}

    async def process_frame(self, jpeg_bytes: bytes) -> dict:
        if self.mock:
            return {"faces": [], "primary_face": None}
        
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return {"faces": [], "primary_face": None}

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)

        faces = []
        for (top, right, bottom, left), encoding in zip(locations, encodings):
            face_id = hashlib.sha256(encoding[:8].tobytes()).hexdigest()[:12]
            name = await self._match_face(encoding)
            emotion = await self._detect_emotion(frame, (left, top, right, bottom))

            faces.append({
                "face_id": face_id,
                "name": name,
                "emotion": emotion or "neutral",
                "confidence": 0.9,
            })

        return {
            "faces": faces,
            "primary_face": max(faces, key=lambda f: f["confidence"]) if faces else None,
        }

    async def _match_face(self, encoding) -> str | None:
        if not self.known_faces and self.memory:
            # Load from SQLite on first use
            faces = await self.memory.long.all_faces()
            for f in faces:
                # Note: We don't have encodings in DB, so we match by face_id only
                # In production, you'd store encodings or use a different matching strategy
                pass
        
        # Compare with known faces
        if not self.known_faces:
            return None
        
        import face_recognition as fr
        for face_id, data in self.known_faces.items():
            if fr.compare_faces([data["encoding"]], encoding, tolerance=0.6)[0]:
                return data.get("name")
        return None

    async def _detect_emotion(self, frame, bbox) -> str | None:
        try:
            x, y, x2, y2 = bbox
            roi = frame[y:y2, x:x2]
            if roi.size == 0:
                return None
            result = DeepFace.analyze(roi, actions=["emotion"], enforce_detection=False, silent=True)
            if result:
                emotions = result[0]["emotion"]
                return max(emotions, key=emotions.get).lower()
        except Exception:
            pass
        return None
```

### 3.9 Add Piper TTS Module

Create `philosopher/tts/engine.py`:
```python
"""Server-side TTS using Piper (invoked via subprocess)."""
import subprocess
import tempfile
import os

class PiperTTS:
    def __init__(self, model_path: str = "", voice: str = "es_ES-carlfm-x_low", mock: bool = False):
        self.model_path = model_path
        self.voice = voice
        self.mock = mock
        
        if not self.model_path:
            # Auto-download default voice to ./data/piper_models/
            self.model_path = self._ensure_model_downloaded(voice)

    def _ensure_model_downloaded(self, voice: str) -> str:
        model_dir = "./data/piper_models"
        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, f"{voice}.onnx")
        
        if not os.path.exists(model_file):
            # Download logic here (wget/curl the model and config)
            # For now, just return the path and expect user to download
            pass
        
        return model_file

    async def synthesize(self, text: str) -> bytes:
        if self.mock:
            # Return silent WAV header + some PCM zeros
            return b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 40
        
        config_path = self.model_path.replace(".onnx", ".onnx.json")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        try:
            subprocess.run(
                [
                    "piper",
                    "--model", self.model_path,
                    "--config", config_path,
                    "--output_file", output_path,
                ],
                input=text.encode(),
                check=True,
                capture_output=True,
            )

            with open(output_path, "rb") as f:
                return f.read()
        except FileNotFoundError:
            # Piper binary not installed
            return b""
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
```

### 3.10 Modify Orchestrator for WebSocket

Edit `philosopher/core/orchestrator.py`. Add methods:
```python
async def handle_speech_started(self, toy_id: str):
    """Toy detected speech start. Clear audio buffer."""
    if hasattr(self, 'stt'):
        self.stt.clear_buffer(toy_id)

async def handle_audio_chunk(self, toy_id: str, pcm_bytes: bytes):
    """Receive audio chunk from toy via WebSocket."""
    if hasattr(self, 'stt'):
        self.stt.add_audio_chunk(toy_id, pcm_bytes)

async def handle_video_frame(self, toy_id: str, jpeg_bytes: bytes):
    """Receive video frame from toy via WebSocket."""
    if hasattr(self, 'vision'):
        return await self.vision.process_frame(jpeg_bytes)
    return None

async def process_speech(self, toy_id: str):
    """Called when toy signals speech ended."""
    if not hasattr(self, 'stt') or not hasattr(self, 'ws_manager'):
        return
    
    result = await self.stt.transcribe(toy_id, confidence_threshold=self.settings.stt.confidence_threshold)

    if result["low_confidence"]:
        await self.ws_manager.send_json(toy_id, {
            "type": "text",
            "content": "Hmm... I didn't quite catch that. Could you repeat?",
            "emotion": "neutral",
        })
        return

    if not result["text"]:
        return  # Empty transcription, do nothing

    # Run split-graph streaming
    await self._stream_response(toy_id, result["text"])

async def _stream_response(self, toy_id: str, user_message: str):
    """Split-graph streaming: nodes 1-3, then external LLM stream, then node_store."""
    
    # Step 1: Run nodes 1-3 via LangGraph for context building
    state = AgentState(user_message=user_message)
    # TODO: Add face info from last known frame
    
    # Run perception, memory, prompt nodes
    state = await node_perception(state, self.graph.ctx)
    state = await node_memory(state, self.graph.ctx)
    state = await node_prompt(state, self.graph.ctx)
    
    # Step 2: Stream LLM tokens
    messages = [LLMMessage(role=m["role"], content=m["content"]) for m in state.short_context[-5:]]
    messages.append(LLMMessage(role="user", content=user_message))
    
    buffer = ""
    full_response = ""
    
    async for token in self.llm.chat_stream(
        messages, 
        system_prompt=state.system_prompt,
        max_tokens=250,
        temperature=self.settings.llm.temperature,
    ):
        buffer += token
        full_response += token
        
        # Check for sentence terminator
        if any(buffer.endswith(t) for t in ['.', '!', '?', '\n']):
            sentence = buffer.strip()
            if sentence:
                # Step 3: Format per sentence
                state.llm_response = sentence
                state = await node_format(state, self.graph.ctx)
                formatted = state.formatted
                
                # Step 4: Send to toy
                await self.ws_manager.send_json(toy_id, {
                    "type": "text",
                    "content": formatted,
                    "emotion": state.emotion or "neutral",
                })
                await self.ws_manager.send_json(toy_id, {
                    "type": "servo",
                    "emotion": state.emotion or "neutral",
                })
                
                # Step 5: Synthesize audio
                if hasattr(self, 'tts') and self.settings.tts.enabled:
                    audio_bytes = await self.tts.synthesize(formatted)
                    if audio_bytes:
                        # Send full WAV per sentence (not chunked)
                        await self.ws_manager.send_binary(toy_id, 0x03, audio_bytes)
            
            buffer = ""
    
    # Handle remaining buffer (sentence without terminator at end)
    if buffer.strip():
        state.llm_response = buffer.strip()
        state = await node_format(state, self.graph.ctx)
        formatted = state.formatted
        await self.ws_manager.send_json(toy_id, {
            "type": "text",
            "content": formatted,
            "emotion": state.emotion or "neutral",
        })
        if hasattr(self, 'tts') and self.settings.tts.enabled:
            audio_bytes = await self.tts.synthesize(formatted)
            if audio_bytes:
                await self.ws_manager.send_binary(toy_id, 0x03, audio_bytes)
    
    # Step 6: Store full interaction once at end
    state.llm_response = full_response
    state.formatted = full_response  # Re-format full response for storage
    state = await node_format(state, self.graph.ctx)
    state = await node_store(state, self.graph.ctx)
```

### 3.11 Update Main Entry Point

Edit `philosopher/main.py`:
```python
from philosopher.api.ws_server import WebSocketManager
from philosopher.stt.engine import StreamingSTT
from philosopher.vision.engine import VisionProcessor
from philosopher.tts.engine import PiperTTS

async def _run():
    orch = Orchestrator()
    await orch.init()
    
    # Initialize new modules
    orch.stt = StreamingSTT(
        model_size=orch.settings.stt.model,
        device=orch.settings.stt.device,
        mock=os.getenv("MOCK_MODE", "false").lower() == "true",
    )
    orch.vision = VisionProcessor(
        memory_manager=orch.memory,
        mock=os.getenv("MOCK_MODE", "false").lower() == "true",
    )
    orch.tts = PiperTTS(
        model_path=orch.settings.tts.model_path,
        voice=orch.settings.tts.voice,
        mock=os.getenv("MOCK_MODE", "false").lower() == "true",
    )

    # Add WebSocket support
    ws_manager = WebSocketManager(orch)
    orch.ws_manager = ws_manager

    app = create_app(orch)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket, toy_id: str = "default"):
        await ws_manager.handle_client(websocket, toy_id)

    config = uvicorn.Config(app, host=settings.api.host, port=settings.api.port)
    server = uvicorn.Server(config)
    await server.serve()
```

---

## 4. TOY — ADD WEBSOCKET CLIENT & SIMPLIFY

### 4.1 Replace Protocol

Create `toy_client/protocol/ws_client.py`:
```python
"""WebSocket client for toy-server communication."""
import aiohttp
import json

class ToyWebSocketClient:
    def __init__(self, server_url: str, toy_id: str = "toy_01", reconnect_interval=5.0, max_reconnect=10):
        self.url = server_url
        self.toy_id = toy_id
        self.reconnect_interval = reconnect_interval
        self.max_reconnect = max_reconnect
        self.session: aiohttp.ClientSession | None = None
        self.ws: aiohttp.ClientWebSocketResponse | None = None
        self._reconnect_count = 0

    async def connect(self):
        self.session = aiohttp.ClientSession()
        self.ws = await self.session.ws_connect(f"{self.url}/ws?toy_id={self.toy_id}")
        self._reconnect_count = 0
        # Start heartbeat
        asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self):
        while self.ws and not self.ws.closed:
            await self.send_json({"type": "ping", "timestamp": asyncio.get_event_loop().time()})
            await asyncio.sleep(5)

    async def _reconnect(self):
        if self._reconnect_count >= self.max_reconnect:
            return
        self._reconnect_count += 1
        await asyncio.sleep(min(self.reconnect_interval * self._reconnect_count, 60))
        await self.connect()

    async def send_audio(self, pcm_bytes: bytes):
        if self.ws:
            await self.ws.send_bytes(b'\x01' + pcm_bytes)

    async def send_frame(self, jpeg_bytes: bytes):
        if self.ws:
            await self.ws.send_bytes(b'\x02' + jpeg_bytes)

    async def send_json(self, data: dict):
        if self.ws:
            await self.ws.send_str(json.dumps(data))

    async def receive(self):
        if self.ws:
            msg = await self.ws.receive()
            if msg.type == aiohttp.WSMsgType.TEXT:
                return json.loads(msg.data)
            elif msg.type == aiohttp.WSMsgType.BINARY:
                return {"type": "binary", "frame_type": msg.data[0], "payload": msg.data[1:]}
            elif msg.type == aiohttp.WSMsgType.CLOSED:
                return {"type": "closed"}
        return {"type": "none"}

    async def close(self):
        if self.ws:
            await self.ws.close()
        if self.session:
            await self.session.close()
```

### 4.2 Replace TTS with Audio Player

Create `toy_client/audio/player.py`:
```python
"""WAV audio player using PyAudio."""
import pyaudio
import asyncio
import wave
import io

class AudioPlayer:
    def __init__(self, rate: int = 22050, channels: int = 1):
        self.rate = rate
        self.channels = channels
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self._play_queue = asyncio.Queue()
        self._playing = False

    def init(self):
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            output=True,
            frames_per_buffer=1024,
        )

    async def play_wav(self, wav_bytes: bytes):
        """Parse WAV and queue for playback."""
        await self._play_queue.put(wav_bytes)
        if not self._playing:
            self._playing = True
            asyncio.create_task(self._playback_loop())

    async def _playback_loop(self):
        while True:
            try:
                wav_bytes = await asyncio.wait_for(self._play_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if self._play_queue.empty():
                    self._playing = False
                    break
                continue
            
            # Parse WAV header
            try:
                with io.BytesIO(wav_bytes) as f:
                    with wave.open(f, 'rb') as wav:
                        # Read all frames and play
                        frames = wav.readframes(wav.getnframes())
                        # Write in chunks to avoid blocking
                        chunk_size = 1024
                        for i in range(0, len(frames), chunk_size):
                            chunk = frames[i:i + chunk_size]
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(None, self.stream.write, chunk)
            except Exception:
                # Fallback: treat as raw PCM
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.stream.write, wav_bytes)

    def close(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()
```

### 4.3 Add Microphone Capture with VAD

Create `toy_client/audio/capture.py`:
```python
"""USB microphone capture with energy-based VAD."""
import pyaudio
import numpy as np
import asyncio

class MicrophoneCapture:
    def __init__(self, rate=16000, chunk=1024, threshold=300, silence=2.0):
        self.rate = rate
        self.chunk = chunk
        self.threshold = threshold
        self.silence = silence
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.input_device_index = None

    async def init(self):
        # Enumerate devices and find USB mic
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            name = info.get("name", "").lower()
            if ("usb" in name or "audio" in name) and info.get("maxInputChannels", 0) > 0:
                self.input_device_index = i
                break
        
        if self.input_device_index is None:
            print("[WARNING] USB microphone not found, using default device")
        
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            input_device_index=self.input_device_index,
            frames_per_buffer=self.chunk,
        )

    def _energy(self, pcm_bytes: bytes) -> float:
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        return np.sqrt(np.mean(samples.astype(np.float32) ** 2))

    async def capture_stream(self):
        """Yield accumulated audio bytes when speech is detected and silence follows."""
        buffer = bytearray()
        is_speaking = False
        silence_frames = 0
        silence_threshold_frames = int(self.rate / self.chunk * self.silence)

        while True:
            pcm_bytes = self.stream.read(self.chunk, exception_on_overflow=False)
            energy = self._energy(pm_bytes)
            
            if energy > self.threshold:
                if not is_speaking:
                    is_speaking = True
                    silence_frames = 0
                    # Signal speech start
                    yield {"type": "speech_started", "audio": b""}
                buffer.extend(pcm_bytes)
            else:
                if is_speaking:
                    buffer.extend(pcm_bytes)
                    silence_frames += 1
                    if silence_frames >= silence_threshold_frames:
                        # Speech ended
                        is_speaking = False
                        audio_data = bytes(buffer)
                        buffer = bytearray()
                        yield {"type": "speech_ended", "audio": audio_data}
                else:
                    # Not speaking, discard
                    pass
            
            await asyncio.sleep(0)  # Yield control

    def close(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()
```

### 4.4 Simplify Camera

Edit `toy_client/vision/camera.py`:
```python
"""Simplified camera: capture only, no processing. Configurable FPS with rounding."""
import cv2
import os

class Camera:
    def __init__(self, fps: float = None, quality: int = 60):
        # Read from env if not provided
        if fps is None:
            fps = float(os.getenv("PHILOSOPHER_CAMERA_FPS", "0.5"))
        
        # Round to nearest 0.5 step, clamp 0.5-5.0
        self.fps = max(0.5, min(5.0, round(fps * 2) / 2))
        
        self.quality = int(os.getenv("PHILOSOPHER_CAMERA_QUALITY", str(quality)))
        self.quality = max(30, min(90, self.quality))
        self.cap = None

    async def init(self):
        # Try V4L2 backends for CSI camera
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
        if not self.cap.isOpened():
            print("[WARNING] Camera not available, entering mock mode")
            self.cap = None
            return

        # Force resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    async def capture(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    async def encode(self, frame) -> bytes:
        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return buf.tobytes() if ret else b''

    def close(self):
        if self.cap:
            self.cap.release()
```

### 4.5 Modify Servos for Sequential Movement

Edit `toy_client/hardware/servos.py`:
```python
"""Sequential servo controller with power safety."""
import asyncio

class ServoController:
    def __init__(self, head=12, left=13, right=18, mock=False):
        self.pins = {"head": head, "left_arm": left, "right_arm": right}
        self.mock = mock
        self.moving = False
        self.queue = asyncio.Queue()
        self._pos = {k: 0 for k in self.pins}

    async def init(self):
        if not self.mock:
            try:
                import OPi.GPIO as GPIO
                GPIO.setmode(GPIO.BCM)
                for pin in self.pins.values():
                    GPIO.setup(pin, GPIO.OUT)
                    GPIO.output(pin, GPIO.LOW)
                # Initialize PWM for all servos
                self._pwm = {}
                for name, pin in self.pins.items():
                    pwm = GPIO.PWM(pin, 50)  # 50Hz for servos
                    pwm.start(7.5)  # Neutral position (1.5ms pulse)
                    self._pwm[name] = pwm
            except Exception as e:
                print(f"[WARNING] GPIO init failed: {e}. Entering mock mode.")
                self.mock = True
        
        print("[WARNING] Servos share MicroUSB power. Only one moves at a time.")
        asyncio.create_task(self._movement_worker())
        return self

    async def move(self, name: str, angle: int):
        await self.queue.put((name, angle))

    async def _movement_worker(self):
        while True:
            name, angle = await self.queue.get()
            self.moving = True
            
            if self.mock:
                print(f"[Servo] {name} -> {angle}")
            else:
                try:
                    # Convert angle (-90 to 90) to duty cycle (2.5 to 12.5)
                    duty = 7.5 + (angle / 90.0) * 5.0
                    duty = max(2.5, min(12.5, duty))
                    
                    if hasattr(self, '_pwm') and name in self._pwm:
                        self._pwm[name].ChangeDutyCycle(duty)
                        await asyncio.sleep(0.5)  # Time to reach position
                        self._pwm[name].ChangeDutyCycle(0)  # Stop PWM to save power
                except Exception as e:
                    print(f"[ERROR] Servo movement failed: {e}")
            
            self._pos[name] = angle
            await asyncio.sleep(0.3)  # Power rail recovery
            self.moving = False

    async def animate(self, emotion: str):
        poses = {
            "happy": [("head", 15), ("right_arm", 45)],
            "sad": [("head", -15)],  # Single servo for speed
            "surprised": [("head", 25), ("right_arm", 35), ("left_arm", 35)],
            "angry": [("head", -5), ("right_arm", 20)],
            "fear": [("head", -10), ("left_arm", -15)],
            "neutral": [("head", 0)],
        }
        for servo, angle in poses.get(emotion, [("head", 5)]):
            await self.move(servo, angle)

    def close(self):
        if not self.mock:
            try:
                import OPi.GPIO as GPIO
                if hasattr(self, '_pwm'):
                    for pwm in self._pwm.values():
                        pwm.stop()
                GPIO.cleanup()
            except Exception:
                pass
```

### 4.6 Rewrite Toy Main Loop

Edit `toy_client/main.py`:
```python
import asyncio
import os
from toy_client.protocol.ws_client import ToyWebSocketClient
from toy_client.audio.capture import MicrophoneCapture
from toy_client.audio.player import AudioPlayer
from toy_client.vision.camera import Camera
from toy_client.hardware.servos import ServoController

class Toy:
    def __init__(self):
        self.server_url = os.getenv("PHILOSOPHER_SERVER_URL", "ws://localhost:8080")
        self.toy_id = os.getenv("PHILOSOPHER_TOY_ID", "toy_01")
        self.mock = os.getenv("PHILOSOPHER_MOCK", "false").lower() == "true"

    async def init(self):
        self.ws = ToyWebSocketClient(self.server_url, self.toy_id)
        await self.ws.connect()

        self.mic = MicrophoneCapture()
        await self.mic.init()

        self.player = AudioPlayer()
        self.player.init()

        self.camera = Camera()
        await self.camera.init()

        self.servos = await ServoController(mock=self.mock).init()

    async def run(self):
        await asyncio.gather(
            self._audio_task(),
            self._video_task(),
            self._receive_task(),
        )

    async def _audio_task(self):
        async for event in self.mic.capture_stream():
            if event["type"] == "speech_started":
                await self.ws.send_json({"type": "speech_started"})
            elif event["type"] == "speech_ended":
                # Send accumulated audio
                await self.ws.send_audio(event["audio"])
                await self.ws.send_json({"type": "speech_ended"})

    async def _video_task(self):
        interval = 1.0 / self.camera.fps
        while True:
            frame = await self.camera.capture()
            if frame is not None:
                jpeg = await self.camera.encode(frame)
                await self.ws.send_frame(jpeg)
            await asyncio.sleep(interval)

    async def _receive_task(self):
        while True:
            msg = await self.ws.receive()
            if msg["type"] == "text":
                print(f"[Philosopher] {msg['content']}")
            elif msg["type"] == "servo":
                await self.servos.animate(msg["emotion"])
            elif msg["type"] == "binary" and msg["frame_type"] == 0x03:
                await self.player.play_wav(msg["payload"])
            elif msg["type"] == "error":
                print(f"[Error] {msg['message']}")
            elif msg["type"] == "closed":
                break

async def main():
    toy = Toy()
    await toy.init()
    try:
        await toy.run()
    except KeyboardInterrupt:
        pass
    finally:
        await toy.ws.close()
        toy.mic.close()
        toy.player.close()
        toy.camera.close()
        await toy.servos.close()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 5. STREAMING LLM — SPLIT-GRAPH EXECUTION

### 5.1 Modify LLM Client for Streaming

Edit `philosopher/llm/client.py`. Add:
```python
async def chat_stream(
    self,
    messages: list[LLMMessage],
    system_prompt: str | None = None,
    max_tokens: int = 250,
    temperature: float | None = None,
) -> AsyncGenerator[str, None]:
    """Stream LLM tokens as they're generated."""
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.extend([m.to_openai() for m in messages])

    response = await self.client.chat.completions.create(
        model=self.cfg.model,
        messages=msgs,
        max_tokens=max_tokens,
        temperature=temperature or self.cfg.temperature,
        stream=True,  # Enable streaming
    )

    async for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
```

### 5.2 Streaming Architecture

The streaming flow in `orchestrator.py`:
1. **Nodes 1-3** (perception, memory, prompt): Run via `AgentGraph` to build system prompt and context
2. **External stream**: Call `llm.chat_stream()` directly with messages + system prompt
3. **Sentence buffer**: Accumulate tokens, split on `.`, `!`, `?`, `\n`
4. **Node 5** (format): Run per sentence via direct async call (not through graph)
5. **Send**: Text + servo + WAV audio to toy immediately per sentence
6. **Node 6** (store): Run once after full response accumulated

This keeps the existing graph intact for HTTP API fallback while enabling sentence-level streaming on WebSocket.

---

## 6. WEBSOCKET PROTOCOL

### Connection
- URL: `ws://<server>:<port>/ws?toy_id=<id>`
- Binary frames for media, text frames for JSON control messages

### Toy → Server (Binary)
| Type Byte | Payload | Meaning |
|-----------|---------|---------|
| `0x01` | PCM 16-bit LE, 16kHz, mono | Audio chunk |
| `0x02` | JPEG bytes | Video frame |

### Toy → Server (JSON Text)
```json
{"type": "speech_started"}
{"type": "speech_ended"}
{"type": "sensor", "sensor": "button", "action": "pressed"}
{"type": "ping", "timestamp": 1717700000.0}
```

### Server → Toy (JSON Text)
```json
{"type": "text", "content": "Hola Juan, me alegra verte.", "emotion": "happy", "turn": 5}
{"type": "servo", "emotion": "happy"}
{"type": "pong", "timestamp": 1717700000.0}
{"type": "error", "message": "STT timeout"}
```

### Server → Toy (Binary)
| Type Byte | Payload | Meaning |
|-----------|---------|---------|
| `0x03` | WAV file bytes (22050 Hz, 16-bit, mono) | Synthesized speech audio |

---

## 7. ERROR HANDLING

| Scenario | Action |
|----------|--------|
| LLM connection error mid-stream | Yield `[connection error]`, send fallback text once, continue session |
| Vision/DeepFace crash | Log error, return `neutral`, don't crash WebSocket |
| Empty transcription at speech_ended | Do nothing |
| WebSocket disconnect mid-response | Cancel tasks, clean buffers |
| Piper TTS binary not found | Return empty bytes, log error, toy gets text but no audio |
| STT model fails to load | Return `{"error": "model load failed"}` |
| Camera not available | Enter mock mode, return `None` frames |
| Servo GPIO fails | Enter mock mode, print to stdout |

---

## 8. CONFIGURATION

### Server `.env`
```bash
# LLM (external only)
PHILOSOPHER_LLM_PROVIDER=local
PHILOSOPHER_LLM_BASE_URL=http://192.168.1.100:1234/v1
PHILOSOPHER_LLM_API_KEY=not-needed
PHILOSOPHER_LLM_MODEL=llama-3.1-8b

# STT (configurable quality)
PHILOSOPHER_STT_MODEL=tiny          # tiny | base | small
PHILOSOPHER_STT_DEVICE=cpu
PHILOSOPHER_STT_CONFIDENCE_THRESHOLD=-0.5

# Camera
PHILOSOPHER_CAMERA_FPS=0.5          # Any float, rounds to 0.5 steps, clamped 0.5-5.0
PHILOSOPHER_CAMERA_QUALITY=60       # JPEG quality: 30-90

# TTS (Piper — requires system binary)
PHILOSOPHER_TTS_ENABLED=true
PHILOSOPHER_TTS_VOICE=es_ES-carlfm-x_low
PHILOSOPHER_TTS_MODEL_PATH=         # Empty = auto-download to ./data/piper_models/

# Memory, API, Logging (unchanged)
PHILOSOPHER_MEMORY_DB_PATH=./data/philosopher_memory.db
PHILOSOPHER_API_HOST=0.0.0.0
PHILOSOPHER_API_PORT=8080
PHILOSOPHER_LOG_LEVEL=INFO
PHILOSOPHER_LOG_FORMAT=rich

# Mock mode for testing (no hardware)
MOCK_MODE=false
```

### Toy `.env`
```bash
PHILOSOPHER_SERVER_URL=ws://192.168.1.50:8080/ws
PHILOSOPHER_TOY_ID=toy_01
PHILOSOPHER_MOCK=false

# Configurable quality
PHILOSOPHER_CAMERA_FPS=0.5          # Any float, rounds to 0.5 steps
PHILOSOPHER_CAMERA_QUALITY=60       # 30-90
```

---

## 9. TESTING STRATEGY

### Keep Existing Tests

All existing tests should continue passing throughout migration:
```bash
cd philosopher-server
pytest tests/ -v  # Must still pass (28 tests)

cd philosopher-toy
pytest tests/ -v  # Must still pass (3 tests)
```

### Add New Tests

**Server:**
- `test_ws_connection.py` — WebSocket connect/disconnect, heartbeat
- `test_stt_engine.py` — Mock audio transcription, confidence threshold
- `test_vision_engine.py` — Mock JPEG processing, face matching
- `test_tts_engine.py` — Mock synthesis, WAV output
- `test_sentence_buffer.py` — Token accumulation, split on terminators
- `test_face_registration.py` — Background name extraction
- `test_camera_fps_rounding.py` — 0.7→0.5, 1.2→1.0, 6.0→5.0
- `test_streaming_llm.py` — Token streaming
- `test_split_graph.py` — Nodes 1-3 → stream → store

**Toy:**
- `test_ws_client.py` — WebSocket client, reconnection
- `test_audio_capture.py` — VAD logic with synthetic audio
- `test_audio_player.py` — WAV parsing, queue playback
- `test_camera.py` — FPS rounding, quality settings
- `test_servos.py` — Mock mode animation, sequential movement with delay

---

## 10. FALLBACK STRATEGY

If WebSocket fails at any point:
1. Keep HTTP API running (`api/app.py`)
2. Server-side STT can run in batch mode (accumulate all audio, then POST to /chat)
3. Toy does NOT fall back to HTTP POST — WebSocket is the only protocol for v2

This ensures the server always works via HTTP for diagnostics, but the toy requires WebSocket.

---

## 11. KNOWN LIMITATIONS & FUTURE WORK

1. **True streaming STT deferred to v3** — Batch approach achieves <1.5s latency. True streaming (word-by-word) would save ~100-200ms but adds 3-5x complexity.
2. **Auto-language detection deferred to v3** — Static language per personality for now.
3. **Face registration is automatic but simple** — Only extracts 1-3 word replies. Complex name introductions require manual DB insert.
4. **No wake word** — Requires button press or continuous listening.
5. **Piper TTS requires system binary** — Not installable via pip. Auto-download helper downloads model files but user must install `piper-tts` binary.

---

## 12. DELIVERABLES

1. WebSocket server endpoint alongside existing HTTP
2. Server-side STT (faster-whisper) with configurable model
3. Server-side vision (face_recognition + DeepFace)
4. Server-side TTS (Piper) with audio streaming (full WAV per sentence)
5. Simplified toy: WebSocket client, capture-only camera, sequential servos, PyAudio player
6. Zero-latency face registration (background extraction)
7. Split-graph streaming (nodes 1-3 → external LLM stream → store)
8. All existing tests still pass (28 + 3)
9. New tests for WebSocket, streaming, audio, face registration
10. Working mock-mode demo
11. Updated migration guide (this document)

**Quality metric**: The toy must respond within 1.5 seconds of speech ending, with configurable quality parameters.