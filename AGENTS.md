# AGENTS.md — Philosopher Project

> Comprehensive reference for AI agents working on the Philosopher conversational AI ecosystem.
> Last reviewed: 2026-06-06 (updated to **v2** — WebSocket streaming, all AI server-side)

> ### ⚠️ v2 architecture — read this first
> The project migrated from v1 (HTTP, AI split across both boards) to **v2: WebSocket streaming with ALL AI on the server**. `MIGRATION_GUIDE.md` is the authoritative design doc; `/CLAUDE.md` has the up-to-date quick reference. **Where the older prose below still describes v1, the source code and `MIGRATION_GUIDE.md` win.** Key v2 facts:
> - **Server = Raspberry Pi 4 (4GB)** runs *everything*: STT (`faster-whisper`), vision (`face_recognition` for identity + FER+ ONNX for emotion via `onnxruntime`), TTS (**Piper**, a subprocess binary), LangGraph, memory, and an external-LLM proxy.
> - **Toy = Banana Pi BPi-M2 Zero (512MB)** is a **thin I/O client with zero ML**: mic/camera capture → stream over WebSocket; play WAV; move servos.
> - **Transport is WebSocket** (`/ws`), not HTTP POST. HTTP `/chat` survives only for diagnostics/fallback.
> - Sections that say "No WebSocket", "STT requires internet", or that put DeepFace/STT/Edge-TTS *on the toy* are **stale v1**.

---

## 1. Project Overview

Philosopher is a **conversational AI agent with personality, dual memory (short/long term), facial recognition, and emotion detection**. It is embedded in a smart plush toy.

The system is **two independent Python projects** plus an external LLM. The toy streams sensor data to the server over **WebSocket**; the server does all inference and streams speech/commands back:

```
+---------------+  HTTP   +--------------------------+  WebSocket  +----------------------+
| External LLM  | <-----> | philosopher-server       | <--------> | philosopher-toy       |
| (LM Studio,   |         | Raspberry Pi 4 (4GB)     |            | Banana Pi M2Z (512MB) |
|  OpenAI, etc.)|         | STT·vision·TTS·LangGraph |            | mic·camera·spkr·servos|
+---------------+         +--------------------------+            +----------------------+
  Model ~8B params              Brain (all AI)                       Body (I/O only, no ML)
```

### Key Design Decisions

1. **Two separate projects**: Each has its own `pyproject.toml`, dependencies, and tests. Deployed on different hardware (server = RPi4 4GB, toy = Banana Pi 512MB).
2. **All AI is server-side**: The 512MB toy runs zero ML. STT, vision/emotion, and TTS all execute on the RPi4 server. The toy only captures, streams, plays, and actuates.
3. **Personalities are YAML-only**: No code in language folders. Add a language by creating a folder with YAML files.
4. **WebSocket streaming**: The toy connects to `/ws` and exchanges binary media + JSON control frames. HTTP `/chat` is kept for diagnostics/fallback only.
5. **LLM-agnostic, external only**: Any OpenAI-compatible API endpoint (LM Studio, OpenAI, Ollama). The server streams tokens; no local LLM.
6. **Semantic memory without embeddings**: Uses keyword matching + recency scoring instead of vector embeddings (optimized for limited RAM).
7. **Mock mode**: `MOCK_MODE=true` (server) / `PHILOSOPHER_MOCK=true` (toy) for development without models or hardware.

---

## 2. Project 1: philosopher-server

The brain. Coordinates intelligence, memory, personality, and exposes a FastAPI REST interface.

### 2.1 Pipeline (LangGraph)

Every user message flows through 6 nodes in sequence:

```
[User Input] --> perception --> memory --> prompt --> think --> format --> store
                    |           |          |        |        |        |
                    v           v          v        v        v        v
               Lookup face  Retrieve   Build     LLM     Apply   Save to
               in SQLite    context    system    gen     style   memory
                          (short+long)  prompt   text           (RAM+SQLite)
```

1. **perception** (`core/nodes.py:17`): Looks up `face_id` in SQLite `known_faces`. Sets `face_name` and `is_new_face`.
2. **memory** (`core/nodes.py:27`): Retrieves short-term context (last 10 messages) + long-term memories (keyword search, top 5).
3. **prompt** (`core/nodes.py:37`): Builds system prompt = personality YAML + face name + emotion + relevant memories.
4. **think** (`core/nodes.py:47`): Calls LLM with constructed prompt + conversation history. Max 250 tokens per response.
5. **format** (`core/nodes.py:63`): Applies personality style (max length, cleans markdown).
6. **store** (`core/nodes.py:72`): Saves interaction to short-term buffer (deque) and long-term SQLite.

### 2.2 Module Reference

| Module | Files | Responsibility |
|--------|-------|--------------|
| `config/` | `settings.py`, `personalities/<lang>/*.yaml` | Pydantic Settings + personality prompts |
| `core/` | `state.py`, `nodes.py`, `graph.py`, `orchestrator.py` | LangGraph pipeline + state machine |
| `llm/` | `client.py` | OpenAI-compatible async client |
| `memory/` | `short_term.py`, `long_term.py`, `manager.py` | Circular buffer + SQLite semantic store |
| `personality/` | `engine.py` | YAML loader + prompt builder + response formatter |
| `api/` | `app.py` | FastAPI REST endpoints |
| `events/` | `bus.py` | Async pub/sub event bus (17 event types) |
| `utils/` | `logger.py` | structlog + rich logging |

### 2.3 Configuration (`.env`)

```bash
# LLM Provider
PHILOSOPHER_LLM_PROVIDER=local          # local | openai | ollama | custom
PHILOSOPHER_LLM_BASE_URL=http://192.168.1.100:1234/v1
PHILOSOPHER_LLM_API_KEY=lm-studio
PHILOSOPHER_LLM_MODEL=llama-3.1-8b

# Personality
PHILOSOPHER_PERSONALITY=filosofo        # YAML filename without extension
PHILOSOPHER_NAME=Philosopher
PHILOSOPHER_LANGUAGE=es                 # selects subfolder in personalities/

# Memory
PHILOSOPHER_MEMORY_DB_PATH=./data/philosopher_memory.db

# API Server
PHILOSOPHER_API_ENABLED=true
PHILOSOPHER_API_HOST=0.0.0.0
PHILOSOPHER_API_PORT=8080

# Logging
PHILOSOPHER_LOG_LEVEL=INFO              # DEBUG | INFO | WARNING | ERROR
PHILOSOPHER_LOG_FORMAT=rich             # json | console | rich
```

Settings are validated via Pydantic (`config/settings.py`). `temperature` is clamped to `[0.0, 2.0]`.

### 2.4 API Endpoints

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET | `/health` | — | `{status, llm, memory}` |
| POST | `/chat` | `{message, face_id?, face_name?, emotion?}` | `{response, emotion, turn}` |
| GET | `/personalities` | — | `["filosofo", "curioso", ...]` |
| POST | `/personality` | `?name=filosofo` | `{personality}` |
| GET | `/memory/stats` | — | `{total_memories, known_faces, by_type}` |
| GET | `/sessions` | — | per-person interaction stats (`SessionTracker`) |
| GET | `/dashboard` `/dashboard/stream` | — | live dashboard page + SSE feed (`DashboardHub`) |

### 2.5 Memory Architecture

**Short-term** (`memory/short_term.py`):
- `deque` circular buffer in RAM, max 10 messages
- Lost on restart
- Stores `role`, `content`, `timestamp`, `metadata`

**Long-term** (`memory/long_term.py`):
- SQLite with WAL mode
- Tables: `memories`, `known_faces`
- Keyword search: extracts words >3 chars from content, scores by overlap + recency decay (1 week half-life)
- `similarity_threshold` default: 0.6

### 2.6 Personalities (YAML Structure)

Located at `philosopher/config/personalities/<lang>/`. Each YAML contains:

```yaml
name: "Philosopher"
role: "companion"
language: "es"
description: "..."
system_prompt: |
  You are Philosopher, a wise plush toy...
traits:
  empathy: 9
  curiosity: 7
  humor: 5
  formality: 2
  energy: 4
  optimism: 6
speech_patterns:
  greetings: [...]
  farewells: [...]
emotion_responses:
  happy: [...]
  sad: [...]
  angry: [...]
  surprised: [...]
  neutral: [...]
  fear: [...]
style:
  max_response_length: 200
  use_asterisks_for_actions: true
  ask_questions: true
  use_metaphors: true
```

**Available personalities (es)**: `filosofo`, `curioso`, `poetico`, `amigo`, `sabio`

### 2.7 Event Bus (`events/bus.py`)

Singleton async pub/sub with 17 event types:

```python
USER_TEXT, USER_VOICE, FACE_DETECTED, FACE_RECOGNIZED,
EMOTION_DETECTED, SPEECH_STARTED, SPEECH_ENDED,
TTS_STARTED, TTS_ENDED, BUTTON_PRESSED, TOUCH_DETECTED,
RESPONSE_READY, THINKING_STARTED, THINKING_ENDED,
WAKE_UP, SLEEP, ERROR, SHUTDOWN
```

Usage:
```python
bus = EventBus()
bus.subscribe(EventType.FACE_DETECTED, handler)
await bus.emit(Event(EventType.FACE_DETECTED, {"face_id": "abc"}))
```

### 2.8 Testing

```bash
cd philosopher-server
pytest tests/ -v
```

- Unit suite (`tests/unit/`) now also covers the v2 modules: `test_stt`, `test_tts`, `test_vision`, `test_ws`, `test_face_registration` (plus config, memory, personality, state, events); integration in `tests/integration/test_nodes.py`
- Uses `pytest-asyncio` with `asyncio_mode = auto`
- Fixtures in `tests/conftest.py` provide mock settings + temp DB + event bus
- Run server-side AI in mock with `MOCK_MODE=true` (STT/vision/TTS return stubs)

---

## 3. Project 2: philosopher-toy

The body. Runs on Banana Pi. Handles camera, microphone, servos, LEDs, and TTS.

> **v2 note:** The toy no longer does any face/emotion detection, STT, or TTS. Those moved to the server. The loop below is **stale v1** — see the v2 loop and module table immediately after it.

### 3.1 Main Loop — v2 (`banana_client/main.py`)

Three concurrent asyncio tasks over one WebSocket (`asyncio.gather`):

```
audio_task:   mic capture + energy VAD → on speech_started/speech_ended,
              stream PCM (0x01) + JSON control to server
video_task:   capture frame → JPEG encode → stream (0x02) at configured FPS
receive_task: handle server frames →
                {type:text}  → print
                {type:servo} → servos.animate(emotion)   (one servo at a time)
                0x03 WAV     → AudioPlayer.play_wav()
```

<details><summary>Stale v1 loop (do not implement)</summary>

```
1. Capture frame; 2. DeepFace emotion; 3. mic VAD; 4. Google STT;
5. HTTP POST /chat; 6. receive text; 7. Edge TTS + ffplay; 8. animate servos.
```
</details>

### 3.2 Module Reference — v2

| Module | Files | Role |
|--------|-------|------|
| `protocol/` | `ws_client.py` | WebSocket client: heartbeat, **auto-reconnect** (backoff), resilient sends |
| `vision/` | `camera.py` | **Capture + JPEG encode only** (no detection). FPS rounds to 0.5 steps, clamped 0.5–5.0 |
| `audio/` | `capture.py`, `player.py` | Mic capture + VAD (**half-duplex gating** while speaking); PyAudio WAV playback (reopens at the WAV's real rate). STT/TTS are server-side. *(v1 `stt.py`/`tts.py`/`protocol/client.py` removed.)* |
| `hardware/` | `servos.py` | Head (GPIO 12), Left arm (GPIO 13), Right arm (GPIO 18). **Sequential moves** (power) |

### 3.3 Configuration (`.env`)

```bash
PHILOSOPHER_SERVER_URL=ws://192.168.1.50:8080   # WebSocket; code default ws://localhost:8080
PHILOSOPHER_TOY_ID=banana_01
PHILOSOPHER_CAMERA_FPS=0.5        # rounds to 0.5 steps, clamped 0.5-5.0
PHILOSOPHER_CAMERA_QUALITY=60     # JPEG quality 30-90
PHILOSOPHER_MOCK=false            # Set true for dev without hardware
```

### 3.4 Hardware Details — v2

**Servos** (`hardware/servos.py`):
- Pins: head=GPIO 12, left_arm=GPIO 13, right_arm=GPIO 18
- Library: OPi.GPIO (falls back to mock mode on import error)
- **Sequential movement only** — the 512MB toy's MicroUSB rail browns out if servos start together
- Emotion animations: `happy`, `sad`, `surprised`, `angry`, `fear`, `neutral` (driven by `{type:servo}` from the server)
- **Head = gaze** (pan-only): `look(offset_x)` (from `{type:look, offset_x}` per video frame) tracks the face; server recenters (`look(0)`) after 4 s with no face; `idle_loop` adds ±3° drift so it's not frozen.
- **Emotion = arms**: `animate` (reply gesture, `{type:servo}`) and `react` (brief reaction on emotion *change*, `{type:react}`) use the arms; the head stays on its gaze base. **Exception**: `react` expresses negative emotion (sad/angry) as a horizontal head **shake** ("no") — legible on a pan-only head — then settles back on the base. Positive emotion never wiggles the head.

**Camera** (`vision/camera.py`) — **capture + JPEG encode only, no ML**:
- Resolution 640x480; FPS configurable (`PHILOSOPHER_CAMERA_FPS`), rounded to 0.5 steps
- Encodes JPEG and streams it (`0x02`) to the server; **all face/emotion runs server-side** (`philosopher-server/philosopher/vision/`)

**Microphone** (`audio/capture.py`):
- Energy-based VAD; emits `speech_started`/`speech_ended` and streams raw PCM (`0x01`) to the server
- **No STT on the toy** — `faster-whisper` runs server-side

**Speaker** (`audio/player.py`):
- Receives WAV (`0x03`) from the server and plays it via PyAudio
- **No TTS on the toy** — Piper runs server-side (`audio/tts.py` is legacy v1)

### 3.5 Testing

```bash
cd philosopher-toy
pytest tests/ -v
```

- Toy `tests/`: `test_protocol`, `test_ws_client`, `test_audio`, `test_camera`, `test_hardware`
- All tests run in mock mode (no hardware required)

---

## 4. Dependencies

### philosopher-server

```toml
dependencies = [
    "langgraph>=0.2.0",
    "langchain-core>=0.3.0",
    "openai>=1.30.0",
    "httpx>=0.27.0",
    "aiohttp>=3.9.0",
    "aiosqlite>=0.20.0",
    "numpy>=1.26.0",
    "python-dotenv>=1.0.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "pyyaml>=6.0.1",
    "structlog>=24.1.0",
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "websockets>=12.0",
    "faster-whisper>=1.0.0",   # STT
    "onnxruntime>=1.17.0",     # FER+ emotion (replaced deepface/TensorFlow)
    "face-recognition>=1.3.0", # identity
    "opencv-python>=4.9.0",    # JPEG decode
    "Pillow>=10.0.0",
]
```

Dev: `pytest`, `pytest-asyncio`, `pytest-cov`, `pytest-mock`. Also needs the **`piper`** system binary for TTS.

### philosopher-toy

```toml
dependencies = [
    "aiohttp>=3.9.0",        # WebSocket client
    "numpy>=1.26.0",         # VAD energy math
    "opencv-python>=4.9.0",  # camera capture + JPEG encode
    "pyaudio>=0.2.14",       # mic capture + WAV playback
]
```

Dev: `pytest`, `pytest-asyncio`, `pytest-mock`. (No edge-tts / speechrecognition / deepface — all moved server-side in v2.)

---

## 5. Running the Projects

### philosopher-server

```bash
cd philosopher-server
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your LLM endpoint; install the `piper` binary for real TTS
python scripts/download_models.py       # pre-fetch FER+/Piper/Whisper (optional)

python -m philosopher.main --server     # WebSocket + HTTP server mode
python -m philosopher.main              # Interactive chat mode (text only)
```

CLI args: `--server`, `--host`, `--port`, `--personality`, `--language`, `--llm-url`, `--llm-model`, `--debug`

### philosopher-toy

```bash
cd philosopher-toy
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with the server URL (ws://...)

python -m banana_client.main

# Mock mode (no hardware):
PHILOSOPHER_MOCK=true python -m banana_client.main
```

---

## 6. Data Flow Example — v2 (WebSocket streaming)

```
User speaks to the toy
    |
    v  toy streams over WebSocket
[Toy] mic VAD -> speech_started, PCM chunks (0x01), speech_ended
[Toy] camera -> JPEG frames (0x02) at PHILOSOPHER_CAMERA_FPS
    |
    v
[Server] vision: decode JPEG -> face_recognition (identity) + FER+ ONNX (emotion)
         -> sends {type:servo, emotion} back to the toy immediately
[Server] stt: on speech_ended, faster-whisper transcribes accumulated PCM
[Server] _stream_response (split-graph):
           nodes 1-3 (perception/memory/prompt) build context
           llm.chat_stream() yields tokens -> buffered into sentences
           per sentence: node_format -> send {type:text} + {type:servo}
                         -> Piper synth -> send WAV (0x03)
           node_store once at the end (persists + background name extraction)
    |
    v  per sentence, as soon as ready
[Toy] plays WAV (0x03) via PyAudio; animates servos from {type:servo}
```

The toy never computes `face_id`/`emotion`/text itself — it only ships sensor frames and plays back what the server streams.

---

## 7. File Tree

```
philosopher-server/
├── pyproject.toml
├── .env.example
├── README.md
├── philosopher/
│   ├── __init__.py
│   ├── main.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   └── personalities/
│   │       ├── es/{filosofo,curioso,poetico,amigo,sabio}.yaml
│   │       └── en/philosopher.yaml
│   ├── core/
│   │   ├── __init__.py, state.py, nodes.py, graph.py, orchestrator.py
│   ├── llm/
│   │   ├── __init__.py, client.py
│   ├── memory/
│   │   ├── __init__.py, short_term.py, long_term.py, manager.py
│   ├── personality/
│   │   ├── __init__.py, engine.py
│   ├── api/
│   │   ├── __init__.py, app.py
│   ├── events/
│   │   ├── __init__.py, bus.py
│   └── utils/
│       ├── __init__.py, logger.py
└── tests/
    ├── conftest.py
    ├── unit/{test_config,test_memory,test_personality,test_state,test_events}.py
    └── integration/test_nodes.py

philosopher-toy/
├── pyproject.toml
├── .env.example
├── README.md
├── banana_client/
│   ├── __init__.py
│   ├── main.py
│   ├── protocol/
│   │   ├── __init__.py, client.py
│   ├── vision/
│   │   ├── __init__.py, camera.py
│   ├── audio/
│   │   ├── __init__.py, stt.py, tts.py
│   ├── hardware/
│   │   ├── __init__.py, servos.py
│   └── utils/
│       └── __init__.py
└── tests/
    ├── test_protocol.py
    └── test_hardware.py
```

---

## 8. Critical Implementation Details

### Pipeline execution (`core/graph.py`)

The 6-node pipeline is a LangGraph `StateGraph` run via `ainvoke`, with each node
registered as an **async** function (LangGraph awaits it on the caller's loop).
⚠️ Do **not** reintroduce a *sync* wrapper: the earlier version wrapped each node
in `asyncio.get_event_loop().run_until_complete()` inside a worker thread, which
raised *"no current event loop in thread"* and silently turned every HTTP `/chat`
reply into an empty error response (this path had no test). LangGraph is kept for
declarative edges (future conditional branches), state tracing, and LangSmith.

### Face Recognition — encoding distance (`philosopher-server/philosopher/vision/engine.py`)

Identity is matched by **Euclidean distance over the full 128-d `face_recognition` encoding** (default tolerance 0.6), which is robust to lighting/angle changes. Stored encodings live in `known_faces.encoding` (BLOB of float64 bytes); `_match_face` loads them and returns the closest match's `face_id` + name, so a returning person keeps the same identity. Only **new** faces mint a fresh id (`SHA256(full encoding)[:12]`) and get persisted via `memory.long.add_face()`.

Note: the comparison is pure numpy — dlib/`face_recognition` is only needed to *generate* encodings from the image (in `process_frame`), so matching is unit-testable without dlib. *(v1 used a fragile `SHA256(first 8 bytes)` hash as the identity — replaced.)*

### Memory Keyword Extraction (`memory/long_term.py:51`)

Keywords are naively extracted: words > 3 characters, lowercased, first 20 words only. No stemming, no stopword removal. Search scores by Jaccard-ish overlap (`|query_keywords ∩ memory_keywords| / |query_keywords|`) plus a recency bonus.

### TTS — v2 (server-side Piper, `philosopher-server/philosopher/tts/engine.py`)

The server synthesizes **one WAV per sentence** with Piper (subprocess) and streams it to the toy as a `0x03` binary frame; the toy just plays it. Piper needs the `piper` binary on PATH and a voice model (auto-downloaded to `./data/piper_models/`, or pinned via `PHILOSOPHER_TTS_MODEL_PATH`). If the binary or model is missing, `synthesize()` returns empty bytes — the toy gets text but no audio, never a crash. *(v1 used Edge-TTS + ffplay on the toy — gone.)*

### Error Handling in LLM Client (`llm/client.py:81`)

Network errors (`ConnectError`, `TimeoutException`) return `LLMResponse` with `finish_reason` set to error string. The `node_format` node detects non-`"stop"` finish reasons and returns a fallback message: *"Hmm... I did not quite understand. Could you repeat?"*

---

## 9. Known Limitations & Future Work

- **STT is offline & server-side**: `faster-whisper` on the RPi4 (no internet needed). ~~Google Speech Recognition~~ (v1).
- **No wake word**: Currently requires continuous listening / VAD. No "Hey Philosopher" detection.
- **No LED implementation**: `hardware/leds.py` does not exist yet.
- **Face registration is automatic but simple**: Background task extracts a name from 1–3 word replies after `node_store`. Complex introductions still need a manual `known_faces` insert.
- **Keyword memory is primitive**: No semantic embeddings. Could upgrade to `sentence-transformers` if RAM allows.
- **No dashboard**: No web UI for viewing memory stats or conversations.
- **TTS depends on the `piper` binary** on the *server* (not pip-installable; not on the toy). Without it the toy gets text but no audio.
- **Emotion = FER+ ONNX** (`vision/emotion.py`, run via `onnxruntime`). DeepFace/TensorFlow was removed (too heavy on ARM, contended Whisper for the 4 cores). The ~35MB model auto-downloads to `./data/fer_models/` on first use; set `PHILOSOPHER_CAMERA_EMOTION_MODEL_PATH` to pin it. Any load failure degrades to "neutral".

---

## 10. Lint & Format

- **philosopher-server**: Uses `ruff` (line-length 100, target py310)
- **philosopher-toy**: No linter configured yet
- Both use `pytest` with `asyncio_mode = auto`

---

## 11. Quick Reference

| Task | Command |
|------|---------|
| Install server | `cd philosopher-server && pip install -e ".[dev]"` |
| Install toy | `cd philosopher-toy && pip install -e ".[dev]"` |
| Pre-download server models | `cd philosopher-server && python scripts/download_models.py` |
| Run server API | `python -m philosopher.main --server` |
| Run server chat | `python -m philosopher.main` |
| Run toy | `python -m banana_client.main` |
| Run toy (mock) | `PHILOSOPHER_MOCK=true python -m banana_client.main` |
| Test server | `cd philosopher-server && pytest tests/ -v` |
| Test toy | `cd philosopher-toy && pytest tests/ -v` |
| Switch personality | POST `/personality?name=curioso` or CLI `--personality curioso` |
| View memory stats | GET `/memory/stats` |
