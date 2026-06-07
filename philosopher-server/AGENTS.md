# AGENTS.md — Philosopher Server

> Reference for AI agents working on the Philosopher server (the "brain").
> Last reviewed: 2026-06-06 (updated to **v2**)

> ### ⚠️ v2 — this server now does ALL the AI
> Runs on a **Raspberry Pi 4 (4GB)**. Added modules beyond the v1 prose below:
> - `stt/engine.py` — `faster-whisper` batch STT (transcribe at `speech_ended`).
> - `vision/engine.py` — `face_recognition` (identity) + `vision/emotion.py` FER+ ONNX (emotion, via `onnxruntime`) on JPEG frames. DeepFace removed.
> - `tts/engine.py` — **Piper** via subprocess (`piper` binary, not pip). One WAV per sentence.
> - `api/ws_server.py` — `WebSocketManager`; route is `/ws?toy_id=<id>`.
> - `core/orchestrator.py` — `_stream_response()` does **split-graph streaming** (nodes 1–3 → external LLM stream → per-sentence format+send → `node_store` once).
> The HTTP `/chat` path still runs the full 6-node graph and is kept for diagnostics. `MIGRATION_GUIDE.md` is authoritative; `/CLAUDE.md` has the quick reference.

---

## 1. Overview

The **philosopher-server** is the intelligence layer of the Philosopher conversational AI ecosystem. It runs on **any computer** with Python 3.10+ and connects to:
- An LLM provider (LM Studio, OpenAI, Ollama, etc.) via HTTP
- A toy client (Banana Pi) via REST API

```
LLM Computer <-> Philosopher Server <-> Toy Client (Banana Pi)
```

**Key Design Decisions:**
- Hardware-agnostic: no physical hardware dependencies
- LLM-agnostic: any OpenAI-compatible API endpoint
- Personalities are YAML-only: add languages by creating folders with YAML files
- Semantic memory without embeddings: keyword matching + recency scoring (optimized for limited RAM)

---

## 2. Architecture

### 2.1 LangGraph Pipeline

Every user message flows through 6 nodes in sequence:

```
[User Input] --> perception --> memory --> prompt --> think --> format --> store
                    |           |          |        |        |        |
                    v           v          v        v        v        v
               Lookup face  Retrieve   Build     LLM     Apply   Save to
               in SQLite    context    system    gen     style   memory
                          (short+long)  prompt   text           (RAM+SQLite)
```

| Node | File | Purpose |
|------|------|---------|
| perception | `core/nodes.py:17` | Looks up `face_id` in SQLite `known_faces`, sets `face_name` and `is_new_face` |
| memory | `core/nodes.py:27` | Retrieves short-term context (last 10 messages) + long-term memories (keyword search, top 5) |
| prompt | `core/nodes.py:37` | Builds system prompt = personality YAML + face name + emotion + relevant memories |
| think | `core/nodes.py:47` | Calls LLM with prompt + history. Max 250 tokens per response |
| format | `core/nodes.py:63` | Applies personality style (max length, cleans markdown) |
| store | `core/nodes.py:72` | Saves interaction to short-term buffer (deque) and long-term SQLite |

### 2.2 Module Reference

| Module | Files | Responsibility |
|--------|-------|--------------|
| `config/` | `settings.py`, `personalities/<lang>/*.yaml` | Pydantic Settings + personality prompts |
| `core/` | `state.py`, `nodes.py`, `graph.py`, `orchestrator.py` | LangGraph pipeline + state machine |
| `llm/` | `client.py` | OpenAI-compatible async client (incl. `chat_stream()` for v2 streaming) |
| `memory/` | `short_term.py`, `long_term.py`, `manager.py` | **Per-face** short-term deques + SQLite semantic store |
| `personality/` | `engine.py` | YAML loader + prompt builder + formatter + localized `fallback()` |
| `stt/` | `engine.py` | **v2** — `faster-whisper` batch STT |
| `vision/` | `engine.py`, `emotion.py` | **v2** — `face_recognition` (identity) + FER+ ONNX (emotion) on JPEG frames |
| `tts/` | `engine.py` | **v2** — Piper TTS via subprocess |
| `api/` | `app.py`, `ws_server.py` | FastAPI REST (`/sessions` incl.) + **WebSocket** (`/ws`) |
| `events/` | `bus.py`, `session_tracker.py` | Pub/sub bus **wired** to `SessionTracker` (per-person stats → `/sessions`) |
| `utils/` | `logger.py` | structlog + rich logging |

---

## 3. Configuration

Settings are loaded from `.env` and validated via Pydantic (`config/settings.py`).

```bash
# LLM Provider
PHILOSOPHER_LLM_PROVIDER=local          # local | openai | ollama | custom
PHILOSOPHER_LLM_BASE_URL=http://192.168.1.100:1234/v1
PHILOSOPHER_LLM_API_KEY=lm-studio
PHILOSOPHER_LLM_MODEL=llama-3.1-8b
PHILOSOPHER_LLM_MAX_TOKENS=4096
PHILOSOPHER_LLM_TEMPERATURE=0.7         # clamped to [0.0, 2.0]
PHILOSOPHER_LLM_TIMEOUT=30

# Personality
PHILOSOPHER_PERSONALITY=filosofo        # YAML filename without extension
PHILOSOPHER_NAME=Philosopher
PHILOSOPHER_LANGUAGE=es                 # selects subfolder in personalities/

# Memory
PHILOSOPHER_MEMORY_DB=sqlite            # sqlite | postgres
PHILOSOPHER_MEMORY_DB_PATH=./data/philosopher_memory.db
PHILOSOPHER_MEMORY_SHORT_TERM_LIMIT=10
PHILOSOPHER_MEMORY_LONG_TERM_TOP_K=5
PHILOSOPHER_MEMORY_SIMILARITY_THRESHOLD=0.6

# API Server
PHILOSOPHER_API_ENABLED=true
PHILOSOPHER_API_HOST=0.0.0.0
PHILOSOPHER_API_PORT=8080

# Logging
PHILOSOPHER_LOG_LEVEL=INFO              # DEBUG | INFO | WARNING | ERROR
PHILOSOPHER_LOG_FORMAT=rich             # json | console | rich
```

---

## 4. API Endpoints

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET | `/` | — | `{name, version}` |
| GET | `/health` | — | `{status, llm, memory}` |
| POST | `/chat` | `{message, face_id?, face_name?, emotion?}` | `{response, emotion, turn}` |
| GET | `/personalities` | — | `["filosofo", "curioso", ...]` |
| POST | `/personality` | `?name=filosofo` | `{personality}` |
| GET | `/memory/stats` | — | `{total_memories, known_faces, by_type}` |
| GET | `/sessions` | — | per-person stats from `SessionTracker` (messages, responses, emotions, last_seen) |
| GET | `/dashboard` | — | self-contained HTML dashboard page (`events/dashboard.py`) |
| GET | `/dashboard/stream` | — | SSE (`text/event-stream`) live event feed via `DashboardHub` |
| WS | `/ws?toy_id=<id>` | binary `0x01` audio / `0x02` JPEG + JSON control | streams `0x03` WAV + `{type:text\|servo\|look\|react}` |

---

## 5. Memory Architecture

### 5.1 Short-term Memory (`memory/short_term.py`)
- `deque` circular buffer in RAM, max 10 messages
- Lost on restart
- Stores `role`, `content`, `timestamp`, `metadata`

### 5.2 Long-term Memory (`memory/long_term.py`)
- SQLite with WAL mode
- Tables: `memories`, `known_faces`
- **Keyword extraction**: words >3 characters, lowercased, first 20 words only. No stemming, no stopword removal.
- **Search scoring**: Jaccard-ish overlap (`|query_keywords ∩ memory_keywords| / |query_keywords|`) plus recency bonus (1 week half-life)
- `similarity_threshold` default: 0.6

---

## 6. Personalities (YAML)

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

---

## 7. Event Bus (`events/bus.py`)

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

---

## 8. Critical Implementation Details

### Pipeline execution (`core/graph.py`)

The 6-node pipeline is a LangGraph `StateGraph` run via `ainvoke`, with each node
an **async** function (awaited on the caller's loop). ⚠️ Do **not** reintroduce a
*sync* wrapper: the earlier `run_until_complete()`-in-a-worker-thread version
raised *"no current event loop in thread"* and silently made every HTTP `/chat`
reply empty (this path had no test). LangGraph is kept for declarative edges,
state tracing, and LangSmith.

### Error Handling in LLM Client (`llm/client.py:81`)

Network errors (`ConnectError`, `TimeoutException`) return `LLMResponse` with `finish_reason` set to error string. The `node_format` node detects non-`"stop"` finish reasons and returns a fallback message: *"Hmm... I did not quite understand. Could you repeat?"*

### Memory Keyword Extraction (`memory/long_term.py:51`)

Keywords are naively extracted: words > 3 characters, lowercased, first 20 words only. No stemming, no stopword removal.

---

## 9. Testing

```bash
cd philosopher-server
pytest tests/ -v
```

- 28 tests total: 22 unit + 6 integration
- Coverage: config, memory, personality, state, events, LangGraph nodes
- Uses `pytest-asyncio` with `asyncio_mode = auto`
- Fixtures in `tests/conftest.py` provide mock settings + temp DB + event bus

---

## 10. Dependencies

```toml
dependencies = [
    "langgraph>=0.2.0", "langchain-core>=0.3.0", "openai>=1.30.0",
    "httpx>=0.27.0", "aiohttp>=3.9.0", "aiosqlite>=0.20.0",
    "numpy>=1.26.0", "python-dotenv>=1.0.0", "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0", "pyyaml>=6.0.1", "structlog>=24.1.0",
    "fastapi>=0.109.0", "uvicorn>=0.27.0", "websockets>=12.0",
]
```

Dev: `pytest`, `pytest-asyncio`, `pytest-cov`, `pytest-mock`

Lint: `ruff` (line-length 100, target py310)

---

## 11. Quick Reference

| Task | Command |
|------|---------|
| Install | `cd philosopher-server && pip install -e ".[dev]"` |
| Run API server | `python -m philosopher.main --server` |
| Run interactive chat | `python -m philosopher.main` |
| Test | `pytest tests/ -v` |
| Switch personality | POST `/personality?name=curioso` or CLI `--personality curioso` |
| View memory stats | GET `/memory/stats` |

---

## 12. Known Limitations & Future Work

- **Keyword memory is primitive**: No semantic embeddings. Could upgrade to `sentence-transformers` if RAM allows.
- **No dashboard**: No web UI for viewing memory stats or conversations.
- **Face registration is automatic but simple**: Background task extracts a name from 1–3 word replies; complex introductions still need a manual `known_faces` insert.
- **Emotion = FER+ ONNX** (`vision/emotion.py`, `onnxruntime`). DeepFace/TensorFlow was removed (too heavy on ARM, contended Whisper for cores). Model auto-downloads to `./data/fer_models/`; pin via `PHILOSOPHER_CAMERA_EMOTION_MODEL_PATH`; failures degrade to "neutral".
