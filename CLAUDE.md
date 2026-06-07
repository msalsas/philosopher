# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Big picture (this is v2 — WebSocket streaming, all AI server-side)

Philosopher is a conversational AI agent (personality + dual memory + face/emotion awareness) embedded in a smart plush toy. It is **two independent Python projects** (each its own `pyproject.toml`, deps, tests) plus an external LLM:

- **`philosopher-server/`** — the brain. Runs on a **Raspberry Pi 4 (4GB)**. Does **all** the compute: STT, vision, TTS, LangGraph pipeline, memory, personality, and proxies an external LLM. Exposes WebSocket (for the toy) + HTTP (for diagnostics).
- **`philosopher-toy/`** — the body. Runs on a **Banana Pi BPi-M2 Zero (512MB)** inside the toy. A **dumb I/O node with zero ML**: captures mic+camera, streams to the server over WebSocket, plays back WAV audio, moves servos. Its 512MB budget is why no inference runs here.
- **External LLM** — any OpenAI-compatible endpoint (LM Studio / OpenAI / Ollama), ~8B model. The server only streams tokens from it; no local LLM.

```
External LLM  ──HTTP──▶  philosopher-server (RPi4, the brain)  ◀──WebSocket──▶  philosopher-toy (Banana Pi, I/O only)
                         STT · vision · TTS · LangGraph · memory          mic · camera · speaker · servos
```

### Server-side AI modules (the heavy lifting, all here)
- `stt/engine.py` — `faster-whisper` batch STT. Toy VAD sends `speech_started`/`speech_ended`; server accumulates PCM chunks and transcribes once at `speech_ended` (no server-side VAD).
- `vision/engine.py` — `face_recognition` (dlib) for identity + `vision/emotion.py` (FER+ ONNX via `onnxruntime`) for emotion, on JPEG frames decoded with OpenCV. (DeepFace/TensorFlow was removed — too heavy on ARM and contended Whisper for cores.)
- `tts/engine.py` — **Piper**, invoked as a `subprocess` (`piper` binary, NOT a pip package). Synthesizes one WAV per sentence, sent to the toy as a binary frame.
- `core/orchestrator.py` — the WebSocket request path. See "split-graph streaming" below.

### Two execution paths through the pipeline (important)
1. **HTTP `/chat`** (`api/app.py` → `orchestrator.process`) runs the full 6-node LangGraph graph (`perception → memory → prompt → think → format → store`). Kept for diagnostics/fallback.
2. **WebSocket** (`orchestrator._stream_response`) uses **split-graph streaming**: runs nodes 1–3 (perception/memory/prompt) for context, then calls `llm.chat_stream()` directly, buffers tokens into sentences, runs `node_format` *per sentence* and ships text+servo+WAV to the toy immediately, then runs `node_store` **once** at the end. This is the real-time path the toy uses — don't assume `/chat` and the WS path share code beyond nodes 1–3 and format/store. The latest face/emotion seen on a toy (`handle_video_frame` stores `orchestrator.last_vision[toy_id]`) is injected into the streamed reply's `AgentState`, so perception resolves a name and the prompt reflects the user's emotion.

### WebSocket protocol (`api/ws_server.py`)
`ws://<server>:<port>/ws?toy_id=<id>`. Binary frames are `type_byte + payload`:
- Toy→server: `0x01` PCM audio (16kHz mono), `0x02` JPEG frame.
- Server→toy: `0x03` WAV audio.
- JSON control both ways: `speech_started`, `speech_ended`, `ping`/`pong`, and server→toy `{type:text|servo|look|error}`.

**Body language — the head only pans (horizontal), so it is split by purpose:**
- **Head = gaze.** Per video frame the server sends `{type:"look", offset_x}` (face's horizontal position −1..+1); `ServoController.look()` sets the head's **base** position to track the face. Pan-only at camera FPS (~1–2 s loop) → "turns to look at you", not smooth tracking. After `RECENTER_AFTER` (4 s) with no face the server sends `look(0)` to recenter. A slow idle drift (`idle_loop`, ±3°) keeps it from being a frozen statue.
- **Emotion = arms (mostly).** Conversational gestures (`{type:"servo", emotion}`, during replies → `animate`) and brief ambient reactions on emotion *change* (`{type:"react", emotion}` → `react`) are **arm** gestures; the head stays on its gaze base. Turning the head away to "act" an emotion would break eye contact.
- **Exception — negative emotion uses a head shake.** `react` expresses sad/angry as a horizontal head **shake** ("no"), which reads correctly on a pan-only head; positive emotions never wiggle the head (it would look like negation). The shake settles back on the gaze base.

## Commands

```bash
# Server (the brain — RPi4). Needs the `piper` binary on PATH for real TTS.
cd philosopher-server && pip install -e ".[dev]"
python scripts/download_models.py       # pre-fetch FER+/Piper/Whisper models (optional)
python -m philosopher.main --server     # WebSocket + HTTP (default :8080) — production mode
python -m philosopher.main              # interactive terminal chat (text only, no STT/TTS/vision)
MOCK_MODE=true python -m philosopher.main --server   # stub STT/vision/TTS, no models needed
pytest tests/ -v                        # full suite
pytest tests/unit/test_stt.py -v        # single file
pytest tests/unit/test_stt.py::test_name -v          # single test
ruff check .                            # lint (line-length 100, py310; rules: E,F,I,W,N,UP,B,C4,SIM,PTH)

# Toy (the body — Banana Pi)
cd philosopher-toy && pip install -e ".[dev]"
python -m banana_client.main
PHILOSOPHER_MOCK=true python -m banana_client.main   # no hardware needed
pytest tests/ -v
```

`MOCK_MODE` (server) / `PHILOSOPHER_MOCK` (toy) let everything run without models or hardware — use them for dev. Both projects are also installed as console scripts (`philosopher-server`, `philosopher-toy`).

## Configuration

Server settings are `PHILOSOPHER_*` env vars (from `.env`), validated by nested Pydantic models in `philosopher-server/philosopher/config/settings.py`. Note the prefixes don't all match their section name:
- LLM → `PHILOSOPHER_LLM_*`, Memory → `PHILOSOPHER_MEMORY_*`, API → `PHILOSOPHER_API_*`
- STT → `PHILOSOPHER_STT_*` (`MODEL=tiny|base|small`, `DEVICE`, `CONFIDENCE_THRESHOLD`)
- **Vision/camera → `PHILOSOPHER_CAMERA_*`** (`FPS`, `QUALITY`) — the class is `VisionSettings` but the env prefix is `CAMERA`.
- TTS → `PHILOSOPHER_TTS_*` (`VOICE`, `MODEL_PATH`, `ENABLED`)
- Personality/logging use the bare `PHILOSOPHER_` prefix (`PHILOSOPHER_PERSONALITY`, `PHILOSOPHER_LANGUAGE`, `PHILOSOPHER_LOG_LEVEL`, …).

`temperature` is validated to `[0.0, 2.0]`. `get_settings()` is `lru_cache`d and creates `./data/` on first call.

## Conventions that aren't obvious from the code

- **Personalities are YAML-only, no code.** Add one by dropping a YAML in `philosopher-server/philosopher/config/personalities/<lang>/`; add a language by creating a new `<lang>` folder. `PHILOSOPHER_LANGUAGE` picks the folder.
- **Memory has no embeddings by design.** Long-term recall (`memory/long_term.py`, SQLite + WAL) uses naive keyword overlap + recency decay, chosen for RAM-constrained hardware. Don't reach for `sentence-transformers` without weighing the RAM cost.
- **The pipeline is a LangGraph `StateGraph` of 6 async nodes** (`core/graph.py`), run via `ainvoke`. Nodes are registered as **async** functions, so LangGraph awaits them on the caller's loop. ⚠️ Don't reintroduce a *sync* wrapper: the earlier version wrapped each node in `get_event_loop().run_until_complete()` inside a worker thread → "no current event loop in thread" → every HTTP `/chat` reply was silently empty (and untested). LangGraph supports async nodes natively. Kept for declarative edges (future branches), state tracing, and LangSmith.
- **Face registration is zero-latency / off the critical path.** `node_store` spawns a background task that regex-extracts a name from short (1–3 word) replies. No LLM call on the response path.
- **Servos move one at a time, sequentially.** The 512MB toy's MicroUSB power rail browns out if multiple servos start together — `hardware/servos.py` queues moves with delays. Don't parallelize servo motion.
- **TTS is a binary, not a package.** Real audio requires the `piper` executable installed on the RPi; without it, `tts/engine.py` returns empty bytes and the toy gets text but no speech.
- **Multi-person is first-class.** Several people use one toy. Both long-term *and* short-term memory are keyed by `face_id` (`ShortTermMemory` has one deque per person; unknown → `_anon` bucket) so one person's recent turns never leak into another's context. Face identity is by encoding distance (robust). **Limitation:** the spoken turn is attributed to the **dominant face** at `speech_ended` (no voice diarization).
- **Both reply paths share assembly + fallbacks.** HTTP `node_think` and WS `_stream_response` both call `build_messages(state)` (incl. new-face introduction / name-ask) so they can't drift. Fallbacks are **localized** via the personality YAML (`fallbacks:` section → `personality.fallback(key)`), not hardcoded English.
- **LLM stream errors are never spoken.** `chat_stream` lets exceptions propagate; `_stream_response` catches them and, if nothing usable was produced, sends one localized fallback (`personality.fallback("error")`) — it does **not** yield `[error]` text into the spoken stream.
- **The mic is half-duplex (anti-echo).** `MicrophoneCapture` takes a `gate` callable; the toy gates it on `AudioPlayer.is_playing` (+~300 ms tail) so it never transcribes its own TTS. No echo cancellation needed.
- **The event bus is wired, with a live dashboard.** `events/bus.py` has two consumers: `SessionTracker` (per-person stats → `GET /sessions`) and `DashboardHub` (SSE fan-out). Orchestrator emits `FACE_RECOGNIZED`/`EMOTION_DETECTED`/`USER_TEXT`/`RESPONSE_READY`. The dashboard is a self-contained page at **`GET /dashboard`** (HTML in `events/dashboard.py`) that polls `/sessions` and live-tails **`GET /dashboard/stream`** (text/event-stream). Add a new subscriber to extend it. *(Note: SSE can't be tested via Starlette `TestClient` — `iter_lines` blocks on the infinite stream; the `DashboardHub` fan-out is unit-tested directly instead.)*

## Models & deployment (Raspberry Pi 4)

The server uses three ML models; all **auto-download on first use** and can be **pre-fetched** to avoid a network hit during the first conversation:

| Model | Lib | Lands in | Pin with |
|-------|-----|----------|----------|
| FER+ emotion (~35 MB) | `onnxruntime` | `./data/fer_models/` | `PHILOSOPHER_CAMERA_EMOTION_MODEL_PATH` |
| Piper voice (~28 MB + json) | `piper` binary | `./data/piper_models/` | `PHILOSOPHER_TTS_MODEL_PATH` |
| faster-whisper (`tiny`…) | `faster-whisper` | HF cache | (size via `PHILOSOPHER_STT_MODEL`) |

Pre-download all three: `python scripts/download_models.py` (idempotent; each step independent).

ARM64 install notes:
- **`face_recognition`/dlib has no PyPI ARM64 wheel.** On Raspberry Pi OS install from piwheels first — `pip install dlib --index-url https://www.piwheels.org/simple` — before falling back to a 2–4 h source build. MediaPipe face detection is a no-dlib fallback if you only need presence, not identity.
- **`piper` is a system binary**, not a pip package — install it separately on the Pi.

## Gotchas (learned the hard way)

- **Everything degrades to a safe default; it does not crash the socket.** Emotion → `"neutral"` if onnxruntime/model missing; TTS → empty bytes (text, no audio) if the `piper` binary/model missing; STT → a mock `"Hello world"` if `faster-whisper` failed to load. So "it runs but does nothing" usually means a missing model/binary, not a logic bug — check those first.
- **`faster-whisper` returns a one-shot generator.** Materialize `segments` to a list before using it twice (`stt/engine.py`); reading it for `text` and again for `confidence` silently yields nothing the second time (and `min([])` raises).
- **Tests must isolate the SQLite DB.** The default DB (`./data/philosopher_memory.db`) is WAL-mode and shared; tests hitting it concurrently block on the lock and *look* like a hang. Use a `tmp_path` DB via `monkeypatch.setenv("PHILOSOPHER_MEMORY_DB_PATH", …)` + `get_settings.cache_clear()` (see `tests/integration/test_stream.py`).
- **The WS sentence splitter is `buffer.endswith(('.','!','?','\n'))`.** A token whose terminator is followed by whitespace (e.g. `". "`) won't split until the next terminator, so sentences can batch. Fine for real LLM streams; matters when writing fake token streams in tests.

## Doc map — what's authoritative

The prose docs were written for **v1** and have since been corrected with a v2 banner at the top, but the body text may still describe v1 in places (HTTP-only, AI on the toy, Edge-TTS, "28 + 3" test counts). Order of authority:

1. **Source code** (`philosopher/`, `banana_client/`) — ground truth.
2. **`MIGRATION_GUIDE.md`** — authoritative v2 design (explicitly overrides `REWRITE_PROMPT.md`).
3. **This `CLAUDE.md`** — up-to-date quick reference.
4. **`AGENTS.md` / `CONTEXT.md` / `README.md`** — useful for the unchanged parts (memory, personality, event bus, LangGraph), but verify transport/where-AI-runs against the code.
