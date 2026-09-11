# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Big picture (this is v2 — WebSocket streaming, all AI server-side)

Philosopher is a conversational AI agent (personality + dual memory + face/emotion awareness) embedded in a smart plush toy. It is **two independent Python projects** (each its own `pyproject.toml`, deps, tests) plus an external LLM:

- **`philosopher-server/`** — the brain. Runs on a host with real CPU. Does **all** the compute: STT, vision, TTS, LangGraph pipeline, memory, personality, and proxies an external LLM. Exposes WebSocket (for the toy) + HTTP (for diagnostics). *(Designed for a Raspberry Pi 4, but in practice it runs on a **laptop** — the RPi4's response times were too slow. Start it with `run.sh`.)*
- **`philosopher-toy/`** — the body. Runs on a **Raspberry Pi Zero WH (512MB)** inside the toy. A **dumb I/O node with zero ML**: captures mic+camera, streams to the server over WebSocket, plays back WAV audio, moves servos. Its 512MB budget is why no inference runs here.
- **External LLM** — any OpenAI-compatible endpoint (Ollama, llama.cpp `server`, vLLM, LM Studio, OpenAI — pick any), ~8B model. The server only streams tokens from it; no local LLM.

```
External LLM  ──HTTP──▶  philosopher-server (laptop, the brain)  ◀──WebSocket──▶  philosopher-toy (Pi Zero WH, I/O only)
                         STT · vision · TTS · LangGraph · memory          mic · camera · speaker · servos
```

### Server-side AI modules (the heavy lifting, all here)
- `stt/engine.py` — `faster-whisper` batch STT. Toy VAD sends `speech_started`/`speech_ended`; server accumulates PCM chunks and transcribes once at `speech_ended` (no server-side VAD).
- `vision/engine.py` — `face_recognition` (dlib) for identity + `vision/emotion.py` (FER+ ONNX via `onnxruntime`) for emotion, on JPEG frames decoded with OpenCV. (DeepFace/TensorFlow was removed — too heavy on ARM and contended Whisper for cores.) **Three cheap gates run before the expensive dlib path**, designed to *fail open* (run the full path on doubt) to avoid blinding the toy — but see the ⚠️ Haar caveat below: (1) static-frame skip (reuse last result when the frame barely changed), (2) downscale to `detect_width` for detection (boxes scaled back; `offset_x` is normalized so scale-invariant), (3) an OpenCV **Haar presence check** — no face → dlib is skipped entirely (the common "nobody around" frame). ⚠️ **Turn this gate OFF (`PHILOSOPHER_CAMERA_PRESENCE_GATE=false`) for the shipped NoIR OV5647** — Haar can't detect its blurry/IR-tinted frames and wrongly reports "no face", skipping dlib and blinding recognition; dlib alone handles those frames fine (verified on-device). All tunable via `PHILOSOPHER_CAMERA_*` (`DETECT_WIDTH`, `PRESENCE_GATE`, `SKIP_SIMILAR`, `SKIP_THRESHOLD`). The dlib pipeline lives in `_recognize()`.
- `tts/` — **provider-selectable TTS** via `PHILOSOPHER_TTS_PROVIDER` (all env-driven, one WAV per sentence sent to the toy as a binary frame): **kokoro** (`kokoro_engine.py`, **default**; local/offline neural via kokoro-onnx — no torch — the natural voice we shipped), **piper** (`engine.py`; the `piper` binary kept **resident** in `--json-input` mode so the voice model loads once), **edge** (`edge_engine.py`, Microsoft online neural voices via edge-tts+ffmpeg — most natural but needs internet). The **voice + kokoro lang follow `PHILOSOPHER_LANGUAGE`** (es→`em_santa`/`es`, en→`am_adam`/`en-us`) unless `PHILOSOPHER_TTS_VOICE`/`_LANG` are set — so language is a single knob. All share a post pitch/formant shift (`PHILOSOPHER_TTS_PITCH`, <1 = deeper). The WS path synthesizes **per sentence for progressive playback**.
- `core/orchestrator.py` — the WebSocket request path. See "split-graph streaming" below.

### Two execution paths through the pipeline (important)
1. **HTTP `/chat`** (`api/app.py` → `orchestrator.process`) runs the full 6-node LangGraph graph (`perception → memory → prompt → think → format → store`). Kept for diagnostics/fallback.
2. **WebSocket** (`orchestrator._stream_response`) uses **split-graph streaming**: runs nodes 1–3 (perception/memory/prompt) for context, then calls `llm.chat_stream()` directly, buffers tokens into sentences, runs `node_format` *per sentence* and ships text+servo+WAV to the toy immediately, then runs `node_store` **once** at the end. This is the real-time path the toy uses — don't assume `/chat` and the WS path share code beyond nodes 1–3 and format/store. The latest face/emotion seen on a toy (`handle_video_frame` stores `orchestrator.last_vision[toy_id]`) is injected into the streamed reply's `AgentState`, so perception resolves a name and the prompt reflects the user's emotion.

### WebSocket protocol (`api/ws_server.py`)
`ws://<server>:<port>/ws?toy_id=<id>`. Binary frames are `type_byte + payload`:
- Toy→server: `0x01` PCM audio (16kHz mono), `0x02` JPEG frame.
- Server→toy: `0x03` WAV audio.
- JSON control both ways: `speech_started`, `speech_ended`, `ping`/`pong`, and server→toy `{type:text|servo|look|error}`.

**Body language (the head servo is on the Pi Zero's BCM12 via *hardware* PWM — PWM0 — for smooth, jitter-free motion; the signal is *released* at rest, `duty 0`, so a held servo never draws current and browns out the 5V rail alongside the audio amp):**
- **Gaze (`look`) barely tracks in the shipped setup.** The server sends `{type:"look", offset_x}` per video frame and `ServoController.look()` pans the head toward the face (capped at `MAX_PAN`, small). But the toy's default camera mode is **on-speech** (one frame per utterance), so `look` fires ~once when you talk — a glance toward you, not continuous tracking — and, released at rest, it doesn't hold on you. Real continuous face-follow needs `stream` camera mode **and** a held servo, i.e. more power headroom (a powered USB hub). What you actually see is the idle "look around" (`idle_loop`, ±`IDLE_DRIFT`, alternating sides, spaced 12–30 s).
- **Emotion gestures are arm poses, and arms are no-ops on this rig.** `{type:"servo"}`→`animate` and `{type:"react"}`→`react` drive the arms (disabled here) or a head shake; **`PHILOSOPHER_SERVO_GAZE_ONLY`** (on for the plush) skips them entirely, since a gesture during the reply coincides with the amp and browns out the board. So on the current build the head only does gaze glances + idle drift.

## Commands

```bash
# Server (the brain — runs on a laptop). Needs the `piper` binary on PATH for real TTS.
cd philosopher-server && pip install -e ".[dev]"
python scripts/download_models.py       # pre-fetch FER+/Piper/Whisper models (optional)
python scripts/check_runtime_deps.py    # readiness gate: dlib, piper binary, haar cascade, models (exit≠0 if missing)
python scripts/validate_hardware.py --faces DIR --emotions DIR --audio DIR --llm --tts  # measure REAL face/emotion accuracy + end-to-end latency on the Pi (exit≠0 if a target is missed)
python -m philosopher.main --server     # WebSocket + HTTP (default :8080) — production mode
python -m philosopher.main              # interactive terminal chat (text only, no STT/TTS/vision)
MOCK_MODE=true python -m philosopher.main --server   # stub STT/vision/TTS, no models needed
python scripts/fake_toy.py              # drive the WebSocket end-to-end with no hardware (sends frame+audio, prints text/servo/WAV); --wav/--jpeg for real input
pytest tests/ -v                        # full suite
pytest tests/unit/test_stt.py -v        # single file
pytest tests/unit/test_stt.py::test_name -v          # single test
ruff check .                            # lint (line-length 100, py310; rules: E,F,I,W,N,UP,B,C4,SIM,PTH)

# Toy (the body — Raspberry Pi Zero WH)
cd philosopher-toy && pip install -e ".[dev]"
python -m toy_client.main
PHILOSOPHER_MOCK=true python -m toy_client.main   # no hardware needed
pytest tests/ -v
```

`MOCK_MODE` (server) / `PHILOSOPHER_MOCK` (toy) let everything run without models or hardware — use them for dev. Both projects are also installed as console scripts (`philosopher-server`, `philosopher-toy`).

## Configuration

Server settings are `PHILOSOPHER_*` env vars (from `.env`), validated by nested Pydantic models in `philosopher-server/philosopher/config/settings.py`. Note the prefixes don't all match their section name:
- LLM → `PHILOSOPHER_LLM_*`, Memory → `PHILOSOPHER_MEMORY_*`, API → `PHILOSOPHER_API_*`
- STT → `PHILOSOPHER_STT_*` (`MODEL=tiny|base|small`, `DEVICE`, `COMPUTE_TYPE`, `CONFIDENCE_THRESHOLD`). `COMPUTE_TYPE` is decoupled from `DEVICE`: `int8` for CPU (the laptop), `float16` on a CUDA GPU.
- **Vision/camera → `PHILOSOPHER_CAMERA_*`** (`FPS`, `QUALITY`) — the class is `VisionSettings` but the env prefix is `CAMERA`.
- TTS → `PHILOSOPHER_TTS_*` (`VOICE`, `MODEL_PATH`, `ENABLED`)
- Personality/logging use the bare `PHILOSOPHER_` prefix (`PHILOSOPHER_PERSONALITY`, `PHILOSOPHER_LANGUAGE`, `PHILOSOPHER_LOG_LEVEL`, …).

`temperature` is validated to `[0.0, 2.0]`. `get_settings()` is `lru_cache`d and creates `./data/` on first call.

## Conventions that aren't obvious from the code

- **Personalities are YAML-only, no code.** Add one by dropping a YAML in `philosopher-server/philosopher/config/personalities/<lang>/`; add a language by creating a new `<lang>` folder. `PHILOSOPHER_LANGUAGE` picks the folder.
- **Memory has no embeddings by design.** Long-term recall (`memory/long_term.py`, SQLite + WAL) uses naive keyword overlap + recency decay, chosen for RAM-constrained hardware. Don't reach for `sentence-transformers` without weighing the RAM cost.
- **The pipeline is a LangGraph `StateGraph` of 6 async nodes** (`core/graph.py`), run via `ainvoke`. Nodes are registered as **async** functions, so LangGraph awaits them on the caller's loop. ⚠️ Don't reintroduce a *sync* wrapper: the earlier version wrapped each node in `get_event_loop().run_until_complete()` inside a worker thread → "no current event loop in thread" → every HTTP `/chat` reply was silently empty (and untested). LangGraph supports async nodes natively. Kept for declarative edges (future branches), state tracing, and LangSmith.
- **Face registration is off the critical path.** `node_learn_name` is the last pipeline node (`store → learn_name → END`): it asks the LLM for the speaker's own name and stores it against the current face. On the WS path it runs **fire-and-forget after the reply is sent**, so it adds no response latency; on HTTP `/chat` it runs inline as the final graph node.
- **The head servo is wired and works; it runs on *hardware* PWM and *releases* at rest.** Head = BCM12 (PWM0) driven via sysfs hardware PWM (`/sys/class/pwm/pwmchip0/pwm0`) — smooth, no software-PWM tremor; needs `dtoverlay=pwm,pin=12,func=4` + `dtparam=audio=off` in the Pi's config.txt (the toy exports/permits the channel at startup via passwordless `sudo -n`). Every move ramps then sets `duty 0`, because a **held** servo draws enough current to brown out the 5V rail alongside the audio amp (the whole "reboots mid-conversation" saga). Arm servos (BCM13/18) exist in the pin map but are disabled here (no HW-PWM channel) — `animate` on them is a no-op. Pins are per-servo env overridable (`PHILOSOPHER_SERVO_*_PIN`, 0 = off). See `HARDWARE_RUNBOOK.md`.
- **TTS backend depends on the provider.** `kokoro` (default) needs the `.[tts-kokoro]` extra + system `espeak-ng` + its model (`download_models.py`); `piper` is a **system binary** on PATH (not pip); `edge` needs `.[tts-edge]` + system `ffmpeg` + internet. Any provider returning empty bytes (missing binary/model/network) degrades to text-only — the toy shows text but stays silent.
- **Multi-person is first-class.** Several people use one toy. Both long-term *and* short-term memory are keyed by `face_id` (`ShortTermMemory` has one deque per person; unknown → `_anon` bucket) so one person's recent turns never leak into another's context. Face identity is by encoding distance (robust). **Limitation:** the spoken turn is attributed to the **dominant face** at `speech_ended` (no voice diarization).
- **Identity is the face; the name is just a label on it — and a name is NOT a unique key** (two people can be "Pedro"). When a freshly-minted face gets a name (`node_learn_name`) that matches an existing record, it is folded in **only if the biometrics agree**: `VisionProcessor.maybe_merge` merges into the known face only when their encodings are within `merge_band` (`PHILOSOPHER_CAMERA_MERGE_BAND`, default 0.68, must be ≥ `tolerance`) — "same person whose face drifted". A genuine namesake (encodings far apart) stays a separate record. The merge reconciles **both** the DB (`LongTermMemory.merge_face`: reassign memories, refresh encoding, delete the duplicate) **and** vision's in-memory `known_faces` (else the next frame re-mints the deleted id). Without biometrics available (mock/no encoding) it never merges — the safe default. *(`merge_band` needs tuning on real hardware. The name itself is extracted by the LLM — `node_learn_name` sends the message with a "give me the speaker's own name or NONE" prompt — so any phrasing/language works, no hardcoded patterns.)*
- **Both reply paths share assembly + fallbacks.** HTTP `node_think` and WS `_stream_response` both call `build_messages(state)` (incl. new-face introduction / name-ask) so they can't drift. Fallbacks are **localized** via the personality YAML (`fallbacks:` section → `personality.fallback(key)`), not hardcoded English.
- **LLM stream errors are never spoken.** `chat_stream` lets exceptions propagate; `_stream_response` catches them and, if nothing usable was produced, sends one localized fallback (`personality.fallback("error")`) — it does **not** yield `[error]` text into the spoken stream.
- **The mic is half-duplex (anti-echo).** `MicrophoneCapture` takes a `gate` callable; the toy gates it on `AudioPlayer.is_playing` (+~300 ms tail) so it never transcribes its own TTS. No echo cancellation needed.
- **The event bus is wired, with a live dashboard.** `events/bus.py` has two consumers: `SessionTracker` (per-person stats → `GET /sessions`) and `DashboardHub` (SSE fan-out). Handlers are dispatched fire-and-forget but **never silently** — each task gets a done-callback that logs exceptions (`logger.error`), so a raising subscriber surfaces in the log instead of vanishing as an unretrieved task exception, and never stops the other handlers. Orchestrator emits `FACE_RECOGNIZED`/`EMOTION_DETECTED`/`USER_TEXT`/`RESPONSE_READY`. The dashboard is a self-contained page at **`GET /dashboard`** (HTML in `events/dashboard.py`) that polls `/sessions` and live-tails **`GET /dashboard/stream`** (text/event-stream): a "Present now" panel (faces seen within `SessionTracker.PRESENCE_WINDOW` = 10 s, exposed as `present`/`current_emotion` in the snapshot), a per-person emotion histogram, and a live 60 s emotion chart. **Charts are vanilla `<canvas>` — no CDN/external JS on purpose** (the Pi may be offline). Add a new subscriber to extend it. *(Note: SSE can't be tested via Starlette `TestClient` — `iter_lines` blocks on the infinite stream; the `DashboardHub` fan-out is unit-tested directly instead.)*

## Models & deployment (server host — a laptop)

The server uses three ML models; all **auto-download on first use** and can be **pre-fetched** to avoid a network hit during the first conversation:

| Model | Lib | Lands in | Pin with |
|-------|-----|----------|----------|
| FER+ emotion (~35 MB) | `onnxruntime` | `./data/fer_models/` | `PHILOSOPHER_CAMERA_EMOTION_MODEL_PATH` |
| Piper voice (~28 MB + json) | `piper` binary | `./data/piper_models/` | `PHILOSOPHER_TTS_MODEL_PATH` |
| faster-whisper (`tiny`…) | `faster-whisper` | HF cache | (size via `PHILOSOPHER_STT_MODEL`) |

Pre-download all three: `python scripts/download_models.py` (idempotent; each step independent).

ARM64 install notes (full bring-up sequence in `HARDWARE_RUNBOOK.md`):
- **`face_recognition`/dlib has no PyPI ARM64 wheel.** On **Raspberry Pi OS Bookworm (Python 3.11)** install from piwheels first — `pip install dlib --index-url https://www.piwheels.org/simple` — before a 2–4 h source build. On **Debian Trixie / Python 3.13** piwheels has no dlib wheel; use **Miniforge + `conda install -c conda-forge dlib`** on a 3.11 env (prebuilt aarch64). The stale `face_recognition` wrapper also needs `pkg_resources` (gone in setuptools≥81) and its model blobs — both now pinned in `pyproject.toml`. MediaPipe is a no-dlib fallback if you only need presence, not identity.
- **Use `opencv-python-headless`, not `opencv-python`.** Both devices are display-less; the GUI build links `libGL.so.1` (absent on a headless Pi) and fails `import cv2`. Both `pyproject.toml`s pin headless.
- **`piper` is a system binary**, not a pip package — install it separately on the Pi (the tarball ships bundled libs, so move the whole folder + symlink the binary; see runbook).

## Gotchas (learned the hard way)

- **Everything degrades to a safe default; it does not crash the socket.** Emotion → `"neutral"` if onnxruntime/model missing; TTS → empty bytes (text, no audio) if the `piper` binary/model missing; STT → **drops the utterance** (empty text + `unavailable:true`, so the toy stays silent) if `faster-whisper` failed to load — it does *not* fabricate text. The `"Hello world"` stub is **only** for `mock=True` (dev). A failed STT load is logged loudly (`logger.error`) once at startup, and every engine's readiness is surfaced on **`GET /health`** (`stt`/`vision`/`tts` each report `ok`/`mock`/`degraded`/`unavailable`; any non-ok engine rolls the top-level `status` up to `degraded`). So "it runs but does nothing" usually means a missing model/binary, not a logic bug — check `/health` and the startup log first.
- **`.env` only works because `get_settings()` calls `load_dotenv()` first.** The nested settings models (`LLMSettings`, etc.) are built via `default_factory` and read **`os.environ`, not the `.env` file** — only the top-level `Settings` has `env_file`. Without the `load_dotenv()` in `get_settings()`, `PHILOSOPHER_*` in `.env` are silently ignored on a plain `python -m` run and the app falls back to defaults (e.g. localhost LLM → "Connection error"). Don't remove that call. (systemd's `EnvironmentFile` injects the vars directly, so the deployed path doesn't depend on it.)
- **`faster-whisper` returns a one-shot generator.** Materialize `segments` to a list before using it twice (`stt/engine.py`); reading it for `text` and again for `confidence` silently yields nothing the second time (and `min([])` raises).
- **Tests must isolate the SQLite DB.** The default DB (`./data/philosopher_memory.db`) is WAL-mode and shared; tests hitting it concurrently block on the lock and *look* like a hang. Use a `tmp_path` DB via `monkeypatch.setenv("PHILOSOPHER_MEMORY_DB_PATH", …)` + `get_settings.cache_clear()` (see `tests/integration/test_stream.py`).
- **The WS sentence splitter is `buffer.endswith(('.','!','?','\n'))`.** A token whose terminator is followed by whitespace (e.g. `". "`) won't split until the next terminator, so sentences can batch. Fine for real LLM streams; matters when writing fake token streams in tests.

## Known limitations / not built yet

Roadmap-y gaps that the code itself won't tell you (absence isn't self-documenting):
- **No *toy-side* wake word (but a server-side wake *gate* exists).** The toy has no acoustic hotword engine — no ML on the 512MB board by design, so it streams all speech and the server transcribes every utterance. Layered on top is an **opt-in server-side wake gate** (`PHILOSOPHER_WAKE_ENABLED`, off by default): `orchestrator._apply_wake_gate` runs a per-toy sleep/wake state machine over the *transcript* — the toy starts asleep and only gets an LLM+TTS reply after an **activation phrase**, and returns to sleep on a **deactivation phrase** or after `PHILOSOPHER_SLEEP_TIMEOUT` (default 30 min) of inactivity. Phrases are localized in the personality YAML (`wake:` → `activate`/`deactivate`), matched accent/case-insensitively. While asleep the server still runs STT (that's the cost of no toy-side detection) but skips LLM+TTS and releases the toy's mic gate with a silent `{type:"idle"}` frame. `WAKE_UP` remains an unused event type, not hotword detection.
- **No LEDs yet.** `toy_client/hardware/leds.py` does not exist; servos (head + arms) are the only expressive output.
- **Name learning needs a detected face on the turn the name is stated.** The LLM extracts the name from any phrasing/language, but if that turn was *anonymous* (no face captured — e.g. the on-speech frame missed the person), there's no face to attach the name to. The biometric dedup band (`merge_band`) still needs tuning on real hardware.
- **One speaker per turn.** The spoken turn is attributed to the dominant face at `speech_ended` — no voice diarization.
- **Keyword memory, no embeddings** — by design for the RAM budget (overlap + recency, not semantic). Revisit `sentence-transformers` only if a bigger host makes the RAM cost acceptable.

## Reference

The longer reference material lives here (this file is canonical; the `AGENTS.md`
files and subproject `README.md`s are stubs that point here).

### Module map
**Server (`philosopher-server/philosopher/`)** — `config/` (Pydantic settings + `personalities/<lang>/*.yaml`), `core/` (`state·nodes·graph·orchestrator` — the LangGraph pipeline + WS streaming path), `llm/` (OpenAI-compatible async client), `memory/` (`short_term·long_term·manager`), `personality/` (YAML loader + prompt builder + formatter), `stt/` · `vision/` (+`emotion.py`) · `tts/` (the server-side AI engines), `api/` (`app.py` HTTP + `ws_server.py` WebSocket), `events/` (`bus·session_tracker·dashboard`), `utils/`.
**Toy (`philosopher-toy/toy_client/`)** — `protocol/ws_client.py` (heartbeat + auto-reconnect), `vision/camera.py` (capture + JPEG encode only), `audio/` (`capture.py` mic+VAD, `player.py` WAV playback), `hardware/servos.py` (head=GPIO12 on hardware PWM, ramped + released at rest; arms in the pin map but disabled). **Pi Zero WH bring-up + gotchas: `HARDWARE_RUNBOOK.md` §B** (the body is only partly verified on real hardware): OTG gadget↔host, armhf install, audio backend. **Audio I/O uses the `arecord`/`aplay` binaries (not PyAudio, which pegs the CPU enumerating devices on the 512MB board) — verified working end-to-end on the Pi Zero WH (the toy converses).** Speaker volume is set from `PHILOSOPHER_SPEAKER_VOLUME` via `amixer` at startup. Mic tuning env: `PHILOSOPHER_MIC_GAIN` / `PHILOSOPHER_VAD_THRESHOLD` / `PHILOSOPHER_MIC_DEBUG` / `PHILOSOPHER_{MIC,SPEAKER}_ALSA_DEVICE`. Camera is the **OV5647 MIPI CSI** on the Pi Zero WH — it **works** (`camera_auto_detect=0` + `dtoverlay=ov5647` in config.txt; a reversed ribbon at the board end gives `i2c ... -5`/`probe failed`). `camera.py` grabs each JPEG via the **`rpicam-still`** binary (cv2.VideoCapture can't read the libcamera CSI stream), no OpenCV on the toy. The head servo is wired and works on hardware PWM (BCM12/PWM0, smooth, released at rest); arm servos are disabled. Mechanical mounting of the head servo is still being finalized.

### HTTP API endpoints (`api/app.py`)
| Method | Path | Notes |
|--------|------|-------|
| GET | `/health` | `{status, llm, memory, stt, vision, tts}` — `status` is `degraded` if any engine isn't ok |
| POST | `/chat` | `{message, face_id?, face_name?, emotion?}` → full 6-node graph (diagnostics/fallback path) |
| GET | `/personalities` · POST `/personality?name=` | list / switch personality |
| GET | `/memory/stats` | `{total_memories, known_faces, by_type}` |
| GET | `/sessions` | per-person stats (`SessionTracker`) |
| GET | `/dashboard` · `/dashboard/stream` | dashboard page + SSE feed (`DashboardHub`) |
| WS | `/ws?toy_id=<id>` | the real-time toy transport (see protocol above) |

CLI (`python -m philosopher.main`): `--server --host --port --personality --language --llm-url --llm-model --debug`.

### Personality YAML (`config/personalities/<lang>/<name>.yaml`)
Top-level keys: `name, role, language, description, system_prompt, traits{empathy,curiosity,humor,formality,energy,optimism}, speech_patterns{greetings,farewells,…}, emotion_responses{happy,sad,angry,surprised,neutral,fear}, style{max_response_length,use_asterisks_for_actions,ask_questions,use_metaphors}, fallbacks{error,not_understood,…}, face_registration{…}`. `fallbacks` is what makes fallback text localized (`personality.fallback(key)`); `face_registration` tunes the name-ask. Personality slugs are language-neutral (same key across languages): `philosopher, curious, poetic, friend, wise`, available in both `es/` and `en/`.

### Memory architecture
- **Short-term** (`memory/short_term.py`): in-RAM `deque` per `face_id` (unknown → `_anon`), lost on restart.
- **Long-term** (`memory/long_term.py`): SQLite + WAL. Tables `memories` and `known_faces(face_id, name, first_seen, last_seen, encounters, encoding)`. Recall = keyword overlap (words >3 chars) + recency decay (~1-week half-life), `threshold` 0.6. No embeddings (RAM budget).

### Example `.env`
```bash
# Server (PHILOSOPHER_*): LLM_PROVIDER, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL,
#   PERSONALITY, LANGUAGE, MEMORY_DB_PATH, API_HOST=0.0.0.0, API_PORT=8080,
#   STT_MODEL/DEVICE/COMPUTE_TYPE, CAMERA_FPS/QUALITY/DETECT_WIDTH/PRESENCE_GATE/MERGE_BAND, TTS_VOICE
# Toy: PHILOSOPHER_SERVER_URL=ws://<server-ip>:8080  (base only; client appends /ws?toy_id=)
#      PHILOSOPHER_TOY_ID=toy_01, PHILOSOPHER_CAMERA_FPS=0.5, PHILOSOPHER_MOCK=false
```

### Data-flow trace (WebSocket, the real path)
```
Toy: mic VAD → speech_started, PCM (0x01), speech_ended ; camera → JPEG (0x02) @ FPS
Server: vision decodes JPEG → face id + emotion → {look}/{servo} back immediately
        stt: on speech_ended, faster-whisper transcribes accumulated PCM
        _stream_response: nodes 1–3 build context → llm.chat_stream() → buffer sentences
          per sentence: node_format → {type:text}+{servo} + Piper WAV (0x03) → toy plays it
          node_store then node_learn_name at the end (persist + LLM name learning, fire-and-forget)
```

## Doc map — what's authoritative

1. **Source code** (`philosopher/`, `toy_client/`) — ground truth.
2. **This `CLAUDE.md`** — the canonical, up-to-date guide (quick reference + the Reference section above). The root `AGENTS.md` and each subproject `README.md`/`AGENTS.md` are **stubs that point here**.
3. **`HARDWARE_RUNBOOK.md`** — bring-up on real devices (server on a laptop + Pi Zero WH toy): dlib/piper/model setup, `check_runtime_deps.py`, autostart (`run.sh install`), the hardware-PWM config.txt step, and an end-to-end smoke checklist with per-step triage.
