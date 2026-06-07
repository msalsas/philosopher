# Philosopher Server

The "brain" of the Philosopher conversational AI — personality, dual memory, and emotional awareness. Runs on a **Raspberry Pi 4 (4GB)** and does **all** the compute: STT, vision, TTS, the LangGraph pipeline, memory, and proxying an external LLM. The plush-toy client (`philosopher-toy`, a 512MB Banana Pi) is a thin I/O node that talks to this server over WebSocket.

## Architecture

```
External LLM  <--HTTP--  Philosopher Server (RPi4)  <--WebSocket-->  Toy (Banana Pi, I/O only)
                         STT · vision · TTS · LangGraph · memory
```

- **LLM**: any OpenAI-compatible endpoint (LM Studio, OpenAI, Ollama). The server only streams tokens; no local LLM.
- **STT**: `faster-whisper`, batch — transcribes accumulated PCM when the toy signals `speech_ended`.
- **Vision**: `face_recognition` (identity) + FER+ ONNX via `onnxruntime` (emotion) on JPEG frames.
- **TTS**: **Piper**, invoked as a subprocess (the `piper` binary must be installed; it is not a pip package). One WAV per sentence, streamed to the toy.

## Quick Start

```bash
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your LLM endpoint (and install the `piper` binary for real TTS)

# Pre-download the FER+ emotion model so the Pi doesn't fetch it at startup
# (requires internet; idempotent — skips if already present)
python scripts/download_models.py

python -m philosopher.main --server     # WebSocket + HTTP (default :8080)
python -m philosopher.main              # interactive terminal chat (text only)
MOCK_MODE=true python -m philosopher.main --server   # no models needed
```

> The emotion model also auto-downloads to `./data/fer_models/` on first use if you skip the step above. Pre-downloading just avoids a network fetch during the first conversation. Pin a path with `PHILOSOPHER_CAMERA_EMOTION_MODEL_PATH` to use an existing copy.

## Pipeline

Two paths share the 6 LangGraph nodes (`perception → memory → prompt → think → format → store`):

- **HTTP `/chat`** runs the full graph (diagnostics/fallback).
- **WebSocket** uses *split-graph streaming*: nodes 1–3 for context, then the external LLM streams tokens, sentences are formatted and shipped (text + servo + WAV) as they complete, and `store` runs once at the end.

## Personalities

YAML files organized by language in `philosopher/config/personalities/<lang>/` (e.g. `es/filosofo.yaml`). No code — add a language by creating a new folder. `PHILOSOPHER_LANGUAGE` selects it.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check (LLM + memory) |
| POST | `/chat` | Text in, text out (full graph) |
| GET | `/personalities` | List personalities for current language |
| POST | `/personality` | Switch personality |
| GET | `/memory/stats` | Memory statistics |
| WS | `/ws?toy_id=<id>` | Toy streaming (audio/video/text/WAV) |

## Testing

```bash
pytest tests/ -v
ruff check .
```
