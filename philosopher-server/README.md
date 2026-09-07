# Philosopher Server

The **brain** of the Philosopher conversational AI — personality, dual memory, and
emotional awareness. Runs on a **laptop** (a real CPU) and does **all** the
compute: STT, vision, TTS, the LangGraph pipeline, memory, and proxying an external
OpenAI-compatible LLM. The plush-toy client (`philosopher-toy`, a 512 MB Raspberry Pi Zero WH)
is a thin I/O node that talks to this server over WebSocket.

```bash
pip install -e ".[dev]"
cp .env.example .env                         # set your LLM endpoint; install the `piper` binary for real TTS
python scripts/download_models.py            # pre-fetch FER+/Piper/Whisper (optional)

python -m philosopher.main --server          # WebSocket + HTTP (default :8080)
python -m philosopher.main                   # interactive terminal chat (text only)
MOCK_MODE=true python -m philosopher.main --server   # no models/hardware needed
```

**Full docs — architecture, configuration, HTTP/WS API, conventions, gotchas — are
in the canonical guide at the repo root: [`../CLAUDE.md`](../CLAUDE.md).**
Deploying on real hardware: [`../HARDWARE_RUNBOOK.md`](../HARDWARE_RUNBOOK.md).
