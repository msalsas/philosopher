# AGENTS.md — philosopher-server

> This is a stub. The canonical guide for the whole repo lives at the root:
> **[`../CLAUDE.md`](../CLAUDE.md)** — architecture, the two execution paths, the
> WebSocket protocol, conventions, gotchas, and the full Reference (module map,
> HTTP API, personality YAML, memory schema, `.env`, data-flow).

`philosopher-server` is the brain (runs on a laptop): STT (`faster-whisper`), vision
(`face_recognition` + FER+ ONNX), TTS (Piper subprocess), the LangGraph pipeline,
memory, and an external-LLM proxy. WebSocket (`/ws`) is the real path; HTTP `/chat`
is the diagnostics fallback.

- Canonical guide: **[`../CLAUDE.md`](../CLAUDE.md)**
- Real-device bring-up + smoke checklist: **[`../HARDWARE_RUNBOOK.md`](../HARDWARE_RUNBOOK.md)**
