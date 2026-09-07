# AGENTS.md — Philosopher

> **The canonical guide for this repo is [`CLAUDE.md`](./CLAUDE.md).** It holds the
> architecture, the two execution paths, the WebSocket protocol, conventions,
> gotchas, and a full Reference section (module map, HTTP API, personality YAML,
> memory schema, `.env`, data-flow). This `AGENTS.md` is intentionally a thin
> pointer so there is a single source of truth.

Quick orientation: Philosopher is a conversational AI plush toy — two Python
projects (`philosopher-server`, the brain on a laptop; `philosopher-toy`,
a zero-ML I/O client on a 512 MB Raspberry Pi Zero WH) talking over WebSocket, plus an
external OpenAI-compatible LLM. All AI (STT, vision, TTS, LangGraph, memory) runs
on the server.

Other docs:
- **[`CLAUDE.md`](./CLAUDE.md)** — canonical guide (start here).
- **[`HARDWARE_RUNBOOK.md`](./HARDWARE_RUNBOOK.md)** — bring-up on real devices + smoke checklist.
