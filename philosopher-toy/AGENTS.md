# AGENTS.md — philosopher-toy

> This is a stub. The canonical guide for the whole repo lives at the root:
> **[`../CLAUDE.md`](../CLAUDE.md)** — architecture, the WebSocket protocol,
> body-language/servo behavior, conventions, gotchas, and the full Reference
> (module map, `.env`, data-flow).

`philosopher-toy` is the **body**: a zero-ML I/O client on a **Raspberry Pi Zero WH
(512 MB)** inside a plush toy. It captures mic + camera, streams them to the server
over WebSocket, plays back the WAV audio the server sends, and moves the head servo.
**All AI (STT, vision, TTS, LLM, memory) runs on the server, not here** — the 512 MB
budget is why no inference runs on the toy.

- Audio I/O uses the `arecord`/`aplay` binaries (not PyAudio); camera grabs JPEGs via
  `rpicam-still` (OV5647 MIPI CSI); the head servo is on BCM12 hardware PWM (arms
  disabled). Details + bring-up: **[`../HARDWARE_RUNBOOK.md`](../HARDWARE_RUNBOOK.md)**.
- Canonical guide: **[`../CLAUDE.md`](../CLAUDE.md)**
