# Philosopher Toy

Thin I/O client for the Philosopher conversational AI. Runs on a **Banana Pi BPi-M2 Zero (512MB)** embedded inside a plush toy. It performs **zero ML** — all intelligence lives in `philosopher-server`. The toy just captures audio/video, streams it to the server over WebSocket, plays back synthesized speech, and moves servos.

## Hardware

- **Microphone**: captured, VAD-gated, streamed as raw PCM to the server (STT happens server-side)
- **Camera**: frames captured and JPEG-encoded, streamed to the server (face/emotion happens server-side)
- **Speaker**: plays WAV audio received from the server (TTS happens server-side, via Piper)
- **Servos**: head + arms, moved **one at a time** on emotion commands (shared MicroUSB power rail)

## Architecture

```
[Mic] --VAD--> PCM  ─┐
[Camera] ----> JPEG ─┤── WebSocket ──> Philosopher Server (RPi4) ──HTTP──> External LLM
[Speaker] <--- WAV  ─┤
[Servos] <--- cmds  ─┘
```

The toy connects to `ws://<server>:<port>/ws?toy_id=<id>`. Binary frames: `0x01` audio and `0x02` video (toy→server), `0x03` WAV (server→toy). JSON control: `speech_started`/`speech_ended`/`ping` (toy→server), `text`/`servo`/`error` (server→toy).

## Quick Start

```bash
pip install -e ".[dev]"
export PHILOSOPHER_SERVER_URL=ws://server-ip:8080
python -m banana_client.main
```

## Mock Mode

Run without physical hardware:

```bash
PHILOSOPHER_MOCK=true python -m banana_client.main
```
