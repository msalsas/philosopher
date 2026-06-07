# Philosopher Toy

Thin I/O client for the Philosopher conversational AI. Runs on a **Banana Pi
BPi-M2 Zero (512 MB)** embedded inside a plush toy. It performs **zero ML** — all
intelligence lives in `philosopher-server`. The toy just captures audio/video,
streams it to the server over WebSocket, plays back synthesized speech, and moves
servos. It connects to `ws://<server>:<port>/ws?toy_id=<id>`.

```bash
pip install -e ".[dev]"
export PHILOSOPHER_SERVER_URL=ws://<server-ip>:8080   # base URL only; client appends /ws?toy_id=
python -m banana_client.main
PHILOSOPHER_MOCK=true python -m banana_client.main     # no hardware needed
```

**Full docs — architecture, the WebSocket protocol, body-language/servo behavior,
conventions — are in the canonical guide at the repo root:
[`../CLAUDE.md`](../CLAUDE.md).**
