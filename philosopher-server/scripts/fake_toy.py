#!/usr/bin/env python3
"""A fake toy: drive the server's WebSocket end-to-end with no hardware.

The toy is just I/O, so its whole side of the protocol is a handful of frames.
This script speaks that protocol (see api/ws_server.py) from a laptop or the Pi
itself, so you can exercise the real streaming path — perception, STT, LLM
stream, per-sentence TTS — before the Banana Pi exists.

It plays one "turn":
    1. sends a camera frame      (0x02 JPEG)   -> expect {look}/{servo} back
    2. speech_started + audio    (0x01 PCM)
    3. speech_ended                            -> server transcribes + streams
    4. prints every server frame: {text}/{servo}/{look}/{react}/{error} + 0x03 WAV
    5. ping                                     -> expect {pong}

Against a MOCK_MODE server, any audio transcribes to "Hello world", so you get a
reply with no real models. Against a real server, pass a 16 kHz mono WAV with
actual speech (silence/short clips transcribe to empty and draw no reply).

    # simplest: synthetic frame + dummy audio against a MOCK_MODE server
    MOCK_MODE=true python -m philosopher.main --server   # (other shell)
    python scripts/fake_toy.py

    # realistic: your own frame + speech, against a real server
    python scripts/fake_toy.py --url ws://192.168.1.50:8080 \
        --jpeg me.jpg --wav hola.wav
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
import wave
from pathlib import Path

try:
    import websockets
except ImportError:
    sys.exit("websockets not installed — run `pip install -e .` in philosopher-server")

SAMPLE_RATE = 16000
TX, RX = "→ sent", "←  recv"


def _load_jpeg(path: str | None) -> bytes:
    if path:
        return Path(path).read_bytes()
    # No file: synthesize a frame (cv2 is a server dep, present in this venv).
    import cv2
    import numpy as np
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return cv2.imencode(".jpg", frame)[1].tobytes()


def _load_pcm(path: str | None) -> bytes:
    if not path:
        return b"\x00\x01" * 16000  # ~1s of dummy 16-bit samples (enough for MOCK)
    with wave.open(path, "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            sys.exit(f"{path} must be 16 kHz mono 16-bit PCM WAV")
        return w.readframes(w.getnframes())


async def _receiver(ws) -> None:
    """Print everything the server sends until cancelled."""
    try:
        async for msg in ws:
            if isinstance(msg, bytes):
                kind = msg[0] if msg else -1
                tag = "WAV audio (0x03)" if kind == 0x03 else f"binary 0x{kind:02x}"
                print(f"{RX}  {tag}: {len(msg) - 1} bytes")
            else:
                data = json.loads(msg)
                t = data.get("type", "?")
                extra = data.get("content") or data.get("emotion") or data.get("offset_x")
                print(f"{RX}  {t:<7} {extra if extra is not None else ''}".rstrip())
    except asyncio.CancelledError:
        pass
    except websockets.ConnectionClosed:
        print(f"{RX}  <connection closed by server>")


async def run(args) -> None:
    uri = f"{args.url.rstrip('/')}/ws?toy_id={args.toy_id}"
    print(f"connecting to {uri}")
    async with websockets.connect(uri) as ws:
        rx = asyncio.create_task(_receiver(ws))

        # 1. A camera frame — server runs vision and should reply look/servo.
        await ws.send(b"\x02" + _load_jpeg(args.jpeg))
        print(f"{TX}  0x02 JPEG frame")
        await asyncio.sleep(1.0)

        # 2. Speak: start, stream PCM in ~1s chunks, end.
        await ws.send(json.dumps({"type": "speech_started"}))
        print(f"{TX}  speech_started")
        pcm = _load_pcm(args.wav)
        for i in range(0, len(pcm), 32000):
            await ws.send(b"\x01" + pcm[i:i + 32000])
            await asyncio.sleep(0.05)
        print(f"{TX}  0x01 PCM  ({len(pcm)} bytes)")
        await ws.send(json.dumps({"type": "speech_ended"}))
        print(f"{TX}  speech_ended  — waiting {args.wait}s for the streamed reply…")

        # 3. Collect the streamed response.
        await asyncio.sleep(args.wait)

        # 4. Heartbeat.
        await ws.send(json.dumps({"type": "ping", "timestamp": 123}))
        print(f"{TX}  ping")
        await asyncio.sleep(0.5)

        rx.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await rx
    print("done.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="ws://localhost:8080", help="server base URL (no /ws)")
    p.add_argument("--toy-id", default="fake_toy", help="toy id query param")
    p.add_argument("--jpeg", help="JPEG frame to send (default: synthetic)")
    p.add_argument("--wav", help="16 kHz mono WAV to send as speech (default: dummy)")
    p.add_argument("--wait", type=float, default=8.0,
                   help="seconds to wait for the streamed reply (default 8)")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
