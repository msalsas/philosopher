"""Banana Pi client entry point."""
from __future__ import annotations

import asyncio
import os

from banana_client.audio.capture import MicrophoneCapture
from banana_client.audio.player import AudioPlayer
from banana_client.hardware.servos import ServoController
from banana_client.protocol.ws_client import ToyWebSocketClient
from banana_client.vision.camera import Camera


class Toy:
    """Main toy controller integrating all hardware modules."""

    def __init__(self) -> None:
        self.server_url = os.getenv("PHILOSOPHER_SERVER_URL", "ws://localhost:8080")
        self.toy_id = os.getenv("PHILOSOPHER_TOY_ID", "banana_01")
        self.mock = os.getenv("PHILOSOPHER_MOCK", "false").lower() == "true"
        self.ws = None
        self.mic = None
        self.player = None
        self.camera = None
        self.servos = None

    async def init(self):
        print("[Toy] Initializing...")
        self.ws = ToyWebSocketClient(self.server_url, self.toy_id)
        await self.ws.connect()

        self.player = AudioPlayer()
        self.player.init()

        # Suppress the mic while the toy is speaking (anti-echo / half-duplex).
        self.mic = MicrophoneCapture(gate=lambda: bool(self.player and self.player.is_playing))
        await self.mic.init()

        self.camera = Camera()
        await self.camera.init()

        self.servos = await ServoController(mock=self.mock).init()
        print("[Toy] Ready!")

    async def run(self):
        await asyncio.gather(
            self._audio_task(),
            self._video_task(),
            self._receive_task(),
            self.servos.idle_loop(),   # subtle idle motion so it isn't a statue
        )

    async def _audio_task(self):
        async for event in self.mic.capture_stream():
            if event["type"] == "speech_started":
                await self.ws.send_json({"type": "speech_started"})
            elif event["type"] == "speech_ended":
                await self.ws.send_audio(event["audio"])
                await self.ws.send_json({"type": "speech_ended"})

    async def _video_task(self):
        interval = 1.0 / self.camera.fps
        while True:
            frame = await self.camera.capture()
            if frame is not None:
                jpeg = await self.camera.encode(frame)
                await self.ws.send_frame(jpeg)
            await asyncio.sleep(interval)

    async def _receive_task(self):
        while True:
            msg = await self.ws.receive()
            if msg["type"] == "text":
                print(f"[Philosopher] {msg['content']}")
            elif msg["type"] == "look":
                await self.servos.look(msg.get("offset_x", 0.0))
            elif msg["type"] == "react":
                await self.servos.react(msg["emotion"])
            elif msg["type"] == "servo":
                await self.servos.animate(msg["emotion"])
            elif msg["type"] == "binary" and msg.get("frame_type") == 0x03:
                await self.player.play_wav(msg["payload"])
            elif msg["type"] == "error":
                print(f"[Error] {msg['message']}")
            elif msg["type"] == "closed":
                print("[Toy] Connection lost, reconnecting...")
                if not await self.ws.reconnect():
                    print("[Toy] Reconnect failed, giving up.")
                    break

    async def stop(self):
        if self.ws:
            await self.ws.close()
        if self.mic:
            self.mic.close()
        if self.player:
            self.player.close()
        if self.camera:
            self.camera.close()
        if self.servos:
            self.servos.close()


async def _amain():
    toy = Toy()
    await toy.init()
    try:
        await toy.run()
    except KeyboardInterrupt:
        print("\n[Toy] Goodbye...")
    finally:
        await toy.stop()


def main():
    """Sync entry point for the `philosopher-toy` console script."""
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
