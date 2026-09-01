"""Banana Pi client entry point."""
from __future__ import annotations

import asyncio
import os
import time

try:
    from dotenv import load_dotenv
except ImportError:  # optional: config still works via real env vars
    load_dotenv = None

from toy_client.audio.capture import MicrophoneCapture
from toy_client.audio.player import AudioPlayer
from toy_client.hardware.servos import ServoController
from toy_client.protocol.ws_client import ToyWebSocketClient
from toy_client.vision.camera import Camera


class Toy:
    """Main toy controller integrating all hardware modules."""

    # Fail-safe: if a reply never arrives (server error / lost frame), stop
    # gating the mic after this long so the toy can't go deaf forever.
    RESPONSE_TURN_TIMEOUT = 30.0

    def __init__(self) -> None:
        self.server_url = os.getenv("PHILOSOPHER_SERVER_URL", "ws://localhost:8080")
        self.toy_id = os.getenv("PHILOSOPHER_TOY_ID", "toy_01")
        self.mock = os.getenv("PHILOSOPHER_MOCK", "false").lower() == "true"
        self.ws = None
        self.mic = None
        self.player = None
        self.camera = None
        self.servos = None
        # Half-duplex per TURN: gate the mic from the moment we send
        # speech_ended until the reply's audio arrives, so we don't pile new
        # utterances onto a server that's still busy transcribing/thinking.
        self._awaiting_response = False
        self._await_deadline = 0.0

    async def init(self):
        print("[Toy] Initializing...")
        self.ws = ToyWebSocketClient(self.server_url, self.toy_id)
        # The toy often boots before the server (or before the network) is up:
        # a failed first connect must retry with backoff, not crash the process.
        try:
            await self.ws.connect()
        except Exception as exc:
            print(f"[Toy] Server unreachable at {self.server_url} ({exc}), retrying...")
            if not await self.ws.reconnect():
                raise RuntimeError(
                    f"Could not reach server at {self.server_url} after "
                    f"{self.ws.max_reconnect} attempts"
                ) from exc

        self.player = AudioPlayer(mock=self.mock)
        self.player.init()

        # Suppress the mic while the toy is speaking (anti-echo / half-duplex).
        self.mic = MicrophoneCapture(gate=self._mic_gated, mock=self.mock)
        await self.mic.init()

        self.camera = Camera()
        await self.camera.init()

        self.servos = await ServoController(mock=self.mock).init()
        print("[Toy] Ready!")

    async def run(self):
        self._camera_on_speech = os.getenv(
            "PHILOSOPHER_CAMERA_ON_SPEECH", "").lower() in ("1", "true", "yes")
        # return_exceptions=True so one task raising can't cancel the siblings
        # (e.g. a transient error in the audio path must not kill receive/idle).
        tasks = [
            self._resilient(self._audio_task, "audio"),
            self._receive_task(),
            self.servos.idle_loop(),   # subtle idle motion so it isn't a statue
        ]
        # Camera modes (power budget): off (CAMERA_DISABLE), on_speech
        # (CAMERA_ON_SPEECH, one frame per turn — spike lands while the amp is
        # silent, no brownout), or stream (default, continuous, drives gaze).
        if os.getenv("PHILOSOPHER_CAMERA_DISABLE", "").lower() in ("1", "true", "yes"):
            print("[Toy] Camera off (voice-only mode)")
        elif self._camera_on_speech:
            print("[Toy] Camera on-speech mode (one frame per turn)")
        else:
            tasks.insert(1, self._resilient(self._video_task, "video"))
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _resilient(self, task_fn, name):
        """Keep a sensor loop alive: if it dies (e.g. a send blew up mid-drop),
        log and restart it rather than leaving the mic/camera silent forever."""
        while True:
            try:
                await task_fn()
                return  # clean completion (shouldn't happen for infinite loops)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[Toy] {name} task error ({exc}); restarting in 1s")
                await asyncio.sleep(1)

    def _mic_gated(self) -> bool:
        """True = suppress the mic. Half-duplex: silent while the toy is
        speaking (anti-echo) AND while a turn is being processed server-side."""
        if self.player and self.player.is_playing:
            return True
        if self._awaiting_response:
            if time.monotonic() < self._await_deadline:
                return True
            self._awaiting_response = False  # fail-safe: resume listening
        return False

    async def _audio_task(self):
        async for event in self.mic.capture_stream():
            if event["type"] == "speech_started":
                await self.ws.send_json({"type": "speech_started"})
                # On-speech: grab one frame (background, so it doesn't stall the
                # audio loop) while the amp is still silent.
                if getattr(self, "_camera_on_speech", False) and self.camera:
                    asyncio.create_task(self._capture_and_send())
            elif event["type"] == "speech_ended":
                await self.ws.send_audio(event["audio"])
                await self.ws.send_json({"type": "speech_ended"})
                # Stop listening until the reply comes back (or we time out).
                self._awaiting_response = True
                self._await_deadline = time.monotonic() + self.RESPONSE_TURN_TIMEOUT

    async def _video_task(self):
        interval = 1.0 / self.camera.fps
        while True:
            frame = await self.camera.capture()
            if frame is not None:
                jpeg = await self.camera.encode(frame)
                await self.ws.send_frame(jpeg)
            await asyncio.sleep(interval)

    async def _capture_and_send(self):
        """Grab and ship one frame (on-speech mode)."""
        try:
            frame = await self.camera.capture()
            if frame is not None:
                await self.ws.send_frame(await self.camera.encode(frame))
        except Exception as exc:
            print(f"[Camera] on-speech capture failed: {exc}")

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
                # Reply audio arrived: the turn is done processing. Clear the
                # gate; playback itself keeps the mic suppressed (is_playing).
                self._awaiting_response = False
                await self.player.play_wav(msg["payload"])
            elif msg["type"] == "idle":
                self._awaiting_response = False  # silent mic-gate release (no reply)
            elif msg["type"] == "error":
                self._awaiting_response = False
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
    try:
        await toy.init()  # inside try so a failed init still closes the session
        await toy.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        # Ctrl+C reaches asyncio.run as a task cancellation, surfacing here as
        # CancelledError (not KeyboardInterrupt) — catch both so shutdown is clean.
        print("\n[Toy] Goodbye...")
    finally:
        await toy.stop()


def main():
    """Sync entry point for the `philosopher-toy` console script."""
    # Config is read from os.environ (see Toy.__init__); load a local .env
    # first so it works on a plain run too. Real env vars win (override=False).
    if load_dotenv is not None:
        load_dotenv(override=False)
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        # asyncio.run re-raises KeyboardInterrupt after cancelling the main task;
        # swallow it so a Ctrl+C exits 0 with no traceback.
        pass


if __name__ == "__main__":
    main()
