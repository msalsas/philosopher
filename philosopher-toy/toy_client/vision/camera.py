"""Camera capture via the rpicam binary (libcamera) -- no OpenCV/picamera2.

On modern Raspberry Pi OS the CSI camera works through libcamera, not plain
V4L2, so cv2.VideoCapture returns no frames. We grab each JPEG by invoking
``rpicam-still`` as a subprocess -- the same lightweight, dependency-free
pattern the toy already uses for audio (arecord/aplay) -- which fits the low
frame rate without loading heavy libraries on the 512MB board.
"""
from __future__ import annotations

import asyncio
import os
import shutil


class Camera:
    """CSI camera capture via rpicam-still -- returns JPEG bytes per frame."""

    def __init__(self, cam_id: int = 0, width: int = 640, height: int = 480,
                 fps: float | None = None) -> None:
        if fps is None:
            fps = float(os.getenv("PHILOSOPHER_CAMERA_FPS", "0.5"))
        self.fps = max(0.5, min(5.0, round(fps * 2) / 2))
        self.quality = max(30, min(90, int(os.getenv("PHILOSOPHER_CAMERA_QUALITY", "60"))))
        self.width = int(os.getenv("PHILOSOPHER_CAMERA_WIDTH", str(width)))
        self.height = int(os.getenv("PHILOSOPHER_CAMERA_HEIGHT", str(height)))
        # ms the sensor runs before the shot so AGC/AWB settle (each rpicam-still
        # starts the pipeline fresh). Small, tunable; fine for the low frame rate.
        self.warmup_ms = int(os.getenv("PHILOSOPHER_CAMERA_WARMUP_MS", "500"))
        self.cam_id = cam_id
        self._bin: str | None = None
        self._mock = False

    async def init(self):
        self._bin = shutil.which("rpicam-still") or shutil.which("libcamera-still")
        if not self._bin:
            print("[WARNING] Camera not available (no rpicam), entering mock mode")
            self._mock = True
        return self

    async def capture(self):
        """Return one JPEG frame as bytes, or None on failure / mock."""
        if self._mock or not self._bin:
            return None
        try:
            proc = await asyncio.create_subprocess_exec(
                self._bin, "-n", "-t", str(self.warmup_ms),
                "--width", str(self.width), "--height", str(self.height),
                "-q", str(self.quality), "-e", "jpg", "-o", "-",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=8.0)
            if proc.returncode == 0 and out[:2] == b"\xff\xd8":  # JPEG SOI marker
                return out
        except (asyncio.TimeoutError, FileNotFoundError, OSError):
            pass
        return None

    async def encode(self, frame) -> bytes:
        # rpicam already returns JPEG bytes; pass them through unchanged.
        return bytes(frame) if isinstance(frame, (bytes, bytearray)) else b""

    def close(self):
        return None
