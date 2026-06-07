"""Simplified camera: capture only, no processing. Configurable FPS with rounding."""
from __future__ import annotations

import os

import cv2


class Camera:
    """USB camera capture only — no face detection on toy."""

    def __init__(self, cam_id: int = 0, width: int = 640, height: int = 480, fps: float | None = None) -> None:
        if fps is None:
            fps = float(os.getenv("PHILOSOPHER_CAMERA_FPS", "0.5"))
        self.fps = max(0.5, min(5.0, round(fps * 2) / 2))
        self.quality = int(os.getenv("PHILOSOPHER_CAMERA_QUALITY", "60"))
        self.quality = max(30, min(90, self.quality))
        self.cam_id = cam_id
        self.width = width
        self.height = height
        self._cap = None

    async def init(self):
        self._cap = cv2.VideoCapture(self.cam_id, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
        if not self._cap.isOpened():
            print("[WARNING] Camera not available, entering mock mode")
            self._cap = None
            return self
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return self

    async def capture(self):
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    async def encode(self, frame) -> bytes:
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return buf.tobytes() if ret else b""

    def close(self):
        if self._cap:
            self._cap.release()
