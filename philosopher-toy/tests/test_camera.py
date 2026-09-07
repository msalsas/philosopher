"""Tests for camera FPS rounding (nearest 0.1, clamped to [0.1, 5.0])."""
from __future__ import annotations

from toy_client.vision.camera import Camera


class TestCameraFPS:
    def test_fps_rounding_0_74(self):
        c = Camera(fps=0.74)
        assert c.fps == 0.7

    def test_fps_rounding_1_23(self):
        c = Camera(fps=1.23)
        assert c.fps == 1.2

    def test_fps_rounding_1_75(self):
        c = Camera(fps=1.75)
        assert c.fps == 1.8

    def test_fps_clamp_max(self):
        c = Camera(fps=6.0)
        assert c.fps == 5.0

    def test_fps_clamp_min(self):
        c = Camera(fps=0.02)
        assert c.fps == 0.1

    def test_quality_clamp(self):
        import os
        os.environ["PHILOSOPHER_CAMERA_QUALITY"] = "100"
        c = Camera()
        assert c.quality == 90
        del os.environ["PHILOSOPHER_CAMERA_QUALITY"]
