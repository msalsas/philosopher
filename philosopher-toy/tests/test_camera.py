"""Tests for camera FPS rounding."""
from __future__ import annotations

import pytest

from toy_client.vision.camera import Camera


class TestCameraFPS:
    def test_fps_rounding_0_7(self):
        c = Camera(fps=0.7)
        assert c.fps == 0.5

    def test_fps_rounding_1_2(self):
        c = Camera(fps=1.2)
        assert c.fps == 1.0

    def test_fps_rounding_1_7(self):
        c = Camera(fps=1.7)
        assert c.fps == 1.5

    def test_fps_rounding_2_3(self):
        c = Camera(fps=2.3)
        assert c.fps == 2.5

    def test_fps_clamp_max(self):
        c = Camera(fps=6.0)
        assert c.fps == 5.0

    def test_fps_clamp_min(self):
        c = Camera(fps=0.1)
        assert c.fps == 0.5

    def test_quality_clamp(self):
        import os
        os.environ["PHILOSOPHER_CAMERA_QUALITY"] = "100"
        c = Camera()
        assert c.quality == 90
        del os.environ["PHILOSOPHER_CAMERA_QUALITY"]
