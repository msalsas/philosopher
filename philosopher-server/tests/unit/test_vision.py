"""Tests for vision engine."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from philosopher.vision.engine import VisionProcessor


def _jpeg(width: int = 640, height: int = 480, value: int = 30) -> bytes:
    """A solid-color JPEG frame (no face) for exercising the cheap gates."""
    img = np.full((height, width, 3), value, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


class TestVisionProcessor:
    @pytest.mark.asyncio
    async def test_mock_frame(self):
        vp = VisionProcessor(mock=True)
        result = await vp.process_frame(b"fake jpeg")
        assert result["faces"] == []
        assert result["primary_face"] is None

    @pytest.mark.asyncio
    async def test_invalid_jpeg(self):
        vp = VisionProcessor(mock=False)
        result = await vp.process_frame(b"not a jpeg")
        assert result["faces"] == []
        assert result["primary_face"] is None

    @pytest.mark.asyncio
    async def test_match_face_by_distance(self):
        # Encoding comparison is pure numpy distance — no dlib needed.
        import numpy as np
        vp = VisionProcessor(mock=False)
        vp._loaded = True  # skip DB load; inject a known face directly
        ref = np.linspace(0, 1, 128, dtype=np.float64)
        vp.known_faces = {"abc": {"name": "Ana", "encoding": ref}}

        # A near-identical encoding matches and resolves the stored name.
        fid, name = await vp._match_face(ref + 0.001)
        assert fid == "abc"
        assert name == "Ana"

        # A distant encoding is rejected (returns no match).
        fid2, name2 = await vp._match_face(ref + 5.0)
        assert fid2 is None
        assert name2 is None

    @pytest.mark.asyncio
    async def test_presence_gate_skips_dlib(self, monkeypatch):
        # When the cheap presence check sees no face, the expensive dlib path
        # (_recognize) must never run.
        vp = VisionProcessor(mock=False, presence_gate=True)
        monkeypatch.setattr(vp, "_has_face_fast", lambda frame: False)

        async def _boom(*a, **k):
            raise AssertionError("_recognize must not run when no face is present")

        monkeypatch.setattr(vp, "_recognize", _boom)
        result = await vp.process_frame(_jpeg())
        assert result == {"faces": [], "primary_face": None}

    @pytest.mark.asyncio
    async def test_blank_frame_has_no_face(self):
        # Real Haar cascade on a solid frame: no face → empty, no dlib needed.
        vp = VisionProcessor(mock=False, presence_gate=True, skip_similar=False)
        result = await vp.process_frame(_jpeg())
        assert result["faces"] == []
        assert result["primary_face"] is None

    @pytest.mark.asyncio
    async def test_similar_frame_reuses_last_result(self, monkeypatch):
        vp = VisionProcessor(mock=False, presence_gate=False, skip_similar=True)
        calls = {"n": 0}

        async def _fake_recognize(*a, **k):
            calls["n"] += 1
            return [{"face_id": "f1", "name": "Ana", "emotion": "happy",
                     "offset_x": 0.0, "confidence": 0.9}]

        monkeypatch.setattr(vp, "_recognize", _fake_recognize)
        frame = _jpeg(value=30)
        first = await vp.process_frame(frame)
        second = await vp.process_frame(frame)  # identical → skipped
        assert calls["n"] == 1
        assert second is first
        assert second["primary_face"]["name"] == "Ana"

        # A clearly different frame is processed again.
        await vp.process_frame(_jpeg(value=200))
        assert calls["n"] == 2

    def test_downscale_scales_box_invariantly(self):
        vp = VisionProcessor(mock=False, detect_width=320)
        big = np.zeros((720, 1280, 3), dtype=np.uint8)
        small, scale = vp._downscale(big)
        assert small.shape[1] == 320
        assert scale == pytest.approx(1280 / 320)
        # A frame already within the cap is left untouched.
        small2, scale2 = vp._downscale(np.zeros((240, 300, 3), dtype=np.uint8))
        assert scale2 == 1.0 and small2.shape[1] == 300

    def test_offset_x(self):
        # 640px-wide frame, center at 320. Offset normalized to [-1, 1].
        assert VisionProcessor._offset_x(310, 330, 640) == 0.0    # centered
        assert VisionProcessor._offset_x(470, 490, 640) == 0.5    # right of center
        assert VisionProcessor._offset_x(150, 170, 640) == -0.5   # left of center
        assert VisionProcessor._offset_x(900, 900, 640) == 1.0    # clamped to +1
        assert VisionProcessor._offset_x(0, 0, 0) == 0.0          # no width
