"""Tests for vision engine."""
from __future__ import annotations

import pytest

from philosopher.vision.engine import VisionProcessor


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

    def test_offset_x(self):
        # 640px-wide frame, center at 320. Offset normalized to [-1, 1].
        assert VisionProcessor._offset_x(310, 330, 640) == 0.0    # centered
        assert VisionProcessor._offset_x(470, 490, 640) == 0.5    # right of center
        assert VisionProcessor._offset_x(150, 170, 640) == -0.5   # left of center
        assert VisionProcessor._offset_x(900, 900, 640) == 1.0    # clamped to +1
        assert VisionProcessor._offset_x(0, 0, 0) == 0.0          # no width
