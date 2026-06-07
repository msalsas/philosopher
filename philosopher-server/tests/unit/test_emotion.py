"""Tests for the FER+ ONNX emotion classifier (hermetic — no model load)."""
from __future__ import annotations

import numpy as np

from philosopher.vision.emotion import _FERPLUS_LABELS, EmotionClassifier


class TestEmotionClassifier:
    def test_mock_returns_none(self):
        clf = EmotionClassifier(mock=True)
        assert clf.predict(np.zeros((64, 64, 3), dtype=np.uint8)) is None

    def test_disabled_returns_none_without_loading(self):
        clf = EmotionClassifier(model_path="/nonexistent/model.onnx")
        clf._disabled = True  # simulate missing onnxruntime/model
        assert clf.predict(np.zeros((64, 64, 3), dtype=np.uint8)) is None

    def test_labels_map_to_supported_vocab(self):
        supported = {"happy", "sad", "angry", "surprised", "neutral", "fear"}
        assert len(_FERPLUS_LABELS) == 8
        assert set(_FERPLUS_LABELS) <= supported
