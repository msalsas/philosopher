"""Lightweight emotion classifier using the FER+ ONNX model.

Replaces DeepFace (TensorFlow) with a single ~35MB ONNX model run via
onnxruntime — far lighter on the Raspberry Pi 4 / ARM64 and frees CPU cores
that would otherwise be contended with Whisper STT. Any failure (missing model,
onnxruntime not installed, bad ROI) degrades gracefully to ``None`` so the
caller falls back to "neutral".
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

# FER+ output order (emotion-ferplus-8), mapped to this project's emotion vocab
# (happy/sad/angry/surprised/neutral/fear) used by personalities and servos.
_FERPLUS_LABELS = [
    "neutral",    # neutral
    "happy",      # happiness
    "surprised",  # surprise
    "sad",        # sadness
    "angry",      # anger
    "angry",      # disgust -> closest supported pose
    "fear",       # fear
    "angry",      # contempt -> closest supported pose
]

_DEFAULT_MODEL_PATH = "./data/fer_models/emotion-ferplus-8.onnx"
_MODEL_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/"
    "body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
)


class EmotionClassifier:
    """Classifies the dominant emotion of a face ROI via the FER+ ONNX model."""

    def __init__(self, model_path: str = "", mock: bool = False) -> None:
        self.mock = mock
        self.model_path = model_path or _DEFAULT_MODEL_PATH
        self._session = None
        self._input_name: str | None = None
        self._disabled = mock

    def _ensure_session(self) -> bool:
        """Lazily load the ONNX session. Returns False if unavailable."""
        if self._session is not None:
            return True
        if self._disabled:
            return False
        try:
            import onnxruntime as ort
        except Exception:
            self._disabled = True
            return False

        path = Path(self.model_path)
        if not path.exists() and not self._download_model(path):
            self._disabled = True
            return False

        try:
            self._session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
        except Exception:
            self._disabled = True
            return False
        return True

    def _download_model(self, path: Path) -> bool:
        """One-time download of the FER+ model. Returns False on failure."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(_MODEL_URL, str(path))  # noqa: S310
            return path.exists()
        except Exception:
            return False

    def predict(self, face_bgr: np.ndarray) -> str | None:
        """Return the dominant emotion label, or None if unavailable."""
        if not self._ensure_session():
            return None
        try:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
            tensor = resized.astype(np.float32).reshape(1, 1, 64, 64)
            logits = self._session.run(None, {self._input_name: tensor})[0][0]
            idx = int(np.argmax(logits))
            if 0 <= idx < len(_FERPLUS_LABELS):
                return _FERPLUS_LABELS[idx]
        except Exception:
            pass
        return None
