"""Server-side face detection and emotion recognition."""
from __future__ import annotations

import hashlib

import cv2
import numpy as np

from philosopher.vision.emotion import EmotionClassifier


class VisionProcessor:
    """Processes JPEG frames from the toy for face detection and emotion."""

    def __init__(self, memory_manager=None, mock: bool = False,
                 emotion_model_path: str = "", tolerance: float = 0.6):
        self.memory = memory_manager
        self.mock = mock
        self.tolerance = tolerance
        # face_id -> {"name": str|None, "encoding": np.ndarray|None}
        self.known_faces: dict[str, dict] = {}
        self._loaded = False
        self.emotion = EmotionClassifier(model_path=emotion_model_path, mock=mock)

    async def process_frame(self, jpeg_bytes: bytes) -> dict:
        if self.mock:
            return {"faces": [], "primary_face": None}

        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return {"faces": [], "primary_face": None}

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            import face_recognition
            locations = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, locations)
        except Exception:
            return {"faces": [], "primary_face": None}

        width = frame.shape[1]
        faces = []
        for (top, right, bottom, left), encoding in zip(
            locations, encodings, strict=False,
        ):
            face_id, name = await self._match_face(encoding)
            if face_id is None:
                # New face: mint a stable id from the full encoding and persist it.
                face_id = hashlib.sha256(encoding.tobytes()).hexdigest()[:12]
                self.known_faces[face_id] = {"name": None, "encoding": encoding}
                if self.memory:
                    await self.memory.long.add_face(face_id, encoding.tobytes())
            emotion = await self._detect_emotion(frame, (left, top, right, bottom))

            faces.append({
                "face_id": face_id,
                "name": name,
                "emotion": emotion or "neutral",
                # Horizontal position for head tracking: -1 (left) .. 0 .. +1 (right).
                "offset_x": self._offset_x(left, right, width),
                "confidence": 0.9,
            })

        return {
            "faces": faces,
            "primary_face": max(faces, key=lambda f: f["confidence"]) if faces else None,
        }

    @staticmethod
    def _offset_x(left: int, right: int, width: int) -> float:
        """Face-center horizontal offset normalized to [-1, 1] (0 = centered)."""
        if width <= 0:
            return 0.0
        center = (left + right) / 2.0
        return max(-1.0, min(1.0, (center - width / 2.0) / (width / 2.0)))

    async def _load_known(self) -> None:
        if self._loaded:
            return
        if self.memory:
            for f in await self.memory.long.all_faces():
                enc = f.get("encoding")
                self.known_faces[f["face_id"]] = {
                    "name": f.get("name"),
                    "encoding": np.frombuffer(enc, dtype=np.float64) if enc else None,
                }
        self._loaded = True

    async def _match_face(self, encoding) -> tuple[str | None, str | None]:
        """Return (face_id, name) of the closest known face within tolerance.

        Uses Euclidean distance over the 128-d encoding — dlib is only needed to
        *generate* encodings, not to compare them, so this stays testable.
        """
        await self._load_known()
        best_id, best_name, best_dist = None, None, self.tolerance
        for fid, data in self.known_faces.items():
            e = data.get("encoding")
            if e is None or getattr(e, "shape", None) != encoding.shape:
                continue
            dist = float(np.linalg.norm(e - encoding))
            if dist <= best_dist:
                best_dist, best_id, best_name = dist, fid, data.get("name")
        return best_id, best_name

    async def _detect_emotion(self, frame, bbox) -> str | None:
        x, y, x2, y2 = bbox
        roi = frame[max(0, y):y2, max(0, x):x2]
        if roi.size == 0:
            return None
        return self.emotion.predict(roi)
