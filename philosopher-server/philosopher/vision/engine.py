"""Server-side face detection and emotion recognition."""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from philosopher.vision.emotion import EmotionClassifier


class VisionProcessor:
    """Processes JPEG frames from the toy for face detection and emotion.

    The expensive step is dlib (``face_recognition`` locations + encodings). To
    keep the RPi4 responsive, three cheap gates run before it:
      1. **Static-frame skip** — a frame that barely changed vs the last one
         reuses the previous result (no work at all).
      2. **Downscale** — detection runs on a width-capped copy; boxes are scaled
         back for the emotion crop. ``offset_x`` is normalized, so scale-invariant.
      3. **Presence gate** — a cheap OpenCV Haar check; if it sees no face, dlib
         is skipped entirely (the common "nobody around" case).
    All three fail *open* (run the full path) so they can never silently blind
    the toy.
    """

    _EMPTY: dict = {"faces": [], "primary_face": None}

    def __init__(self, memory_manager=None, mock: bool = False,
                 emotion_model_path: str = "", tolerance: float = 0.6,
                 detect_width: int = 480, presence_gate: bool = True,
                 skip_similar: bool = True, skip_threshold: float = 2.0,
                 merge_band: float = 0.68):
        self.memory = memory_manager
        self.mock = mock
        self.tolerance = tolerance
        # A new face whose name collides with a known one is merged into it only
        # if their encodings are within this (relaxed, > tolerance) distance —
        # i.e. "probably the same person whose face drifted", not a namesake.
        self.merge_band = merge_band
        # face_id -> {"name": str|None, "encoding": np.ndarray|None}
        self.known_faces: dict[str, dict] = {}
        self._loaded = False
        self.emotion = EmotionClassifier(model_path=emotion_model_path, mock=mock)
        # Optimization knobs.
        self.detect_width = detect_width
        self.presence_gate = presence_gate
        self.skip_similar = skip_similar
        self.skip_threshold = skip_threshold
        self._haar: cv2.CascadeClassifier | None = None
        self._last_thumb: np.ndarray | None = None
        self._last_result: dict = self._EMPTY

    def status(self) -> dict:
        """Engine readiness for /health.

        dlib is imported lazily inside ``_recognize`` (fail-open), so probe it
        here once to report whether real recognition is actually available vs.
        silently returning no faces.
        """
        if self.mock:
            return {"status": "mock"}
        try:
            import face_recognition  # noqa: F401
            dlib_ok = True
        except Exception:  # noqa: BLE001
            dlib_ok = False
        emotion_ok = self.emotion._ensure_session()
        if dlib_ok:
            return {"status": "ok", "face_recognition": True, "emotion": emotion_ok}
        return {"status": "degraded", "face_recognition": False, "emotion": emotion_ok}

    async def process_frame(self, jpeg_bytes: bytes) -> dict:
        if self.mock:
            return {"faces": [], "primary_face": None}

        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return {"faces": [], "primary_face": None}

        # Gate 1: skip frames near-identical to the last processed one.
        thumb = self._thumb(frame)
        if (self.skip_similar and self._last_thumb is not None
                and self._frame_delta(thumb) < self.skip_threshold):
            return self._last_result

        # Gate 2: downscale for detection.
        small, scale = self._downscale(frame)

        # White-balance the detection frame: the NoIR camera's purple cast
        # otherwise defeats the HOG face detector (verified on-device).
        small = self._white_balance(small)

        # Gate 3: cheap presence check before the expensive dlib path.
        if self.presence_gate and not self._has_face_fast(small):
            return self._remember(thumb, {"faces": [], "primary_face": None})

        faces = await self._recognize(frame, small, scale)
        result = {
            "faces": faces,
            "primary_face": max(faces, key=lambda f: f["confidence"]) if faces else None,
        }
        return self._remember(thumb, result)

    async def _recognize(self, frame, small, scale: float) -> list[dict]:
        """Run the dlib pipeline on the downscaled frame; the expensive path."""
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        try:
            import face_recognition
            locations = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, locations)
        except Exception:
            return []

        small_w = small.shape[1]
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
            # Scale the box back to full-res coords for a sharper emotion crop.
            bbox = (int(left * scale), int(top * scale),
                    int(right * scale), int(bottom * scale))
            emotion = await self._detect_emotion(frame, bbox)

            faces.append({
                "face_id": face_id,
                "name": name,
                "emotion": emotion or "neutral",
                # Horizontal position for head tracking: -1 (left) .. 0 .. +1 (right).
                # Normalized, so computing it in downscaled coords is identical.
                "offset_x": self._offset_x(left, right, small_w),
                "confidence": 0.9,
            })
        return faces

    @staticmethod
    def _white_balance(frame: np.ndarray) -> np.ndarray:
        """Gray-world white balance: neutralizes a colour cast (e.g. the NoIR
        camera's purple/IR tint) so it doesn't defeat face detection."""
        b, g, r = cv2.split(frame.astype(np.float32))
        mb, mg, mr = (b.mean() or 1.0, g.mean() or 1.0, r.mean() or 1.0)
        k = (mb + mg + mr) / 3.0
        b = np.clip(b * k / mb, 0, 255)
        g = np.clip(g * k / mg, 0, 255)
        r = np.clip(r * k / mr, 0, 255)
        return cv2.merge([b, g, r]).astype(np.uint8)

    def _downscale(self, frame) -> tuple[np.ndarray, float]:
        """Return (downscaled frame, scale) where full = small * scale."""
        width = frame.shape[1]
        if self.detect_width <= 0 or width <= self.detect_width:
            return frame, 1.0
        scale = width / self.detect_width
        height = int(frame.shape[0] / scale)
        small = cv2.resize(frame, (self.detect_width, height), interpolation=cv2.INTER_AREA)
        return small, scale

    def _has_face_fast(self, frame_bgr) -> bool:
        """Cheap Haar presence check. Fails open (True) if the cascade is missing."""
        cascade = self._haar_cascade()
        if cascade is None or cascade.empty():
            return True
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        found = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30),
        )
        return len(found) > 0

    def _haar_cascade(self) -> cv2.CascadeClassifier | None:
        if self._haar is None:
            path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(str(path))
        return self._haar

    @staticmethod
    def _thumb(frame) -> np.ndarray:
        """Tiny grayscale signature of a frame for cheap change detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (32, 24), interpolation=cv2.INTER_AREA).astype(np.int16)

    def _frame_delta(self, thumb: np.ndarray) -> float:
        return float(np.mean(np.abs(thumb - self._last_thumb)))

    def _remember(self, thumb: np.ndarray, result: dict) -> dict:
        self._last_thumb = thumb
        self._last_result = result
        return result

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

    def _distance(self, id_a: str, id_b: str) -> float | None:
        """Euclidean distance between two known faces' encodings, or None."""
        a = self.known_faces.get(id_a, {}).get("encoding")
        b = self.known_faces.get(id_b, {}).get("encoding")
        if a is None or b is None or getattr(a, "shape", None) != getattr(b, "shape", None):
            return None
        return float(np.linalg.norm(a - b))

    def maybe_merge(self, src_id: str, dst_id: str, name: str) -> bytes | None:
        """Biometric guard for name-collision dedup.

        If the freshly-minted `src` face is within `merge_band` of the known
        `dst` face, treat them as the same (drifted) person: repoint the in-memory
        map so future frames resolve to `dst`, and return src's encoding bytes for
        the DB merge. Otherwise return None — a genuine namesake, keep separate.
        """
        dist = self._distance(src_id, dst_id)
        if dist is None or dist > self.merge_band:
            return None
        src = self.known_faces.pop(src_id, None)
        enc = src.get("encoding") if src else None
        if enc is None:
            return None
        # The canonical id now answers to the encoding we just saw.
        self.known_faces[dst_id] = {"name": name, "encoding": enc}
        return enc.tobytes()

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
