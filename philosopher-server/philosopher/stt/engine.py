"""Server-side batch STT using faster-whisper."""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class StreamingSTT:
    """Batch STT that accumulates audio and transcribes on speech_ended."""

    def __init__(self, model_size: str = "tiny", device: str = "cpu",
                 compute_type: str = "int8", mock: bool = False):
        self.mock = mock
        self.model_size = model_size
        self.model = None
        self.load_error: str | None = None
        self._warned_unavailable = False
        if not mock:
            try:
                from faster_whisper import WhisperModel
                self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            except Exception as exc:  # noqa: BLE001
                # Real mode but the model didn't load. This is NOT a benign
                # degrade: without STT the toy can't hear anything. Make it loud
                # here (once) and surface it on /health — never silently pretend.
                self.load_error = repr(exc)
                logger.error(
                    "STT model '%s' failed to load (%s) — speech will be DROPPED, "
                    "not transcribed. Check faster-whisper install / model cache.",
                    model_size, exc,
                )
        self.audio_buffers: dict[str, bytearray] = {}

    def status(self) -> dict:
        """Engine readiness for /health."""
        if self.mock:
            return {"status": "mock", "model": self.model_size}
        if self.model is None:
            return {"status": "unavailable", "model": self.model_size, "error": self.load_error}
        return {"status": "ok", "model": self.model_size}

    def add_audio_chunk(self, toy_id: str, pcm_bytes: bytes):
        if toy_id not in self.audio_buffers:
            self.audio_buffers[toy_id] = bytearray()
        self.audio_buffers[toy_id].extend(pcm_bytes)

    def clear_buffer(self, toy_id: str):
        self.audio_buffers[toy_id] = bytearray()

    async def transcribe(
        self, toy_id: str, language: str = "es", confidence_threshold: float = -0.5,
    ) -> dict:
        if self.mock:
            return {
                "text": "Hello world",
                "is_final": True,
                "confidence": 0.9,
                "low_confidence": False,
            }

        if self.model is None:
            # Real mode, no model: drop the utterance rather than fabricate text.
            # Returning empty makes the caller a no-op (the toy stays silent),
            # which is the safe failure — far better than answering phantom speech.
            if not self._warned_unavailable:
                logger.warning(
                    "STT unavailable (model failed to load); dropping speech for %s "
                    "(this warning is logged once)", toy_id,
                )
                self._warned_unavailable = True
            self.audio_buffers[toy_id] = bytearray()
            return {
                "text": "", "is_final": True, "confidence": 0.0,
                "low_confidence": False, "unavailable": True,
            }

        buffer = self.audio_buffers.get(toy_id, bytearray())
        if len(buffer) < 16000:
            return {
                "text": "",
                "is_final": True,
                "confidence": 0.0,
                "low_confidence": False,
            }

        audio_np = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
        segments, info = self.model.transcribe(
            audio_np, language=language, condition_on_previous_text=False,
        )

        # `segments` is a one-shot generator — materialize it once, otherwise the
        # second pass is empty (and min([]) raises ValueError).
        seg_list = list(segments)
        text = " ".join(seg.text for seg in seg_list)
        confidence = min((seg.avg_logprob for seg in seg_list), default=-1.0)

        self.audio_buffers[toy_id] = bytearray()

        return {
            "text": text.strip(),
            "is_final": True,
            "confidence": confidence,
            "low_confidence": confidence < confidence_threshold,
        }
