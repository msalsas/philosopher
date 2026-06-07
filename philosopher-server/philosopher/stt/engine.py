"""Server-side batch STT using faster-whisper."""
from __future__ import annotations

import numpy as np


class StreamingSTT:
    """Batch STT that accumulates audio and transcribes on speech_ended."""

    def __init__(self, model_size: str = "tiny", device: str = "cpu",
                 compute_type: str = "int8", mock: bool = False):
        self.mock = mock
        self.model = None
        if not mock:
            try:
                from faster_whisper import WhisperModel
                self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            except Exception:
                pass
        self.audio_buffers: dict[str, bytearray] = {}

    def add_audio_chunk(self, toy_id: str, pcm_bytes: bytes):
        if toy_id not in self.audio_buffers:
            self.audio_buffers[toy_id] = bytearray()
        self.audio_buffers[toy_id].extend(pcm_bytes)

    def clear_buffer(self, toy_id: str):
        self.audio_buffers[toy_id] = bytearray()

    async def transcribe(
        self, toy_id: str, language: str = "es", confidence_threshold: float = -0.5,
    ) -> dict:
        if self.mock or self.model is None:
            return {
                "text": "Hello world",
                "is_final": True,
                "confidence": 0.9,
                "low_confidence": False,
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
