"""Agent state for LangGraph processing pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentState:
    """State that flows through the LangGraph pipeline."""
    user_message: str = ""
    face_id: str | None = None
    face_name: str | None = None
    emotion: str | None = None
    is_new_face: bool = False
    system_prompt: str = ""
    short_context: list[dict] = field(default_factory=list)
    long_memories: list[dict] = field(default_factory=list)
    llm_response: str = ""
    formatted: str = ""
    error: str | None = None
    turn: int = 0
    pending_audio: bool = False
    servo_emotion: str | None = None

    @property
    def message(self) -> str:
        return self.user_message

    @classmethod
    def from_dict(cls, d: dict) -> AgentState:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return {
            "user_message": self.user_message,
            "face_id": self.face_id,
            "face_name": self.face_name,
            "emotion": self.emotion,
            "is_new_face": self.is_new_face,
            "system_prompt": self.system_prompt,
            "short_context": self.short_context,
            "long_memories": self.long_memories,
            "llm_response": self.llm_response,
            "formatted": self.formatted,
            "error": self.error,
            "turn": self.turn,
            "pending_audio": self.pending_audio,
            "servo_emotion": self.servo_emotion,
        }
