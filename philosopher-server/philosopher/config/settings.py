"""Centralized configuration using Pydantic Settings."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PHILOSOPHER_LLM_")
    provider: Literal["local", "openai", "ollama", "custom"] = "local"
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    model: str = "llama-3.1-8b"
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 30

    @field_validator("temperature")
    @classmethod
    def _validate(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v


class MemorySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PHILOSOPHER_MEMORY_")
    db: Literal["sqlite", "postgres"] = "sqlite"
    db_path: str = "./data/philosopher_memory.db"
    short_term_limit: int = 10
    long_term_top_k: int = 5
    similarity_threshold: float = 0.6


class PersonalitySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PHILOSOPHER_")
    personality: str = "filosofo"
    name: str = "Philosopher"
    language: str = "es"


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PHILOSOPHER_API_")
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080


class LoggingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PHILOSOPHER_")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console", "rich"] = "rich"


class STTSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PHILOSOPHER_STT_")
    model: str = "tiny"
    device: str = "cpu"
    confidence_threshold: float = -0.5


class VisionSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PHILOSOPHER_CAMERA_")
    fps: float = 0.5
    quality: int = 60
    # Path to the FER+ emotion ONNX model. Empty = auto-download to ./data/fer_models/
    emotion_model_path: str = ""


class TTSSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PHILOSOPHER_TTS_")
    model_path: str = ""
    voice: str = "es_ES-carlfm-x_low"
    enabled: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    llm: LLMSettings = Field(default_factory=LLMSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    personality: PersonalitySettings = Field(default_factory=PersonalitySettings)
    api: APISettings = Field(default_factory=APISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    stt: STTSettings = Field(default_factory=STTSettings)
    vision: VisionSettings = Field(default_factory=VisionSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)

    def ensure_dirs(self) -> None:
        Path(self.memory.db_path).parent.mkdir(parents=True, exist_ok=True)
        Path("./data").mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
