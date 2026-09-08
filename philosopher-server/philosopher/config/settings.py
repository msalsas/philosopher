"""Centralized configuration using Pydantic Settings."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PHILOSOPHER_LLM_")
    provider: Literal["local", "openai", "ollama", "custom"] = "local"
    # any OpenAI-compatible server (Ollama/llama.cpp/vLLM/LM Studio/OpenAI)
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "not-needed"  # placeholder for local servers; set your real key only for OpenAI
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
    personality: str = "philosopher"
    name: str = "Philosopher"
    language: str = "es"
    # Wake-word gate (opt-in): toy starts asleep, answers only after the
    # activation word; phrases live in the personality YAML (`wake:` section).
    wake_enabled: bool = False
    sleep_timeout: float = 1800.0  # 30 min of inactivity -> back to sleep


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
    # Transcription language. Empty = follow the app language
    # (PHILOSOPHER_LANGUAGE); set PHILOSOPHER_STT_LANGUAGE to override (e.g. force
    # "en" while the personality stays another language).
    language: str = ""
    # faster-whisper quantization. "int8" suits CPU (the RPi4); use "float16" on
    # a CUDA GPU. Decoupled from device so a laptop/GPU can be tuned via env.
    compute_type: str = "int8"
    confidence_threshold: float = -0.5
    # CPU threads for faster-whisper decoding. 0 = auto (all cores). On a
    # power-marginal Pi, lowering this (e.g. 2) shrinks the current spike that
    # can trigger under-voltage throttling -- trading peak speed for consistency.
    cpu_threads: int = 0


class VisionSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PHILOSOPHER_CAMERA_")
    fps: float = 0.5
    quality: int = 60
    # Path to the FER+ emotion ONNX model. Empty = auto-download to ./data/fer_models/
    emotion_model_path: str = ""
    # --- Frame-processing optimizations (cheap gates before expensive dlib) ---
    # Width (px) the frame is downscaled to for detection; 0 disables downscaling.
    detect_width: int = 480
    # Cheap OpenCV Haar presence check before running dlib; skips the expensive
    # encoding path entirely on the (common) frames with nobody in view.
    presence_gate: bool = True
    # Skip frames that barely changed vs the last processed one (static scene).
    skip_similar: bool = True
    # Mean per-pixel grayscale delta (0-255) below which a frame counts as "same".
    skip_threshold: float = 2.0
    # Face-match tolerance: a face is recognized as a known person only if its
    # encoding is within this distance. Lower = stricter (fewer false positives,
    # e.g. a poor camera matching everyone to one person); too low = fails to
    # recognize the same person across frames. face_recognition default is 0.6.
    tolerance: float = 0.6
    # Name-collision merge: a new face is folded into a same-named known face only
    # if their encodings are within this distance. Must be ≥ tolerance; the gap
    # (tolerance..merge_band) is the "same person, face drifted" band. Tune on hardware.
    merge_band: float = 0.68


class TTSSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PHILOSOPHER_TTS_")
    model_path: str = ""
    # Voice + lang default to "" = auto: derived from PHILOSOPHER_LANGUAGE +
    # provider (see _DEFAULTS / resolved_voice / resolved_lang), so switching
    # language is a single knob. Set them only to force a specific voice.
    voice: str = ""
    enabled: bool = True
    # "kokoro" (local neural, shipped), "piper" (local, offline), or "edge"
    # (Microsoft online neural). See .env.example for what each needs.
    provider: str = "kokoro"
    # edge-tts prosody (used only when provider="edge").
    rate: str = "+0%"
    pitch_hz: str = "+0Hz"
    # kokoro lang (espeak code); "" = auto from PHILOSOPHER_LANGUAGE.
    lang: str = ""
    speed: float = 1.0

    # Per-(provider, app-language) default (voice, tts-lang). tts-lang matters
    # only for kokoro (an espeak code: es / en-us); piper/edge ignore it.
    _DEFAULTS = {
        "kokoro": {"es": ("em_santa", "es"), "en": ("am_adam", "en-us")},
        "piper": {"es": ("es_ES-carlfm-x_low", ""), "en": ("en_US-lessac-medium", "")},
        "edge": {"es": ("es-ES-AlvaroNeural", ""), "en": ("en-US-GuyNeural", "")},
    }

    def _default_pair(self, language: str) -> tuple[str, str]:
        by_lang = self._DEFAULTS.get(self.provider, self._DEFAULTS["kokoro"])
        return by_lang.get(language, by_lang.get("es", ("", "")))

    def resolved_voice(self, language: str) -> str:
        """The voice to use: explicit `voice`, else the language/provider default."""
        return self.voice or self._default_pair(language)[0]

    def resolved_lang(self, language: str) -> str:
        """The kokoro espeak lang: explicit `lang`, else the language default."""
        return self.lang or self._default_pair(language)[1]
    # Piper synthesis params (defaults = piper's own defaults). Higher
    # noise_scale = livelier pitch, higher noise_w = livelier rhythm,
    # length_scale <1 = faster speech.
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w: float = 0.8
    # Post pitch/formant shift on the output WAV. 1.0 = none; <1 = deeper
    # (lower pitch AND formants -> a bigger, plush-animal voice).
    pitch: float = 1.0


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
    # Load .env into the process environment FIRST. The nested settings models
    # (LLMSettings, MemorySettings, …) are built via default_factory and read
    # os.environ, NOT the .env file — only the top-level Settings has env_file.
    # Without this, PHILOSOPHER_* in .env are silently ignored on a plain
    # `python -m` run and the app falls back to defaults (e.g. localhost LLM).
    # override=False so real env vars (e.g. systemd EnvironmentFile) still win.
    load_dotenv(override=False)
    s = Settings()
    s.ensure_dirs()
    return s
