"""Personality engine: loads and applies personality profiles from YAML."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml


class PersonalityEngine:
    """Loads personality YAML files and builds system prompts."""

    # Localized defaults used when a personality YAML has no `fallbacks` section.
    _DEFAULT_FALLBACKS = {
        "not_understood": "Perdona, no te he entendido bien. ¿Puedes repetirlo?",
        "error": "Disculpa, ahora mismo no consigo responder. Probemos de nuevo.",
    }

    def __init__(self, settings) -> None:
        self.s = settings
        self._data: dict[str, Any] | None = None
        self.load(self.s.personality)

    def load(self, name: str | None = None) -> dict[str, Any]:
        name = name or self.s.personality
        lang = self.s.language
        base = Path(__file__).parent.parent / "config" / "personalities"
        f = base / lang / f"{name}.yaml"
        if not f.exists():
            f = base / "es" / "filosofo.yaml"
        self._data = yaml.safe_load(f.open(encoding="utf-8"))
        return self._data

    @property
    def system_prompt(self) -> str:
        return self._data.get("system_prompt", "You are a helpful assistant.")

    @property
    def name(self) -> str:
        return self.s.name

    def greeting(self, emotion: str | None = None) -> str:
        if emotion:
            r = self._data.get("emotion_responses", {}).get(emotion, [])
            if r:
                return random.choice(r)
        g = self._data.get("speech_patterns", {}).get("greetings", ["Hello."])
        return random.choice(g)

    def farewell(self) -> str:
        f = self._data.get("speech_patterns", {}).get("farewells", ["Goodbye."])
        return random.choice(f)

    def build_prompt(
        self, emotion: str | None = None, face_name: str | None = None,
        memories: list | None = None,
    ) -> str:
        parts = [self.system_prompt]
        if face_name:
            parts.append(f"You are talking to {face_name}.")
        if emotion:
            parts.append(f"The person seems {emotion}.")
        if memories:
            for m in memories[:3]:
                parts.append(f"Memory: {m.get('summary', '')[:150]}")
        return "\n".join(parts)

    def format_response(self, text: str) -> str:
        # Allow up to 2x the configured length, but cut on a word boundary
        # (with an ellipsis) instead of mid-word.
        max_len = self._data.get("style", {}).get("max_response_length", 200) * 2
        text = text.strip()
        if len(text) <= max_len:
            return text
        return text[:max_len].rsplit(" ", 1)[0].rstrip() + "…"

    def fallback(self, key: str) -> str:
        """Localized fallback phrase (from the YAML `fallbacks:` section)."""
        fb = self._data.get("fallbacks", {}) or {}
        return fb.get(key) or self._DEFAULT_FALLBACKS.get(key, "...")

    def get_name_question(self) -> str:
        return self._data.get("face_registration", {}).get(
            "ask_name_prompt", "Hola, ¿cómo te llamas?"
        )

    def list_all(self) -> list[str]:
        base = Path(__file__).parent.parent / "config" / "personalities"
        lang_dir = base / self.s.language
        if lang_dir.exists():
            return [f.stem for f in lang_dir.glob("*.yaml")]
        return []
