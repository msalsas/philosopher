"""Personality engine: loads and applies personality profiles from YAML."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml


class PersonalityEngine:
    """Loads personality YAML files and builds system prompts."""

    # Neutral last-resort defaults (English). Every shipped personality YAML
    # defines its own localized `fallbacks`, so these should never actually fire.
    _DEFAULT_FALLBACKS = {
        "not_understood": "Sorry, I didn't catch that. Could you repeat it?",
        "error": "Sorry, I can't respond right now. Let's try again.",
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
            f = base / "es" / "philosopher.yaml"
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

    # Wake-gate phrases (from the YAML `wake:` section). Neutral English
    # last-resort; every shipped personality defines its own localized `wake`.
    _DEFAULT_WAKE = ["wake up"]
    _DEFAULT_SLEEP = ["goodbye", "good night"]

    def wake_words(self) -> list[str]:
        w = self._data.get("wake", {}) or {}
        return w.get("activate") or self._DEFAULT_WAKE

    def sleep_words(self) -> list[str]:
        w = self._data.get("wake", {}) or {}
        return w.get("deactivate") or self._DEFAULT_SLEEP

    def build_prompt(
        self, emotion: str | None = None, face_name: str | None = None,
        memories: list | None = None,
    ) -> str:
        parts = [self.system_prompt]
        if face_name:
            parts.append(
                f"You are talking to {face_name}. Address them by their name "
                f"naturally now and then (not in every sentence).")
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
            "ask_name_prompt", "Hi, what's your name?"
        )

    # Self-introduction openers for name learning, per language (multilingual:
    # never hardcode these in the pipeline). `prefixes` are explicit ("me llamo")
    # and trusted always; `soft` ("soy") only right after we asked; `not_names`
    # are one-word replies to ignore. A personality YAML's `face_registration`
    # may override any of these; otherwise the language default applies.
    _DEFAULT_NAME_PREFIXES = {
        "es": ["me llamo", "mi nombre es"],
        "en": ["my name is", "i am called", "they call me"],
    }
    _DEFAULT_NAME_SOFT = {"es": ["soy"], "en": ["i am", "i'm"]}
    _DEFAULT_NOT_NAMES = {
        "es": ["no", "si", "hola", "que", "tal", "nada", "bien", "mal", "vale",
               "ok", "claro", "bueno", "gracias", "adios", "yo", "eh", "oye"],
        "en": ["no", "yes", "hi", "hello", "what", "nothing", "fine", "good",
               "ok", "okay", "sure", "thanks", "bye", "me", "hey"],
    }

    def _fr(self, key: str, default_map: dict) -> list[str]:
        fr = self._data.get("face_registration", {}) or {}
        return fr.get(key) or default_map.get(self.s.language, [])

    def name_prefixes(self) -> list[str]:
        return self._fr("name_prefixes", self._DEFAULT_NAME_PREFIXES)

    def name_soft_prefixes(self) -> list[str]:
        return self._fr("name_soft_prefixes", self._DEFAULT_NAME_SOFT)

    def not_names(self) -> list[str]:
        return self._fr("not_names", self._DEFAULT_NOT_NAMES)

    def list_all(self) -> list[str]:
        base = Path(__file__).parent.parent / "config" / "personalities"
        lang_dir = base / self.s.language
        if lang_dir.exists():
            return [f.stem for f in lang_dir.glob("*.yaml")]
        return []
