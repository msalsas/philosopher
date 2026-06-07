"""Tests for configuration module."""
from __future__ import annotations

import pytest

from philosopher.config.settings import LLMSettings, MemorySettings, Settings, get_settings


class TestLLMSettings:
    def test_defaults(self):
        s = LLMSettings()
        assert s.provider == "local"
        assert s.temperature == 0.7

    def test_temperature_validation(self):
        with pytest.raises(ValueError):
            LLMSettings(temperature=5.0)
        with pytest.raises(ValueError):
            LLMSettings(temperature=-1.0)
        s = LLMSettings(temperature=1.5)
        assert s.temperature == 1.5


class TestSettings:
    def test_construction(self):
        s = Settings()
        assert s.llm is not None
        assert s.memory is not None
        assert s.personality is not None

    def test_ensure_dirs(self, tmp_dir):
        s = Settings(memory=MemorySettings(db_path=str(tmp_dir / "test.db")))
        s.ensure_dirs()
        assert (tmp_dir).exists()

    def test_singleton(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
