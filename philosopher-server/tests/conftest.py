"""Shared pytest fixtures."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from philosopher.config.settings import (
    LLMSettings,
    MemorySettings,
    PersonalitySettings,
    Settings,
)
from philosopher.events.bus import EventBus


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as t:
        yield Path(t)


@pytest.fixture
def mock_settings(tmp_dir: Path) -> Settings:
    return Settings(
        llm=LLMSettings(provider="local", base_url="http://test:1234/v1",
                       api_key="test", model="test-model", max_tokens=100),
        memory=MemorySettings(db_path=str(tmp_dir / "test.db"),
                             short_term_limit=5, similarity_threshold=0.3),
        personality=PersonalitySettings(personality="philosopher", language="es", name="Test"),
    )


@pytest_asyncio.fixture
async def event_bus():
    EventBus._instance = None
    bus = EventBus()
    await bus.start()
    yield bus
    await bus.stop()
