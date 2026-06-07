"""Generic LLM client supporting local and cloud providers."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import httpx
from openai import AsyncOpenAI


@dataclass
class LLMMessage:
    role: str
    content: str
    name: str | None = None

    def to_openai(self) -> dict[str, Any]:
        m: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            m["name"] = self.name
        return m


@dataclass
class LLMResponse:
    content: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""


class LLMClient:
    """OpenAI-compatible LLM client."""

    def __init__(self, settings) -> None:
        self.cfg = settings.llm
        self._client: AsyncOpenAI | None = None
        self._lock = asyncio.Lock()

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.cfg.base_url,
                api_key=self.cfg.api_key,
                timeout=httpx.Timeout(self.cfg.timeout, connect=5.0),
                max_retries=2,
            )
        return self._client

    async def chat(
        self,
        messages: list[LLMMessage],
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        async with self._lock:
            try:
                msgs: list[dict[str, Any]] = []
                if system_prompt:
                    msgs.append({"role": "system", "content": system_prompt})
                msgs.extend([m.to_openai() for m in messages])

                resp = await self.client.chat.completions.create(
                    model=self.cfg.model,
                    messages=msgs,
                    temperature=temperature or self.cfg.temperature,
                    max_tokens=max_tokens or self.cfg.max_tokens,
                )
                c = resp.choices[0]
                return LLMResponse(
                    content=c.message.content or "",
                    usage={
                        "prompt": resp.usage.prompt_tokens if resp.usage else 0,
                        "completion": resp.usage.completion_tokens if resp.usage else 0,
                        "total": resp.usage.total_tokens if resp.usage else 0,
                    },
                    finish_reason=c.finish_reason or "",
                )
            except httpx.ConnectError:
                return LLMResponse(finish_reason="connection_error")
            except httpx.TimeoutException:
                return LLMResponse(finish_reason="timeout")
            except Exception as e:
                return LLMResponse(finish_reason=f"error:{e}")

    async def chat_stream(
        self,
        messages: list[LLMMessage],
        system_prompt: str | None = None,
        max_tokens: int = 250,
        temperature: float | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream LLM tokens as they're generated."""
        msgs: list[dict[str, Any]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.extend([m.to_openai() for m in messages])

        # Exceptions propagate to the caller (e.g. _stream_response), which sends
        # a single localized fallback. Never yield error text into the stream —
        # it would be spoken aloud by the toy.
        response = await self.client.chat.completions.create(
            model=self.cfg.model,
            messages=msgs,
            max_tokens=max_tokens,
            temperature=temperature or self.cfg.temperature,
            stream=True,
        )
        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def health_check(self) -> dict[str, Any]:
        try:
            models = await self.client.models.list()
            return {"status": "ok", "models": [m.id for m in models.data]}
        except Exception as e:
            return {"status": "error", "error": str(e)}
