"""Thin client for an OpenAI-compatible chat completions endpoint.

Groq, OpenRouter, Together, Fireworks, and a locally-proxied Gemini all
expose the same /chat/completions shape, so one small httpx wrapper (no
provider SDK) lets the deployment swap providers via env vars alone.
"""
import json
import os
from dataclasses import dataclass
from typing import List, Optional

import httpx


class LLMError(Exception):
    pass


@dataclass
class LLMConfig:
    api_key: str
    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "llama-3.3-70b-versatile"
    timeout_seconds: float = 20.0
    # Some providers (Gemini's OpenAI-compat endpoint) spend the completion
    # token budget on hidden reasoning by default, starving the actual JSON
    # output, unless this is set. Others (Groq) reject the field outright if
    # present at all. So it's opt-in per-deployment via env var, not a
    # hardcoded default -- see .env.example.
    reasoning_effort: str = ""

    @classmethod
    def from_env(cls) -> "LLMConfig":
        api_key = os.environ.get("LLM_API_KEY", "")
        if not api_key:
            raise LLMError(
                "LLM_API_KEY is not set. Set it to a Groq/OpenRouter/other "
                "OpenAI-compatible API key (see .env.example)."
            )
        return cls(
            api_key=api_key,
            base_url=os.environ.get("LLM_BASE_URL", cls.base_url),
            model=os.environ.get("LLM_MODEL", cls.model),
            timeout_seconds=float(os.environ.get("LLM_TIMEOUT_SECONDS", cls.timeout_seconds)),
            reasoning_effort=os.environ.get("LLM_REASONING_EFFORT", cls.reasoning_effort),
        )


class LLMClient:
    """Callable-injectable client so tests can substitute a stub."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()

    def complete_json(self, messages: List[dict], temperature: float = 0.2, max_tokens: int = 2048) -> str:
        """Sends a chat completion request asking for a JSON object reply.
        Returns the raw text content (caller parses/validates it)."""
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.config.reasoning_effort:
            payload["reasoning_effort"] = self.config.reasoning_effort
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=self.config.timeout_seconds) as client:
                resp = client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as e:
            raise LLMError(f"LLM request timed out: {e}") from e
        except httpx.HTTPError as e:
            raise LLMError(f"LLM request failed: {e}") from e

        if resp.status_code != 200:
            raise LLMError(f"LLM request returned {resp.status_code}: {resp.text[:500]}")

        try:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise LLMError(f"Unexpected LLM response shape: {e}") from e
