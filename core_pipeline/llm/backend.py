"""
LLM Backend Abstraction Layer.

Supports provider switching via environment variables.
Pattern reused from consumer-signal-agentic-platform.

Tier strategy:
  - parser (lightweight): claude-haiku-4-5-20251001
  - generator (high-quality): claude-sonnet-4-20250514
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import anthropic

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    provider: str
    model: str
    max_tokens: int = 4096
    temperature: float = 0.0


# Default tier configuration
DEFAULT_TIERS = {
    "parser": LLMConfig(
        provider="anthropic",
        model=os.getenv("PARSER_MODEL", "claude-haiku-4-5-20251001"),
        max_tokens=4096,
        temperature=0.0,
    ),
    "generator": LLMConfig(
        provider="anthropic",
        model=os.getenv("GENERATOR_MODEL", "claude-sonnet-4-6"),
        max_tokens=4096,
        temperature=0.3,
    ),
}


class LLMBackend(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        pass

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        system: str | None = None,
        response_schema: dict | None = None,
    ) -> dict:
        pass


class AnthropicBackend(LLMBackend):
    """Anthropic Claude API backend."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)
        return response.content[0].text

    def generate_structured(
        self,
        prompt: str,
        system: str | None = None,
        response_schema: dict | None = None,
    ) -> dict:
        system_msg = system or ""
        system_msg += (
            "\n\nRespond ONLY with valid JSON. "
            "No markdown fences, no preamble, no explanation. "
            "Escape all special characters in string values properly. "
            "Keep string values concise (under 100 chars each)."
        )

        raw = self.generate(prompt=prompt, system=system_msg)

        # Strip markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Try direct parse first
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Repair common JSON issues
        repaired = self._repair_json(cleaned)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as e:
            logger.warning("JSON repair failed: %s", e)
            # Last resort: extract partial JSON
            return self._extract_partial_json(cleaned)

    @staticmethod
    def _repair_json(text: str) -> str:
        """Attempt to repair common JSON issues from LLM output."""
        import re

        # Remove control characters inside strings (newlines, tabs)
        # Replace unescaped newlines within JSON strings
        text = re.sub(r'(?<=": ")(.*?)(?="[,}\]])', 
                       lambda m: m.group(0).replace('\n', ' ').replace('\r', '').replace('\t', ' '),
                       text, flags=re.DOTALL)

        # Fix trailing commas before closing brackets
        text = re.sub(r',\s*([}\]])', r'\1', text)

        # Try to close unclosed strings/arrays/objects
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')

        if open_braces > 0 or open_brackets > 0:
            # Truncate at last complete item
            last_complete = max(text.rfind('}'), text.rfind(']'))
            if last_complete > 0:
                text = text[:last_complete + 1]
                # Re-balance
                open_braces = text.count('{') - text.count('}')
                open_brackets = text.count('[') - text.count(']')
                text += ']' * open_brackets + '}' * open_braces

        return text

    @staticmethod
    def _extract_partial_json(text: str) -> dict:
        """Extract whatever valid JSON we can from the output."""
        # Try to find the outermost JSON object
        start = text.find('{')
        if start == -1:
            return {}

        # Try progressively shorter substrings
        for end in range(len(text), start, -1):
            try:
                candidate = text[start:end]
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        return {}


# Provider registry
_PROVIDERS = {
    "anthropic": AnthropicBackend,
}


def get_backend(tier: str) -> LLMBackend:
    """Get LLM backend for specified tier.

    Args:
        tier: 'parser' or 'generator'

    Returns:
        Configured LLMBackend instance
    """
    config = DEFAULT_TIERS.get(tier)
    if config is None:
        raise ValueError(f"Unknown tier: {tier}. Available: {list(DEFAULT_TIERS.keys())}")

    backend_cls = _PROVIDERS.get(config.provider)
    if backend_cls is None:
        raise ValueError(
            f"Unknown provider: {config.provider}. Available: {list(_PROVIDERS.keys())}"
        )

    logger.info("Initializing %s backend for tier '%s' (model: %s)",
                config.provider, tier, config.model)
    return backend_cls(config)
