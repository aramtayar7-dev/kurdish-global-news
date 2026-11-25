"""
Kurdish Global News Bot — OpenRouter Kurdish-Optimized Free Fallback Engine

Provides resilient async OpenRouter REST API fallback with:
  - Multi-key rotation & cooldown handling across free-tier OpenRouter keys
  - Ordered failover across Central Kurdish (Sorani)-capable free models
    (Gemma 3 27B -> Llama 3.3 70B -> DeepSeek Chat v3 -> Qwen 3 235B -> OpenRouter Free)
  - Automatic JSON fallback retry when a free model rejects `response_format={"type": "json_object"}` (HTTP 400)
  - Reasoning trace stripping (<think>...</think>) and robust markdown fence extraction
  - Authentic Central Kurdish (Sorani) neutrality and linguistic integrity
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Optional

import aiohttp

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_APP_TITLE,
    OPENROUTER_APP_URL,
    OPENROUTER_BASE_URL,
    OPENROUTER_KURDISH_FREE_MODELS,
    REQUEST_TIMEOUT,
)

logger = logging.getLogger("processing.openrouter")

# Regex to strip thinking/reasoning traces emitted by reasoning models
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class OpenRouterError(Exception):
    """Raised when all OpenRouter keys, models, or retry attempts fail."""


class OpenRouterQuotaExhaustedError(OpenRouterError):
    """Raised when OpenRouter credits are exhausted (HTTP 402) or all models are rate-limited."""


def _parse_api_keys(raw_key: Optional[str | list[str]]) -> list[str]:
    """Normalize comma-separated or list API keys cleanly without duplicates."""
    items = raw_key if isinstance(raw_key, list) else (raw_key or "").split(",")
    keys: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            continue
        k = item.strip()
        if k and k not in seen:
            keys.append(k)
            seen.add(k)
    return keys


class OpenRouterClient:
    """Async OpenAI-compatible client for OpenRouter free models with multi-model failover."""

    def __init__(
        self,
        api_key: Optional[str | list[str]] = None,
        models: Optional[list[str]] = None,
        base_url: Optional[str] = None,
        session: Optional[aiohttp.ClientSession] = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        raw_key = OPENROUTER_API_KEY if api_key is None else api_key
        self._api_keys = _parse_api_keys(raw_key)
        self._models = (
            list(models)
            if models is not None and len(models) > 0
            else list(OPENROUTER_KURDISH_FREE_MODELS)
        )
        self._base_url = (base_url or OPENROUTER_BASE_URL).strip()
        self._session = session
        self._current_key_index: int = 0
        self._timeout_seconds = max(10.0, timeout_seconds)
        self._key_cooldowns: dict[str, float] = {}

    @property
    def has_keys(self) -> bool:
        return bool(self._api_keys)

    @property
    def api_keys(self) -> list[str]:
        return self._api_keys

    @property
    def models(self) -> list[str]:
        return self._models

    def _select_available_key(self) -> Optional[str]:
        if not self._api_keys:
            return None
        now = time.monotonic()
        total = len(self._api_keys)
        for offset in range(total):
            idx = (self._current_key_index + offset) % total
            candidate = self._api_keys[idx]
            if now >= self._key_cooldowns.get(candidate, 0.0):
                self._current_key_index = idx
                return candidate
        # If all keys cooling, pick the one expiring soonest
        soonest_idx = min(
            range(total),
            key=lambda i: self._key_cooldowns.get(self._api_keys[i], 0.0),
        )
        self._current_key_index = soonest_idx
        return self._api_keys[soonest_idx]

    def _mark_key_cooldown(self, key: str, seconds: float = 60.0) -> None:
        self._key_cooldowns[key] = time.monotonic() + max(0.0, seconds)

    async def _get_session(self) -> aiohttp.ClientSession:
        loop = asyncio.get_running_loop()
        sess = self._session
        sess_loop = getattr(sess, "_loop", None)
        loop_mismatch = (
            isinstance(sess_loop, asyncio.AbstractEventLoop)
            and sess_loop != loop
        )
        if sess is None or getattr(sess, "closed", False) or loop_mismatch:
            timeout = aiohttp.ClientTimeout(total=max(REQUEST_TIMEOUT, self._timeout_seconds))
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"Content-Type": "application/json"},
            )
        return self._session

    async def cleanup(self) -> None:
        """Close the underlying aiohttp session if open."""
        if self._session and not self._session.closed:
            await self._session.close()

    @staticmethod
    def _strip_markdown_and_think(raw_text: str) -> str:
        """Strip reasoning <think> blocks, markdown fences, and stray preamble."""
        cleaned = _THINK_BLOCK_RE.sub("", raw_text).strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()

    @classmethod
    def _parse_json_safely(cls, raw_content: str) -> dict[str, Any]:
        """Attempt multi-pass JSON extraction and recovery from LLM response text."""
        cleaned = cls._strip_markdown_and_think(raw_content)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # Try locating outer JSON object boundaries {...}
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start : end + 1]
            try:
                parsed = json.loads(snippet)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        raise OpenRouterError(f"Failed to parse valid JSON from OpenRouter output: {raw_content[:200]}")

    async def generate_json(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> dict[str, Any]:
        """
        Execute chat completion on OpenRouter, failing over across free Kurdish-capable models.
        Automatically retries without response_format if HTTP 400 is returned.
        """
        if not self._api_keys:
            raise OpenRouterError("OPENROUTER_API_KEY is not configured.")

        session = await self._get_session()
        last_error = ""

        # Failover across configured models
        for model in self._models:
            key = self._select_available_key()
            if not key:
                break

            headers = {
                "Authorization": f"Bearer {key}",
                "HTTP-Referer": OPENROUTER_APP_URL,
                "X-Title": OPENROUTER_APP_TITLE,
                "Content-Type": "application/json",
            }

            base_payload: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            # Attempt 1: Structured JSON object response_format
            # Attempt 2: If 400, raw JSON instruction without response_format
            attempts_for_model = [
                {"response_format": {"type": "json_object"}},
                {},  # Fallback format-less attempt
            ]

            for attempt_idx, fmt_override in enumerate(attempts_for_model):
                payload = dict(base_payload)
                payload.update(fmt_override)

                try:
                    async with session.post(
                        self._base_url,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self._timeout_seconds),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json(content_type=None)
                            choices = data.get("choices", [])
                            if not choices:
                                last_error = f"Model {model} returned empty choices list."
                                logger.warning("OpenRouter model %s returned empty choices.", model)
                                break  # try next model

                            msg = choices[0].get("message", {})
                            content = str(msg.get("content", "")).strip()
                            if not content:
                                last_error = f"Model {model} returned empty message content."
                                logger.warning("OpenRouter model %s returned empty content.", model)
                                break  # try next model

                            parsed = self._parse_json_safely(content)
                            logger.info(
                                "OpenRouter fallback successfully served request using model: %s",
                                model,
                            )
                            return parsed

                        resp_text = await resp.text()
                        status = resp.status

                        # If 400 and we sent response_format, immediately retry this model without response_format
                        if status == 400 and attempt_idx == 0:
                            logger.info(
                                "OpenRouter model %s rejected response_format (HTTP 400); retrying without format...",
                                model,
                            )
                            continue

                        # If 429 (Rate Limit), mark key cooldown and try next model
                        if status == 429:
                            self._mark_key_cooldown(key, 60.0)
                            last_error = f"Model {model} HTTP 429: {resp_text[:120]}"
                            logger.warning("OpenRouter model %s hit HTTP 429; advancing to next model.", model)
                            break

                        # If 402 (Payment / Credits required for free model)
                        if status == 402:
                            last_error = f"Model {model} HTTP 402: {resp_text[:120]}"
                            logger.warning("OpenRouter model %s returned HTTP 402; advancing to next model.", model)
                            break

                        # If 404 (Model removed or unavailable)
                        if status == 404:
                            last_error = f"Model {model} HTTP 404: {resp_text[:120]}"
                            logger.warning("OpenRouter model %s returned HTTP 404; advancing to next model.", model)
                            break

                        # 5xx or other upstream error
                        last_error = f"Model {model} HTTP {status}: {resp_text[:120]}"
                        logger.warning("OpenRouter model %s returned HTTP %d; advancing...", model, status)
                        break

                except asyncio.TimeoutError:
                    last_error = f"Model {model} request timed out after {self._timeout_seconds}s."
                    logger.warning("OpenRouter model %s timed out; advancing to next model.", model)
                    break
                except Exception as ex:
                    last_error = f"Model {model} network error: {str(ex)[:120]}"
                    logger.warning("OpenRouter model %s error (%s); advancing...", model, ex)
                    break

        raise OpenRouterError(f"All OpenRouter models failed. Last error: {last_error}")
