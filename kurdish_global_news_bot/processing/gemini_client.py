"""
Kurdish Global News Bot — Shared Gemini REST API Client

Provides reusable async Gemini generateContent JSON calls with:
  - Comma-separated multi-key rotation on HTTP 401/403/429 or quota errors
  - Automatic exponential backoff on HTTP 429 rate limits
  - Robust JSON extraction (stripping accidental markdown code fences)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

import aiohttp

from config import GEMINI_API_KEY, GEMINI_MODEL, REQUEST_TIMEOUT

logger = logging.getLogger("processing.gemini")

GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiAPIError(Exception):
    """Raised when all Gemini API keys or retry attempts fail."""


class GeminiClient:
    """Shared async Gemini REST API client with key rotation and 429 backoff."""

    def __init__(
        self,
        api_key: Optional[str | list[str]] = None,
        model: Optional[str] = None,
        session: Optional[aiohttp.ClientSession] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
        base_backoff: float = 1.5,
        max_rate_limit_retries: int = 2,
    ) -> None:
        raw_key = GEMINI_API_KEY if api_key is None else api_key
        if isinstance(raw_key, list):
            self._api_keys = [k.strip() for k in raw_key if k and k.strip()]
        elif isinstance(raw_key, str):
            self._api_keys = [k.strip() for k in raw_key.split(",") if k.strip()]
        else:
            self._api_keys = []

        self._model = (GEMINI_MODEL if model is None else model).strip() or "gemini-2.5-flash"
        self._session = session
        self._semaphore = semaphore
        self._current_key_index: int = 0
        self._base_backoff = base_backoff
        self._max_rate_limit_retries = max_rate_limit_retries

    @property
    def has_keys(self) -> bool:
        return bool(self._api_keys)

    @property
    def api_keys(self) -> list[str]:
        return self._api_keys

    @property
    def model(self) -> str:
        return self._model

    @property
    def current_key(self) -> str:
        if not self._api_keys:
            return ""
        return self._api_keys[self._current_key_index]

    async def _get_session(self) -> aiohttp.ClientSession:
        loop = asyncio.get_running_loop()
        if (
            self._session is None
            or self._session.closed
            or getattr(self._session, "_loop", None) != loop
        ):
            timeout = aiohttp.ClientTimeout(total=max(REQUEST_TIMEOUT, 30))
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
    def _strip_code_fences(raw_text: str) -> str:
        clean = raw_text.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        elif clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        return clean.strip()

    async def generate_json(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> dict[str, Any]:
        """
        Invoke Gemini generateContent and parse JSON output.
        Rotates keys and applies exponential backoff on HTTP 429 rate limits.
        """
        if not self._api_keys:
            raise GeminiAPIError("GEMINI_API_KEY is not configured.")

        generation_config: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        }
        if "2.5" in self._model:
            generation_config["thinkingConfig"] = {"thinkingBudget": 0}

        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_content}]}],
            "generationConfig": generation_config,
        }
        endpoint = f"{GEMINI_API_BASE_URL}/{self._model}:generateContent"

        if self._semaphore is not None:
            async with self._semaphore:
                return await self._execute_with_retries(endpoint, payload)
        return await self._execute_with_retries(endpoint, payload)

    async def _execute_with_retries(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        session = await self._get_session()
        total_keys = len(self._api_keys)
        total_cycles = self._max_rate_limit_retries + 1
        max_attempts = total_keys * total_cycles
        last_error = ""
        rate_limit_hits = 0

        for attempt in range(max_attempts):
            key = self._api_keys[self._current_key_index]
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": key,
            }

            try:
                async with session.post(endpoint, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json(content_type=None)
                        candidates = data.get("candidates", [])
                        if not candidates:
                            raise GeminiAPIError("Gemini API returned no candidates.")
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if not parts:
                            raise GeminiAPIError("Gemini API returned empty content parts.")

                        raw_text = str(parts[0].get("text", "")).strip()
                        clean_text = self._strip_code_fences(raw_text)
                        parsed = json.loads(clean_text)
                        if not isinstance(parsed, dict):
                            raise GeminiAPIError("Gemini JSON output was not an object.")
                        return parsed

                    err_text = await response.text()
                    last_error = f"HTTP {response.status}: {err_text[:180]}"
                    is_rate_or_auth = (
                        response.status in (401, 403, 429)
                        or "quota" in err_text.lower()
                        or "rate" in err_text.lower()
                    )

                    if is_rate_or_auth and attempt < max_attempts - 1:
                        old_idx = self._current_key_index
                        self._current_key_index = (self._current_key_index + 1) % total_keys
                        logger.warning(
                            "Gemini key [%d] hit HTTP %d. Rotating to key [%d].",
                            old_idx,
                            response.status,
                            self._current_key_index,
                        )
                        if response.status == 429 or "rate" in err_text.lower() or "quota" in err_text.lower():
                            rate_limit_hits += 1
                            # Apply exponential backoff after cycling keys or repeatedly hitting 429
                            if total_keys == 1 or rate_limit_hits >= total_keys:
                                exponent = max(0, rate_limit_hits - total_keys) if total_keys > 1 else (rate_limit_hits - 1)
                                delay = min(self._base_backoff * (2 ** exponent), 16.0)
                                if delay > 0:
                                    logger.info("Gemini HTTP 429 backoff: sleeping %.1fs...", delay)
                                    await asyncio.sleep(delay)
                        continue

                    if response.status >= 500 and attempt < max_attempts - 1:
                        self._current_key_index = (self._current_key_index + 1) % total_keys
                        await asyncio.sleep(min(self._base_backoff, 2.0))
                        continue

                    raise GeminiAPIError(f"Gemini API error ({last_error})")

            except json.JSONDecodeError as e:
                raise GeminiAPIError(f"Gemini returned invalid JSON: {e}") from e
            except GeminiAPIError:
                raise
            except asyncio.TimeoutError as e:
                last_error = "Gemini request timed out"
                if attempt < max_attempts - 1:
                    self._current_key_index = (self._current_key_index + 1) % total_keys
                    continue
                raise GeminiAPIError(last_error) from e
            except Exception as e:
                last_error = str(e)[:180]
                if attempt < max_attempts - 1:
                    self._current_key_index = (self._current_key_index + 1) % total_keys
                    continue
                raise GeminiAPIError(f"Gemini request failed: {last_error}") from e

        raise GeminiAPIError(f"All Gemini API keys/retries exhausted. Last error: {last_error}")
