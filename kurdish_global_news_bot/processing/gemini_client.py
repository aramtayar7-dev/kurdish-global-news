"""
Kurdish Global News Bot — Shared Gemini REST API Client

Provides reusable async Gemini generateContent JSON calls with:
  - Comma-separated multi-key loading from GEMINI_API_KEYS / GEMINI_API_KEY
  - Immediate key rotation & per-key HTTP 429 cooldown timestamps (prevents
    hammering an already exhausted key)
  - Request pacing between consecutive API calls and fast circuit-breaking when
    all configured keys are exhausted/in cooldown
  - Robust JSON extraction (stripping accidental markdown code fences)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

import aiohttp

from config import (
    ENABLE_OPENROUTER_FALLBACK,
    GEMINI_API_KEY,
    GEMINI_KEY_COOLDOWN_SECONDS,
    GEMINI_MODEL,
    GEMINI_QUOTA_COOLDOWN_SECONDS,
    GEMINI_REQUEST_PACING_SECONDS,
    OPENROUTER_API_KEY,
    REQUEST_TIMEOUT,
)
from processing.openrouter_client import (
    NEUTRAL_EDITORIAL_SYSTEM_INSTRUCTION,
    OpenRouterClient,
    OpenRouterError,
)

logger = logging.getLogger("processing.gemini")

GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def resolve_gemini_system_instruction(
    system_prompt: Optional[str] = None,
    enforce_editorial_policy: bool = False,
) -> str:
    """Resolve the Gemini systemInstruction text using the shared editorial policy."""
    raw = (system_prompt or "").strip()
    if not raw:
        return NEUTRAL_EDITORIAL_SYSTEM_INSTRUCTION
    if enforce_editorial_policy and NEUTRAL_EDITORIAL_SYSTEM_INSTRUCTION not in raw:
        return f"{NEUTRAL_EDITORIAL_SYSTEM_INSTRUCTION}\n\n{raw}"
    return raw


class GeminiAPIError(Exception):
    """Raised when all Gemini API keys or retry attempts fail."""


class GeminiQuotaExhaustedError(GeminiAPIError):
    """Raised immediately when every configured Gemini API key is in HTTP 429 cooldown."""


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


class GeminiClient:
    """Shared async Gemini REST API client with multi-key rotation and 429 cooldowns."""

    def __init__(
        self,
        api_key: Optional[str | list[str]] = None,
        model: Optional[str] = None,
        session: Optional[aiohttp.ClientSession] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
        base_backoff: float = 1.5,
        max_rate_limit_retries: int = 2,
        cooldown_store: Optional[dict[str, float]] = None,
        cooldown_seconds: Optional[float] = None,
        quota_cooldown_seconds: Optional[float] = None,
        pacing_delay: Optional[float] = None,
    ) -> None:
        raw_key = GEMINI_API_KEY if api_key is None else api_key
        self._api_keys = _parse_api_keys(raw_key)

        self._model = (GEMINI_MODEL if model is None else model).strip() or "gemini-2.5-flash"
        self._session = session
        self._semaphore = semaphore
        self._current_key_index: int = 0
        self._base_backoff = max(0.0, base_backoff)
        self._max_rate_limit_retries = max(0, max_rate_limit_retries)

        self._cooldowns: dict[str, float] = {} if cooldown_store is None else cooldown_store
        self._cooldown_seconds = (
            GEMINI_KEY_COOLDOWN_SECONDS if cooldown_seconds is None else max(0.0, cooldown_seconds)
        )
        self._quota_cooldown_seconds = (
            GEMINI_QUOTA_COOLDOWN_SECONDS
            if quota_cooldown_seconds is None
            else max(self._cooldown_seconds, quota_cooldown_seconds)
        )
        # Skip inter-request sleep in fast unit tests (base_backoff == 0) unless overridden
        if pacing_delay is not None:
            self._pacing_delay = max(0.0, pacing_delay)
        elif self._base_backoff == 0.0:
            self._pacing_delay = 0.0
        else:
            self._pacing_delay = max(0.0, GEMINI_REQUEST_PACING_SECONDS)

        self._last_request_ts: float = 0.0
        self._exhausted_warned_until: float = 0.0
        self._rate_limit_hits_cycle: int = 0
        self._exhausted_events_cycle: int = 0
        self._openrouter_client: Optional[OpenRouterClient] = None
        self._openrouter_rescued_cycle: int = 0

    def consume_cycle_stats(self) -> dict[str, int]:
        """Return and reset per-cycle Gemini HTTP 429 & quota exhaustion counters."""
        stats = {
            "http_429_hits": int(self._rate_limit_hits_cycle),
            "quota_exhausted_events": int(self._exhausted_events_cycle),
            "openrouter_rescued": int(self._openrouter_rescued_cycle),
        }
        self._rate_limit_hits_cycle = 0
        self._exhausted_events_cycle = 0
        self._openrouter_rescued_cycle = 0
        return stats

    @property
    def has_keys(self) -> bool:
        return bool(self._api_keys)

    @property
    def has_fallback(self) -> bool:
        """Return True if an alternative LLM provider (such as OpenRouter) is configured and ready."""
        import config

        if not getattr(config, "ENABLE_OPENROUTER_FALLBACK", True):
            return False
        return bool(
            getattr(config, "OPENROUTER_API_KEY", "")
            or getattr(config, "OPENROUTER_API_KEYS", [])
        )

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

    def _is_key_cooling(self, key: str, now: Optional[float] = None) -> bool:
        ref = time.monotonic() if now is None else now
        return ref < self._cooldowns.get(key, 0.0)

    def all_keys_in_cooldown(self, now: Optional[float] = None) -> bool:
        """Return True when configured keys exist and every key is currently cooling down."""
        if not self._api_keys:
            return False
        ref = time.monotonic() if now is None else now
        return all(self._is_key_available(k, ref) is False for k in self._api_keys)

    def _is_key_available(self, key: str, now: float) -> bool:
        return now >= self._cooldowns.get(key, 0.0)

    def _mark_key_cooldown(self, key: str, seconds: float) -> float:
        until = time.monotonic() + max(0.0, seconds)
        prev = self._cooldowns.get(key, 0.0)
        self._cooldowns[key] = max(prev, until)
        return self._cooldowns[key]

    def _select_available_key_index(self, now: float) -> Optional[int]:
        """Return first non-cooling key index starting from current index, else None."""
        total = len(self._api_keys)
        if total == 0:
            return None
        for offset in range(total):
            idx = (self._current_key_index + offset) % total
            if self._is_key_available(self._api_keys[idx], now):
                return idx
        return None

    def _earliest_cooldown_expiry(self) -> float:
        if not self._api_keys:
            return 0.0
        return min(self._cooldowns.get(k, 0.0) for k in self._api_keys)

    async def _pace_request(self) -> None:
        if self._pacing_delay <= 0 or self._last_request_ts <= 0:
            self._last_request_ts = time.monotonic()
            return
        elapsed = time.monotonic() - self._last_request_ts
        remaining = self._pacing_delay - elapsed
        if remaining > 0:
            await asyncio.sleep(remaining)
        self._last_request_ts = time.monotonic()

    async def _get_session(self) -> aiohttp.ClientSession:
        loop = asyncio.get_running_loop()
        sess = self._session
        sess_loop = getattr(sess, "_loop", None)
        loop_mismatch = (
            isinstance(sess_loop, asyncio.AbstractEventLoop)
            and sess_loop != loop
        )
        if sess is None or getattr(sess, "closed", False) or loop_mismatch:
            timeout = aiohttp.ClientTimeout(total=max(REQUEST_TIMEOUT, 30))
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"Content-Type": "application/json"},
            )
        return self._session

    def _get_openrouter_client(self) -> Optional[OpenRouterClient]:
        """Lazy-initialize OpenRouter client if fallback is enabled and credentials exist."""
        import config

        if not getattr(config, "ENABLE_OPENROUTER_FALLBACK", True):
            return None
        or_keys = getattr(config, "OPENROUTER_API_KEY", "")
        if not or_keys and not getattr(config, "OPENROUTER_API_KEYS", []):
            return None
        if self._openrouter_client is None:
            self._openrouter_client = OpenRouterClient(
                api_key=or_keys or getattr(config, "OPENROUTER_API_KEYS", []),
                session=self._session,
            )
        return self._openrouter_client

    async def cleanup(self) -> None:
        """Close the underlying aiohttp session if open."""
        if self._openrouter_client:
            await self._openrouter_client.cleanup()
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
        Rotates cooled-down keys immediately on HTTP 429 and short-circuits
        when every key is already in cooldown.
        Fails over cleanly to OpenRouter free models if all Gemini keys are unavailable.
        """
        resolved_system_prompt = resolve_gemini_system_instruction(system_prompt)
        if not self._api_keys:
            if openrouter := self._get_openrouter_client():
                logger.info("No Gemini keys configured; delegating to OpenRouter fallback...")
                try:
                    res = await openrouter.generate_json(
                        resolved_system_prompt, user_content, temperature, max_tokens
                    )
                    self._openrouter_rescued_cycle += 1
                    return res
                except Exception as exc:
                    logger.warning("OpenRouter fallback failed: %s", exc)
            raise GeminiAPIError("GEMINI_API_KEY is not configured.")

        now = time.monotonic()
        if self.all_keys_in_cooldown(now):
            self._exhausted_events_cycle += 1
            if now >= self._exhausted_warned_until:
                earliest = self._earliest_cooldown_expiry()
                wait_left = max(1.0, earliest - now)
                logger.warning(
                    "All %d Gemini API key(s) are in HTTP 429 cooldown (%.0fs left); "
                    "skipping Gemini requests cleanly until reset.",
                    len(self._api_keys),
                    wait_left,
                )
                self._exhausted_warned_until = earliest
            if openrouter := self._get_openrouter_client():
                logger.info("All Gemini keys in 429 cooldown; failing over to OpenRouter free models...")
                try:
                    res = await openrouter.generate_json(
                        resolved_system_prompt, user_content, temperature, max_tokens
                    )
                    self._openrouter_rescued_cycle += 1
                    return res
                except Exception as exc:
                    logger.warning("OpenRouter fallback failed after Gemini 429 cooldown: %s", exc)
            raise GeminiQuotaExhaustedError("All Gemini API keys are cooling down after HTTP 429.")

        generation_config: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        }
        if "2.5" in self._model:
            generation_config["thinkingConfig"] = {"thinkingBudget": 0}

        payload = {
            "systemInstruction": {"parts": [{"text": resolved_system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_content}]}],
            "generationConfig": generation_config,
        }
        endpoint = f"{GEMINI_API_BASE_URL}/{self._model}:generateContent"

        try:
            if self._semaphore is not None:
                async with self._semaphore:
                    return await self._execute_with_retries(endpoint, payload)
            return await self._execute_with_retries(endpoint, payload)
        except (GeminiQuotaExhaustedError, GeminiAPIError) as gemini_err:
            if openrouter := self._get_openrouter_client():
                logger.info(
                    "Gemini execution failed (%s); failing over to OpenRouter free models...",
                    gemini_err,
                )
                try:
                    res = await openrouter.generate_json(
                        resolved_system_prompt, user_content, temperature, max_tokens
                    )
                    self._openrouter_rescued_cycle += 1
                    return res
                except Exception as fallback_err:
                    logger.warning("OpenRouter fallback failed after Gemini error: %s", fallback_err)
            raise

    async def _execute_with_retries(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        session = await self._get_session()
        total_keys = len(self._api_keys)
        # Try every non-cooling key plus limited all-keys-busy backoff passes
        max_attempts = max(total_keys, total_keys + self._max_rate_limit_retries)
        last_error = ""
        rate_limit_hits = 0

        for attempt in range(max_attempts):
            now = time.monotonic()
            avail_idx = self._select_available_key_index(now)
            if avail_idx is not None:
                self._current_key_index = avail_idx
            else:
                # All keys are cooling down; only wait/retry within this call if configured
                if self._base_backoff <= 0 or attempt >= max_attempts - 1:
                    break
                exponent = max(0, rate_limit_hits - total_keys)
                delay = min(self._base_backoff * (2 ** exponent), 8.0)
                if delay > 0:
                    logger.info("All Gemini keys cooling down; transient backoff sleep %.1fs...", delay)
                    await asyncio.sleep(delay)
                # Pick key whose cooldown expires soonest
                self._current_key_index = min(
                    range(total_keys), key=lambda i: self._cooldowns.get(self._api_keys[i], 0.0)
                )

            await self._pace_request()
            key = self._api_keys[self._current_key_index]
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": key,
            }

            try:
                async with session.post(endpoint, headers=headers, json=payload) as response:
                    if response.status == 200:
                        self._cooldowns.pop(key, None)
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
                    err_lower = err_text.lower()
                    last_error = f"HTTP {response.status}: {err_text[:180]}"
                    is_auth_err = response.status in (401, 403)
                    is_daily_quota = (
                        "per day" in err_lower
                        or "daily" in err_lower
                        or "resource_exhausted" in err_lower
                    )
                    is_rate_or_auth = (
                        response.status in (401, 403, 429)
                        or "quota" in err_lower
                        or "rate" in err_lower
                    )

                    if is_rate_or_auth:
                        rate_limit_hits += 1
                        self._rate_limit_hits_cycle += 1
                        old_idx = self._current_key_index
                        cd = (
                            self._quota_cooldown_seconds
                            if (is_auth_err or is_daily_quota)
                            else self._cooldown_seconds
                        )
                        self._mark_key_cooldown(key, cd)

                        next_avail = self._select_available_key_index(time.monotonic())
                        if next_avail is not None:
                            self._current_key_index = next_avail
                            logger.warning(
                                "Gemini key [%d] hit HTTP %d (%.0fs cooldown). Rotating immediately to key [%d].",
                                old_idx,
                                response.status,
                                cd,
                                self._current_key_index,
                            )
                            continue

                        # All keys hit 429 / auth; rotate index and apply bounded backoff only if retries remain
                        self._current_key_index = (self._current_key_index + 1) % total_keys
                        if attempt < max_attempts - 1 and self._base_backoff > 0:
                            exponent = max(0, rate_limit_hits - total_keys)
                            delay = min(self._base_backoff * (2 ** exponent), 8.0)
                            logger.warning(
                                "Gemini key [%d] hit HTTP %d; all %d key(s) rate-limited. Backing off %.1fs.",
                                old_idx,
                                response.status,
                                total_keys,
                                delay,
                            )
                            await asyncio.sleep(delay)
                            continue
                        break

                    if response.status >= 500 and attempt < max_attempts - 1:
                        self._current_key_index = (self._current_key_index + 1) % total_keys
                        if self._base_backoff > 0:
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

        if self.all_keys_in_cooldown():
            self._exhausted_events_cycle += 1
            earliest = self._earliest_cooldown_expiry()
            wait_left = max(1.0, earliest - time.monotonic())
            logger.warning(
                "All %d Gemini API key(s) exhausted after HTTP 429/quota errors (%.0fs cooldown active). Last error: %s",
                total_keys,
                wait_left,
                last_error[:120],
            )
            self._exhausted_warned_until = earliest
            raise GeminiQuotaExhaustedError(f"All Gemini API keys exhausted. Last error: {last_error}")

        raise GeminiAPIError(f"All Gemini API keys/retries exhausted. Last error: {last_error}")
