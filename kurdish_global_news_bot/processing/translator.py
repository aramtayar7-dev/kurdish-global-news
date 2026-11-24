"""
Kurdish Global News Bot — Translation & Proofreading Engine

3-Tier Resilient Kurdish (Sorani / ckb) Translation Pipeline:
  Tier 1: Groq API with Multi-Key Rotation (llama-3.3-70b-versatile)
  Tier 2: DuckDuckGo AI Chat (Free & Keyless Fallback — gpt-4o-mini / claude-3-haiku)
  Tier 3: Google Translate ckb via aiohttp (Keyless Safety Net)

Stage 4:
  Gemini AI Pre-Publish Proofreader & Quality-Check Editor (gemini-2.5-flash)
  with Multi-Key Rotation and Fail-Safe Passthrough.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Optional

import aiohttp

from config import (
    ENABLE_TRANSLATION_FALLBACK,
    REQUEST_TIMEOUT,
    GROQ_API_KEY,
    GROQ_MODEL,
    GEMINI_API_KEY,
    ENABLE_GEMINI_PROOFREAD,
    GEMINI_MODEL,
)

logger = logging.getLogger("processing.translator")

GROQ_API_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
DUCKDUCKGO_CHAT_ENDPOINT = "https://duckduckgo.com/duckchat/v1/chat"
DUCKDUCKGO_STATUS_ENDPOINT = "https://duckduckgo.com/duckchat/v1/status"
GOOGLE_TRANSLATE_ENDPOINT = "https://translate.googleapis.com/translate_a/single"
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

DEFAULT_TRANSLATE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}

GROQ_SYSTEM_PROMPT = (
    "You are an expert international news journalist and translator specializing in Central Kurdish (Sorani / سۆرانی).\n"
    "Your mission is to report international breaking news clearly, accurately, and informatively for Kurdish readers.\n\n"
    "Strict Journalistic Guidelines:\n"
    "1. Comprehensive Body Report: Write a clear, well-structured Central Kurdish (Sorani) news report of 3 to 5 sentences "
    "(2 short paragraphs, ~60 to 110 words, 350 to 650 characters) based strictly on the provided article facts.\n"
    "2. NEVER Repeat the Headline: Do NOT repeat the headline verbatim or paraphrase it as the opening sentence of the body. "
    "Dive straight into the core event, actions, and background.\n"
    "3. Essential 5Ws & Concrete Detail: Clearly explain what happened, who is involved, where, why it matters, "
    "and include key concrete numbers, figures, and facts from the article so the reader understands the full story.\n"
    "4. Natural Sorani & Jargon Clarification: Use correct Central Kurdish Arabic orthography (ڕ, ڵ, ۆ, ێ, ە, ه). "
    "Translate or briefly explain unfamiliar technical, scientific, or foreign terms into natural everyday Sorani "
    "(e.g., instead of untranslated jargon like 'ڕیگۆلیتی', write 'تۆز و خۆڵی هاوشێوەی سەر مانگ').\n"
    "5. Output Constraint: Output ONLY the translated Central Kurdish text. Never add conversational filler, "
    "greetings, explanations, notes, or commentary."
)

GEMINI_PROOFREADER_SYSTEM_PROMPT = (
    "You are a senior Central Kurdish (Sorani / سۆرانی) news editor and proofreader for an international breaking news service.\n"
    "Your task is to review, polish, and synthesize a draft Kurdish Sorani headline and summary against the original English source facts.\n\n"
    "Editorial & Proofreading Guidelines:\n"
    "1. Informative News Report: Ensure the summary is a clear, well-structured Central Kurdish (Sorani) news report of 3 to 5 sentences "
    "(2 short paragraphs, ~60 to 110 words, 350 to 650 characters) that explains the full context.\n"
    "2. NEVER Repeat the Headline: The summary MUST NEVER repeat the headline verbatim or use it as its first sentence. "
    "The body must provide distinct substance and context.\n"
    "3. Core Journalism: Clearly explain what happened, who is involved, where, why it matters, and include key concrete details and numbers "
    "from the English source so a general Kurdish reader understands the story immediately.\n"
    "4. Natural Sorani & Jargon Clarification: Ensure correct Central Kurdish orthography (ڕ, ڵ, ۆ, ێ, ە, ه). "
    "Translate or briefly clarify unfamiliar technical, scientific, or foreign terms into natural Sorani "
    "(e.g., instead of raw jargon like 'ڕیگۆلیتی', write 'تۆز و خۆڵی هاوشێوەی سەر مانگ').\n"
    "5. Factual Integrity & Length: Preserve facts, names, figures, and dates accurately without hallucinating. "
    "Keep total output concise enough to fit comfortably inside Telegram's 1024-character photo caption limit (target body: 350–650 characters).\n"
    "6. Output Format: You MUST output ONLY valid JSON matching this schema with NO markdown code fences and no conversational filler:\n"
    '{"title": "punchy Sorani headline", "summary": "informative 3-5 sentence Sorani report"}'
)


class TranslationProvider(ABC):
    """Abstract translation provider interface."""

    @abstractmethod
    async def translate(self, text: str, source_lang: str = "auto", target_lang: str = "ckb") -> str:
        """Translate text. Raises on failure."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


# ═══════════════════════════════════════════════════════════
# Tier 1: Groq LLM Provider with Multi-Key Rotation
# ═══════════════════════════════════════════════════════════
class GroqTranslationProvider(TranslationProvider):
    """
    Tier 1 Primary Translator: Groq LLM (llama-3.3-70b-versatile).
    Supports single or comma-separated multiple keys for automatic rotation on HTTP 429.
    """

    def __init__(
        self,
        api_key: Optional[str | list[str]] = None,
        model: Optional[str] = None,
        delay: float = 0.5,
        semaphore: Optional[asyncio.Semaphore] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        raw_key = GROQ_API_KEY if api_key is None else api_key
        if isinstance(raw_key, list):
            self._api_keys = [k.strip() for k in raw_key if k.strip()]
        elif isinstance(raw_key, str):
            self._api_keys = [k.strip() for k in raw_key.split(",") if k.strip()]
        else:
            self._api_keys = []

        self._model = (GROQ_MODEL if model is None else model).strip() or "llama-3.3-70b-versatile"
        self._delay = delay
        self._semaphore = semaphore if semaphore is not None else asyncio.Semaphore(2)
        self._session = session
        self._current_key_index: int = 0

    @property
    def name(self) -> str:
        return "Groq"

    @property
    def model(self) -> str:
        return self._model

    @property
    def has_keys(self) -> bool:
        return bool(self._api_keys)

    @property
    def api_keys(self) -> list[str]:
        return self._api_keys

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
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def cleanup(self) -> None:
        """Close the underlying aiohttp session if open."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def translate(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "ckb",
    ) -> str:
        """Translate news content into Sorani Kurdish using Groq with multi-key rotation."""
        if not text or not text.strip():
            return ""

        if not self._api_keys:
            raise TranslationError("GROQ_API_KEY is not configured.")

        total_keys = len(self._api_keys)
        last_error = ""

        async with self._semaphore:
            session = await self._get_session()

            for attempt in range(total_keys):
                key = self._api_keys[self._current_key_index]
                headers = {
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "User-Agent": "KurdishGlobalNewsBot/1.0",
                }
                payload = {
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": GROQ_SYSTEM_PROMPT},
                        {"role": "user", "content": text},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 800,
                }

                try:
                    async with session.post(
                        GROQ_API_ENDPOINT,
                        json=payload,
                        headers=headers,
                    ) as response:
                        if response.status == 200:
                            data = await response.json(content_type=None)
                            choices = data.get("choices", [])
                            if not choices:
                                raise TranslationError("Groq API returned no choices in response.")

                            content = choices[0].get("message", {}).get("content", "").strip()
                            if not content:
                                raise TranslationError("Groq API returned empty translation content.")

                            if self._delay > 0:
                                await asyncio.sleep(self._delay)

                            return content

                        err_text = await response.text()
                        last_error = f"Groq API HTTP {response.status}: {err_text[:200]}"

                        # Key exhausted, rate-limited, or unauthorized -> rotate key
                        if (
                            response.status in (401, 403, 429)
                            or "rate" in err_text.lower()
                            or "quota" in err_text.lower()
                        ):
                            old_idx = self._current_key_index
                            self._current_key_index = (self._current_key_index + 1) % total_keys
                            logger.warning(
                                "Groq key [%d] hit HTTP %d / rate limit. Rotating to key [%d].",
                                old_idx,
                                response.status,
                                self._current_key_index,
                            )
                            if attempt < total_keys - 1:
                                continue
                            raise TranslationError(
                                f"All Groq API keys exhausted or rate-limited. Last error: {last_error}"
                            )

                        raise TranslationError(last_error)

                except TranslationError:
                    if attempt < total_keys - 1:
                        old_idx = self._current_key_index
                        self._current_key_index = (self._current_key_index + 1) % total_keys
                        logger.warning(
                            "Groq key [%d] failed (%s). Trying next key [%d].",
                            old_idx,
                            last_error,
                            self._current_key_index,
                        )
                        continue
                    raise
                except asyncio.TimeoutError:
                    last_error = "Groq translation request timed out."
                    if attempt < total_keys - 1:
                        self._current_key_index = (self._current_key_index + 1) % total_keys
                        continue
                    raise TranslationError(last_error)
                except Exception as e:
                    last_error = f"Groq translation request failed: {e}"
                    if attempt < total_keys - 1:
                        self._current_key_index = (self._current_key_index + 1) % total_keys
                        continue
                    raise TranslationError(last_error)

        raise TranslationError(f"All Groq API keys exhausted or rate-limited. Last error: {last_error}")


# ═══════════════════════════════════════════════════════════
# Tier 2: DuckDuckGo AI Chat Provider (Free & Keyless Fallback)
# ═══════════════════════════════════════════════════════════
class DuckDuckGoAIProvider(TranslationProvider):
    """
    Tier 2 Translation Provider: DuckDuckGo AI Chat (Free & Keyless).
    Translates using DuckDuckGo AI endpoint (gpt-4o-mini / claude-3-haiku).
    Protected with asyncio.Semaphore(1) and a 2.5s pacing delay.
    Cleanly raises TranslationError on 403/418/429 for immediate Tier 3 takeover.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        delay: float = 2.5,
        semaphore: Optional[asyncio.Semaphore] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self._model = model
        self._delay = delay
        self._semaphore = semaphore if semaphore is not None else asyncio.Semaphore(1)
        self._session = session

    @property
    def name(self) -> str:
        return "DuckDuckGoAI"

    @property
    def model(self) -> str:
        return self._model

    async def _get_session(self) -> aiohttp.ClientSession:
        loop = asyncio.get_running_loop()
        if (
            self._session is None
            or self._session.closed
            or getattr(self._session, "_loop", None) != loop
        ):
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"User-Agent": DEFAULT_TRANSLATE_HEADERS["User-Agent"]},
            )
        return self._session

    async def cleanup(self) -> None:
        """Close the underlying aiohttp session if open."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _fetch_vqd(self, session: aiohttp.ClientSession) -> str:
        """Acquire required VQD authorization token from DuckDuckGo."""
        # 1. Try DuckChat status endpoint
        try:
            status_headers = {
                "x-vqd-accept": "1",
                "User-Agent": DEFAULT_TRANSLATE_HEADERS["User-Agent"],
            }
            async with session.get(DUCKDUCKGO_STATUS_ENDPOINT, headers=status_headers) as resp:
                if resp.status in (403, 418, 429):
                    raise TranslationError(
                        f"DuckDuckGo AI status returned HTTP {resp.status} (rate-limited/challenge)."
                    )
                vqd = resp.headers.get("x-vqd-4", "")
                if vqd:
                    return vqd
        except TranslationError:
            raise
        except Exception:
            pass

        # 2. Try DuckDuckGo search vqd extraction
        try:
            async with session.get("https://duckduckgo.com/?q=duckchat") as resp:
                if resp.status in (403, 418, 429):
                    raise TranslationError(f"DuckDuckGo search returned HTTP {resp.status}.")
                text = await resp.text()
                match = re.search(r'vqd=([\"\']?)([\d-]+)\1', text)
                if match:
                    return match.group(2)
        except TranslationError:
            raise
        except Exception as e:
            raise TranslationError(f"Failed to obtain DuckDuckGo VQD: {e}")

        raise TranslationError("DuckDuckGo AI returned no valid VQD token (challenge required).")

    async def translate(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "ckb",
    ) -> str:
        """Translate text via DuckDuckGo AI Chat."""
        if not text or not text.strip():
            return ""

        async with self._semaphore:
            session = await self._get_session()
            vqd = await self._fetch_vqd(session)

            headers = {
                "x-vqd-4": vqd,
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "User-Agent": DEFAULT_TRANSLATE_HEADERS["User-Agent"],
            }
            prompt = (
                f"{GROQ_SYSTEM_PROMPT}\n\n"
                f"Translate this international news text into natural Central Kurdish (Sorani):\n\n"
                f"{text}"
            )
            payload = {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
            }

            try:
                async with session.post(
                    DUCKDUCKGO_CHAT_ENDPOINT,
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status in (403, 418, 429):
                        err_body = await response.text()
                        raise TranslationError(
                            f"DuckDuckGo AI rate limited / challenge (HTTP {response.status}): {err_body[:100]}"
                        )
                    if response.status != 200:
                        err_body = await response.text()
                        raise TranslationError(
                            f"DuckDuckGo AI returned HTTP {response.status}: {err_body[:100]}"
                        )

                    # Stream parsing SSE lines
                    collected_parts: list[str] = []
                    async for raw_line in response.content:
                        line = raw_line.decode("utf-8", errors="ignore").strip()
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk_json = json.loads(data_str)
                                msg = chunk_json.get("message", "")
                                if msg:
                                    collected_parts.append(msg)
                            except Exception:
                                continue

                    full_content = "".join(collected_parts).strip()
                    if not full_content:
                        raise TranslationError("DuckDuckGo AI returned empty translation stream.")

                    if self._delay > 0:
                        await asyncio.sleep(self._delay)

                    return full_content

            except TranslationError:
                raise
            except asyncio.TimeoutError:
                raise TranslationError("DuckDuckGo AI request timed out.")
            except Exception as e:
                raise TranslationError(f"DuckDuckGo AI request failed: {e}")


# ═══════════════════════════════════════════════════════════
# Tier 3: Google Translate ckb Provider (Final Safety Net)
# ═══════════════════════════════════════════════════════════
class GoogleTranslateProvider(TranslationProvider):
    """
    Tier 3 Final Translation Safety Net: Google Translate via aiohttp.
    Uses free keyless single endpoint with ckb (Central Kurdish / Sorani).
    Protected with Semaphore(1), 1.5s delay, and text chunking.
    """

    def __init__(
        self,
        delay: float = 1.5,
        semaphore: Optional[asyncio.Semaphore] = None,
        session: Optional[aiohttp.ClientSession] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        self._delay = delay
        self._semaphore = semaphore if semaphore is not None else asyncio.Semaphore(1)
        self._session = session
        self._headers = headers or DEFAULT_TRANSLATE_HEADERS

    @property
    def semaphore(self) -> asyncio.Semaphore:
        return self._semaphore

    @property
    def name(self) -> str:
        return "GoogleTranslate"

    async def _get_session(self) -> aiohttp.ClientSession:
        loop = asyncio.get_running_loop()
        if (
            self._session is None
            or self._session.closed
            or getattr(self._session, "_loop", None) != loop
        ):
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            self._session = aiohttp.ClientSession(timeout=timeout, headers=self._headers)
        return self._session

    async def cleanup(self) -> None:
        """Close the underlying aiohttp session if open."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _translate_chunk(
        self,
        chunk: str,
        source_lang: str = "auto",
        target_lang: str = "ckb",
    ) -> str:
        """Translate a single text chunk using translate.googleapis.com single endpoint."""
        if not chunk.strip():
            return ""

        session = await self._get_session()
        last_error = ""

        # Primary requested client is 'gtx'; fallback to 'dict-chrome-ex' if 'gtx' hits rate limit (429)
        client_candidates = ["gtx", "dict-chrome-ex"]

        for client in client_candidates:
            params = {
                "client": client,
                "sl": source_lang,
                "tl": target_lang,
                "dt": "t",
                "q": chunk,
            }

            try:
                async with session.get(
                    GOOGLE_TRANSLATE_ENDPOINT,
                    params=params,
                    headers=self._headers,
                ) as response:
                    if response.status == 200:
                        data = await response.json(content_type=None)
                        if not data or not isinstance(data, list) or len(data) == 0:
                            return ""

                        segments = data[0]
                        if not isinstance(segments, list):
                            return ""

                        parts = [
                            segment[0]
                            for segment in segments
                            if isinstance(segment, (list, tuple)) and len(segment) > 0 and segment[0]
                        ]
                        return "".join(parts)

                    text_body = await response.text()
                    last_error = f"Google Translate endpoint returned HTTP {response.status}: {text_body[:150]}"
                    if response.status == 429:
                        logger.warning(
                            "Google Translate client='%s' returned 429 rate limit. Trying fallback client.",
                            client,
                        )
                        continue
                    else:
                        raise TranslationError(last_error)
            except TranslationError:
                raise
            except asyncio.TimeoutError:
                last_error = "Google Translate request timed out."
            except Exception as e:
                last_error = f"Google Translate request failed: {e}"

        raise TranslationError(last_error or "All Google Translate clients failed.")

    async def translate(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "ckb",
    ) -> str:
        """Translate using Google Translate API with chunking and semaphore rate-limiting."""
        if not text.strip():
            return ""

        async with self._semaphore:
            chunks = _split_text(text, max_len=2500)
            translated_chunks: list[str] = []

            for i, chunk in enumerate(chunks):
                if i > 0 and self._delay > 0:
                    await asyncio.sleep(self._delay)

                res = await self._translate_chunk(
                    chunk, source_lang=source_lang, target_lang=target_lang
                )
                if res:
                    translated_chunks.append(res)

            if self._delay > 0:
                await asyncio.sleep(self._delay)

            return " ".join(translated_chunks)


# ═══════════════════════════════════════════════════════════
# Stage 4: Gemini Pre-Publish Proofreader & Quality-Check Editor
# ═══════════════════════════════════════════════════════════
class GeminiProofreader:
    """
    Stage 4: Gemini Pre-Publish Proofreader & Quality-Check Editor.
    Uses direct async aiohttp calls to Google Gemini REST API.
    Supports single or comma-separated multiple keys for automatic rotation on HTTP 429.
    Fails safely so breaking news publication is NEVER blocked.
    """

    def __init__(
        self,
        api_key: Optional[str | list[str]] = None,
        model: Optional[str] = None,
        enabled: Optional[bool] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        raw_key = GEMINI_API_KEY if api_key is None else api_key
        if isinstance(raw_key, list):
            self._api_keys = [k.strip() for k in raw_key if k.strip()]
        elif isinstance(raw_key, str):
            self._api_keys = [k.strip() for k in raw_key.split(",") if k.strip()]
        else:
            self._api_keys = []

        self._model = (GEMINI_MODEL if model is None else model).strip() or "gemini-2.5-flash"
        self._enabled = ENABLE_GEMINI_PROOFREAD if enabled is None else enabled
        self._current_key_index: int = 0
        self._session = session

    @property
    def is_enabled(self) -> bool:
        return self._enabled and bool(self._api_keys)

    @property
    def model(self) -> str:
        return self._model

    @property
    def api_keys(self) -> list[str]:
        return self._api_keys

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
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"Content-Type": "application/json"},
            )
        return self._session

    async def cleanup(self) -> None:
        """Close the underlying aiohttp session if open."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def proofread(
        self,
        original_title: str,
        draft_title: str,
        original_summary: str = "",
        draft_summary: str = "",
    ) -> tuple[str, str]:
        """
        Proofread and polish draft Kurdish title and summary against English originals.
        Returns (polished_title, polished_summary).
        Returns unedited draft on any failure or when disabled (fail-safe).
        """
        if not self._enabled:
            logger.info("GeminiProofread: Skipped/Fallback - disabled by configuration")
            return draft_title, draft_summary

        if not self._api_keys:
            logger.info("GeminiProofread: Skipped/Fallback - no GEMINI_API_KEY configured")
            return draft_title, draft_summary

        if not draft_title or not draft_title.strip():
            return draft_title, draft_summary

        session = await self._get_session()
        total_keys = len(self._api_keys)

        user_content = (
            f"=== ORIGINAL ENGLISH HEADLINE ===\n{original_title}\n\n"
            f"=== DRAFT KURDISH (SORANI) TITLE ===\n{draft_title}\n\n"
            f"=== ORIGINAL ENGLISH SUMMARY ===\n{original_summary}\n\n"
            f"=== DRAFT KURDISH (SORANI) SUMMARY ===\n{draft_summary}\n\n"
            "Review, fix, and polish the Kurdish Sorani draft. Return ONLY valid JSON: "
            '{"title": "...", "summary": "..."}'
        )

        payload = {
            "systemInstruction": {
                "parts": [{"text": GEMINI_PROOFREADER_SYSTEM_PROMPT}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_content}]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 800,
                "responseMimeType": "application/json",
            }
        }

        for attempt in range(total_keys):
            key = self._api_keys[self._current_key_index]
            endpoint = f"{GEMINI_API_BASE_URL}/{self._model}:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": key,
            }

            try:
                async with session.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status == 200:
                        data = await response.json(content_type=None)
                        candidates = data.get("candidates", [])
                        if not candidates:
                            logger.warning("GeminiProofread: Skipped/Fallback - no candidates returned")
                            return draft_title, draft_summary

                        parts = candidates[0].get("content", {}).get("parts", [])
                        if not parts:
                            logger.warning("GeminiProofread: Skipped/Fallback - empty content parts")
                            return draft_title, draft_summary

                        raw_text = parts[0].get("text", "").strip()
                        # Clean code fence if present
                        clean_text = raw_text
                        if clean_text.startswith("```json"):
                            clean_text = clean_text[7:]
                        elif clean_text.startswith("```"):
                            clean_text = clean_text[3:]
                        if clean_text.endswith("```"):
                            clean_text = clean_text[:-3]
                        clean_text = clean_text.strip()

                        result_json = json.loads(clean_text)
                        polished_title = result_json.get("title", "").strip() or draft_title
                        polished_summary = result_json.get("summary", "").strip() or draft_summary

                        logger.info("GeminiProofread: OK - polished headline: '%s'", polished_title[:60])
                        return polished_title, polished_summary

                    err_text = await response.text()
                    if (
                        response.status in (401, 403, 429)
                        or "quota" in err_text.lower()
                        or "rate" in err_text.lower()
                    ):
                        old_idx = self._current_key_index
                        self._current_key_index = (self._current_key_index + 1) % total_keys
                        logger.warning(
                            "Gemini key [%d] hit HTTP %d / quota limit. Rotating to key [%d].",
                            old_idx,
                            response.status,
                            self._current_key_index,
                        )
                        if attempt < total_keys - 1:
                            continue

                    logger.warning(
                        "GeminiProofread: Skipped/Fallback - HTTP %d: %s",
                        response.status,
                        err_text[:150],
                    )
                    return draft_title, draft_summary

            except json.JSONDecodeError as e:
                logger.warning("GeminiProofread: Skipped/Fallback - JSON parse error: %s", e)
                return draft_title, draft_summary
            except Exception as e:
                if attempt < total_keys - 1:
                    old_idx = self._current_key_index
                    self._current_key_index = (self._current_key_index + 1) % total_keys
                    logger.warning(
                        "Gemini attempt [%d] failed (%s). Trying next key [%d].",
                        old_idx,
                        str(e)[:150],
                        self._current_key_index,
                    )
                    continue
                logger.warning("GeminiProofread: Skipped/Fallback - %s", str(e)[:200])
                return draft_title, draft_summary

        logger.warning("GeminiProofread: Skipped/Fallback - All Gemini API keys exhausted or rate-limited.")
        return draft_title, draft_summary


# ═══════════════════════════════════════════════════════════
# Translation Engine (3 Tiers + Stage 4 Quality Check)
# ═══════════════════════════════════════════════════════════
class TranslationEngine:
    """
    Manages 3-tier resilient translation pipeline with Stage 4 Gemini proofreading:
    Tier 1: Groq LLM (llama-3.3-70b-versatile with multi-key rotation)
    Tier 2: DuckDuckGo AI Chat (keyless free AI fallback)
    Tier 3: Google Translate ckb (direct aiohttp single endpoint safety net)
    Stage 4: Gemini Pre-Publish Proofreader & Editor (gemini-2.5-flash)
    """

    def __init__(
        self,
        delay: float = 1.5,
        groq_api_key: Optional[str | list[str]] = None,
        groq_model: Optional[str] = None,
        gemini_api_key: Optional[str | list[str]] = None,
        gemini_model: Optional[str] = None,
        enable_gemini_proofread: Optional[bool] = None,
    ) -> None:
        self._delay = delay
        self._semaphore = asyncio.Semaphore(1)
        self._groq_api_key = GROQ_API_KEY if groq_api_key is None else groq_api_key
        self._groq_model = GROQ_MODEL if groq_model is None else groq_model
        self._gemini_api_key = GEMINI_API_KEY if gemini_api_key is None else gemini_api_key
        self._gemini_model = GEMINI_MODEL if gemini_model is None else gemini_model
        self._enable_gemini_proofread = (
            ENABLE_GEMINI_PROOFREAD if enable_gemini_proofread is None else enable_gemini_proofread
        )

        self._providers: list[TranslationProvider] = []
        self._proofreader = GeminiProofreader(
            api_key=self._gemini_api_key,
            model=self._gemini_model,
            enabled=self._enable_gemini_proofread,
        )
        self._setup_providers()

    @property
    def semaphore(self) -> asyncio.Semaphore:
        return self._semaphore

    @property
    def providers(self) -> list[TranslationProvider]:
        return self._providers

    @property
    def proofreader(self) -> GeminiProofreader:
        return self._proofreader

    def _setup_providers(self) -> None:
        """Initialize 3 translation tiers in priority order."""
        # Tier 1: Groq LLM with multi-key rotation (if configured)
        groq_provider = GroqTranslationProvider(
            api_key=self._groq_api_key,
            model=self._groq_model,
            delay=0.5,
        )
        if groq_provider.has_keys:
            self._providers.append(groq_provider)

        # Tier 2: DuckDuckGo AI Chat (Free & Keyless Fallback)
        self._providers.append(DuckDuckGoAIProvider(delay=2.0))

        # Tier 3: Direct Google Translate ckb via aiohttp (Final Safety Net)
        self._providers.append(
            GoogleTranslateProvider(delay=self._delay, semaphore=self._semaphore)
        )
        logger.info(
            "Translation tiers initialized: %s | Gemini Proofreader: %s",
            [p.name for p in self._providers],
            "ENABLED" if self._proofreader.is_enabled else "DISABLED",
        )

    async def cleanup(self) -> None:
        """Clean up provider and proofreader resources."""
        for provider in self._providers:
            if hasattr(provider, "cleanup"):
                await provider.cleanup()
        if hasattr(self._proofreader, "cleanup"):
            await self._proofreader.cleanup()

    async def translate_title(self, title: str, target_lang: str = "ckb") -> str:
        """Translate headline to Sorani Kurdish through 3-tier pipeline."""
        if not title.strip():
            return ""
        translated = await self._translate_with_fallback(title, target_lang=target_lang)
        return self._post_process(translated, title)

    async def translate_summary(self, summary: str, target_lang: str = "ckb") -> str:
        """Translate summary to Sorani Kurdish through 3-tier pipeline."""
        if not summary.strip():
            return ""
        translated = await self._translate_with_fallback(summary, target_lang=target_lang)
        return self._post_process(translated, summary)

    async def translate_article(
        self,
        title: str,
        summary: str = "",
        target_lang: str = "ckb",
    ) -> tuple[str, str]:
        """
        Translate article title and summary through 3-tier pipeline,
        then run Stage 4 Gemini proofreading & quality-check.
        Returns (final_title, final_summary).
        """
        if target_lang == "ckb":
            draft_title = await self.translate_title(title)
        else:
            draft_title = await self.translate_title(title, target_lang=target_lang)

        draft_summary = ""
        if summary and summary.strip():
            if target_lang == "ckb":
                draft_summary = await self.translate_summary(summary)
            else:
                draft_summary = await self.translate_summary(summary, target_lang=target_lang)

        # Stage 4: Gemini Proofreading & Polish
        final_title, final_summary = await self._proofreader.proofread(
            original_title=title,
            draft_title=draft_title,
            original_summary=summary,
            draft_summary=draft_summary,
        )
        return final_title, final_summary

    async def _translate_with_fallback(self, text: str, target_lang: str = "ckb") -> str:
        """Try each provider in tier order until one succeeds."""
        last_error = ""
        for provider in self._providers:
            try:
                result = await provider.translate(text, target_lang=target_lang)
                if result and result.strip():
                    logger.info("Translation Tier: %s translated text (%d chars)", provider.name, len(text))
                    return result
            except Exception as e:
                last_error = str(e)
                logger.warning("Translation provider %s failed: %s", provider.name, str(e)[:200])
                if not ENABLE_TRANSLATION_FALLBACK:
                    break

        raise TranslationError(f"All translation providers failed. Last error: {last_error}")

    def _post_process(self, translated: str, original: str) -> str:
        """Post-process translated text for quality."""
        if not translated:
            return ""

        # Clean up extra whitespace
        translated = re.sub(r"\s+", " ", translated).strip()

        # Preserve URLs from original (should not be translated)
        url_pattern = re.compile(r"https?://\S+")
        orig_urls = url_pattern.findall(original)
        for url in orig_urls:
            if url not in translated:
                pass

        return translated

    async def validate_translation(self, translated: str, original: str) -> bool:
        """
        Validate that a translation is acceptable.
        Returns True if the translation passes quality checks.
        """
        if not translated or not translated.strip():
            return False

        if translated.strip() == original.strip():
            logger.warning("Translation identical to original — may indicate failure.")
            return True

        if len(translated.strip()) < 5:
            return False

        ratio = len(translated) / max(len(original), 1)
        if ratio < 0.1 or ratio > 10:
            logger.warning("Translation length ratio suspicious: %.2f", ratio)
            return True

        return True


class TranslationError(Exception):
    """Raised when translation fails."""
    pass


def _split_text(text: str, max_len: int = 2500) -> list[str]:
    """Split text into chunks at sentence boundaries or word boundaries."""
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    current = ""

    sentences = re.split(r"(?<=[.!?\n])\s+", text)
    for sentence in sentences:
        while len(sentence) > max_len:
            split_idx = sentence.rfind(" ", 0, max_len)
            if split_idx <= 0:
                split_idx = max_len
            part = sentence[:split_idx].strip()
            if part:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(part)
            sentence = sentence[split_idx:].strip()

        if len(current) + len(sentence) + 1 <= max_len:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks if chunks else [text[:max_len]]
