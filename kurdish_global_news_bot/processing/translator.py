"""
Kurdish Global News Bot — Gemini Primary Translation & Kurdish Editorial Engine

Single-Pass Gemini Primary Pipeline:
  - Tier 1: Gemini AI (gemini-2.5-flash with multi-key rotation & HTTP 429 exponential backoff)
            Performs Translation + Summarization + Journalistic Editing (or Native Kurdish Rewrite)
            in ONE high-quality JSON call (`max_tokens=800`).
  - Tier 2 (Safety Net ONLY if all Gemini keys fail): Keyless Google Translate `ckb` via aiohttp.
"""

from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from typing import Optional

import aiohttp

from config import (
    ENABLE_TRANSLATION_FALLBACK,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    REQUEST_TIMEOUT,
)
from processing.gemini_client import GEMINI_API_BASE_URL, GeminiAPIError, GeminiClient

logger = logging.getLogger("processing.translator")

GOOGLE_TRANSLATE_ENDPOINT = "https://translate.googleapis.com/translate_a/single"

DEFAULT_TRANSLATE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}

GEMINI_TRANSLATION_SYSTEM_PROMPT = (
    "You are an expert international news journalist and senior Central Kurdish (Sorani / سۆرانی) editor "
    "working for Kurdish Global News — an independent, non-partisan international news organization.\n"
    "Translate and edit the given international news article into publication-ready, BBC/Reuters-style Central Kurdish (Sorani) in ONE step.\n\n"
    "Editorial Identity & Strict Neutrality Rules (Mandatory for ALL Sources):\n"
    "- Kurdish Global News is neither aligned with nor hostile toward Iran, the United States, Israel, Iraq, the Kurdistan Region, or any political, military, religious, or ideological faction.\n"
    "- NEVER reproduce propaganda slogans, insults, slurs, mockery, curses, sectarian labels, dehumanization, glorification, or demonization from the original source.\n"
    "- Report verifiable actions and facts cleanly, attributing disputed claims or allegations directly to their sources (distinguishing confirmed facts vs claims/unverified reports) without inventing details or creating artificial false equivalence.\n"
    "- Mandatory Neutral Terminology Normalization:\n"
    "  * Whenever referring to the Kurdistan Region of Iraq, normalize 'شمال العراق', 'Northern Iraq', 'باکووری عێراق', or 'باکوری عێراق' -> 'هەرێمی کوردستان'.\n"
    "  * Normalize anti-US epithets ('العدو الأمريكي', 'الاحتلال الأمريكي', 'الشيطان الأكبر', 'ئەمریکای دوژمن', 'دوژمنی ئەمریکی') -> 'ئەمریکا' or 'هێزەکانی ئەمریکا'.\n"
    "  * Normalize anti-Israel epithets ('الكيان الصهيوني', 'العدو الصهيوني', 'الكيان المؤقت', 'قەوارەی زایۆنی', 'دوژمنی زایۆنی') -> 'ئیسرائیل' or 'سوپای ئیسرائیل'.\n"
    "  * Normalize partisan Syria labels ('سوريا المحتلة', 'سووریای داگیرکراو', 'سووریای داگیرکراوە') -> 'سووریا'.\n"
    "  * Strip partisan insults, slurs, or derogatory adjectives against world leaders or officials (e.g., 'المجرم ترامب' -> 'دۆناڵد ترەمپ', 'تاوانبار نەتەنیاهۆ' -> 'بێنیامین نەتەنیاهۆ') and refer to every leader strictly by neutral name and official title.\n"
    "  * Neutralize derogatory labels against Iran, Iraq, Kurdistan Region, or any state/group while preserving independently verifiable facts of real territorial occupation, military presence, or disputed borders.\n\n"
    "Strict Journalistic & Sorani Quality Rules:\n"
    "1. Informative 3-4 Sentence Body: Write a clear, complete Sorani news report of 3 to 4 complete sentences (350 to 650 characters) "
    "so the entire post + metadata fits comfortably inside Telegram's 1024-character photo caption limit without cutting off mid-sentence.\n"
    "2. Headline & No Repetition: Write a crisp, punchy, neutral Sorani headline. NEVER repeat the headline verbatim or paraphrase it as the first sentence of the summary.\n"
    "3. 5Ws & Facts: Clearly convey what happened, who is involved, where, why it matters, and exact numbers/facts strictly from the source text. Do NOT invent facts, casualties, victories, or quotations.\n"
    "4. Accurate Terms (No Mistranslation): Translate roles and ages faithfully — NEVER translate 'man', 'official', or an older adult (e.g., a 62-year-old) as 'گەنج'; use accurate Sorani terms such as 'پیاوێکی تەمەن ٦٢ ساڵ' or 'بەرپرسێک'.\n"
    "5. Natural Sorani (No Raw Jargon): Replace untranslated foreign jargon (such as 'ئیفێمیزم' or 'ڕیگۆلیتی') with clear natural Sorani phrasing (e.g., 'دەربڕینی ناڕاستەوخۆ' or 'تۆز و خۆڵی هاوشێوەی سەر مانگ').\n"
    "6. Proper Sorani Orthography: Always use proper Sorani characters (ڕ, ڵ, ۆ, ێ, ە, ه, ک, ی) rather than Arabic ي or ك.\n"
    "7. No Agency Datelines: NEVER copy or prepend news-agency datelines (e.g., 'قامیشلی، سوریا (نۆرس پرێس) –', 'هەولێر -', 'واشنتۆن (ڕۆیتەرز) -', 'Qamishli, Syria (North Press) –') at the start of the summary. Begin directly with the first clean news sentence.\n"
    "8. Output Format: Respond ONLY with valid JSON matching this schema (NO markdown fences, NO commentary):\n"
    '{"title": "punchy Sorani headline", "summary": "3 to 4 complete Sorani sentences"}'
)

GEMINI_KURDISH_REWRITE_SYSTEM_PROMPT = (
    "You are a senior Central Kurdish (Sorani / سۆرانی) broadcast news editor at Kurdish Global News, "
    "an independent, non-partisan international news organization.\n"
    "The provided article is ALREADY written in Kurdish Sorani. Do NOT translate from English; instead summarize and rewrite it cleanly in neutral, objective BBC/Reuters-style journalistic Sorani.\n\n"
    "Editorial & Neutrality Rules:\n"
    "1. Non-Partisan Objectivity: Remove partisan hype, propaganda slogans, insults, slurs, or emotional epithets toward Iran, US, Israel, Iraq, the Kurdistan Region, or any leader/group (e.g. strip labels like 'تاوانبار'/'المجرم' before leader names), attributing disputed claims to their sources.\n"
    "2. Neutral Terminology: Normalize 'باکووری عێراق'/'باکوری عێراق'/'شمال العراق' -> 'هەرێمی کوردستان'; 'ئەمریکای دوژمن'/'دوژمنی ئەمریکی' -> 'ئەمریکا'; 'قەوارەی زایۆنی'/'دوژمنی زایۆنی' -> 'ئیسرائیل'; 'سووریای داگیرکراو' -> 'سووریا', while preserving documented factual military/territorial realities.\n"
    "3. Headline & Structure: Craft a concise, neutral Sorani headline and a 3 to 4 complete-sentence news report (350 to 650 characters) that never repeats the headline verbatim.\n"
    "4. Complete Sentences & Orthography: End every sentence with proper punctuation (. or ؟ or !) and proper Sorani letters (ڕ, ڵ, ۆ, ێ, ە, ه, ک, ی).\n"
    "5. No Agency Datelines: Strip any raw news-agency or city datelines (e.g., 'قامیشلی، سوریا (نۆرس پرێس) –', 'هەولێر -', 'واشنتۆن (ڕۆیتەرز) -') from the start of the summary so it starts cleanly with the first sentence.\n"
    "6. Output Format: Respond ONLY with valid JSON matching this schema (NO markdown fences, NO commentary):\n"
    '{"title": "clean journalistic Sorani headline", "summary": "3 to 4 complete neutral Sorani sentences"}'
)


class TranslationError(Exception):
    """Raised when translation fails across all providers."""


def normalize_sorani_chars(text: str) -> str:
    """Normalize Arabic variants (ك, ي) into standard Central Kurdish Sorani characters (ک, ی)."""
    if not text:
        return ""
    cleaned = text.replace("ك", "ک").replace("ي", "ی")
    return re.sub(r"[ \t]+", " ", cleaned).strip()


# Leading agency dateline patterns (Sorani/Arabic and English wire services)
_DATELINE_WITH_AGENCY_RE = re.compile(
    r"^\s*[^\n()—–\-\.\!\?]{1,55}?\([^()\n]{2,45}\)\s*(?:[-–—−]+|:)?\s+",
)
_DATELINE_CITY_DASH_RE = re.compile(
    r"^\s*[\w\u0600-\u06FF]+(?:[\s,،]+[\w\u0600-\u06FF]+){0,2}\s+[-–—−]+\s+",
)


def strip_news_dateline(text: str) -> str:
    """
    Strip leading raw news-agency datelines such as:
      - 'قامیشلی، سوریا (نۆرس پرێس) – '
      - 'هەولێر - '
      - 'واشنتۆن (ڕۆیتەرز) - '
      - 'Qamishli, Syria (North Press) – '
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    for _ in range(2):
        updated = _DATELINE_WITH_AGENCY_RE.sub("", cleaned, count=1)
        if updated == cleaned:
            updated = _DATELINE_CITY_DASH_RE.sub("", cleaned, count=1)
        updated = updated.strip()
        if not updated or updated == cleaned:
            break
        cleaned = updated
    return cleaned


_SORANI_DISTINCT_RE = re.compile(r"[پچژڤگڵڕێۆە]")
_ARABIC_ONLY_LETTERS_RE = re.compile(r"[أإآىًٌٍ]")
_ARABIC_STOPWORDS_RE = re.compile(
    r"(?:^|[\s،.؟!:\-\(\)])"
    r"(?:في|على|إلى|الى|عن|مع|أن|إن|التي|الذي|هذا|هذه|أعلنت|أفادت|قالت|أكدت|"
    r"خلال|عبر|منذ|كما|فقد|عاجل|بينما|حيث|ذلك|تلك|بعدما|عقب|نحو|ضد|بشأن|"
    r"وزارة|الدفاع|العراقية|الأمريكي|الأمريكية|القوات|الجيش|الحكومة)"
    r"(?=$|[\s،.؟!:\-\(\)])"
)
_SORANI_STOPWORDS_RE = re.compile(
    r"(?:^|[\s،.؟!:\-\(\)])"
    r"(?:لە|بۆ|بە|کە|لەسەر|لەگەڵ|دوای|پێش|ئەم|ئەو|بوو|بوون|دەبێت|دەکات|دەڵێت|"
    r"ڕایگەیاند|هەرێمی|هەولێر|کوردستان|ئەمریکا|ئیسرائیل|ئێران|عێراق|تورکیا|"
    r"سووریا|شەڕ|سەربازی|حکوومەت|سەرۆک|وەزیری|ئەمڕۆ|هێزەکانی|سوپای|نوێ|"
    r"پوختە|هەواڵ|سەردێر)"
    r"(?=$|[\s،.؟!:\-\(\)])"
)


def is_valid_kurdish_sorani(text: str) -> bool:
    """
    Strictly return True ONLY when text is genuine Central Kurdish (Sorani).
    Rejects English, untranslated Arabic (even though both share Unicode U+0600-U+06FF),
    empty strings, and heavily Latin-mixed outputs.
    """
    if not text or not text.strip():
        return False

    # Ignore URLs and HTML formatting tags when measuring natural language script ratio
    cleaned = re.sub(r"https?://\S+", " ", text)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned).strip()
    if not cleaned:
        return False

    arabic_script_chars = re.findall(r"[\u0600-\u06FF]", cleaned)
    latin_chars = re.findall(r"[A-Za-z]", cleaned)
    total_alpha = len(arabic_script_chars) + len(latin_chars)
    if total_alpha == 0 or len(arabic_script_chars) < 3:
        return False

    ratio = len(arabic_script_chars) / total_alpha
    if ratio < 0.70:
        return False

    # Arabic teh marbuta ('ة') never occurs in standard Central Kurdish Sorani
    if "\u0629" in cleaned:
        return False

    sorani_markers = _SORANI_DISTINCT_RE.findall(cleaned)
    if not sorani_markers:
        return False

    arabic_stops = len(_ARABIC_STOPWORDS_RE.findall(cleaned))
    sorani_stops = len(_SORANI_STOPWORDS_RE.findall(cleaned))
    arabic_only_diac = len(_ARABIC_ONLY_LETTERS_RE.findall(cleaned))

    if arabic_stops >= 1 and arabic_stops >= sorani_stops:
        return False
    if arabic_only_diac >= 1 and len(sorani_markers) < 2:
        return False
    if arabic_only_diac >= 2 and sorani_stops == 0:
        return False
    if len(arabic_script_chars) >= 35 and len(sorani_markers) < 2 and sorani_stops == 0:
        return False

    return True


def is_kurdish_sorani(text: str) -> bool:
    """Backward-compatible alias enforcing strict Central Kurdish (Sorani) validation."""
    return is_valid_kurdish_sorani(text)


def trim_at_sentence_boundary(text: str, max_len: int = 650) -> str:
    """Trim text cleanly at the last full sentence boundary (., ؟, !) without mid-sentence ellipsis."""
    clean = (text or "").strip()
    if len(clean) <= max_len:
        return clean

    window = clean[:max_len]
    cut_point = max(window.rfind("."), window.rfind("؟"), window.rfind("!"))
    if cut_point > 0:
        return window[: cut_point + 1].strip()

    # No sentence punctuation within limit: clean word boundary without '...'
    last_space = window.rfind(" ")
    if last_space > max_len // 2:
        return window[:last_space].strip() + "."
    return window.strip()


class TranslationProvider(ABC):
    """Abstract translation provider interface."""

    @abstractmethod
    async def translate(self, text: str, source_lang: str = "auto", target_lang: str = "ckb") -> str:
        """Translate text. Raises TranslationError on failure."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


# ═══════════════════════════════════════════════════════════
# Tier 1: Gemini Primary Translation, Summarization & Editing
# ═══════════════════════════════════════════════════════════
class GeminiTranslationProvider(TranslationProvider):
    """
    Tier 1 Primary Engine: Gemini (`gemini-2.5-flash`) with multi-key rotation
    and automatic exponential backoff on HTTP 429 rate limits.
    Performs translation, summarization, and journalistic editing in one JSON pass.
    """

    def __init__(
        self,
        api_key: Optional[str | list[str]] = None,
        model: Optional[str] = None,
        session: Optional[aiohttp.ClientSession] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
        base_backoff: float = 1.5,
        cooldown_store: Optional[dict[str, float]] = None,
    ) -> None:
        self._client = GeminiClient(
            api_key=GEMINI_API_KEY if api_key is None else api_key,
            model=GEMINI_MODEL if model is None else model,
            session=session,
            semaphore=semaphore,
            base_backoff=base_backoff,
            cooldown_store=cooldown_store,
        )

    @property
    def name(self) -> str:
        return "Gemini"

    @property
    def model(self) -> str:
        return self._client.model

    @property
    def api_keys(self) -> list[str]:
        return self._client.api_keys

    @property
    def has_keys(self) -> bool:
        return self._client.has_keys

    @property
    def current_key(self) -> str:
        return self._client.current_key

    async def cleanup(self) -> None:
        await self._client.cleanup()

    async def translate_article(
        self,
        title: str,
        summary: str = "",
        is_kurdish: Optional[bool] = None,
    ) -> tuple[str, str]:
        """
        Single-pass Gemini translation + summarization + journalistic editing.
        If `is_kurdish` is True, directly rewrites/summarizes native Sorani text.
        Returns (final_title, final_summary).
        """
        if not self._client.has_keys:
            raise TranslationError("GEMINI_API_KEY is not configured.")

        native_kurdish = (
            is_kurdish
            if is_kurdish is not None
            else is_kurdish_sorani(f"{title} {summary}")
        )

        system_prompt = (
            GEMINI_KURDISH_REWRITE_SYSTEM_PROMPT
            if native_kurdish
            else GEMINI_TRANSLATION_SYSTEM_PROMPT
        )

        user_content = (
            f"=== ORIGINAL HEADLINE ===\n{title}\n\n"
            f"=== ARTICLE FACTS / CONTENT ===\n{(summary or '')[:3000]}\n\n"
            "Produce the final Sorani Kurdish headline and 3-4 sentence report (350-650 chars) as JSON: "
            '{"title": "...", "summary": "..."}'
        )

        try:
            data = await self._client.generate_json(
                system_prompt=system_prompt,
                user_content=user_content,
                temperature=0.1,
                max_tokens=800,
            )
        except GeminiAPIError as e:
            raise TranslationError(str(e)) from e

        from processing.content_formatter import normalize_neutral_terminology

        final_title = normalize_neutral_terminology(
            normalize_sorani_chars(str(data.get("title", "")).strip())
        )
        final_summary = normalize_neutral_terminology(
            strip_news_dateline(normalize_sorani_chars(str(data.get("summary", "")).strip()))
        )

        if not final_title or not is_valid_kurdish_sorani(final_title):
            raise TranslationError("Gemini headline failed strict Sorani Kurdish verification.")
        if summary and summary.strip():
            if not final_summary or not is_valid_kurdish_sorani(final_summary):
                raise TranslationError("Gemini summary failed strict Sorani Kurdish verification.")

        return final_title, trim_at_sentence_boundary(final_summary, max_len=650)

    async def translate(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "ckb",
    ) -> str:
        """Translate a standalone text snippet into Sorani Kurdish via Gemini."""
        if not text or not text.strip():
            return ""
        if not self._client.has_keys:
            raise TranslationError("GEMINI_API_KEY is not configured.")

        try:
            data = await self._client.generate_json(
                system_prompt=(
                    "Translate the following news text into natural, journalistic Central Kurdish (Sorani). "
                    'Respond ONLY with JSON: {"text": "..."}'
                ),
                user_content=text,
                temperature=0.1,
                max_tokens=800,
            )
        except GeminiAPIError as e:
            raise TranslationError(str(e)) from e

        translated = normalize_sorani_chars(str(data.get("text", "")).strip())
        if not translated:
            raise TranslationError("Gemini returned empty translation text.")
        return translated


# ═══════════════════════════════════════════════════════════
# Tier 2 (Fallback ONLY): Keyless Google Translate (ckb)
# ═══════════════════════════════════════════════════════════
class GoogleTranslateProvider(TranslationProvider):
    """
    Lightweight Non-LLM Safety Net Fallback: Direct async Google Translate (tl=ckb)
    via aiohttp. Used ONLY when all Gemini API keys are unavailable or exhausted.
    """

    def __init__(
        self,
        delay: float = 1.5,
        semaphore: Optional[asyncio.Semaphore] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self._delay = delay
        self._semaphore = semaphore if semaphore is not None else asyncio.Semaphore(1)
        self._headers = dict(DEFAULT_TRANSLATE_HEADERS)
        self._session = session

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
        if self._session and not self._session.closed:
            await self._session.close()

    async def _translate_chunk(
        self,
        chunk: str,
        source_lang: str = "auto",
        target_lang: str = "ckb",
    ) -> str:
        session = await self._get_session()
        clients = ["gtx", "dict-chrome-ex"]
        last_error = ""

        for client in clients:
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

            return normalize_sorani_chars(" ".join(translated_chunks))


# ═══════════════════════════════════════════════════════════
# Translation Engine (Gemini Primary + GoogleTranslate Fallback)
# ═══════════════════════════════════════════════════════════
class TranslationEngine:
    """
    Manages the Gemini-first Kurdish (Sorani) translation & editing pipeline:
      - Primary: GeminiTranslationProvider (single-pass translation, summarization & editing, max_tokens=800)
      - Fallback: GoogleTranslateProvider (ckb) ONLY if all Gemini keys fail
    """

    def __init__(
        self,
        delay: float = 1.5,
        gemini_api_key: Optional[str | list[str]] = None,
        gemini_model: Optional[str] = None,
        base_backoff: float = 1.5,
        cooldown_store: Optional[dict[str, float]] = None,
    ) -> None:
        self._delay = delay
        self._semaphore = asyncio.Semaphore(1)
        self._gemini_provider = GeminiTranslationProvider(
            api_key=gemini_api_key,
            model=gemini_model,
            semaphore=self._semaphore,
            base_backoff=base_backoff,
            cooldown_store=cooldown_store,
        )
        self._google_fallback = GoogleTranslateProvider(
            delay=self._delay,
            semaphore=self._semaphore,
        )

        self._providers: list[TranslationProvider] = []
        if self._gemini_provider.has_keys:
            self._providers.append(self._gemini_provider)
        self._providers.append(self._google_fallback)

        logger.info(
            "Translation engine initialized (Gemini-first, no Groq). Active providers: %s",
            [p.name for p in self._providers],
        )

    @property
    def semaphore(self) -> asyncio.Semaphore:
        return self._semaphore

    @property
    def providers(self) -> list[TranslationProvider]:
        return self._providers

    @property
    def gemini_provider(self) -> GeminiTranslationProvider:
        return self._gemini_provider

    async def cleanup(self) -> None:
        """Clean up provider HTTP sessions."""
        await self._gemini_provider.cleanup()
        await self._google_fallback.cleanup()

    async def translate_title(self, title: str, target_lang: str = "ckb") -> str:
        """Translate headline to Sorani Kurdish through fallback chain."""
        if not title.strip():
            return ""
        translated = await self._translate_with_fallback(title, target_lang=target_lang)
        cleaned = self._post_process(translated, title)
        if not is_valid_kurdish_sorani(cleaned):
            raise TranslationError("Translated headline failed strict Sorani Kurdish verification.")
        return cleaned

    async def translate_summary(self, summary: str, target_lang: str = "ckb") -> str:
        """Translate summary to Sorani Kurdish through fallback chain."""
        if not summary.strip():
            return ""
        translated = await self._translate_with_fallback(summary, target_lang=target_lang)
        cleaned = self._post_process(translated, summary)
        if not is_valid_kurdish_sorani(cleaned):
            raise TranslationError("Translated summary failed strict Sorani Kurdish verification.")
        return cleaned

    async def translate_article(
        self,
        title: str,
        summary: str = "",
        target_lang: str = "ckb",
    ) -> tuple[str, str]:
        """
        Translate or rewrite an article in a single Gemini pass when available;
        fall back cleanly to Google Translate (`ckb`) ONLY if Gemini fails or has no keys.
        NEVER falls back to raw English or Arabic source text.
        For strictly verified native Sorani articles, uses Kurdish editorial rewrite.
        """
        native_kurdish = is_valid_kurdish_sorani(f"{title} {summary}")
        has_source_summary = bool(summary and summary.strip())

        # Tier 1: Single high-quality Gemini pass (translates+summarizes OR rewrites native Kurdish)
        if self._gemini_provider.has_keys:
            try:
                final_title, final_summary = await self._gemini_provider.translate_article(
                    title=title,
                    summary=summary,
                    is_kurdish=native_kurdish,
                )
                if not is_valid_kurdish_sorani(final_title) or (
                    has_source_summary and (not final_summary or not is_valid_kurdish_sorani(final_summary))
                ):
                    raise TranslationError("Gemini output failed strict Sorani Kurdish verification.")
                logger.info(
                    "Gemini single-pass %s completed for headline '%s'",
                    "Kurdish editorial rewrite" if native_kurdish else "Sorani translation",
                    title[:50],
                )
                return final_title, final_summary
            except TranslationError as e:
                logger.warning("Gemini primary translation failed (%s). Activating fallback.", str(e)[:180])
                if not ENABLE_TRANSLATION_FALLBACK:
                    raise

        # Native Kurdish safety fallback (only when input itself passed strict Sorani verification)
        if native_kurdish:
            from processing.content_formatter import normalize_neutral_terminology
            clean_title = normalize_neutral_terminology(normalize_sorani_chars(title))
            clean_summary = (
                trim_at_sentence_boundary(
                    normalize_neutral_terminology(strip_news_dateline(normalize_sorani_chars(summary))),
                    max_len=650,
                )
                if has_source_summary
                else ""
            )
            if is_valid_kurdish_sorani(clean_title) and (
                not has_source_summary or (clean_summary and is_valid_kurdish_sorani(clean_summary))
            ):
                logger.info("Using verified native Sorani text directly after Gemini fallback.")
                return clean_title, clean_summary

        # Tier 2 Fallback: Lightweight Google Translate ckb safety net
        fallback_title = await self._google_fallback.translate(title, target_lang=target_lang)
        fallback_title = self._post_process(fallback_title, title)
        if not is_valid_kurdish_sorani(fallback_title):
            raise TranslationError("Fallback headline failed strict Sorani Kurdish verification.")

        fallback_summary = ""
        if has_source_summary:
            # Keep fallback summary concise so caption stays within Telegram limit
            trimmed_source = trim_at_sentence_boundary(strip_news_dateline(summary.strip()), max_len=1200)
            translated_sum = await self._google_fallback.translate(trimmed_source, target_lang=target_lang)
            cleaned_sum = strip_news_dateline(self._post_process(translated_sum, trimmed_source))
            fallback_summary = trim_at_sentence_boundary(cleaned_sum, max_len=650)
            if not fallback_summary or not is_valid_kurdish_sorani(fallback_summary):
                raise TranslationError("Fallback summary failed strict Sorani Kurdish verification.")

        return fallback_title, fallback_summary

    async def _translate_with_fallback(self, text: str, target_lang: str = "ckb") -> str:
        """Try active providers in tier order until one succeeds."""
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
        """Post-process translated text for Sorani orthography and neutral terminology."""
        if not translated:
            return ""
        from processing.content_formatter import normalize_neutral_terminology
        cleaned = normalize_sorani_chars(re.sub(r"\s+", " ", translated).strip())
        return normalize_neutral_terminology(cleaned)

    async def validate_translation(self, translated: str, original: str) -> bool:
        """Validate that a translation is strictly acceptable Central Kurdish (Sorani)."""
        if not translated or not translated.strip():
            return False
        if len(translated.strip()) < 5:
            return False
        if not is_valid_kurdish_sorani(translated):
            logger.warning("Translation failed strict Sorani Kurdish verification: '%s'", translated[:60])
            return False
        ratio = len(translated) / max(len(original), 1)
        if ratio < 0.1 or ratio > 10:
            logger.warning("Translation length ratio suspicious: %.2f", ratio)
            return True
        return True


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
