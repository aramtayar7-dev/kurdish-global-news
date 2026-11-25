"""
Kurdish Global News Bot — Nightly Gemini Digest Generator

Synthesizes the top 5-7 distinct published stories from the past 24 hours
into a crisp Central Kurdish (Sorani) nightly roundup using Gemini exclusively.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import html
import json
import logging
from typing import Optional

from config import CHANNEL_HANDLE, GEMINI_API_KEY, GEMINI_MODEL
from processing.content_formatter import normalize_neutral_terminology, resolve_category_display
from processing.gemini_client import GeminiAPIError, GeminiClient
from processing.translator import normalize_sorani_chars, trim_at_sentence_boundary
from storage.database import get_kurdistan_date_str
from storage.models import Article

logger = logging.getLogger("processing.digest")

KURDISH_MONTHS: dict[int, str] = {
    1: "کانوونی دووەم",
    2: "شوبات",
    3: "ئازار",
    4: "نیسان",
    5: "ئایار",
    6: "حوزەیران",
    7: "تەممووز",
    8: "ئاب",
    9: "ئەیلوول",
    10: "تشرینی یەکەم",
    11: "تشرینی دووەم",
    12: "کانوونی یەکەم",
}

DIGEST_HEADER_PREFIX = "🌙 پوختەی گرنگترین هەواڵەکانی ئەمڕۆ"

GEMINI_DIGEST_SYSTEM_PROMPT = (
    "You are the managing editor of a Central Kurdish (Sorani / سۆرانی) international news channel.\n"
    "Given up to 10 published news candidates from the past 24 hours, select the 5 to 7 MOST IMPORTANT distinct stories "
    "and write a crisp, authoritative nightly news roundup in Sorani Kurdish.\n\n"
    "Strict Digest Guidelines:\n"
    "1. Prioritization & Order: Place major Kurdistan stories (category='kurdistan', any of the 4 parts of Kurdistan) FIRST, "
    "followed by the biggest global military, diplomatic, economic, AI/tech, cyber, or space developments.\n"
    "2. Distinct Stories: Never include duplicate or overlapping coverage of the same event.\n"
    "3. Scannable Bullet Summaries: Write 1 to 2 crisp, complete Sorani sentences per bullet (about 75 to 125 characters per bullet) "
    "so the entire digest fits comfortably between 650 and 920 characters total (including header and footer).\n"
    "4. Emoji Assignment: Use each story's exact matching category emoji (☀️ for kurdistan, ⚔️ for military_conflict, "
    "🌍 for politics_world, 📈 for economy_markets, 🤖 for ai_tech, 🛡️ for cybersecurity, 🚀 for science_space).\n"
    "5. Language Quality & Neutrality: Maintain strict non-partisan Reuters/BBC neutrality without partisan epithets ('هەرێمی کوردستان', 'ئەمریکا', 'ئیسرائیل', 'سووریا'), and use proper Central Kurdish Arabic orthography (ڕ, ڵ, ۆ, ێ, ە, ه, ک, ی). Never hallucinate unstated facts.\n"
    "6. Output Format: Respond ONLY with valid JSON matching this schema (NO markdown fences, NO commentary):\n"
    '{"stories": [{"emoji": "☀️", "summary": "1-2 crisp Sorani sentences ending with a period."}]}'
)


@dataclass
class DigestResult:
    """Output of the nightly Kurdish news digest generator."""
    date_str: str
    date_label: str
    story_count: int
    digest_text: str
    bullets: list[str]
    used_fallback: bool = False


def format_kurdish_date(date_str: str) -> str:
    """Format YYYY-MM-DD cleanly into a Sorani Kurdish display date."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        month_name = KURDISH_MONTHS.get(dt.month, str(dt.month))
        return f"{dt.day}ی {month_name}ی {dt.year}"
    except Exception:
        return date_str


class DailyDigestGenerator:
    """Generates nightly Kurdish Sorani news digests using Gemini exclusively."""

    def __init__(
        self,
        api_key: Optional[str | list[str]] = None,
        model: Optional[str] = None,
        channel_handle: str = CHANNEL_HANDLE,
        base_backoff: float = 1.5,
    ) -> None:
        self._channel_handle = channel_handle or "@KurdishGlobalNews"
        self._gemini = GeminiClient(
            api_key=GEMINI_API_KEY if api_key is None else api_key,
            model=GEMINI_MODEL if model is None else model,
            base_backoff=base_backoff,
        )

    async def cleanup(self) -> None:
        await self._gemini.cleanup()

    def _assemble_digest_text(
        self,
        date_label: str,
        items: list[tuple[str, str]],
    ) -> tuple[str, list[str]]:
        """Format final digest header, emoji bullet items, and @KurdishGlobalNews footer."""
        header = f"<b>{DIGEST_HEADER_PREFIX}</b>\n📅 {html.escape(date_label)}"
        footer = f"━━━━━━━━━━━━━━━\n📡 {html.escape(self._channel_handle)}"

        formatted_bullets: list[str] = []
        for emoji, raw_summary in items:
            clean_summary = normalize_neutral_terminology(normalize_sorani_chars(raw_summary)).strip()
            if clean_summary and clean_summary[-1] not in ".؟!":
                clean_summary += "."
            formatted_bullets.append(f"{emoji} {html.escape(clean_summary)}")

        body = "\n\n".join(formatted_bullets)
        full_text = f"{header}\n\n{body}\n\n{footer}"
        return full_text, formatted_bullets

    def _build_fallback_digest(
        self,
        articles: list[Article],
        date_str: str,
        date_label: str,
    ) -> DigestResult:
        """Build a clean Kurdish roundup from already-translated DB headlines/summaries if Gemini errors out."""
        selected = articles[: min(6, len(articles))]
        items: list[tuple[str, str]] = []
        for art in selected:
            _, emoji = resolve_category_display(art.category)
            headline = normalize_sorani_chars(art.translated_title or art.original_title).strip()
            summary = normalize_sorani_chars(art.translated_summary or "").strip()
            first_sentence = ""
            if summary:
                first_sentence = trim_at_sentence_boundary(summary, max_len=110)
            bullet = first_sentence if (first_sentence and len(first_sentence) >= 40) else headline
            bullet = trim_at_sentence_boundary(bullet, max_len=130)
            items.append((emoji, bullet))

        full_text, bullets = self._assemble_digest_text(date_label, items)
        return DigestResult(
            date_str=date_str,
            date_label=date_label,
            story_count=len(items),
            digest_text=full_text,
            bullets=bullets,
            used_fallback=True,
        )

    async def generate_digest(
        self,
        articles: list[Article],
        digest_date: Optional[str] = None,
    ) -> Optional[DigestResult]:
        """
        Synthesize 5-7 top distinct stories into a nightly Sorani Kurdish digest.
        Returns None and logs SKIPPED if fewer than 3 articles exist in the last 24h.
        """
        if len(articles) < 3:
            logger.info("SKIPPED: fewer than 3 articles in the last 24h (count=%d).", len(articles))
            return None

        date_str = digest_date or get_kurdistan_date_str()
        date_label = format_kurdish_date(date_str)
        candidates = articles[:10]

        if not self._gemini.has_keys:
            logger.warning("GEMINI_API_KEY not configured; building extractive digest fallback.")
            return self._build_fallback_digest(candidates, date_str, date_label)

        candidate_payload = []
        for idx, art in enumerate(candidates, start=1):
            _, emoji = resolve_category_display(art.category)
            candidate_payload.append({
                "index": idx,
                "category": art.category or "politics_world",
                "emoji": emoji,
                "source": art.source_name or "",
                "title": art.translated_title or art.original_title,
                "summary": (art.translated_summary or art.original_summary or "")[:350],
            })

        user_content = (
            f"Date: {date_label} ({date_str})\n"
            f"Candidates ({len(candidate_payload)} stories from last 24h):\n"
            f"{json.dumps(candidate_payload, ensure_ascii=False)}\n\n"
            "Select the best 5 to 7 distinct stories (Kurdistan first, then top global events) and "
            "write concise 1-2 sentence Sorani bullets targeting 650-920 chars total output."
        )

        try:
            data = await self._gemini.generate_json(
                system_prompt=GEMINI_DIGEST_SYSTEM_PROMPT,
                user_content=user_content,
                temperature=0.2,
                max_tokens=1000,
            )
            raw_stories = data.get("stories")
            if not isinstance(raw_stories, list) or len(raw_stories) == 0:
                raise GeminiAPIError("Gemini digest response missing 'stories' list.")

            items: list[tuple[str, str]] = []
            for entry in raw_stories[:7]:
                if not isinstance(entry, dict):
                    continue
                summary = str(entry.get("summary", "")).strip()
                if not summary:
                    continue
                emoji = str(entry.get("emoji", "🌍")).strip() or "🌍"
                items.append((emoji, summary))

            if len(items) < 3:
                raise GeminiAPIError(f"Gemini returned too few digest bullets ({len(items)}).")

            full_text, bullets = self._assemble_digest_text(date_label, items)
            return DigestResult(
                date_str=date_str,
                date_label=date_label,
                story_count=len(items),
                digest_text=full_text,
                bullets=bullets,
                used_fallback=False,
            )

        except Exception as e:
            logger.warning("Gemini digest synthesis failed (%s); using clean DB fallback.", str(e)[:180])
            return self._build_fallback_digest(candidates, date_str, date_label)
