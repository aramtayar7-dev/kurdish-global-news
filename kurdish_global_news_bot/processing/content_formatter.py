"""
Kurdish Global News Bot — Content Formatter

Formats translated articles for Telegram publication with proper
Kurdish RTL-friendly formatting.
"""

from __future__ import annotations

import html
import logging
import re
from typing import Optional

from storage.models import Article
from config import CATEGORY_EMOJIS

logger = logging.getLogger("processing.formatter")

# Central Kurdish (Sorani) translations for news categories
CATEGORY_LABELS_KURDISH: dict[str, str] = {
    "Wars & Armed Conflicts": "شەڕ و پێکدادانی چەکداری",
    "Military & Defense": "سەربازی و بەرگری",
    "World Leaders & Governments": "سەرکردە و حکومەتەکان",
    "Diplomacy & International Relations": "دیپلۆماسی و پەیوەندییە نێودەوڵەتییەکان",
    "International Politics": "سیاسەتی نێودەوڵەتی",
    "Global Economy & Financial Markets": "ئابووریی جیهانی و بازاڕەکان",
    "AI & Technology": "تەکنەلۆژیا و زیرەکیی دەستکرد",
    "AI & Tech": "تەکنەلۆژیا و زیرەکیی دەستکرد",
    "Science & Research": "زانست و توێژینەوە",
    "Science & Space": "زانست و بۆشایی ئاسمان",
    "Space & Astronomy": "زانست و بۆشایی ئاسمان",
    "Cybersecurity": "ئاسایشی سایبەری",
    "Health & Medical": "تەندروستی و پزیشکی",
    "Climate & Environment": "کەشوهەوا و ژینگە",
    "Natural Disasters & Emergencies": "کارەساتی سروشتی و فریاگوزاری",
    "Energy, Oil & Gas": "وزە، نەوت و گاز",
    "International Organizations": "ڕێکخراوە نێودەوڵەتییەکان",
    "Human Rights & Humanitarian": "مافی مرۆڤ و کاروباری مرۆیی",
    "Criminal Investigations & Security": "لێکۆڵینەوەی تاوان و ئاسایش",
    "International Sports": "وەرزشی نێودەوڵەتی",
    "Business & Corporate": "کاروبار و کۆمپانیاکان",
    "World News": "هەواڵی جیهانی",
    "Other Significant Events": "هەواڵی جیهانی",
}


class ContentFormatter:
    """Formats articles for Telegram publication in Kurdish Sorani."""

    def format_telegram_message(self, article: Article) -> str:
        """
        Format an article as a Telegram HTML message.

        Structure:
        1. Category emoji + Sorani headline
        2. Concise Sorani summary
        3. Category tag
        4. Source name
        5. Publication date (if available)
        6. Original article URL
        """
        parts: list[str] = []

        # 1. Headline
        emoji = CATEGORY_EMOJIS.get(article.category, "📰")
        headline = self._escape_html(article.translated_title or article.original_title)
        parts.append(f"{emoji} <b>{headline}</b>")

        # 2. Summary
        summary = article.translated_summary or article.original_summary
        if summary:
            summary = self._escape_html(self._truncate_summary(summary, max_len=600))
            parts.append(f"\n{summary}")

        # 3. Separator and metadata
        parts.append("\n━━━━━━━━━━━━━━━")

        # 4. Category
        raw_cat = article.category or "Other Significant Events"
        kurdish_cat = CATEGORY_LABELS_KURDISH.get(raw_cat, raw_cat)
        parts.append(f"📂 {self._escape_html(kurdish_cat)}")

        # 5. Source
        if article.source_name:
            parts.append(f"📡 {self._escape_html(article.source_name)}")

        # 6. Date
        if article.publication_date:
            date_str = self._format_date(article.publication_date)
            if date_str:
                parts.append(f"🕐 {date_str}")

        # 7. Original link
        if article.original_url:
            safe_url = self._escape_html(article.original_url)
            parts.append(f'\n🔗 <a href="{safe_url}">بابەتی ئەسڵی</a>')

        return "\n".join(parts)

    def format_caption(self, article: Article) -> str:
        """
        Format a shorter caption for image posts.
        Telegram captions have a 1024 character limit.
        """
        parts: list[str] = []

        emoji = CATEGORY_EMOJIS.get(article.category, "📰")
        headline = self._escape_html(article.translated_title or article.original_title)
        parts.append(f"{emoji} <b>{headline}</b>")

        summary = article.translated_summary or article.original_summary
        if summary:
            # Shorter summary for captions (fits comfortably in 1024 char limit)
            summary = self._escape_html(self._truncate_summary(summary, max_len=650))
            parts.append(f"\n{summary}")

        parts.append("\n━━━━━━━━━━━━━━━")

        raw_cat = article.category or "Other Significant Events"
        kurdish_cat = CATEGORY_LABELS_KURDISH.get(raw_cat, raw_cat)
        parts.append(f"📂 {self._escape_html(kurdish_cat)}")

        if article.source_name:
            parts.append(f"📡 {self._escape_html(article.source_name)}")

        if article.original_url:
            safe_url = self._escape_html(article.original_url)
            parts.append(f'🔗 <a href="{safe_url}">بابەتی ئەسڵی</a>')

        result = "\n".join(parts)

        # Enforce Telegram caption limit
        if len(result) > 1024:
            result = result[:1020] + "..."

        return result

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters for Telegram HTML parse mode."""
        return html.escape(text, quote=False)

    @staticmethod
    def _truncate_summary(text: str, max_len: int = 600) -> str:
        """Truncate summary at a sentence boundary."""
        if len(text) <= max_len:
            return text

        truncated = text[:max_len]
        # Try to cut at a sentence boundary
        last_period = truncated.rfind(".")
        last_question = truncated.rfind("؟")  # Kurdish question mark
        last_excl = truncated.rfind("!")

        cut_point = max(last_period, last_question, last_excl)
        if cut_point > max_len // 2:
            return truncated[:cut_point + 1]

        # Fall back to word boundary
        last_space = truncated.rfind(" ")
        if last_space > max_len // 2:
            return truncated[:last_space] + "..."

        return truncated + "..."

    @staticmethod
    def _format_date(date_str: str) -> str:
        """Format a date string for display."""
        if not date_str:
            return ""
        try:
            from dateutil import parser as dateparser
            dt = dateparser.parse(date_str)
            if dt:
                return dt.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            pass
        # Return raw date if parsing fails
        return date_str[:20] if len(date_str) > 20 else date_str
