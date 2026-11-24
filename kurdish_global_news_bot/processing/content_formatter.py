"""
Kurdish Global News Bot — Content Formatter

Formats translated articles for Telegram publication with accurate Kurdish
category labels, emojis, RTL-friendly layout, and clean sentence-boundary trimming.
"""

from __future__ import annotations

import html
import logging
import re
from typing import Optional

from config import CATEGORY_EMOJIS, CATEGORY_META
from processing.translator import trim_at_sentence_boundary
from storage.models import Article

logger = logging.getLogger("processing.formatter")

# Central Kurdish (Sorani) category labels covering the 7 canonical keys + legacy aliases
CATEGORY_LABELS_KURDISH: dict[str, str] = {
    # Canonical 7 categories
    "kurdistan": "کوردستان",
    "military_conflict": "شەڕ و سەربازی",
    "politics_world": "سیاسەت و جیهان",
    "economy_markets": "ئابووری و بازاڕ",
    "ai_tech": "تەکنەلۆژیا و زیرەکیی دەستکرد",
    "cybersecurity": "ئاسایشی سایبەری",
    "science_space": "زانست و بۆشایی ئاسمان",
    # Legacy aliases
    "Wars & Armed Conflicts": "شەڕ و سەربازی",
    "Military & Defense": "شەڕ و سەربازی",
    "World Leaders & Governments": "سیاسەت و جیهان",
    "Diplomacy & International Relations": "سیاسەت و جیهان",
    "International Politics": "سیاسەت و جیهان",
    "Global Economy & Financial Markets": "ئابووری و بازاڕ",
    "Energy, Oil & Gas": "ئابووری و بازاڕ",
    "Business & Corporate": "ئابووری و بازاڕ",
    "AI & Technology": "تەکنەلۆژیا و زیرەکیی دەستکرد",
    "AI & Tech": "تەکنەلۆژیا و زیرەکیی دەستکرد",
    "Cybersecurity": "ئاسایشی سایبەری",
    "Science & Research": "زانست و بۆشایی ئاسمان",
    "Science & Space": "زانست و بۆشایی ئاسمان",
    "Space & Astronomy": "زانست و بۆشایی ئاسمان",
    "Health & Medical": "سیاسەت و جیهان",
    "Climate & Environment": "سیاسەت و جیهان",
    "Natural Disasters & Emergencies": "سیاسەت و جیهان",
    "International Organizations": "سیاسەت و جیهان",
    "Human Rights & Humanitarian": "سیاسەت و جیهان",
    "Criminal Investigations & Security": "شەڕ و سەربازی",
    "International Sports": "سیاسەت و جیهان",
    "World News": "سیاسەت و جیهان",
    "Other Significant Events": "سیاسەت و جیهان",
}


def resolve_category_display(category: Optional[str]) -> tuple[str, str]:
    """Return (kurdish_label, emoji) for any canonical or legacy category string."""
    if category and category in CATEGORY_META:
        meta = CATEGORY_META[category]
        return meta["label"], meta["emoji"]
    if category:
        label = CATEGORY_LABELS_KURDISH.get(category) or CATEGORY_LABELS_KURDISH.get(category.strip(), "سیاسەت و جیهان")
        emoji = CATEGORY_EMOJIS.get(category, "🌍")
        return label, emoji
    return "سیاسەت و جیهان", "🌍"


class ContentFormatter:
    """Formats articles for Telegram publication in Kurdish Sorani."""

    def _format_headline(self, article: Article) -> str:
        """Escape the headline and prepend the breaking badge prefix when active."""
        raw_title = (article.translated_title or article.original_title or "").strip()
        prefix = "🔴 بەپەلە | " if getattr(article, "is_breaking", False) else ""
        return self._escape_html(f"{prefix}{raw_title}")

    def format_telegram_message(self, article: Article) -> str:
        """
        Format an article as a Telegram HTML text message.
        """
        parts: list[str] = []

        kurdish_cat, emoji = resolve_category_display(article.category)
        headline = self._format_headline(article)
        parts.append(f"{emoji} <b>{headline}</b>")

        summary = article.translated_summary or article.original_summary
        if summary:
            trimmed = self._truncate_summary(summary, max_len=650)
            parts.append(f"\n{self._escape_html(trimmed)}")

        parts.append("\n━━━━━━━━━━━━━━━")
        parts.append(f"📂 {self._escape_html(kurdish_cat)}")

        if article.source_name:
            parts.append(f"📡 {self._escape_html(article.source_name)}")

        if article.publication_date:
            date_str = self._format_date(article.publication_date)
            if date_str:
                parts.append(f"🕐 {date_str}")

        if article.original_url:
            safe_url = self._escape_html(article.original_url)
            parts.append(f'\n🔗 <a href="{safe_url}">بابەتی ئەسڵی</a>')

        return "\n".join(parts)

    def format_caption(self, article: Article) -> str:
        """
        Format a photo caption guaranteed to fit within Telegram's 1024-character limit
        without cutting mid-sentence (`...`) or truncating footer tags.
        """
        kurdish_cat, emoji = resolve_category_display(article.category)
        headline = self._format_headline(article)
        header = f"{emoji} <b>{headline}</b>"

        footer_parts = [
            "\n━━━━━━━━━━━━━━━",
            f"📂 {self._escape_html(kurdish_cat)}",
        ]
        if article.source_name:
            footer_parts.append(f"📡 {self._escape_html(article.source_name)}")
        if article.original_url:
            safe_url = self._escape_html(article.original_url)
            footer_parts.append(f'🔗 <a href="{safe_url}">بابەتی ئەسڵی</a>')
        footer = "\n".join(footer_parts)

        summary = (article.translated_summary or article.original_summary or "").strip()
        if not summary:
            caption = f"{header}\n{footer}"
            return caption if len(caption) <= 1024 else f"{header[:900]}</b>\n{footer}"[:1024]

        # Compute exact budget for the summary inside Telegram's 1024-char caption limit
        fixed_overhead = len(header) + len(footer) + 4  # spacing newlines
        max_summary_budget = min(650, max(120, 1024 - fixed_overhead))

        candidate_summary = self._truncate_summary(summary, max_len=max_summary_budget)
        escaped_summary = self._escape_html(candidate_summary)
        result = f"{header}\n\n{escaped_summary}\n{footer}"

        # Iteratively step down by sentence boundary if HTML escaping expanded past 1024 chars
        while len(result) > 1024 and max_summary_budget > 80:
            max_summary_budget -= 60
            candidate_summary = self._truncate_summary(summary, max_len=max_summary_budget)
            escaped_summary = self._escape_html(candidate_summary)
            result = f"{header}\n\n{escaped_summary}\n{footer}"

        if len(result) > 1024:
            # Drop optional summary cleanly rather than breaking HTML anchor or cutting mid-sentence
            minimal = f"{header}\n{footer}"
            if len(minimal) <= 1024:
                return minimal
            return f"<b>{headline[:200]}</b>\n{footer}"[:1024]

        return result

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters for Telegram HTML parse mode."""
        return html.escape(text, quote=False)

    @staticmethod
    def _truncate_summary(text: str, max_len: int = 650) -> str:
        """Trim summary cleanly at a full sentence boundary (., ؟, !) with no mid-sentence ellipsis."""
        return trim_at_sentence_boundary(text, max_len=max_len)

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
        return date_str[:20] if len(date_str) > 20 else date_str
