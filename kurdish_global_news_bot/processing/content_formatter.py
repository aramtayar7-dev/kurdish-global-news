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
from processing.translator import (
    is_valid_kurdish_sorani,
    strip_news_dateline,
    trim_at_sentence_boundary,
)
from storage.models import Article

logger = logging.getLogger("processing.formatter")

# Attribution qualifier for uncorroborated Tier-3 single-source claims
TIER3_ATTRIBUTION_PREFIX = "بەپێی ڕاپۆرتە سەرەتاییەکان، "
_ATTRIBUTION_PREFIXES = (
    "بەپێی", "سەرچاوە", "میدیا", "لە ڕاپۆرتێکدا", "بەپێی ڕاپۆرتە",
)

# RSS/HTML footer boilerplate stripped before translation and after formatting
_RSS_BOILERPLATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\s*\[\s*(?:\.\.\.|…)\s*\]\s*", re.IGNORECASE),
    re.compile(r"\s*the\s+post\s+.+?\s+appeared\s+first\s+on\s+.+$", re.IGNORECASE),
    re.compile(
        r"(?:^|\s)(?:continue\s+reading|read\s+more(?:\s+at\s+\S+|\s+here)?|"
        r"click\s+here(?:\s+to\s+read\s+more)?|watch\s+live|listen\s+to\s+the\s+episode|"
        r"subscribe\s+(?:to|now)[^.!?\n]*|follow\s+us\s+on[^.!?\n]*|"
        r"sign\s+up\s+for[^.!?\n]*)\s*(?:\.\.\.|…)?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\s)(?:بەردەوام\s+بە\s+لە\s+خوێندنەوە|زیاتر\s+بخوێنەوە|"
        r"تەواوی\s+(?:هەواڵەکە|بابەتەکە)\s+بخوێنەوە|کلیک\s+لێرە\s+بکە|"
        r"ئابوونە\s+بکە|ڕاستەوخۆ\s+سەیر\s+بکە)\s*(?:\.\.\.|…)?\s*[.!؟]?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\s)(?:اقرأ\s+المزيد|تابع\s+القراءة|للمزيد\s+اضغط\s+هنا)\s*(?:\.\.\.|…)?\s*[.!؟]?\s*$",
        re.IGNORECASE,
    ),
)

_AWKWARD_COMMA_AND_RE = re.compile(r"[,،]\s*و\s*[,،]")


def strip_rss_boilerplate(text: str) -> str:
    """
    Strip trailing RSS/web boilerplate ('Continue reading...', 'Read more...',
    'بەردەوام بە لە خوێندنەوە...', '[...]', etc.) cleanly without altering article facts.
    """
    if not text:
        return ""

    cleaned = text.strip()
    for _ in range(3):
        prev = cleaned
        for pat in _RSS_BOILERPLATE_PATTERNS:
            cleaned = pat.sub(" ", cleaned).strip()
        # Strip lingering bracketed ellipses or bare trailing '...' left by truncated RSS descriptions
        cleaned = re.sub(r"(?:\.{3}|…)\s*$", "", cleaned).strip()
        if cleaned == prev:
            break
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()

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


def prepend_tier3_attribution(translated_summary: str) -> str:
    """
    Prepend neutral Kurdish preliminary-attribution qualifier to uncorroborated Tier-3 summaries.
    Skips if the summary already starts with an attribution prefix.
    Returns the modified summary, guaranteed <= 1024 chars.
    """
    if not translated_summary or not translated_summary.strip():
        return translated_summary

    stripped = translated_summary.strip()
    # Check if already attributed
    for prefix in _ATTRIBUTION_PREFIXES:
        if stripped.startswith(prefix):
            return translated_summary

    result = TIER3_ATTRIBUTION_PREFIX + stripped
    # Trim to <= 1024 chars if needed
    if len(result) > 1024:
        result = trim_at_sentence_boundary(result, max_len=1024)
    return result


# Unicode Arabic/Kurdish letter boundaries so punctuation (، ؛ ؟ .) separates cleanly
# while partial substrings inside unrelated words are never matched.
_BOUNDARY_LEFT = r"(?<![A-Za-z0-9_\u0621-\u064A\u0671-\u06D5])"
_BOUNDARY_RIGHT = r"(?![A-Za-z0-9_\u0621-\u064A\u0671-\u06D5])"

# Ordered longest/most-specific first; replaces partisan epithets with neutral journalistic forms
_RAW_NEUTRAL_TERM_RULES: tuple[tuple[str, str], ...] = (
    # Kurdistan Region (normalize geopolitical misnomers to هەرێمی کوردستان)
    (r"(?:إقليم|اقليم)\s+شمال\s+العراق", "هەرێمی کوردستان"),
    (r"هەرێمی\s+باکوو?ری\s+عێراق", "هەرێمی کوردستان"),
    (r"شمال\s+العراق", "هەرێمی کوردستان"),
    (r"باکوو?ری\s+عێراق", "هەرێمی کوردستان"),
    (r"Northern\s+Iraq", "هەرێمی کوردستان"),
    # Israel & Military (ordered specific military/government forms before bare country forms)
    (
        r"(?:سوپای|سوپاى)\s+(?:قەوارەی|قەوارەى|کیانی|کیانى|ڕژێمی|رژێمی|دوژمنی|دوژمنێ|داگیرکەر|داگیرکەری)?\s*(?:زایۆنی|زایۆنى|ئیسرائیل)",
        "سوپای ئیسرائیل",
    ),
    (
        r"(?:هێزە|هێزەکانی|هێزەکانى)\s+(?:(?:داگیرکەر|داگیرکەری|داگیرکەرەکانی)\s+)?(?:قەوارەی|قەوارەى|کیانی|کیانى|دوژمنی|دوژمنێ)\s+(?:زایۆنی|زایۆنى)",
        "هێزەکانی ئیسرائیل",
    ),
    (
        r"(?:هێزە|هێزەکانی|هێزەکانى)\s+(?:داگیرکەر|داگیرکەری|داگیرکەرەکانی)\s+(?:زایۆنی|زایۆنى|ئیسرائیل)",
        "هێزەکانی ئیسرائیل",
    ),
    (
        r"(?:حکومەتی|حکومەتى)\s+(?:قەوارەی|قەوارەى|کیانی|کیانى|دوژمنی)\s+(?:زایۆنی|زایۆنى)",
        "حکومەتی ئیسرائیل",
    ),
    (r"جيش\s+(?:العدو|الكيان|الاحتلال)\s+(?:الصهيوني|الإسرائيلي|الاسرائيلي)", "سوپای ئیسرائیل"),
    (r"قوات\s+(?:العدو|الكيان|الاحتلال)\s+(?:الصهيوني|الإسرائيلي|الاسرائيلي)", "هێزەکانی ئیسرائیل"),
    (
        r"(?:قەوارەی|قەوارەى|کیانی|کیانى|دوژمنی|ڕژێمی|رژێمی|داگیرکەری)\s+(?:زایۆنی|زایۆنى)",
        "ئیسرائیل",
    ),
    (
        r"(?:قەوارەی|قەوارەى|کیانی|کیانى)\s+(?:داگیرکەر|داگیرکەرەکە|کاتی|کاتى)",
        "ئیسرائیل",
    ),
    (r"ئەسرائیلی\s+داگیرکەر", "ئیسرائیل"),
    (r"الكيان\s+الصهيوني", "ئیسرائیل"),
    (r"العدو\s+الصهيوني", "ئیسرائیل"),
    (r"الكيان\s+المؤقت", "ئیسرائیل"),
    # United States & Forces
    (
        r"(?:هێزە|هێزەکانی|هێزەکانى)\s+(?:داگیرکەر|داگیرکەری|داگیرکەرەکانی)\s+(?:ئەمریکا|ئەمریکای|ئەمەریکا|ئەمەریکای)",
        "هێزەکانی ئەمریکا",
    ),
    (r"(?:هێزەکانی|هێزەکانى)\s+(?:ئەمریکای|ئەمریکاى)\s+دوژمن", "هێزەکانی ئەمریکا"),
    (r"(?:هێزەکانی|هێزەکانى)\s+دوژمنی\s+(?:ئەمریکی|ئەمریکى|ئەمەریکی)", "هێزەکانی ئەمریکا"),
    (r"قوات\s+(?:الاحتلال|العدو)\s+(?:الأمريكي|الامريكي)", "هێزەکانی ئەمریکا"),
    (r"(?:ئەمریکای|ئەمریکاى|ئەمەریکای|ئەمەریکاى)\s+دوژمن", "ئەمریکا"),
    (r"(?:داگیرکاری|داگیرکەری)\s+(?:ئەمریکا|ئەمەریکا)", "ئەمریکا"),
    (r"دوژمنی\s+(?:ئەمریکی|ئەمریکى|ئەمەریکی|ئەمەریکى)", "ئەمریکا"),
    (r"شەیتانی\s+گەورە", "ئەمریکا"),
    (r"العدو\s+(?:الأمريكي|الامريكي)", "ئەمریکا"),
    (r"الاحتلال\s+(?:الأمريكي|الامريكي)", "هێزەکانی ئەمریکا"),
    (r"الشيطان\s+(?:الأكبر|الاكبر)", "ئەمریکا"),
    # Syria
    (r"سوو?ریای\s+داگیرکراوە?", "سووریا"),
    (r"سوري[اة]\s+المحتلة", "سووریا"),
    # Strip partisan insults / epithets targeting political leaders
    (r"(?:المجرم|القاتل|الإرهابي|الارهابي|السفاح|الجبان|الطاغية)\s+(?:دونالد\s+)?(?:ترامب|ترمب)", "دۆناڵد ترەمپ"),
    (r"(?:تاوانبار|بکوژ|تیرۆریست|دڕندە|ترسنۆک)\s+(?:دۆناڵد\s+)?(?:ترەمپ|ترامپ)", "دۆناڵد ترەمپ"),
    (r"(?:دۆناڵد\s+)?(?:ترەمپ|ترامپ)ی\s+(?:تاوانبار|بکوژ|تیرۆریست|ترسنۆک)", "دۆناڵد ترەمپ"),
    (r"(?:المجرم|القاتل|الإرهابي|الارهابي|السفاح|الجبان)\s+(?:بنيامين\s+)?(?:نتنياهو|نتنیاهو)", "بێنیامین نەتەنیاهۆ"),
    (r"(?:تاوانبار|بکوژ|تیرۆریست)\s+(?:بێنیامین\s+)?(?:نەتەنیاهۆ|نتنیاهو|نتنياهو)", "بێنیامین نەتەنیاهۆ"),
    (r"(?:المجرم|القاتل|الإرهابي|الارهابي|السفاح)\s+(?:جو\s+)?(?:بايدن|بایدن)", "جۆ بایدن"),
    (r"(?:تاوانبار|بکوژ|تیرۆریست)\s+(?:جۆ\s+)?(?:بایدن|بايدن)", "جۆ بایدن"),
    # Other partisan epithets
    (r"(?:ئێرانی\s+دوژمن|دوژمنی\s+ئێرانی|ڕژێمی\s+مەلاکان)", "ئێران"),
    (r"(?:العدو\s+(?:الإيراني|الايراني)|النظام\s+الصفوي)", "ئێران"),
)

_COMPILED_NEUTRAL_TERM_RULES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            rf"{_BOUNDARY_LEFT}(?:(?P<conj>و)(?=ال))?{pat}(?P<suffix>دا|ەوە|یش)?{_BOUNDARY_RIGHT}",
            re.IGNORECASE,
        ),
        repl,
    )
    for pat, repl in _RAW_NEUTRAL_TERM_RULES
]


def normalize_neutral_terminology(text: str) -> str:
    """
    Deterministically normalize partisan, hostile, or misnamed Kurdish/Arabic
    geopolitical terminology to neutral BBC/Reuters-style Central Kurdish terms.
    Preserves sentence structure, punctuation, numbers, factual occupation/presence
    descriptions, and avoids partial substring corruption. Idempotent.
    """
    if not text:
        return ""

    result = strip_rss_boilerplate(text)
    for pattern, replacement in _COMPILED_NEUTRAL_TERM_RULES:
        result = pattern.sub(
            lambda m, r=replacement: f"{(m.group('conj') + ' ') if m.group('conj') else ''}{r}{m.group('suffix') or ''}",
            result,
        )

    # Fix literal comma-and-comma artifacts (e.g. ', و,' / '، و،') and clean redundant spacing
    result = _AWKWARD_COMMA_AND_RE.sub(" و ", result)
    result = re.sub(r"([،,])(?=[A-Za-z0-9\u0621-\u064A\u0671-\u06D5])", r"\1 ", result)

    # Normalize redundant horizontal spaces cleanly while keeping intentional newlines
    lines = [re.sub(r"[ \t]{2,}", " ", ln).strip() if ln.strip() else "" for ln in result.splitlines()]
    return strip_rss_boilerplate("\n".join(lines).strip())


class ContentFormatter:
    """Formats articles for Telegram publication in Kurdish Sorani."""

    def _format_headline(self, article: Article) -> str:
        """
        Escape the headline from strict Sorani translation only (never raw source text).
        Prepends breaking badge (`🔴 بەپەلە | `) first when active, or threaded update
        tag (`🔄 نوێکردنەوە | `) for non-breaking live event updates.
        """
        raw_title = normalize_neutral_terminology(
            (article.translated_title or "").strip()
        )
        if getattr(article, "is_breaking", False):
            prefix = "🔴 بەپەلە | "
        elif getattr(article, "is_thread_update", False) or getattr(article, "thread_root_message_id", None):
            prefix = "🔄 نوێکردنەوە | "
        else:
            prefix = ""
        return self._escape_html(f"{prefix}{raw_title}")

    def format_telegram_message(self, article: Article) -> str:
        """
        Format an article as a Telegram HTML text message.
        """
        parts: list[str] = []

        kurdish_cat, emoji = resolve_category_display(article.category)
        headline = self._format_headline(article)
        parts.append(f"{emoji} <b>{headline}</b>")

        summary = normalize_neutral_terminology(
            strip_news_dateline(article.translated_summary or "")
        )
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

        summary = normalize_neutral_terminology(
            strip_news_dateline(article.translated_summary or "")
        )
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
