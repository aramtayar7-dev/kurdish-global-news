"""
Kurdish Global News Bot — Content Formatter

Formats translated articles for Telegram publication with accurate Kurdish
category labels, emojis, RTL-friendly layout, and clean sentence-boundary trimming.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import html
import logging
import re
from typing import Optional

from config import CATEGORY_EMOJIS, CATEGORY_META
from processing.translator import (
    is_valid_kurdish_sorani,
    normalize_sorani_orthography,
    strip_news_dateline,
    trim_at_sentence_boundary,
)
from storage.models import Article

logger = logging.getLogger("processing.formatter")

# Attribution qualifier for uncorroborated Tier-3 single-source claims & raw eyewitness X videos
TIER3_ATTRIBUTION_PREFIX = "بەپێی ڕاپۆرتە سەرەتاییەکان، "
X_EYEWITNESS_ATTRIBUTION_PREFIX = "بەپێی گرتە ڤیدیۆ بڵاوکراوەکان، "
_ATTRIBUTION_PREFIXES = (
    "بەپێی", "سەرچاوە", "میدیا", "لە ڕاپۆرتێکدا", "بەپێی ڕاپۆرتە", "بەپێی گرتە",
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
    return _prepend_attribution_with_prefix(translated_summary, TIER3_ATTRIBUTION_PREFIX)


def prepend_x_video_attribution(translated_summary: str) -> str:
    """
    Prepend neutral Kurdish video-footage attribution (`بەپێی گرتە ڤیدیۆ بڵاوکراوەکان، `)
    for uncorroborated X eyewitness/Tier-3 video claims.
    """
    return _prepend_attribution_with_prefix(translated_summary, X_EYEWITNESS_ATTRIBUTION_PREFIX)


def _prepend_attribution_with_prefix(translated_summary: str, prefix_text: str) -> str:
    if not translated_summary or not translated_summary.strip():
        return translated_summary

    stripped = translated_summary.strip()
    for prefix in _ATTRIBUTION_PREFIXES:
        if stripped.startswith(prefix):
            return translated_summary

    result = prefix_text + stripped
    if len(result) > 1024:
        result = trim_at_sentence_boundary(result, max_len=1024)
    return result


# Unicode Arabic/Kurdish letter boundaries so punctuation (، ؛ ؟ .) separates cleanly
# while partial substrings inside unrelated words are never matched.
_BOUNDARY_LEFT = r"(?<![A-Za-z0-9_\u0621-\u064A\u0660-\u0669\u0671-\u06D5\u06F0-\u06F9\u200C])"
_BOUNDARY_RIGHT = r"(?![A-Za-z0-9_\u0621-\u064A\u0660-\u0669\u0671-\u06D5\u06F0-\u06F9\u200C])"

# Targeted Unicode emoji, flag, skin-tone, variation selector, and ZWJ ranges.
# Deliberately avoids \U00010000-\U0010FFFF so non-emoji supplementary characters are preserved.
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002600-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "\U0001F3FB-\U0001F3FF"
    "]"
    "|[\uFE0E\uFE0F]"
    "|\u200D"
    "|\u20E3"
)

PUNCTUATION_PATTERN = re.compile(r"[!?؟۔]{2,}")
_REPEATED_PERIOD_PATTERN = re.compile(r"\.{2,}")

# Explicit country/state names stripped only when tightly wrapped in decorative/sarcastic quotes
_EXPLICIT_COUNTRY_NAMES_ALTERNATION = (
    r"ئیسرائیل|ئسرائیل|إسرائيل|اسرائيل|Israel|"
    r"ئەمریکا|ئەمەریکا|أمريكا|امريكا|الولايات\s+المتحدة|United\s+States|U\.S\.|USA|"
    r"ئێران|إيران|ايران|Iran|"
    r"عێراق|العراق|Iraq|"
    r"سووریا|سوریا|سوريا|سورية|Syria|"
    r"تورکیا|تركيا|Türkiye|Turkey|"
    r"ڕووسیا|روسیا|روسيا|Russia|"
    r"ئۆکرانیا|أوكرانيا|اوكرانيا|Ukraine|"
    r"فەلەستین|فلسطين|Palestine|"
    r"لوبنان|لبنان|Lebanon|"
    r"یەمەن|اليمن|Yemen|"
    r"چین|الصين|China|"
    r"بەریتانیا|بريطانيا|Britain|UK|"
    r"فەڕەنسا|فەرەنسا|فرنسا|France|"
    r"ئەڵمانیا|ألمانيا|المانيا|Germany|"
    r"هەرێمی\s+کوردستان|کوردستان|إقليم\s+كردستان|كردستان|Kurdistan|"
    r"قەوارە[یى]\s+زایۆن[یى]|الكيان\s+الصهيوني|الكيان\s+المؤقت|[ڕر]ژێمی\s+زایۆن[یى]"
)

_SARCASTIC_COUNTRY_QUOTE_RE = re.compile(
    rf"{_BOUNDARY_LEFT}"
    rf"(?:[\"']|«|“|‘|„)\s*"
    rf"(?P<country>{_EXPLICIT_COUNTRY_NAMES_ALTERNATION})"
    rf"\s*(?:[\"']|»|”|’|“)"
    rf"{_BOUNDARY_RIGHT}",
    re.IGNORECASE,
)

# Legitimate direct quotations / statement excerpts protected from editorial rewrites
_DIRECT_QUOTATION_RE = re.compile(
    r"«[^»\n]+»|“[^”\n]+”|‘[^’\n]+’|\"[^\"\n]+\""
)

# Attribution and legal cues that signal reported characterization rather than narrator voice
_ATTRIBUTION_OR_LEGAL_CONTEXT_RE = re.compile(
    r"(?:ڕایگەیاند|گوتی|وتیشی|دەڵێت|ئاماژەی\s+(?:دا|کرد)|بەپێی|بەگوێرەی|بە\s+وتەی|"
    r"جريمة|القانون\s+الدولي|یاسای\s+نێودەوڵەتی|القوانين|"
    r"دادگا|پەیماننامەی|قال|صرح|أعلن|وصفت?ه?\s+بالعدوان|ما\s+وصفه)",
    re.IGNORECASE,
)
_PASSIVE_LABEL_SUFFIX_RE = re.compile(r"^\s+ناو\s*(?:دەبرێت|دەنرێت|برا|دەبەن|دەهێنرێت)")

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
    # Strip editorialized slur qualifiers while preserving verified organization names natural
    # (e.g. حیزبوڵڵا، حەماس، ئەنساروڵڵا، حووسییەکان remain untouched; legal 'تیرۆریستان' preserved)
    (
        r"(?:میلیشیای|گرووپی|ڕێکخراوی|حیزبی)\s+(?:تیرۆریستیی?|الإرهابية?)\s+(?=(?:حیزبوڵڵا|حەماس|ئەنساروڵڵا|حووسییەکان))",
        "__KGN_STRIP__",
    ),
    (r"(?:تیرۆریستیی?|الإرهابية?)", "__KGN_STRIP__"),
    # Unsupported rhetorical attack characterizations (handled context-safely; not replacing legal/quoted usages)
    (r"دەستدرێژییەکانی", "هێرشەکانی"),
    (r"دەستدرێژییەکی", "هێرشێکی"),
    (r"دەستدرێژییە", "هێرشە"),
    (r"دەستدرێژیی?(?!\s*سێکسی)", "هێرش"),
    (r"العدوان", "هێرش"),
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

# Contextual rules (rhetorical attack noun / slur replacements skipped under legal/attributed prefix cues)
_CONTEXT_SENSITIVE_REPLACEMENTS: frozenset[str] = frozenset(
    {"هێرشەکانی", "هێرشێکی", "هێرشە", "هێرش", "__KGN_STRIP__"}
)


def _strip_sarcastic_country_quotes(text: str) -> tuple[str, int]:
    """Remove scare/sarcastic quotation marks wrapped directly around explicit country/state names."""
    count = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return match.group("country")

    return _SARCASTIC_COUNTRY_QUOTE_RE.sub(_replace, text), count


def _protect_direct_quotations(text: str) -> tuple[str, dict[str, str]]:
    """
    Temporarily mask legitimate direct quotations so neutral phrase replacement
    never alters verbatim quoted speech or official statement excerpts.
    Single-token sarcastic country quotes must be unwrapped before calling this.
    """
    placeholders: dict[str, str] = {}

    def _mask(match: re.Match[str]) -> str:
        quoted = match.group(0)
        inner = quoted[1:-1].strip()
        # Only protect multi-word statements or longer non-trivial quotes (not 1-word scare quotes)
        if not inner or (len(inner.split()) < 2 and len(inner) < 24):
            return quoted
        token = f"__KGN_QUOTE_{len(placeholders)}__"
        placeholders[token] = quoted
        return token

    return _DIRECT_QUOTATION_RE.sub(_mask, text), placeholders


def _restore_direct_quotations(text: str, placeholders: dict[str, str]) -> str:
    """Restore masked direct quotation spans verbatim."""
    restored = text
    for token, original in placeholders.items():
        restored = restored.replace(token, original)
    return restored


def _apply_neutral_term_rules(text: str) -> tuple[str, int]:
    """Apply explicit neutral term replacements respecting word boundaries and attribution/legal context."""
    result = text
    replacements = 0
    for pattern, replacement in _COMPILED_NEUTRAL_TERM_RULES:
        is_contextual = replacement in _CONTEXT_SENSITIVE_REPLACEMENTS

        def _sub(m: re.Match[str], r: str = replacement, ctx: bool = is_contextual) -> str:
            nonlocal replacements
            if ctx:
                prefix_window = result[max(0, m.start() - 120) : m.start()]
                clause_prefix = prefix_window.rsplit(".", 1)[-1].rsplit("؟", 1)[-1].rsplit("!", 1)[-1]
                suffix_window = result[m.end() : min(len(result), m.end() + 45)]
                is_passive_labeled = bool(
                    re.search(r"\bبە\s*$", clause_prefix) and _PASSIVE_LABEL_SUFFIX_RE.search(suffix_window)
                )
                if is_passive_labeled or _ATTRIBUTION_OR_LEGAL_CONTEXT_RE.search(clause_prefix):
                    return m.group(0)
            replacements += 1
            if r == "__KGN_STRIP__":
                return ""
            conj = (m.group("conj") + " ") if m.group("conj") else ""
            suffix = m.group("suffix") or ""
            return f"{conj}{r}{suffix}"

        result = pattern.sub(_sub, result)
    return result, replacements


def normalize_neutral_terminology(text: str) -> str:
    """
    Deterministically normalize partisan, hostile, or misnamed Kurdish/Arabic
    geopolitical terminology to neutral BBC/Reuters-style Central Kurdish terms.
    Preserves sentence structure, punctuation, numbers, direct quotations,
    legal/attribution context, factual descriptions, and avoids partial
    substring corruption. Idempotent.
    """
    if not isinstance(text, str) or not text:
        return ""

    result = strip_rss_boilerplate(text)
    result = normalize_sorani_orthography(result, convert_arabic_ky=False)
    result, _ = _strip_sarcastic_country_quotes(result)
    masked, quote_map = _protect_direct_quotations(result)
    masked, _ = _apply_neutral_term_rules(masked)
    result = _restore_direct_quotations(masked, quote_map)

    # Fix literal comma-and-comma artifacts (e.g. ', و,' / '، و،') and clean redundant spacing
    result = _AWKWARD_COMMA_AND_RE.sub(" و ", result)
    result = re.sub(r"([،,])(?=[A-Za-z0-9\u0621-\u064A\u0671-\u06D5])", r"\1 ", result)

    # Normalize redundant horizontal spaces cleanly while keeping intentional newlines
    lines = [re.sub(r"[ \t]{2,}", " ", ln).strip() if ln.strip() else "" for ln in result.splitlines()]
    return strip_rss_boilerplate("\n".join(lines).strip())


def sanitize_neutral_sorani_text(text: str) -> str:
    """
    Deterministic, idempotent post-translation/pre-publish Sorani neutrality sanitizer.

    Removes decorative/targeted emojis, collapses excessive punctuation,
    removes sarcastic quotation marks around explicit country/state names
    without disturbing legitimate quotations/attributions, applies Sorani orthography
    normalization (broken 'ه ' spacing & stray hashtags), applies controlled
    contextual neutral phrase replacements, and normalizes whitespace.
    """
    if not isinstance(text, str) or not text:
        return ""

    # 1. Remove targeted Unicode emoji/ZWJ sequences without corrupting Sorani/Arabic or U+10000+ text
    cleaned = EMOJI_PATTERN.sub("", text)
    cleaned = normalize_sorani_orthography(cleaned, convert_arabic_ky=False)

    # 2. Strip sarcastic quotes wrapped solely around explicit country names
    cleaned, quote_strips = _strip_sarcastic_country_quotes(cleaned)

    # 3. Protect direct multi-word quotations before applying neutral editorial replacements
    masked, quote_map = _protect_direct_quotations(cleaned)
    masked = strip_rss_boilerplate(masked)
    masked, term_replacements = _apply_neutral_term_rules(masked)
    cleaned = _restore_direct_quotations(masked, quote_map)

    # Re-run sarcastic country quote cleanup in case an editorial substitute (e.g. "قەوارەی زایۆنی") normalized to a bare country name
    cleaned, extra_strips = _strip_sarcastic_country_quotes(cleaned)
    quote_strips += extra_strips

    # 4. Normalize punctuation without altering single '?' / '؟' questions, decimals, or dates
    cleaned = PUNCTUATION_PATTERN.sub(".", cleaned)
    cleaned = _REPEATED_PERIOD_PATTERN.sub(".", cleaned)
    cleaned = _AWKWARD_COMMA_AND_RE.sub(" و ", cleaned)
    cleaned = re.sub(r"([،,])(?=[A-Za-z0-9\u0621-\u064A\u0671-\u06D5])", r"\1 ", cleaned)

    # Remove orphaned spaces directly before standard sentence punctuation while keeping decimals/dates intact
    cleaned = re.sub(r"[ \t]+([.!?؟،؛])", r"\1", cleaned)

    # 5. Normalize whitespace deterministically while preserving line structure
    lines = [re.sub(r"[ \t]{2,}", " ", ln).strip() for ln in cleaned.splitlines()]
    result = "\n".join(lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = strip_rss_boilerplate(result.strip())

    if term_replacements or quote_strips:
        logger.debug(
            "Sanitizer applied %d neutral term replacement(s) and stripped %d sarcastic country quote(s).",
            term_replacements,
            quote_strips,
        )

    return result


_DUPLICATE_PREFIX_RE = re.compile(
    r"^\s*(?:(?:🔴|🔄)\s*)?(?:بەپەلە|نوێکردنەوە|عاجل|breaking|urgent)\s*(?:\||:|[-–—])+\s*",
    re.IGNORECASE,
)
_DUPLICATE_PUNCT_RE = re.compile(r"[.!؟?،,؛;:\"'«»“”‘’()\[\]{}\-–—_|/\\]+")


def _normalize_for_duplicate_check(text: str) -> str:
    """Normalize headline or summary for duplicate detection (strip emojis, prefixes, orthography, punctuation, whitespace)."""
    if not isinstance(text, str) or not text.strip():
        return ""
    work = html.unescape(text)
    work = EMOJI_PATTERN.sub(" ", work)
    work = normalize_sorani_orthography(work)
    for _ in range(2):
        updated = _DUPLICATE_PREFIX_RE.sub("", work)
        if updated == work:
            break
        work = updated
    work = _DUPLICATE_PUNCT_RE.sub(" ", work)
    return re.sub(r"\s+", " ", work).strip().lower()


def _is_duplicate_summary(headline: str, summary: str) -> bool:
    """
    Return True when `summary` adds no distinct detail beyond `headline`
    (empty, identical, substring of headline, SequenceMatcher ratio >= 0.85,
    token overlap >= 0.90, or single-sentence near-repeat of headline).
    """
    norm_summary = _normalize_for_duplicate_check(summary)
    if not norm_summary:
        return True

    norm_headline = _normalize_for_duplicate_check(headline)
    if not norm_headline:
        return False

    if norm_summary == norm_headline or norm_summary in norm_headline:
        return True

    if SequenceMatcher(None, norm_headline, norm_summary).ratio() >= 0.85:
        return True

    h_tokens = norm_headline.split()
    s_tokens = norm_summary.split()
    summary_body = re.sub(r"[.!?؟\s]+$", "", EMOJI_PATTERN.sub("", summary or "").strip())
    is_single_sentence = not bool(re.search(r"[.!؟]", summary_body))
    if is_single_sentence and norm_headline in norm_summary and len(s_tokens) <= len(h_tokens) + 4:
        return True

    if h_tokens and s_tokens:
        h_set = set(h_tokens)
        s_set = set(s_tokens)
        overlap = len(h_set & s_set)
        if (overlap / len(s_set)) >= 0.90 and (overlap / len(h_set)) >= 0.90:
            return True
        if (overlap / len(s_set)) >= 0.90 and len(s_tokens) <= len(h_tokens) + 2:
            return True

    return False


class ContentFormatter:
    """Formats articles for Telegram publication in Kurdish Sorani."""

    def _format_headline(self, article: Article) -> str:
        """
        Escape the headline from strict Sorani translation only (never raw source text).
        Prepends breaking badge (`🔴 بەپەلە | `) first when active, or threaded update
        tag (`🔄 نوێکردنەوە | `) for non-breaking live event updates.
        """
        raw_title = sanitize_neutral_sorani_text(
            (article.translated_title or "").strip()
        )
        if getattr(article, "is_breaking", False):
            prefix = "🔴 بەپەلە | "
        elif getattr(article, "is_thread_update", False) or getattr(article, "thread_root_message_id", None):
            prefix = "🔄 نوێکردنەوە | "
        else:
            prefix = ""
        return self._escape_html(f"{prefix}{raw_title}")

    def format_telegram_message(
        self,
        article: Article,
        has_inline_source_button: bool = False,
    ) -> str:
        """
        Format an article as a Telegram HTML text message.
        Omits the bottom text URL line when `has_inline_source_button=True`.
        """
        parts: list[str] = []

        from core.importance_ranker import is_genuine_kurdistan_event, is_politics_or_war

        display_cat = article.category
        if (display_cat or "").strip().lower() in {"kurdistan", "کوردستان", "كردستان"} and not is_genuine_kurdistan_event(article):
            display_cat = "military_conflict" if is_politics_or_war(article) else "politics_world"

        kurdish_cat, emoji = resolve_category_display(display_cat)
        headline = self._format_headline(article)
        parts.append(f"{emoji} <b>{headline}</b>")

        summary = sanitize_neutral_sorani_text(
            strip_news_dateline(article.translated_summary or "")
        )
        if summary and not _is_duplicate_summary(headline, summary):
            trimmed = sanitize_neutral_sorani_text(
                self._truncate_summary(summary, max_len=650)
            )
            if trimmed and not _is_duplicate_summary(headline, trimmed):
                parts.append(f"\n{self._escape_html(trimmed)}")

        parts.append("\n━━━━━━━━━━━━━━━")
        parts.append(f"📂 {self._escape_html(kurdish_cat)}")

        if article.source_name:
            parts.append(f"📡 {self._escape_html(article.source_name)}")

        if article.publication_date:
            date_str = self._format_date(article.publication_date)
            if date_str:
                parts.append(f"🕐 {date_str}")

        if not has_inline_source_button and article.original_url:
            raw_url = article.original_url.strip()
            if raw_url.lower().startswith(("http://", "https://")):
                safe_url = self._escape_html(raw_url)
                parts.append(f'\n🔗 <a href="{safe_url}">سەرچاوە</a>')

        return "\n".join(parts)

    def format_caption(
        self,
        article: Article,
        has_inline_source_button: bool = False,
    ) -> str:
        """
        Format a photo caption guaranteed to fit within Telegram's 1024-character limit
        without cutting mid-sentence (`...`) or truncating footer tags.
        Omits the bottom text URL line when an inline source button is attached.
        """
        from core.importance_ranker import is_genuine_kurdistan_event, is_politics_or_war

        display_cat = article.category
        if (display_cat or "").strip().lower() in {"kurdistan", "کوردستان", "كردستان"} and not is_genuine_kurdistan_event(article):
            display_cat = "military_conflict" if is_politics_or_war(article) else "politics_world"

        kurdish_cat, emoji = resolve_category_display(display_cat)
        headline = self._format_headline(article)
        header = f"{emoji} <b>{headline}</b>"

        footer_parts = [
            "\n━━━━━━━━━━━━━━━",
            f"📂 {self._escape_html(kurdish_cat)}",
        ]
        if article.source_name:
            footer_parts.append(f"📡 {self._escape_html(article.source_name)}")
        if not has_inline_source_button and article.original_url:
            raw_url = article.original_url.strip()
            if raw_url.lower().startswith(("http://", "https://")):
                safe_url = self._escape_html(raw_url)
                footer_parts.append(f'🔗 <a href="{safe_url}">سەرچاوە</a>')
        footer = "\n".join(footer_parts)

        summary = sanitize_neutral_sorani_text(
            strip_news_dateline(article.translated_summary or "")
        )
        if not summary or _is_duplicate_summary(headline, summary):
            caption = f"{header}\n{footer}"
            return caption if len(caption) <= 1024 else f"{header[:900]}</b>\n{footer}"[:1024]

        # Compute exact budget for the summary inside Telegram's 1024-char caption limit
        fixed_overhead = len(header) + len(footer) + 4  # spacing newlines
        max_summary_budget = min(650, max(120, 1024 - fixed_overhead))

        candidate_summary = sanitize_neutral_sorani_text(
            self._truncate_summary(summary, max_len=max_summary_budget)
        )
        if not candidate_summary or _is_duplicate_summary(headline, candidate_summary):
            caption = f"{header}\n{footer}"
            return caption if len(caption) <= 1024 else f"{header[:900]}</b>\n{footer}"[:1024]
        escaped_summary = self._escape_html(candidate_summary)
        result = f"{header}\n\n{escaped_summary}\n{footer}"

        # Iteratively step down by sentence boundary if HTML escaping expanded past 1024 chars
        while len(result) > 1024 and max_summary_budget > 80:
            max_summary_budget -= 60
            candidate_summary = sanitize_neutral_sorani_text(
                self._truncate_summary(summary, max_len=max_summary_budget)
            )
            if not candidate_summary or _is_duplicate_summary(headline, candidate_summary):
                minimal = f"{header}\n{footer}"
                return minimal if len(minimal) <= 1024 else f"<b>{headline[:200]}</b>\n{footer}"[:1024]
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
