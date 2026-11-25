"""
Kurdish Global News Bot — Zero-API Cross-Source, Cross-Language Deduplicator & Live Event Threader

Combines multi-layer deterministic deduplication and Three-Way Event Classification
(NEW_STORY | THREAD_UPDATE | DUPLICATE) across English, Arabic, and Kurdish Sorani
with zero extra Gemini API calls:
  - Canonical URL & Arabic/Kurdish Unicode/diacritics normalization
  - Persistent 48-72h published history lookup (Supabase / SQLite) + same-cycle state
  - Stage A (Fast title similarity): Normalized title fuzzy ratio >= 0.60
  - Event Key / Entities Matching: Gemini-extracted canonical English `event_key` & `event_entities`
  - Stage B (Local semantic event match): Stopword-filtered token & multilingual concept overlap
  - Three-Way Live Event Threading (`THREAD_UPDATE` vs `DUPLICATE`):
      1. Active root story within `EVENT_THREAD_WINDOW_HOURS` (24h) with valid `telegram_message_id`
      2. Thread reply count < `MAX_UPDATES_PER_THREAD` (3)
      3. Elapsed time since last post in thread >= `THREAD_UPDATE_COOLDOWN_MINUTES` (20m),
         bypassed ONLY when strict breaking gate holds (`is_breaking=True` & `score >= 9.0`)
      4. Non-redundant content + deterministic material factual delta (casualties, official
         response, new figures, retaliation, ceasefire, arrest/investigation, or Gemini `update_type`)
  - Post-Translation Sorani Kurdish Safety Gate: catches rephrased Sorani repeats right
    before card generation & Telegram publishing (or downgrades non-material updates).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from rapidfuzz import fuzz

import config
from core.importance_ranker import is_strict_breaking
from storage.database import Database
from storage.models import Article
from storage.utils import (
    normalize_event_entities,
    normalize_event_key,
    parse_utc_timestamp,
    unpack_event_metadata,
)

logger = logging.getLogger("processing.deduplicator")

OUTCOME_NEW_STORY = "NEW_STORY"
OUTCOME_THREAD_UPDATE = "THREAD_UPDATE"
OUTCOME_DUPLICATE = "DUPLICATE"

_POSITIVE_UPDATE_TYPES: set[str] = {
    "casualties_update",
    "official_response",
    "retaliation",
    "escalation",
    "ceasefire",
    "legal_action",
    "investigation_result",
    "material_update",
    "update",
}

_STOPWORDS: set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from", "has", "have",
    "in", "into", "is", "it", "its", "of", "on", "or", "over", "said", "says", "than", "that",
    "the", "their", "this", "to", "under", "was", "were", "will", "with", "amid", "after",
    "before", "near", "new", "more", "report", "reported", "reports", "official", "officials",
    "news", "today", "yesterday", "live", "breaking", "update", "latest",
    # Arabic common function words (written post y->ی and ة->ه normalization too)
    "في", "فی", "من", "على", "علی", "إلى", "الى", "الی", "عن", "مع", "بعد", "قبل", "بين", "بین",
    "أن", "ان", "إن", "الذي", "الذی", "التي", "التی", "هذا", "هذه", "عاجل", "خلال", "حول", "بشان",
    "افادت", "اعلنت", "قالت", "اکدت", "اكدت",
    # Sorani Kurdish stopwords + frequent journalistic auxiliaries
    "لە", "بۆ", "کە", "بە", "و", "ئەم", "ئەو", "لەسەر", "لەلایەن", "ڕایگەیاند", "دەڵێت",
    "ئاشکرای", "کرد", "بوو", "هەیە", "دەکات", "دەبێت", "لەگەڵ", "دوای", "پێش",
    "ئەمڕۆ", "دوێنێ", "هەروەها", "بەڵام", "یان", "چونکە", "ئەوە", "ئەوەی", "تێیدا",
    "بەهۆی", "سەبارەت", "ئاشکرا", "کردەوە", "بوون", "نییە", "بێ",
}

_SORANI_STOPWORDS: set[str] = {
    "لە", "بۆ", "کە", "بە", "و", "ئەم", "ئەو", "لەسەر", "لەلایەن", "ڕایگەیاند",
    "دەڵێت", "ئاشکرای", "کرد", "بوو", "هەیە", "دەکات", "دەبێت", "لەگەڵ", "دوای",
    "پێش", "ئەمڕۆ", "دوێنێ", "هەروەها", "بەڵام", "یان", "چونکە", "ئەوەی", "تێیدا",
    "بەهۆی", "سەبارەت", "ئاشکرا", "کردەوە", "بوون", "نییە", "بێ", "لەم", "بەو",
    "لەلای", "لەنێوان", "بەپێی", "پاش", "دەکرێت", "کراوە", "دەبن", "هەواڵ", "زانیاری",
    "نوێ", "نوێی", "زمانی", "کوردی", "سۆرانی", "پوختە", "پوختەی", "بڵاوکردنەوە",
    "بڵاوکرایەوە", "بڵاوی", "ڕاپۆرت", "ڕاپۆرتی", "ڕاپۆرتێکی", "ئەنجامدرا", "لێدوان",
}

# Broad actors/locations/titles that MUST NOT by themselves merge unrelated stories
_BROAD_CONTEXT_TOKENS: set[str] = {
    "erbil", "baghdad", "kurdistan", "iraq", "iraqi", "krg", "iran", "iranian", "tehran",
    "israel", "israeli", "usa", "us", "american", "washington", "syria", "syrian", "damascus",
    "turkey", "turkish", "ankara", "russia", "russian", "moscow", "ukraine", "ukrainian", "kyiv",
    "trump", "biden", "netanyahu", "barzani", "sudani", "putin", "zelensky", "khamenei",
    "government", "military", "army", "forces", "official", "officials", "minister",
    "ministry", "president", "leader", "leaders", "cabinet", "talks", "meeting", "news",
    "region", "regional", "security", "report", "statement", "updates", "international",
    "هەولێر", "بەغدا", "کوردستان", "هەرێم", "هەرێمی", "عێراق", "ئێران", "تاران",
    "ئیسرائیل", "ئەمریکا", "واشنتن", "سووریا", "تورکیا", "ڕووسیا", "رووسیا", "ئۆکرانیا",
    "حکوومەت", "حکومەت", "حکومەتی", "وەزارەتی", "سەرۆک", "سەرۆکی", "ترەمپ", "بایدن",
    "نەتەنیاهۆ", "سوپا", "سوپای", "هێز", "هێزەکان", "هێزەکانی", "بەرپرس", "بەرپرسان",
    "کۆبوونەوە", "دۆخ", "دۆخی", "ناوچە", "ناوچەکە", "ئەمنی", "جیهان", "کەناڵدا", "هەواڵی",
}

# Canonical multilingual actors, places, and event concepts for paraphrase alignment
_CONCEPT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("loc:erbil", ("erbil", "hewler", "هەولێر", "اربيل", "أربيل")),
    ("loc:kirkuk", ("kirkuk", "کەرکوک", "كركوك")),
    ("loc:sulaymaniyah", ("sulaymaniyah", "slemani", "سلێمانی", "سلیمانیة")),
    ("loc:duhok", ("duhok", "دهۆک", "دهوك")),
    ("loc:kurdistan", ("kurdistan", "krg", "peshmerga", "کوردستان", "پێشمەرگە", "کوردستانی", "کردستان")),
    ("loc:baghdad_iraq", ("iraq", "iraqi", "baghdad", "anbar", "basra", "عێراق", "بەغدا", "العراق", "بغداد")),
    ("loc:iran", ("iran", "iranian", "tehran", "ئێران", "تاران", "إيران", "طهران")),
    ("loc:israel", ("israel", "israeli", "tel aviv", "idf", "ئیسرائیل", "إسرائيل", "اسرائيل")),
    ("loc:lebanon_beirut", ("beirut", "lebanon", "lebanese", "hezbollah", "بەیروت", "لوبنان", "بیروت", "بيروت", "لبنان", "اللبناني", "اللبنانية")),
    ("loc:gaza", ("gaza", "rafah", "hamas", "غەززە", "غزة", "رفح")),
    ("loc:syria", ("syria", "syrian", "damascus", "qamishli", "qamishlo", "rojava", "سووریا", "سوریا", "دیمەشق", "ڕۆژئاوا", "سوريا", "دمشق", "روجافا", "روج افا", "قامشلو", "القامشلي")),
    ("loc:turkey", ("turkey", "turkish", "ankara", "تورکیا", "ئەنقەرە", "تركيا", "أنقرة")),
    ("loc:usa", ("united states", "american", "pentagon", "washington", "white house", "ئەمریکا", "واشنتن", "الأمريكي", "أمريكا", "واشنطن")),
    ("loc:russia", ("russia", "russian", "moscow", "kremlin", "رووسیا", "مۆسکۆ", "روسيا", "موسكو")),
    ("loc:ukraine", ("ukraine", "ukrainian", "kyiv", "ئۆکرانیا", "کیێڤ", "أوكرانيا", "كييف")),
    ("loc:red_sea", ("red sea", "houthi", "bab el mandeb", "البحر الأحمر", "دەریای سوور")),
    ("act:protest_rally", (
        "protest", "protests", "protester", "protesters", "rally", "rallies", "demonstration", "demonstrations", "support",
        "خۆپیشاندان", "گردبوونەوە", "پشتیوانی", "تظاهرات", "تظاهرة", "مظاهرات", "مظاهرة", "وقفة", "احتجاج", "احتجاجات", "مسيرة", "دعم", "دعما",
    )),
    ("act:strike_explosion", (
        "bomb", "bombed", "bombing", "bombardment", "explosion", "explosions", "blast",
        "shelling", "strike", "strikes", "attack", "attacks", "targeted",
        "هێرش", "تەقینەوە", "بۆردومان", "تۆپباران", "قصف", "انفجار", "انفجارات", "ضربات", "هجوم", "استهداف",
    )),
    ("act:air_ops", ("airstrike", "airstrikes", "warplane", "warplanes", "aerial", "fighter jet", "غارات", "فڕۆکەی جەنگی")),
    ("act:missile_drone", ("missile", "missiles", "ballistic", "rocket", "drone", "drones", "مووشەک", "درۆن", "صاروخ", "مسيرة", "طائرات", "فڕۆکەی بێفڕۆکەوان")),
    ("act:ceasefire", ("ceasefire", "truce", "armistice", "ئاگربەست", "وقف إطلاق النار", "هدنة")),
    ("act:summit_talks", ("summit", "talks", "negotiation", "negotiations", "treaty", "accord", "agreement", "کۆبوونەوە", "گفتوگۆ", "ڕێککەوتن", "مباحثات", "قمة", "اتفاق")),
    ("act:salary_budget", ("salary", "salaries", "payroll", "budget", "wage", "wages", "مووچە", "بودجە", "رواتب", "الموازنة", "الرواتب")),
    ("act:sanctions", ("sanctions", "embargo", "tariff", "tariffs", "سزا", "باج", "عقوبات")),
    ("act:earthquake", ("earthquake", "quake", "tsunami", "tremor", "بوومەلەرزە", "زلزال", "هزة")),
    ("act:airport_base", ("airport", "airbase", "depot", "depots", "base", "bases", "فڕۆکەخانە", "بنکە", "مطار", "قاعدة")),
    ("act:cyberattack", ("cyberattack", "ransomware", "breach", "hacked", "zero-day", "سایبەری", "هاک", "سيبراني")),
    ("act:election_parliament", ("election", "elections", "parliament", "bill", "vote", "voting", "هەڵبژاردن", "پەرلەمان", "دەنگدان", "انتخابات", "البرلمان")),
)

# Material development markers across English, Sorani Kurdish, and Arabic
# Normalized forms match normalize_arabic_kurdish_text(...) output cleanly.
_MATERIAL_DEV_MARKERS: tuple[tuple[str, str], ...] = (
    # Casualties & toll changes
    ("toll rises", "casualty_toll"),
    ("death toll", "casualty_toll"),
    ("casualties", "casualty_toll"),
    ("killed", "casualty_toll"),
    ("wounded", "casualty_toll"),
    ("injured", "casualty_toll"),
    ("dead", "casualty_toll"),
    ("کوژراو", "casualty_toll"),
    ("بریندار", "casualty_toll"),
    ("قوربانی", "casualty_toll"),
    ("بەرزبووەوە", "casualty_toll"),
    ("قتلی", "casualty_toll"),
    ("جرحی", "casualty_toll"),
    ("مصابین", "casualty_toll"),
    ("ارتفاع حصیله", "casualty_toll"),
    # Official statements, responses, confirmations & denials
    ("official response", "official_response"),
    ("government response", "official_response"),
    ("official statement", "official_response"),
    ("condemned", "official_response"),
    ("confirmed", "confirmation_denial"),
    ("denies", "confirmation_denial"),
    ("denied", "confirmation_denial"),
    ("وەڵامدانەوە", "official_response"),
    ("بەیاننامە", "official_response"),
    ("ڕاگەیاندراو", "official_response"),
    ("ئیدانەی", "official_response"),
    ("پشتڕاستی کردەوە", "confirmation_denial"),
    ("ڕەتیکردەوە", "confirmation_denial"),
    ("بیان رسمی", "official_response"),
    ("رد رسمی", "official_response"),
    ("ردت", "official_response"),
    ("اکد", "confirmation_denial"),
    ("نفی", "confirmation_denial"),
    # Retaliation, escalations & second waves
    ("retaliation", "retaliation_wave"),
    ("retaliatory", "retaliation_wave"),
    ("second wave", "retaliation_wave"),
    ("fresh strikes", "retaliation_wave"),
    ("new strikes", "retaliation_wave"),
    ("counterattack", "retaliation_wave"),
    ("هێرشی نوێ", "retaliation_wave"),
    ("شەپۆلی دووەم", "retaliation_wave"),
    ("تۆڵە", "retaliation_wave"),
    ("انتقام", "retaliation_wave"),
    ("موجة ثانیه", "retaliation_wave"),
    ("موجه ثانیه", "retaliation_wave"),
    ("ضربات جدیده", "retaliation_wave"),
    # Ceasefires, agreements, evacuations, legal/investigation breakthroughs
    ("ceasefire", "ceasefire_deal"),
    ("agreement reached", "ceasefire_deal"),
    ("truce", "ceasefire_deal"),
    ("arrested", "legal_investigation"),
    ("resigned", "legal_investigation"),
    ("investigation reveals", "legal_investigation"),
    ("evacuation", "emergency_action"),
    ("declared emergency", "emergency_action"),
    ("ئاگربەست", "ceasefire_deal"),
    ("دەستگیرکرا", "legal_investigation"),
    ("دەستی لەکارکێشایەوە", "legal_investigation"),
    ("ئەنجامی لێکۆڵینەوە", "legal_investigation"),
    ("وقف اطلاق النار", "ceasefire_deal"),
    ("اعتقال", "legal_investigation"),
    ("استقاله", "legal_investigation"),
    ("نتائج التحقیق", "legal_investigation"),
)

_ACUTE_EMERGENCY_HINTS_RE = re.compile(
    r"(retaliat|second\s+wave|fresh\s+strikes|airstrike|missile|explosion|assassination|"
    r"ceasefire|death\s+toll|کوژراو|بریندار|تەقینەوە|مووشەک|ئاگربەست|ضربات|انفجار)",
    re.IGNORECASE,
)

_ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_DIGIT_TRANSLATION = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


@dataclass
class DedupResult:
    """
    Outcome of zero-API three-way event classification & deduplication:
      - classification='NEW_STORY' (is_duplicate=False, thread_root_message_id=None)
      - classification='THREAD_UPDATE' (is_duplicate=False, thread_root_message_id=<root_msg_id>)
      - classification='DUPLICATE' (is_duplicate=True, thread_root_message_id=None)
    """
    is_duplicate: bool
    matched_id: Optional[int] = None
    reason: str = ""
    stage: str = "none"
    used_fallback: bool = False
    matched_title: str = ""
    classification: str = OUTCOME_NEW_STORY
    thread_root_message_id: Optional[int] = None
    delta_reason: str = ""


def normalize_url(url: str) -> str:
    """Canonicalize an article URL (strip utm_*, fbclid, ref, fragments, slashes, www./m.)."""
    return Article.canonicalize_url(url) if url else ""


def normalize_arabic_kurdish_text(text: str) -> str:
    """
    Normalize Arabic & Central Kurdish (Sorani) text consistently:
      - Lowercase ASCII letters & normalize Eastern Arabic / Persian numerals to ASCII digits
      - Strip Arabic diacritics (tashkeel) and tatweel
      - Unify أ/إ/آ/ٱ -> ا, ة -> ه, ي/ى -> ی, ك -> ک
      - Strip punctuation and collapse extra whitespace
    """
    if not text:
        return ""
    t = text.lower().translate(_DIGIT_TRANSLATION)
    t = _ARABIC_DIACRITICS_RE.sub("", t)
    t = (
        t.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ٱ", "ا")
        .replace("ة", "ه")
        .replace("ى", "ی")
        .replace("ي", "ی")
        .replace("ك", "ک")
    )
    cleaned = re.sub(r"[^\w\s\u0621-\u064A\u0671-\u06D5]", " ", t)
    return re.sub(r"\s+", " ", cleaned).strip()


_TOKEN_SYNONYM_MAP: dict[str, str] = {}
for _concept_name, _forms in _CONCEPT_GROUPS:
    for _form in _forms:
        _norm_form = normalize_arabic_kurdish_text(_form)
        for _key in (_form.lower(), _norm_form):
            if _key and " " not in _key:
                _TOKEN_SYNONYM_MAP[_key] = _concept_name


def _normalize_unicode_text(text: str) -> str:
    """Backward-compatible alias for Arabic/Kurdish normalization."""
    return normalize_arabic_kurdish_text(text)


def _normalize_for_match(text: str) -> str:
    """Lowercase and normalize Arabic/Kurdish variants + strip punctuation for fast title similarity."""
    return normalize_arabic_kurdish_text(text)


def _stem_token(tok: str) -> str:
    if len(tok) > 5 and tok.isascii():
        if tok.endswith("ies"):
            return tok[:-3] + "y"
        if tok.endswith("ing") or tok.endswith("ment"):
            pass
        elif tok.endswith("ed") and len(tok) > 5:
            return tok[:-2]
        elif tok.endswith("es") and len(tok) > 4:
            return tok[:-2]
        elif tok.endswith("s") and not tok.endswith("ss") and len(tok) > 4:
            return tok[:-1]
    return tok


def _tokenize_content(text: str, extra_stopwords: Optional[set[str]] = None) -> set[str]:
    norm = _normalize_unicode_text(text)
    stops = _STOPWORDS if extra_stopwords is None else (_STOPWORDS | extra_stopwords)
    tokens: set[str] = set()
    for raw in norm.split():
        if raw in stops or len(raw) <= 2:
            continue
        stemmed = _stem_token(raw)
        if stemmed not in stops:
            tokens.add(stemmed)
    return tokens


def _tokenize_sorani(text: str) -> set[str]:
    """Tokenize Sorani Kurdish text stripped of high-frequency Kurdish stopwords."""
    norm = normalize_arabic_kurdish_text(text)
    tokens: set[str] = set()
    for raw in norm.split():
        if raw in _SORANI_STOPWORDS or raw in _STOPWORDS or len(raw) <= 2:
            continue
        tokens.add(raw)
    return tokens


def _char_ngrams(text: str, n: int = 4) -> set[str]:
    compact = re.sub(r"\s+", "", text or "")
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[i: i + n] for i in range(len(compact) - n + 1)}


def _extract_event_concepts(text: str) -> tuple[set[str], set[str], set[str]]:
    """
    Extract (place_or_actor_concepts, action_concepts, proper_entities_or_numbers)
    using zero-API multilingual canonical mapping.
    """
    norm = _normalize_unicode_text(text)
    padded = f" {norm} "
    places: set[str] = set()
    actions: set[str] = set()

    for concept, forms in _CONCEPT_GROUPS:
        matched = False
        for form in forms:
            norm_form = _normalize_unicode_text(form)
            if " " in norm_form:
                if f" {norm_form} " in padded:
                    matched = True
                    break
            elif norm_form in padded.split():
                matched = True
                break
        if matched:
            if concept.startswith("loc:"):
                places.add(concept)
            else:
                actions.add(concept)

    entities: set[str] = set()
    for cap in re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}\b", text or ""):
        low = cap.lower()
        if low not in _STOPWORDS and low not in _TOKEN_SYNONYM_MAP:
            entities.add(low)
    for num in re.findall(r"\b\d{2,}\b", norm):
        entities.add(num)

    return places, actions, entities


def _extract_numbers(text: str) -> set[str]:
    """Extract normalized meaningful numeric figures (excluding pure calendar years)."""
    norm = normalize_arabic_kurdish_text(text)
    nums: set[str] = set()
    for tok in re.findall(r"\b\d+\b", norm):
        if len(tok) == 4 and 2020 <= int(tok) <= 2035:
            continue
        nums.add(tok)
    return nums


def _extract_material_markers(norm_text: str) -> tuple[set[str], set[str]]:
    """Return `(matched_phrase_forms, matched_categories)` from normalized text."""
    padded = f" {norm_text} "
    forms: set[str] = set()
    cats: set[str] = set()
    for raw_phrase, cat in _MATERIAL_DEV_MARKERS:
        p = normalize_arabic_kurdish_text(raw_phrase)
        if not p:
            continue
        if " " in p:
            if f" {p} " in padded:
                forms.add(p)
                cats.add(cat)
        elif p in padded.split():
            forms.add(p)
            cats.add(cat)
    return forms, cats


def _reference_display_title(item: dict[str, Any]) -> str:
    return (
        (item.get("translated_title") or "").strip()
        or (item.get("original_title") or "").strip()
        or f"#{item.get('id', '?')}"
    )


class SemanticDeduplicator:
    """
    Zero-API cross-source & cross-language event deduplicator & Live Event Threader:
      - Classifies candidates cleanly into NEW_STORY | THREAD_UPDATE | DUPLICATE
      - Enforces active 24h window, 3-update thread cap, 20m cooldown (breaking bypass),
        and deterministic material factual delta verification without extra Gemini calls.
    """

    def __init__(
        self,
        db: Database,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        window_hours: Optional[int] = None,
        fast_threshold: float = 0.60,
        semantic_threshold: float = 0.52,
        max_candidates: int = 25,
        base_backoff: float = 0.0,
    ) -> None:
        self._db = db
        self._window_hours = window_hours if window_hours is not None else config.DEDUP_HISTORY_HOURS
        self._fast_threshold = fast_threshold
        self._semantic_threshold = semantic_threshold
        self._max_candidates = max_candidates
        self.uses_gemini = False

    async def cleanup(self) -> None:
        """No-op cleanup (no external API session held)."""
        return None

    async def _load_history(
        self, recent_history: Optional[list[dict[str, Any]]] = None
    ) -> list[dict[str, Any]]:
        if recent_history is not None:
            return list(recent_history)
        return await self._db.get_recent_published_headlines(hours=self._window_hours)

    @staticmethod
    def _compute_title_similarity(title_a: str, title_b: str) -> float:
        """Return normalized [0.0, 1.0] lexical similarity between two titles."""
        norm_a = _normalize_for_match(title_a)
        norm_b = _normalize_for_match(title_b)
        if not norm_a or not norm_b:
            return 0.0
        ratio_seq = fuzz.ratio(norm_a, norm_b) / 100.0
        ratio_tok = fuzz.token_sort_ratio(norm_a, norm_b) / 100.0
        return max(ratio_seq, ratio_tok)

    def _fast_similarity_check(
        self,
        article: Article,
        recent_headlines: list[dict[str, Any]],
    ) -> Optional[tuple[dict[str, Any], float, str]]:
        """Stage A: zero-cost normalized headline similarity check (>= 0.60)."""
        candidate_titles = [
            t for t in (article.original_title, article.translated_title) if t and t.strip()
        ]
        if not candidate_titles:
            return None

        best_score = 0.0
        best_item: Optional[dict[str, Any]] = None

        for item in recent_headlines:
            if article.id is not None and item.get("id") == article.id:
                continue
            ref_titles = [
                t for t in (item.get("original_title", ""), item.get("translated_title", ""))
                if t and t.strip()
            ]
            for cand in candidate_titles:
                for ref in ref_titles:
                    sim = self._compute_title_similarity(cand, ref)
                    if sim > best_score:
                        best_score = sim
                        best_item = item

        if best_item is not None and best_score >= self._fast_threshold:
            matched_title = _reference_display_title(best_item)
            return best_item, best_score, f"Fast title similarity {best_score:.2f} >= {self._fast_threshold:.2f} (matched '{matched_title[:80]}')"
        return None

    def _event_key_and_entities_match_item(
        self,
        article: Article,
        recent_headlines: list[dict[str, Any]],
    ) -> Optional[tuple[dict[str, Any], str, str]]:
        """
        Compare Gemini-extracted canonical English `event_key` and `event_entities`
        against stored 48-72h history & same-cycle items (0 extra API calls).
        Returns `(matched_item, stage, detail_reason)` or None.
        """
        cand_key = normalize_event_key(getattr(article, "event_key", ""))
        cand_entities_list = normalize_event_entities(getattr(article, "event_entities", []))
        if not cand_key and not cand_entities_list:
            cand_key, cand_entities_list = unpack_event_metadata(
                getattr(article, "named_entities", ""),
                cand_key,
                cand_entities_list,
            )
        if not cand_key and not cand_entities_list:
            return None

        cand_key_tokens = {
            tok for tok in cand_key.split("_") if len(tok) >= 2 and tok not in _STOPWORDS
        }
        cand_entity_set = {
            tok for tok in cand_entities_list if len(tok) >= 2 and tok not in _STOPWORDS
        }
        if cand_key_tokens and len(cand_entity_set) < 2:
            cand_entity_set |= cand_key_tokens

        for item in recent_headlines:
            if article.id is not None and item.get("id") == article.id:
                continue
            ref_key, ref_entities_list = unpack_event_metadata(
                item.get("named_entities", ""),
                item.get("event_key", ""),
                item.get("event_entities"),
            )
            if not ref_key and not ref_entities_list:
                continue

            ref_key_tokens = {
                tok for tok in ref_key.split("_") if len(tok) >= 2 and tok not in _STOPWORDS
            }
            ref_entity_set = {
                tok for tok in ref_entities_list if len(tok) >= 2 and tok not in _STOPWORDS
            }
            if ref_key_tokens and len(ref_entity_set) < 2:
                ref_entity_set |= ref_key_tokens

            matched_title = _reference_display_title(item)

            if cand_key and ref_key:
                if cand_key == ref_key and len(cand_key_tokens) >= 2:
                    return (
                        item,
                        "event_key",
                        f"Exact event_key match '{cand_key}' (matched '{matched_title[:80]}')",
                    )
                if cand_key_tokens and ref_key_tokens:
                    shared_k = cand_key_tokens & ref_key_tokens
                    shared_k_specific = shared_k - _BROAD_CONTEXT_TOKENS
                    k_jaccard = len(shared_k) / max(1, len(cand_key_tokens | ref_key_tokens))
                    k_contain = len(shared_k) / max(1, min(len(cand_key_tokens), len(ref_key_tokens)))
                    if (
                        (k_jaccard >= 0.65 and len(shared_k_specific) >= 1)
                        or (k_jaccard >= 0.58 and len(shared_k_specific) >= 2)
                        or (k_contain >= 0.75 and len(shared_k_specific) >= 2)
                    ):
                        return (
                            item,
                            "event_key",
                            f"Event key match '{cand_key}' ~ '{ref_key}' (jaccard={k_jaccard:.2f}; matched '{matched_title[:80]}')",
                        )

            if cand_entity_set and ref_entity_set:
                shared_e = cand_entity_set & ref_entity_set
                shared_e_specific = shared_e - _BROAD_CONTEXT_TOKENS
                e_jaccard = len(shared_e) / max(1, len(cand_entity_set | ref_entity_set))
                if (len(shared_e) >= 3 or e_jaccard >= 0.55) and (
                    len(shared_e_specific) >= 1 or e_jaccard >= 0.85
                ):
                    return (
                        item,
                        "event_entities",
                        f"Event entities overlap ({len(shared_e)} shared, jaccard={e_jaccard:.2f}: {','.join(sorted(shared_e)[:4])}; matched '{matched_title[:80]}')",
                    )

        return None

    def _local_semantic_match_item(
        self,
        article: Article,
        recent_headlines: list[dict[str, Any]],
    ) -> Optional[tuple[dict[str, Any], float, str]]:
        """
        Stage B (zero-API): Compare candidate title/summary against recent published
        headlines using token overlap, multilingual canonical entities/actions, and token-set ratio.
        """
        cand_text = " ".join(
            p for p in (article.original_title, article.translated_title, (article.original_summary or "")[:240])
            if p and p.strip()
        ).strip()
        if not cand_text:
            return None

        cand_tokens = _tokenize_content(f"{article.original_title or ''} {article.translated_title or ''}")
        cand_places, cand_actions, cand_entities = _extract_event_concepts(cand_text)
        cand_norm_title = _normalize_unicode_text(article.original_title or article.translated_title or "")
        cand_key = normalize_event_key(getattr(article, "event_key", ""))
        cand_ev_ents = set(normalize_event_entities(getattr(article, "event_entities", [])))

        best_score = 0.0
        best_item: Optional[dict[str, Any]] = None
        best_detail = ""

        for item in recent_headlines:
            if article.id is not None and item.get("id") == article.id:
                continue

            ref_key, ref_ev_ents_list = unpack_event_metadata(
                item.get("named_entities", ""),
                item.get("event_key", ""),
                item.get("event_entities"),
            )
            if cand_key and ref_key and cand_key != ref_key:
                cand_spec_ents = (cand_ev_ents | set(cand_key.split("_"))) - _BROAD_CONTEXT_TOKENS - _STOPWORDS
                ref_spec_ents = (set(ref_ev_ents_list) | set(ref_key.split("_"))) - _BROAD_CONTEXT_TOKENS - _STOPWORDS
                if cand_spec_ents and ref_spec_ents and not (cand_spec_ents & ref_spec_ents):
                    continue

            ref_title = " ".join(
                t for t in (item.get("original_title", ""), item.get("translated_title", ""))
                if t and t.strip()
            )
            ref_text = " ".join(
                t for t in (
                    ref_title,
                    (item.get("original_summary") or "")[:240],
                    (item.get("translated_summary") or "")[:240],
                )
                if t and t.strip()
            )
            if not ref_text:
                continue

            ref_tokens = _tokenize_content(ref_title or ref_text)
            ref_places, ref_actions, ref_entities = _extract_event_concepts(ref_text)
            ref_norm_title = _normalize_unicode_text(item.get("original_title", "") or item.get("translated_title", ""))

            shared_places = cand_places & ref_places
            shared_actions = cand_actions & ref_actions
            shared_entities = cand_entities & ref_entities
            shared_tokens = cand_tokens & ref_tokens
            shared_non_loc_tokens = {
                tok for tok in shared_tokens
                if not (_TOKEN_SYNONYM_MAP.get(tok, "").startswith("loc:"))
                and tok not in _BROAD_CONTEXT_TOKENS
            }

            token_overlap = (
                len(shared_tokens) / max(1, min(len(cand_tokens), len(ref_tokens)))
                if cand_tokens and ref_tokens
                else 0.0
            )
            token_jaccard = (
                len(shared_tokens) / max(1, len(cand_tokens | ref_tokens))
                if cand_tokens and ref_tokens
                else 0.0
            )
            token_set = (
                fuzz.token_set_ratio(cand_norm_title, ref_norm_title) / 100.0
                if cand_norm_title and ref_norm_title
                else 0.0
            )

            all_cand_concepts = cand_places | cand_actions | cand_entities
            all_ref_concepts = ref_places | ref_actions | ref_entities
            shared_concept_count = len(shared_places) + len(shared_actions) + len(shared_entities)
            concept_jaccard = (
                shared_concept_count / max(1, len(all_cand_concepts | all_ref_concepts))
                if (all_cand_concepts and all_ref_concepts)
                else 0.0
            )

            combined = (0.38 * token_overlap) + (0.32 * token_set) + (0.30 * concept_jaccard)

            concept_anchors = len(shared_places) + len(shared_actions) + (1 if len(shared_entities) >= 1 else 0)
            paraphrase_hit = bool(
                (shared_places or len(shared_entities) >= 2)
                and shared_actions
                and (concept_anchors >= 3 or token_overlap >= 0.28 or token_set >= 0.42)
            )
            lexical_entity_hit = bool(
                len(shared_non_loc_tokens) >= 2
                and len(shared_tokens) >= 3
                and token_overlap >= 0.50
                and token_set >= 0.58
            )
            combined_hit = bool(
                combined >= self._semantic_threshold
                and (len(shared_non_loc_tokens) >= 2 or (bool(shared_actions) and len(shared_tokens) >= 2))
            )

            if paraphrase_hit or lexical_entity_hit or combined_hit:
                effective = max(combined, 0.72 if paraphrase_hit else combined)
                if effective > best_score:
                    best_score = effective
                    best_item = item
                    matched_anchors = sorted(shared_places | shared_actions | shared_entities)[:4]
                    best_detail = (
                        f"Local cross-source match (score={effective:.2f}, jaccard={token_jaccard:.2f}, "
                        f"anchors={','.join(matched_anchors) or 'tokens'})"
                    )

        if best_item is not None:
            matched_title = _reference_display_title(best_item)
            return best_item, best_score, f"{best_detail}; matched '{matched_title[:80]}'"
        return None

    @staticmethod
    def _resolve_thread_context(
        matched_item: dict[str, Any],
        combined_history: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], Optional[int], list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Walk up to the root post of a matched item and return:
        `(root_item, root_message_id, updates_in_thread, all_thread_items)`.
        """
        explicit_root_msg = matched_item.get("thread_root_message_id")
        matched_msg = matched_item.get("telegram_message_id")

        root_msg_id: Optional[int] = (
            int(explicit_root_msg)
            if explicit_root_msg is not None
            else (int(matched_msg) if matched_msg is not None else None)
        )

        root_item = matched_item
        if root_msg_id is not None:
            for item in combined_history:
                if (
                    item.get("telegram_message_id") == root_msg_id
                    and item.get("thread_root_message_id") is None
                ):
                    root_item = item
                    break

            thread_items: list[dict[str, Any]] = []
            seen_ids: set[int] = set()
            for item in combined_history:
                is_root = item.get("telegram_message_id") == root_msg_id and item.get("thread_root_message_id") is None
                is_child = item.get("thread_root_message_id") == root_msg_id
                if is_root or is_child:
                    oid = id(item)
                    if oid not in seen_ids:
                        seen_ids.add(oid)
                        thread_items.append(item)
            if root_item and id(root_item) not in seen_ids:
                thread_items.append(root_item)

            updates = [it for it in thread_items if it.get("thread_root_message_id") == root_msg_id]
            return root_item, root_msg_id, updates, thread_items

        return matched_item, None, [], [matched_item]

    @staticmethod
    def _detect_material_delta(
        article: Article,
        thread_items: list[dict[str, Any]],
    ) -> tuple[bool, str]:
        """
        Zero-API deterministic material-development detector comparing candidate against
        ALL existing posts in the active thread (root + any prior replies).
        Returns `(has_material_delta, delta_description)`.
        """
        cand_raw_parts = [
            article.original_title or "",
            article.original_summary or "",
            article.translated_title or "",
            article.translated_summary or "",
        ]
        cand_joined = " ".join(p for p in cand_raw_parts if p and p.strip())
        cand_norm = normalize_arabic_kurdish_text(cand_joined)
        if not cand_norm:
            return False, ""

        thread_raw_parts: list[str] = []
        cand_title_norm = normalize_arabic_kurdish_text(
            article.translated_title or article.original_title or ""
        )
        max_title_fuzz = 0.0
        for item in thread_items:
            t_parts = [
                item.get("original_title") or "",
                item.get("original_summary") or "",
                item.get("translated_title") or "",
                item.get("translated_summary") or "",
            ]
            thread_raw_parts.extend(p for p in t_parts if p and p.strip())
            ref_title_norm = normalize_arabic_kurdish_text(
                item.get("translated_title") or item.get("original_title") or ""
            )
            if cand_title_norm and ref_title_norm:
                max_title_fuzz = max(max_title_fuzz, fuzz.ratio(cand_title_norm, ref_title_norm) / 100.0)

        thread_joined = " ".join(thread_raw_parts)
        thread_norm = normalize_arabic_kurdish_text(thread_joined)

        # Reject near-verbatim headline copies unconditionally
        if max_title_fuzz >= 0.88:
            return False, ""

        # Check optional Gemini hint (0 extra API calls)
        update_type = (getattr(article, "update_type", "") or "").strip().lower()
        if update_type == "repeat_coverage":
            # Only trust concrete numeric changes if Gemini explicitly flagged repeat coverage
            update_type = ""

        # 1. Quantitative / casualty number deltas absent from earlier thread coverage
        cand_nums = _extract_numbers(cand_joined)
        thread_nums = _extract_numbers(thread_joined)
        new_nums = sorted(cand_nums - thread_nums)

        # 2. Material development markers (EN / Sorani / Arabic)
        cand_markers, cand_marker_cats = _extract_material_markers(cand_norm)
        thread_markers, thread_marker_cats = _extract_material_markers(thread_norm)
        new_markers = sorted(cand_markers - thread_markers)
        new_cats = sorted(cand_marker_cats - thread_marker_cats)

        # 3. Newly introduced proper/institutional English actors
        cand_actors = {
            m.lower()
            for m in re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}\b", cand_joined)
            if m.lower() not in _STOPWORDS and m.lower() not in _BROAD_CONTEXT_TOKENS
        }
        thread_actors = {
            m.lower()
            for m in re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}\b", thread_joined)
            if m.lower() not in _STOPWORDS and m.lower() not in _BROAD_CONTEXT_TOKENS
        }
        new_actors = sorted(cand_actors - thread_actors)

        # Avoid treating minor rephrase of the exact same title with 0 new facts as update
        if max_title_fuzz >= 0.80 and not new_nums and not new_cats:
            return False, ""

        reasons: list[str] = []
        if new_nums:
            reasons.append(f"new_figures={','.join(new_nums[:3])}")
        if new_cats or len(new_markers) >= 1:
            marker_lbl = ",".join((new_cats or new_markers)[:3])
            reasons.append(f"dev_signals={marker_lbl}")
        if update_type in _POSITIVE_UPDATE_TYPES:
            reasons.append(f"update_type={update_type}")
        if len(new_actors) >= 2 and (len(cand_markers) >= 1 or new_nums):
            reasons.append(f"new_actors={','.join(new_actors[:3])}")

        if reasons:
            return True, "; ".join(reasons)
        return False, ""

    def _classify_matched_event(
        self,
        article: Article,
        matched_item: dict[str, Any],
        combined_history: list[dict[str, Any]],
        match_stage: str,
        match_reason: str,
        *,
        allow_pending_cooldown: bool = False,
    ) -> DedupResult:
        """
        Apply the 4 strict anti-spam Threading Guardrails onto a matched event to classify it
        as DUPLICATE, THREAD_UPDATE, or NEW_STORY (when thread window expired with new facts).
        """
        matched_id = int(matched_item["id"]) if matched_item.get("id") is not None else None
        matched_title = _reference_display_title(matched_item)

        if not config.ENABLE_LIVE_EVENT_THREADING:
            return DedupResult(
                is_duplicate=True,
                matched_id=matched_id,
                reason=match_reason,
                stage=match_stage,
                matched_title=matched_title,
                classification=OUTCOME_DUPLICATE,
            )

        root_item, root_msg_id, updates_in_thread, thread_items = self._resolve_thread_context(
            matched_item, combined_history
        )
        root_title = _reference_display_title(root_item)

        has_delta, delta_reason = self._detect_material_delta(article, thread_items)

        # Check 1: Root window age (24h cap)
        now = datetime.now(timezone.utc)
        root_dt = parse_utc_timestamp(root_item.get("published_at") or root_item.get("discovered_at"))
        if root_dt is not None:
            root_age_hours = max(0.0, (now - root_dt).total_seconds() / 3600.0)
            if root_age_hours > config.EVENT_THREAD_WINDOW_HOURS:
                if has_delta:
                    return DedupResult(
                        is_duplicate=False,
                        matched_id=None,
                        reason=(
                            f"Event window expired ({root_age_hours:.1f}h > {config.EVENT_THREAD_WINDOW_HOURS}h) "
                            f"with new material facts ({delta_reason}); publishing as standalone story"
                        ),
                        stage="thread_window_expired",
                        matched_title=root_title,
                        classification=OUTCOME_NEW_STORY,
                        thread_root_message_id=None,
                        delta_reason=delta_reason,
                    )
                return DedupResult(
                    is_duplicate=True,
                    matched_id=matched_id,
                    reason=f"Stale repeat outside {config.EVENT_THREAD_WINDOW_HOURS}h thread window ({match_reason})",
                    stage=match_stage,
                    matched_title=matched_title,
                    classification=OUTCOME_DUPLICATE,
                )

        # Check 2: Active Telegram root message_id must exist to reply to
        if root_msg_id is None:
            cand_raw = f"{article.original_title or ''} {article.original_summary or ''}"
            if (
                allow_pending_cooldown
                and matched_item.get("is_published") is False
                and has_delta
                and bool(_ACUTE_EMERGENCY_HINTS_RE.search(cand_raw))
            ):
                return DedupResult(
                    is_duplicate=False,
                    matched_id=None,
                    reason="Deferring same-cycle acute update pending earlier sibling Telegram message_id",
                    stage="pending_cycle_root",
                    classification=OUTCOME_NEW_STORY,
                )
            return DedupResult(
                is_duplicate=True,
                matched_id=matched_id,
                reason=f"No telegram_message_id anchor on root event ({match_reason})",
                stage=match_stage,
                matched_title=matched_title,
                classification=OUTCOME_DUPLICATE,
            )

        # Check 3: Require genuinely new material development (anti-rephrase guard)
        if not has_delta:
            return DedupResult(
                is_duplicate=True,
                matched_id=matched_id,
                reason=f"No material delta over root story ({match_reason})",
                stage=match_stage,
                matched_title=matched_title,
                classification=OUTCOME_DUPLICATE,
            )

        # Check 4: Thread reply cap (< MAX_UPDATES_PER_THREAD = 3)
        if len(updates_in_thread) >= config.MAX_UPDATES_PER_THREAD:
            return DedupResult(
                is_duplicate=True,
                matched_id=matched_id,
                reason=(
                    f"Thread update cap reached ({len(updates_in_thread)}/{config.MAX_UPDATES_PER_THREAD} "
                    f"updates on root msg_id={root_msg_id})"
                ),
                stage="thread_cap_reached",
                matched_title=root_title,
                classification=OUTCOME_DUPLICATE,
            )

        # Check 5: Cooldown >= THREAD_UPDATE_COOLDOWN_MINUTES (20m), bypassed ONLY when strict breaking
        strict_breaking = is_strict_breaking(article)
        last_dt = root_dt
        for item in thread_items:
            dt = parse_utc_timestamp(item.get("published_at") or item.get("discovered_at"))
            if dt is not None and (last_dt is None or dt > last_dt):
                last_dt = dt

        if last_dt is not None and not strict_breaking:
            elapsed_minutes = max(0.0, (now - last_dt).total_seconds() / 60.0)
            if elapsed_minutes < config.THREAD_UPDATE_COOLDOWN_MINUTES:
                cand_raw = f"{article.original_title or ''} {article.original_summary or ''}"
                is_unscored_acute = (
                    allow_pending_cooldown
                    and float(getattr(article, "importance_score", 0.0) or 0.0) <= 0.0
                    and bool(_ACUTE_EMERGENCY_HINTS_RE.search(cand_raw))
                )
                if not is_unscored_acute:
                    return DedupResult(
                        is_duplicate=True,
                        matched_id=matched_id,
                        reason=(
                            f"Thread update cooldown active ({elapsed_minutes:.1f}m < "
                            f"{config.THREAD_UPDATE_COOLDOWN_MINUTES}m on root msg_id={root_msg_id})"
                        ),
                        stage="thread_cooldown",
                        matched_title=root_title,
                        classification=OUTCOME_DUPLICATE,
                        delta_reason=delta_reason,
                    )

        return DedupResult(
            is_duplicate=False,
            matched_id=int(root_item["id"]) if root_item.get("id") is not None else matched_id,
            reason=(
                f"Material update eligible for Live Thread reply to msg_id={root_msg_id} "
                f"({len(updates_in_thread)}/{config.MAX_UPDATES_PER_THREAD} used; {delta_reason})"
            ),
            stage=match_stage,
            matched_title=root_title,
            classification=OUTCOME_THREAD_UPDATE,
            thread_root_message_id=root_msg_id,
            delta_reason=delta_reason,
        )

    async def check_duplicate(
        self,
        article: Article,
        extra_recent: Optional[list[dict[str, Any]]] = None,
        recent_history: Optional[list[dict[str, Any]]] = None,
        *,
        allow_pending_cooldown: bool = False,
    ) -> DedupResult:
        """
        Classify `article` into NEW_STORY | THREAD_UPDATE | DUPLICATE against
        the last `DEDUP_HISTORY_HOURS` (48-72h) or same-cycle accepted items using 0 Gemini calls.
        """
        db_history = await self._load_history(recent_history)
        combined_history = list(extra_recent or []) + db_history
        if not combined_history:
            return DedupResult(
                is_duplicate=False,
                reason="No recent published headlines in lookback window",
                stage="none",
                classification=OUTCOME_NEW_STORY,
            )

        # Stage 0: Canonical English event_key & event_entities match (if populated)
        ev_match = self._event_key_and_entities_match_item(article, combined_history)
        if ev_match is not None:
            matched_item, stage, reason = ev_match
            return self._classify_matched_event(
                article, matched_item, combined_history, stage, reason,
                allow_pending_cooldown=allow_pending_cooldown,
            )

        # Stage A: Fast normalized headline similarity
        fast_hit = self._fast_similarity_check(article, combined_history)
        if fast_hit is not None:
            matched_item, _, reason = fast_hit
            return self._classify_matched_event(
                article, matched_item, combined_history, "fast", reason,
                allow_pending_cooldown=allow_pending_cooldown,
            )

        # Stage B: Zero-API local semantic / entity / concept match
        sem_hit = self._local_semantic_match_item(article, combined_history)
        if sem_hit is not None:
            matched_item, _, reason = sem_hit
            return self._classify_matched_event(
                article, matched_item, combined_history, "local_semantic", reason,
                allow_pending_cooldown=allow_pending_cooldown,
            )

        return DedupResult(
            is_duplicate=False,
            matched_id=None,
            reason="Distinct event (local zero-API dedup)",
            stage="local_semantic",
            classification=OUTCOME_NEW_STORY,
        )

    async def check_event_match(
        self,
        article: Article,
        extra_recent: Optional[list[dict[str, Any]]] = None,
        recent_history: Optional[list[dict[str, Any]]] = None,
    ) -> DedupResult:
        """
        Post-Gemini-scoring / pre-translation cross-language check on `event_key`,
        `event_entities`, and strict breaking/cooldown rules.
        """
        db_history = await self._load_history(recent_history)
        combined_history = list(extra_recent or []) + db_history
        if not combined_history:
            return DedupResult(is_duplicate=False, stage="event_key", classification=OUTCOME_NEW_STORY)

        # Re-check structured event match OR re-verify pending thread cooldown once score/is_breaking is known
        ev_match = self._event_key_and_entities_match_item(article, combined_history)
        if ev_match is not None:
            matched_item, stage, reason = ev_match
            return self._classify_matched_event(
                article, matched_item, combined_history, stage, reason,
                allow_pending_cooldown=False,
            )

        if getattr(article, "thread_root_message_id", None) is not None or getattr(article, "is_thread_update", False):
            # Re-verify strict cooldown now that real score & is_breaking are set
            fast_hit = self._fast_similarity_check(article, combined_history)
            sem_hit = fast_hit or self._local_semantic_match_item(article, combined_history)
            if sem_hit is not None:
                matched_item, _, reason = sem_hit
                return self._classify_matched_event(
                    article, matched_item, combined_history, "post_scoring_thread", reason,
                    allow_pending_cooldown=False,
                )

        return DedupResult(is_duplicate=False, stage="event_key", classification=OUTCOME_NEW_STORY)

    async def check_post_translation_duplicate(
        self,
        article: Article,
        extra_recent: Optional[list[dict[str, Any]]] = None,
        recent_history: Optional[list[dict[str, Any]]] = None,
    ) -> DedupResult:
        """
        Final Post-Translation Sorani Kurdish Safety Gate right before card generation & publish:
          - If already classified THREAD_UPDATE: verifies the translated Kurdish text genuinely
            adds new facts vs prior Sorani text in that thread (downgrades to DUPLICATE if repeated).
          - If NEW_STORY: compares translated Kurdish title/summary against 48-72h history &
            current cycle to classify NEW_STORY | THREAD_UPDATE | DUPLICATE cleanly.
        """
        db_history = await self._load_history(recent_history)
        combined_history = list(extra_recent or []) + db_history
        if not combined_history:
            return DedupResult(
                is_duplicate=False,
                stage="post_translation_sorani",
                classification=(
                    OUTCOME_THREAD_UPDATE
                    if getattr(article, "thread_root_message_id", None) is not None
                    else OUTCOME_NEW_STORY
                ),
                thread_root_message_id=getattr(article, "thread_root_message_id", None),
            )

        existing_root_msg = getattr(article, "thread_root_message_id", None)
        is_existing_thread_candidate = bool(
            getattr(article, "is_thread_update", False) or existing_root_msg is not None
        )

        # 1. If already earmarked as THREAD_UPDATE, validate Sorani text vs that thread specifically!
        if is_existing_thread_candidate and existing_root_msg is not None:
            thread_posts = [
                it for it in combined_history
                if (article.id is None or it.get("id") != article.id)
                and (
                    (it.get("telegram_message_id") == existing_root_msg and it.get("thread_root_message_id") is None)
                    or it.get("thread_root_message_id") == existing_root_msg
                )
            ]
            if thread_posts:
                cand_ku_title = normalize_arabic_kurdish_text(article.translated_title or "")
                cand_ku_sum = normalize_arabic_kurdish_text((article.translated_summary or "")[:320])
                cand_nums = _extract_numbers(f"{article.translated_title or ''} {article.translated_summary or ''}")
                cand_markers, _ = _extract_material_markers(f"{cand_ku_title} {cand_ku_sum}")
                cand_tokens = _tokenize_sorani(f"{article.translated_title or ''} {article.translated_summary or ''}")

                prior_sorani_text = " ".join(
                    f"{p.get('translated_title') or ''} {p.get('translated_summary') or ''}"
                    for p in thread_posts
                )
                prior_nums = _extract_numbers(prior_sorani_text)
                prior_markers, _ = _extract_material_markers(normalize_arabic_kurdish_text(prior_sorani_text))
                has_sorani_delta = bool((cand_nums - prior_nums) or (cand_markers - prior_markers))

                for p in thread_posts:
                    p_title = normalize_arabic_kurdish_text(p.get("translated_title") or "")
                    p_sum = normalize_arabic_kurdish_text((p.get("translated_summary") or "")[:320])
                    if not p_title and not p_sum:
                        continue
                    p_tokens = _tokenize_sorani(f"{p.get('translated_title') or ''} {p.get('translated_summary') or ''}")
                    shared = cand_tokens & p_tokens
                    contain = len(shared) / max(1, min(len(cand_tokens), len(p_tokens))) if (cand_tokens and p_tokens) else 0.0
                    t_fuzz = fuzz.ratio(cand_ku_title, p_title) / 100.0 if (cand_ku_title and p_title) else 0.0
                    s_fuzz = fuzz.token_set_ratio(cand_ku_sum, p_sum) / 100.0 if (cand_ku_sum and p_sum) else 0.0

                    if (t_fuzz >= 0.72 or (s_fuzz >= 0.78 and contain >= 0.65)) and not has_sorani_delta:
                        matched_title = _reference_display_title(p)
                        return DedupResult(
                            is_duplicate=True,
                            matched_id=int(p["id"]) if p.get("id") is not None else None,
                            reason=(
                                f"Post-translation Sorani check downgraded THREAD_UPDATE to DUPLICATE: "
                                f"Kurdish text repeats '{matched_title[:80]}'"
                            ),
                            stage="post_translation_sorani",
                            matched_title=matched_title,
                            classification=OUTCOME_DUPLICATE,
                            thread_root_message_id=None,
                        )

                # Re-run strict thread check with Sorani text included
                rep = thread_posts[0]
                return self._classify_matched_event(
                    article,
                    rep,
                    combined_history,
                    "post_translation_thread",
                    f"Verified Sorani thread update on msg_id={existing_root_msg}",
                    allow_pending_cooldown=False,
                )

        # 2. Check structured event_key / event_entities first (including translator metadata)
        ev_match = self._event_key_and_entities_match_item(article, combined_history)
        if ev_match is not None:
            matched_item, stage, reason = ev_match
            return self._classify_matched_event(
                article, matched_item, combined_history, stage, reason,
                allow_pending_cooldown=False,
            )

        # 3. Compare stopword-stripped Kurdish title & summary
        cand_ku_title = normalize_arabic_kurdish_text(article.translated_title or "")
        cand_ku_sum = normalize_arabic_kurdish_text((article.translated_summary or "")[:320])
        if not cand_ku_title and not cand_ku_sum:
            return DedupResult(is_duplicate=False, stage="post_translation_sorani", classification=OUTCOME_NEW_STORY)

        cand_title_tokens = _tokenize_sorani(article.translated_title or "")
        cand_all_tokens = _tokenize_sorani(f"{article.translated_title or ''} {article.translated_summary or ''}")
        cand_title_clean_str = " ".join(sorted(cand_title_tokens))
        cand_4grams = _char_ngrams(" ".join(cand_title_tokens), n=4)
        cand_places, cand_actions, cand_entities = _extract_event_concepts(
            f"{article.translated_title or ''} {article.translated_summary or ''}"
        )

        best_score = 0.0
        best_item: Optional[dict[str, Any]] = None
        best_detail = ""

        for item in combined_history:
            if article.id is not None and item.get("id") == article.id:
                continue

            ref_raw_title = (item.get("translated_title") or "").strip()
            ref_raw_sum = (item.get("translated_summary") or "").strip()[:320]
            if not ref_raw_title and not ref_raw_sum:
                continue

            ref_ku_title = normalize_arabic_kurdish_text(ref_raw_title)
            ref_title_tokens = _tokenize_sorani(ref_raw_title)
            ref_all_tokens = _tokenize_sorani(f"{ref_raw_title} {ref_raw_sum}")
            if not cand_all_tokens or not ref_all_tokens:
                continue

            shared_all = cand_all_tokens & ref_all_tokens
            shared_title = cand_title_tokens & ref_title_tokens
            shared_specific = shared_all - _BROAD_CONTEXT_TOKENS

            tok_jaccard = len(shared_all) / max(1, len(cand_all_tokens | ref_all_tokens))
            tok_contain = len(shared_all) / max(1, min(len(cand_all_tokens), len(ref_all_tokens)))

            ref_title_clean_str = " ".join(sorted(ref_title_tokens))
            title_fuzz = (
                max(
                    fuzz.ratio(cand_ku_title, ref_ku_title),
                    fuzz.token_set_ratio(cand_title_clean_str, ref_title_clean_str),
                )
                / 100.0
                if (cand_ku_title and ref_ku_title and cand_title_clean_str and ref_title_clean_str)
                else 0.0
            )
            ref_4grams = _char_ngrams(" ".join(ref_title_tokens), n=4)
            gram_jaccard = (
                len(cand_4grams & ref_4grams) / max(1, len(cand_4grams | ref_4grams))
                if (cand_4grams and ref_4grams)
                else 0.0
            )
            gram_contain = (
                len(cand_4grams & ref_4grams) / max(1, min(len(cand_4grams), len(ref_4grams)))
                if (cand_4grams and ref_4grams)
                else 0.0
            )

            ref_places, ref_actions, ref_entities = _extract_event_concepts(f"{ref_raw_title} {ref_raw_sum}")
            shared_concept_actions = cand_actions & ref_actions
            shared_concept_places = (cand_places & ref_places) | (cand_entities & ref_entities)

            direct_title_dup = bool(len(shared_specific) >= 1 and title_fuzz >= 0.72)
            rephrased_sorani_dup = bool(
                len(shared_specific) >= 2
                and (
                    tok_jaccard >= 0.40
                    or tok_contain >= 0.55
                    or title_fuzz >= 0.62
                    or (gram_jaccard >= 0.36 and len(shared_title) >= 2)
                )
            )
            concept_sorani_dup = bool(
                shared_concept_actions
                and shared_concept_places
                and (gram_contain >= 0.45 or tok_contain >= 0.38 or len(shared_specific) >= 1)
            )

            if direct_title_dup or rephrased_sorani_dup or concept_sorani_dup:
                combined = max(
                    title_fuzz,
                    (0.45 * tok_contain) + (0.30 * gram_jaccard) + (0.25 * title_fuzz),
                )
                if combined > best_score:
                    best_score = combined
                    best_item = item
                    best_detail = (
                        f"Sorani dedup score={combined:.2f}, jaccard={tok_jaccard:.2f}, "
                        f"fuzz={title_fuzz:.2f}, specific={','.join(sorted(shared_specific)[:4]) or 'concepts'}"
                    )

        if best_item is not None:
            matched_title = _reference_display_title(best_item)
            return self._classify_matched_event(
                article,
                best_item,
                combined_history,
                "post_translation_sorani",
                f"Duplicate story blocked: matched '{matched_title[:100]}' ({best_detail})",
                allow_pending_cooldown=False,
            )

        return DedupResult(
            is_duplicate=False,
            reason="Distinct Sorani story",
            stage="post_translation_sorani",
            classification=OUTCOME_NEW_STORY,
        )
