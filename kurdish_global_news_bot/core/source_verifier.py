"""
Kurdish Global News Bot — Source Credibility Tiers, Cross-Source Corroboration
& Single-Source Verification Guard

Deterministic, 100% local (ZERO Gemini API calls):
  - Classifies sources into Tier 1 (official wire/broadcast), Tier 2 (established
    regional/OSINT), and Tier 3 (factional/unverified Telegram).
  - Groups paired outlets (e.g. Sabereen News + Sabereen Plus = "sabereen") so they
    count as ONE editorial faction for corroboration and dominance capping.
  - Computes cross-source corroboration using event_key, event_entities, and
    normalized title overlap — across the current discovery batch AND recent 24h storage.
  - Blocks uncorroborated Tier-3 single-source claims from firing 🔴 بەپەلە alerts.
  - Marks uncorroborated Tier-3 articles for attribution tagging.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from urllib.parse import urlparse

import config
from processing.extractor import ArticleExtractor
from storage.models import Article
from storage.utils import normalize_event_entities, normalize_event_key, unpack_event_metadata

logger = logging.getLogger("core.source_verifier")

# ═══════════════════════════════════════════════════
# Credibility Tier Constants
# ═══════════════════════════════════════════════════
TIER_1 = 1  # Official & Global Tier-1 wire/broadcast
TIER_2 = 2  # Established Regional / Kurdish Media & Structured OSINT
TIER_3 = 3  # Unverified / Factional / Single-Channel Telegram

# ═══════════════════════════════════════════════════
# Source Group Mapping
# ═══════════════════════════════════════════════════
# Maps lowercased source_name → canonical source_group name.
# Outlets belonging to the same organization share the same group.
_SOURCE_GROUP_MAP: dict[str, str] = {
    # Sabereen faction (both channels → "sabereen")
    "sabereen news": "sabereen",
    "sabereen plus": "sabereen",
    # Kurdish media (Telegram + RSS → same group)
    "rudaw": "rudaw",
    "rudaw sorani": "rudaw",
    "rudaw digital": "rudaw",
    "rudaw english": "rudaw",
    "rudaw arabic": "rudaw",
    "kurdistan24": "kurdistan24",
    "kurdistan24 en": "kurdistan24",
    "kurdistan24 sorani": "kurdistan24",
    "kurdistan 24": "kurdistan24",
    "k24": "kurdistan24",
    "nrt": "nrt",
    "nrt tv": "nrt",
    # OSINT / Conflict & X (Twitter) 16-Account Roster
    "liveuamap": "liveuamap",
    "clash report": "clashreport",
    "clashreport": "clashreport",
    "centcom": "centcom",
    "basnews": "basnews",
    "englishbasnews": "basnews",
    "shafaq": "shafaq",
    "shafaq news": "shafaq",
    "shafaqnews": "shafaq",
    "rawafid": "rawafid",
    "rawafiid": "rawafid",
    "aurora intel": "auroraintel",
    "auroraintel": "auroraintel",
    "osinttechnical": "osinttechnical",
    "faytuks": "faytuks",
    "faytuks news": "faytuks",
    "intel crab": "intelcrab",
    "intelcrab": "intelcrab",
    "osintdefender": "osintdefender",
    "sentdefender": "osintdefender",
    "elint news": "elintnews",
    "elintnews": "elintnews",
    "rudawkurdish": "rudaw",
    "rudawenglish": "rudaw",
    "kurdistan24tv": "kurdistan24",
    "kurdistan24_en": "kurdistan24",
    # Global Tier-1 wire services
    "reuters": "reuters",
    "reuters world": "reuters",
    "reuters technology": "reuters",
    "reuters business": "reuters",
    "reuters science": "reuters",
    "ap news": "ap",
    "associated press": "ap",
    "bbc": "bbc",
    "bbc world": "bbc",
    "bbc news": "bbc",
    "bbc technology": "bbc",
    "al jazeera": "aljazeera",
    "al jazeera arabic": "aljazeera",
    "al jazeera - breaking": "aljazeera",
    "ajanews": "aljazeera",
    "الجزيرة": "aljazeera",
    "قناة الجزيرة": "aljazeera",
    "france24": "france24",
    "france24 en": "france24",
    "dw": "dw",
    "dw news": "dw",
    "the guardian": "guardian",
    "the guardian world": "guardian",
    "npr": "npr",
    "npr news": "npr",
    "un news": "un",
    "who": "who",
    "who news": "who",
    "nato": "nato",
    "nato news": "nato",
    "eu news": "eu",
    "iaea": "iaea",
    "iaea news": "iaea",
    "cnn": "cnn",
    "new york times": "nytimes",
    "washington post": "washingtonpost",
    # Defense & cybersecurity
    "defense news": "defensenews",
    "defense news global": "defensenews",
    "military times": "militarytimes",
    "the hacker news": "thehackernews",
    "bleepingcomputer": "bleepingcomputer",
    "cnbc": "cnbc",
    "cnbc world": "cnbc",
    "space.com": "spacecom",
    "sky news": "skynews",
    # Kurdish regional
    "north press agency": "northpress",
    "rojava information center": "ric",
    "pukmedia": "pukmedia",
    "pukmedia en": "pukmedia",
    "pukmedia sorani": "pukmedia",
    "bianet": "bianet",
    "bianet en": "bianet",
    "khrn": "khrn",
    "google news": "googlenews",
    "google news - south kurdistan": "googlenews",
    "google news - kurdistan sorani": "googlenews",
    "google news - west kurdistan": "googlenews",
    "google news - north kurdistan": "googlenews",
    "google news - east kurdistan": "googlenews",
}

# Source group → Tier mapping
_GROUP_TIER_MAP: dict[str, int] = {
    # Tier 1: Official wire/broadcast, Al Jazeera & CENTCOM
    "aljazeera": TIER_1,
    "reuters": TIER_1,
    "ap": TIER_1,
    "bbc": TIER_1,
    "un": TIER_1,
    "who": TIER_1,
    "nato": TIER_1,
    "eu": TIER_1,
    "iaea": TIER_1,
    "cnn": TIER_1,
    "nytimes": TIER_1,
    "washingtonpost": TIER_1,
    "centcom": TIER_1,
    # Tier 2: Established Regional / Kurdish / OSINT & Demoted Western Feeds
    "rudaw": TIER_2,
    "kurdistan24": TIER_2,
    "basnews": TIER_2,
    "shafaq": TIER_2,
    "nrt": TIER_2,
    "rawafid": TIER_2,
    "auroraintel": TIER_2,
    "liveuamap": TIER_2,
    "clashreport": TIER_2,
    "osinttechnical": TIER_2,
    "faytuks": TIER_2,
    "intelcrab": TIER_2,
    "osintdefender": TIER_2,
    "elintnews": TIER_2,
    "defensenews": TIER_2,
    "militarytimes": TIER_2,
    "thehackernews": TIER_2,
    "france24": TIER_2,
    "dw": TIER_2,
    "guardian": TIER_2,
    "npr": TIER_2,
    "bleepingcomputer": TIER_2,
    "cnbc": TIER_2,
    "spacecom": TIER_2,
    "skynews": TIER_2,
    "northpress": TIER_2,
    "ric": TIER_2,
    "pukmedia": TIER_2,
    "bianet": TIER_2,
    "khrn": TIER_2,
    "googlenews": TIER_2,
    # Tier 3: Factional / Unverified
    "sabereen": TIER_3,
    "x_unofficial": TIER_3,
}


_X_STATUS_HANDLE_RE = re.compile(r"^/([A-Za-z0-9_]+)/status/\d+", re.IGNORECASE)
_X_HOSTS = {"x.com", "twitter.com", "mobile.x.com", "mobile.twitter.com", "fxtwitter.com", "vxtwitter.com"}


def extract_x_handle_from_url(url: str) -> str:
    """Extract lowercased X handle from `https://x.com/<handle>/status/<id>` URLs, else empty string."""
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
        host = (parsed.hostname or "").lower().lstrip("www.")
        if host not in _X_HOSTS:
            return ""
        m = _X_STATUS_HANDLE_RE.match(parsed.path or "")
        return m.group(1).lower() if m else ""
    except Exception:
        return ""


def is_x_source_url(url: str) -> bool:
    """Return True if URL belongs to X / Twitter or an X bridge."""
    return bool(extract_x_handle_from_url(url)) or (
        urlparse((url or "").strip()).hostname or ""
    ).lower().lstrip("www.") in _X_HOSTS


def get_x_account_meta(handle_or_source: str, url: str = "") -> Optional[dict[str, object]]:
    """Lookup configured X account metadata by handle or URL."""
    candidates = [extract_x_handle_from_url(url), (handle_or_source or "").strip().lstrip("@")]
    cand_lower = {c.lower() for c in candidates if c}
    for acc in getattr(config, "X_VIDEO_SOURCE_ACCOUNTS", []):
        if str(acc.get("handle", "")).lower() in cand_lower:
            return acc
    return None


def is_official_media_source(source_name: str, url: str = "") -> bool:
    """
    Return True ONLY for recognized official news channels / military commands
    (e.g. CENTCOM, Rudaw, Kurdistan24, BasNews, Shafaq, Reuters, BBC).
    Ordinary/personal/OSINT accounts (e.g. Rawafid, Clash Report, unknown users)
    ALWAYS return False and trigger the strict NO-watermark gate.
    """
    acc = get_x_account_meta(source_name, url=url)
    if acc is not None:
        return bool(acc.get("is_official_media", False))
    handle = extract_x_handle_from_url(url)
    if handle:
        # Unknown X handles discovered via fallback are NEVER official media
        return False
    grp = get_source_group(source_name)
    official_groups = set(getattr(config, "OFFICIAL_MEDIA_SOURCE_GROUPS", set()))
    return grp in official_groups


def get_source_group(source_name: str) -> str:
    """Return the canonical source_group for a given source_name, or the lowered name as default."""
    if not source_name:
        return "unknown"
    key = source_name.strip().lower()
    if key in _SOURCE_GROUP_MAP:
        return _SOURCE_GROUP_MAP[key]
    for prefix, grp in (
        ("sabereen", "sabereen"),
        ("al jazeera", "aljazeera"),
        ("ajanews", "aljazeera"),
        ("الجزيرة", "aljazeera"),
        ("kurdistan24", "kurdistan24"),
        ("kurdistan 24", "kurdistan24"),
        ("rudaw", "rudaw"),
        ("pukmedia", "pukmedia"),
        ("google news", "googlenews"),
        ("nrt", "nrt"),
        ("bianet", "bianet"),
        ("reuters", "reuters"),
        ("bbc", "bbc"),
    ):
        if key.startswith(prefix):
            return grp
    return key


def get_source_tier(source_name: str) -> int:
    """Return the credibility tier (1, 2, or 3) for a source_name."""
    group = get_source_group(source_name)
    return _GROUP_TIER_MAP.get(group, TIER_3)


# ═══════════════════════════════════════════════════
# Corroboration Result
# ═══════════════════════════════════════════════════
@dataclass
class CorroborationResult:
    """Result of cross-source corroboration check for a single article."""
    corroborating_sources: list[str] = field(default_factory=list)
    corroboration_count: int = 0
    highest_corroborating_tier: int = TIER_3
    is_uncorroborated_tier3: bool = False


# ═══════════════════════════════════════════════════
# Title Similarity (lightweight zero-API)
# ═══════════════════════════════════════════════════
_WORD_SPLIT_RE = re.compile(r"[\s\-–—:;,،؛]+")


def _normalized_title_tokens(title: str) -> set[str]:
    """Split a title into normalized lowercase tokens for overlap comparison."""
    if not title:
        return set()
    normed = Article.normalize_title(title) if hasattr(Article, "normalize_title") else title.lower()
    return {t for t in _WORD_SPLIT_RE.split(normed) if len(t) > 2}


def _title_overlap_ratio(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Return Jaccard-like overlap ratio between two token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union) if union else 0.0


def _entity_overlap(ents_a: list[str], ents_b: list[str]) -> int:
    """Count overlapping entity strings (case-insensitive)."""
    set_a = {e.lower().strip() for e in ents_a if e and len(e.strip()) > 1}
    set_b = {e.lower().strip() for e in ents_b if e and len(e.strip()) > 1}
    return len(set_a & set_b)


def _events_match(
    key_a: str,
    ents_a: list[str],
    title_a: str,
    key_b: str,
    ents_b: list[str],
    title_b: str,
) -> bool:
    """Return True if two articles report the same event based on event_key, entities, or title overlap."""
    # 1. Exact event_key match
    if key_a and key_b and key_a == key_b:
        return True

    # 2. Strong entity overlap (>= 2 shared entities)
    if ents_a and ents_b and _entity_overlap(ents_a, ents_b) >= 2:
        return True

    # 3. High title token overlap (>= 50% Jaccard)
    tokens_a = _normalized_title_tokens(title_a)
    tokens_b = _normalized_title_tokens(title_b)
    if tokens_a and tokens_b and _title_overlap_ratio(tokens_a, tokens_b) >= 0.50:
        return True

    return False


# ═══════════════════════════════════════════════════
# Corroboration Engine
# ═══════════════════════════════════════════════════

def compute_corroboration(
    article: Article,
    batch_articles: list[Article],
    recent_history: Optional[list[dict]] = None,
) -> CorroborationResult:
    """
    Compute cross-source corroboration for an article by scanning:
      1. The current discovery batch (before dedup drops duplicates)
      2. Recent 24h published/accepted history from storage

    Returns a CorroborationResult with distinct source_group counts.
    """
    art_group = get_source_group(article.source_name)
    art_tier = get_source_tier(article.source_name)
    art_key = normalize_event_key(getattr(article, "event_key", ""))
    art_ents = normalize_event_entities(getattr(article, "event_entities", []))
    art_title = article.original_title or ""

    seen_groups: set[str] = {art_group}
    best_tier = art_tier

    # Scan current batch
    for other in batch_articles:
        if other is article:
            continue
        if article.id is not None and other.id is not None and article.id == other.id:
            continue
        if article.canonical_url and other.canonical_url and article.canonical_url == other.canonical_url:
            continue

        other_group = get_source_group(other.source_name)
        if other_group in seen_groups:
            continue

        other_key = normalize_event_key(getattr(other, "event_key", ""))
        other_ents = normalize_event_entities(getattr(other, "event_entities", []))
        other_title = other.original_title or ""

        if _events_match(art_key, art_ents, art_title, other_key, other_ents, other_title):
            seen_groups.add(other_group)
            other_tier = get_source_tier(other.source_name)
            best_tier = min(best_tier, other_tier)

    # Scan recent history
    for rec in (recent_history or []):
        rec_source = str(rec.get("source_name", "") or "")
        rec_group = get_source_group(rec_source)
        if rec_group in seen_groups:
            continue

        rec_title = str(rec.get("original_title", "") or rec.get("translated_title", "") or "")
        rec_key, rec_ents = unpack_event_metadata(
            rec.get("named_entities"),
            rec.get("event_key"),
            rec.get("event_entities"),
        )

        if _events_match(art_key, art_ents, art_title, rec_key, rec_ents, rec_title):
            seen_groups.add(rec_group)
            rec_tier = get_source_tier(rec_source)
            best_tier = min(best_tier, rec_tier)

    corroboration_count = len(seen_groups)
    is_uncorroborated_tier3 = (best_tier == TIER_3 and corroboration_count <= 1)

    return CorroborationResult(
        corroborating_sources=sorted(seen_groups),
        corroboration_count=corroboration_count,
        highest_corroborating_tier=best_tier,
        is_uncorroborated_tier3=is_uncorroborated_tier3,
    )


def apply_verification_guard(
    article: Article,
    corroboration: CorroborationResult,
) -> Article:
    """
    Apply the single-source verification guard:
      - Downgrade breaking status (and CRITICAL priority) for uncorroborated Tier-3
        or uncorroborated unofficial X eyewitness claims.
      - Mark article transiently for attribution tagging
        (`_is_uncorroborated_tier3` and `_is_uncorroborated_x_video`).
    Returns the (mutated) article.
    """
    is_unofficial_x = (
        is_x_source_url(getattr(article, "original_url", "") or "")
        and not is_official_media_source(
            getattr(article, "source_name", "") or "",
            getattr(article, "original_url", "") or "",
        )
    )
    is_uncorroborated_x = bool(is_unofficial_x and corroboration.corroboration_count <= 1)

    article._verification_checked = True  # type: ignore[attr-defined]
    article._corroboration_result = corroboration  # type: ignore[attr-defined]
    article._is_uncorroborated_tier3 = bool(corroboration.is_uncorroborated_tier3)  # type: ignore[attr-defined]
    article._is_uncorroborated_x_video = bool(is_uncorroborated_x)  # type: ignore[attr-defined]

    if (corroboration.is_uncorroborated_tier3 or is_uncorroborated_x) and getattr(article, "is_breaking", False):
        logger.info(
            "Downgraded breaking alert: uncorroborated single-source claim from %s "
            "(event_key=%s, corroboration_count=%d, unofficial_x=%s)",
            article.source_name,
            getattr(article, "event_key", ""),
            corroboration.corroboration_count,
            is_uncorroborated_x,
        )
        article.is_breaking = False
        if getattr(article, "priority", "") == "CRITICAL":
            article.priority = "HIGH"

    return article


def select_preferred_source(
    candidates: list[Article],
    batch_articles: Optional[list[Article]] = None,
    recent_history: Optional[list[dict]] = None,
) -> list[Article]:
    """
    When multiple candidates report the same event, prefer the highest-tier source
    (then credibility, then importance_score). Returns deduplicated candidates ordered
    by highest credibility tier first.
    """
    if len(candidates) <= 1:
        return list(candidates)

    result: list[Article] = []

    tier_sorted = sorted(
        candidates,
        key=lambda a: (
            get_source_tier(a.source_name),
            -ArticleExtractor.get_source_credibility(a.source_name),
            -float(getattr(a, "importance_score", 0.0) or 0.0),
        ),
    )

    for art in tier_sorted:
        art_key = normalize_event_key(getattr(art, "event_key", ""))
        art_ents = normalize_event_entities(getattr(art, "event_entities", []))

        is_dup_event = False
        for existing in result:
            ex_key = normalize_event_key(getattr(existing, "event_key", ""))
            ex_ents = normalize_event_entities(getattr(existing, "event_entities", []))
            if _events_match(art_key, art_ents, art.original_title or "", ex_key, ex_ents, existing.original_title or ""):
                is_dup_event = True
                break

        if not is_dup_event:
            result.append(art)

    return result
