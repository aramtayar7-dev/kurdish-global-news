"""
Kurdish Global News Bot — Free-Tier X (Twitter) Context Enricher

Cross-checks short or ambiguous X video captions against:
1) Local articles already fetched in the current cycle batch & 24h DB history (0 network calls)
2) Free DuckDuckGo web/news search fallback (up to 3 external non-X headlines)

Attaches concise `[Corroborating Context]` to `article.original_summary` for accurate
Sorani translation and returns corroboration records for verification gating.
ZERO Gemini API calls.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import urlparse

from rapidfuzz import fuzz

from core.source_verifier import get_source_group, is_x_source_url
from processing.deduplicator import (
    _extract_event_concepts,
    _tokenize_content,
    normalize_arabic_kurdish_text,
)
from storage.models import Article

logger = logging.getLogger(__name__)

_CORROBORATING_MARKER = "[Corroborating Context]"
_VAGUE_X_CAPTION_RE = re.compile(
    r"\b(?:watch|happening\s+now|just\s+in|breaking|video|footage|look|this)\b|"
    r"(?:ڤیدیۆ|شاهد|فيديو|عاجل|لحظة)",
    re.IGNORECASE,
)
_X_HOST_SUFFIXES = ("x.com", "twitter.com", "xcancel.com", "nitter.poast.org", "nitter.privacyredirect.com")


def _is_x_mirror_url(url: str) -> bool:
    if not url:
        return False
    try:
        host = (urlparse(url).hostname or "").lower().lstrip("www.")
    except Exception:
        return False
    return any(host == d or host.endswith(f".{d}") for d in _X_HOST_SUFFIXES)


def needs_context_enrichment(article: Article) -> bool:
    """Return True when an X/social video article has a short or ambiguous caption."""
    if not is_x_source_url(getattr(article, "original_url", "") or ""):
        return False
    summary = (getattr(article, "original_summary", "") or "").strip()
    if _CORROBORATING_MARKER in summary:
        return False
    title = (getattr(article, "original_title", "") or "").strip()
    combined = summary or title
    if len(combined) < 140:
        return True
    if _VAGUE_X_CAPTION_RE.search(combined) and len(combined) < 220:
        return True
    return False


def _match_text_similarity(cand_text: str, ref_text: str) -> bool:
    """Determine if a local batch/DB headline or DDG snippet discusses the same real event."""
    if not cand_text or not ref_text:
        return False

    cand_toks = _tokenize_content(cand_text)
    ref_toks = _tokenize_content(ref_text)
    shared_toks = cand_toks & ref_toks

    cand_places, cand_actions, cand_entities = _extract_event_concepts(cand_text)
    ref_places, ref_actions, ref_entities = _extract_event_concepts(ref_text)

    shared_places = cand_places & ref_places
    shared_actions = cand_actions & ref_actions
    shared_entities = cand_entities & ref_entities

    concept_hits = len(shared_places) + len(shared_actions) + len(shared_entities)
    if (shared_places or shared_entities) and shared_actions and (concept_hits >= 2 or len(shared_toks) >= 2):
        return True

    if len(shared_toks) >= 3:
        return True

    norm_cand = normalize_arabic_kurdish_text(cand_text)
    norm_ref = normalize_arabic_kurdish_text(ref_text)
    if len(shared_toks) >= 2 and fuzz.token_set_ratio(norm_cand, norm_ref) >= 50:
        return True

    return False


def _build_ddg_query(article: Article) -> str:
    """Build a concise keyword query from an X caption without handles or URLs."""
    raw = (getattr(article, "original_title", "") or getattr(article, "original_summary", "") or "").strip()
    cleaned = re.sub(r"https?://\S+", "", raw)
    cleaned = re.sub(r"@[A-Za-z0-9_]+", "", cleaned)
    cleaned = re.sub(r"[#\[\]\(\)\"'`]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    words = cleaned.split()
    return " ".join(words[:10])


async def _default_ddg_search(query: str, max_results: int = 3) -> list[dict[str, str]]:
    """Run free DuckDuckGo news/text search in a worker thread with timeout protection."""
    if not query:
        return []

    def _run() -> list[dict[str, str]]:
        from duckduckgo_search import DDGS

        out: list[dict[str, str]] = []
        with DDGS() as ddgs:
            try:
                news_hits = list(ddgs.news(keywords=query, region="wt-wt", safesearch="off", max_results=6) or [])
            except Exception:
                news_hits = []
            for item in news_hits:
                url = str(item.get("url") or "").strip()
                title = str(item.get("title") or "").strip()
                source = str(item.get("source") or "").strip() or (urlparse(url).hostname or "Web")
                if url and title:
                    out.append({"title": title, "url": url, "source": source, "snippet": str(item.get("body") or "")})

            if not out:
                try:
                    text_hits = list(ddgs.text(keywords=query, region="wt-wt", safesearch="off", max_results=6) or [])
                except Exception:
                    text_hits = []
                for item in text_hits:
                    url = str(item.get("href") or item.get("url") or "").strip()
                    title = str(item.get("title") or "").strip()
                    source = (urlparse(url).hostname or "Web").lstrip("www.")
                    if url and title:
                        out.append({"title": title, "url": url, "source": source, "snippet": str(item.get("body") or "")})
        return out[: max_results * 2]

    try:
        return await asyncio.wait_for(asyncio.to_thread(_run), timeout=6.0)
    except Exception as exc:
        logger.debug("DuckDuckGo X-context fallback failed cleanly: %s", exc)
        return []


async def enrich_x_article_context(
    article: Article,
    batch_articles: Optional[list[Article]] = None,
    recent_history: Optional[list[dict[str, Any]]] = None,
    ddg_search_fn: Optional[Callable[[str], Awaitable[list[dict[str, Any]]]]] = None,
    max_results: int = 3,
    force: bool = False,
) -> list[dict[str, Any]]:
    """
    Find corroborating headlines for an X video post locally first (batch + 24h DB),
    then optionally via free DuckDuckGo search if local evidence is absent or ambiguous.

    Modifies `article.original_summary` in-place to append `[Corroborating Context]`
    (capped at 1500 chars) and returns synthetic/local corroboration records usable by
    `compute_corroboration(...)` / `apply_verification_guard(...)`.
    """
    if not force and not needs_context_enrichment(article):
        return []

    cand_text = f"{getattr(article, 'original_title', '') or ''} {getattr(article, 'original_summary', '') or ''}".strip()
    if not cand_text:
        return []

    own_group = get_source_group(getattr(article, "source_name", "") or "")
    seen_groups: set[str] = {own_group} if own_group else set()
    corroboration_records: list[dict[str, Any]] = []
    context_lines: list[str] = []

    def _add_match(src_name: str, headline: str, ev_key: str = "", ev_ents: Any = None) -> None:
        nonlocal context_lines, corroboration_records
        if len(context_lines) >= max_results:
            return
        src_clean = (src_name or "Source").strip()
        grp = get_source_group(src_clean)
        if grp and grp in seen_groups:
            return
        if grp:
            seen_groups.add(grp)
        headline_clean = re.sub(r"\s+", " ", (headline or "").strip())
        if not headline_clean:
            return
        context_lines.append(f"- {src_clean}: {headline_clean[:180]}")
        corroboration_records.append(
            {
                "source_name": src_clean,
                "original_title": headline_clean,
                "event_key": ev_key or getattr(article, "event_key", "") or "",
                "event_entities": ev_ents if ev_ents is not None else getattr(article, "event_entities", []),
            }
        )

    # 1. Local batch comparison (ZERO network calls)
    for other in batch_articles or []:
        if other is article:
            continue
        other_url = getattr(other, "original_url", "") or ""
        if other_url and other_url == getattr(article, "original_url", ""):
            continue
        other_text = f"{getattr(other, 'original_title', '') or ''} {(getattr(other, 'original_summary', '') or '')[:200]}"
        if _match_text_similarity(cand_text, other_text):
            _add_match(
                getattr(other, "source_name", "") or "",
                getattr(other, "original_title", "") or other_text,
                ev_key=getattr(other, "event_key", "") or "",
                ev_ents=getattr(other, "event_entities", []),
            )

    # 2. Local 24h DB history comparison (ZERO network calls)
    for hist in recent_history or []:
        if len(context_lines) >= max_results:
            break
        hist_title = (hist.get("original_title") or hist.get("translated_title") or "").strip()
        hist_summary = (hist.get("original_summary") or hist.get("translated_summary") or "")[:200]
        hist_text = f"{hist_title} {hist_summary}".strip()
        if hist_text and _match_text_similarity(cand_text, hist_text):
            _add_match(
                str(hist.get("source_name") or ""),
                hist_title or hist_text,
                ev_key=str(hist.get("event_key") or ""),
                ev_ents=hist.get("event_entities"),
            )

    # 3. Free DuckDuckGo web/news fallback if local corroboration wasn't found
    if not context_lines:
        query = _build_ddg_query(article)
        if query:
            search_fn = ddg_search_fn or _default_ddg_search
            try:
                hits = await search_fn(query)
            except Exception as exc:
                logger.debug("DDG enrichment hook failed cleanly: %s", exc)
                hits = []

            for hit in hits or []:
                if len(context_lines) >= max_results:
                    break
                url = str(hit.get("url") or hit.get("href") or "").strip()
                if _is_x_mirror_url(url) or is_x_source_url(url):
                    continue
                title = str(hit.get("title") or "").strip()
                snippet = str(hit.get("snippet") or hit.get("body") or "").strip()
                src = str(hit.get("source") or "").strip() or (urlparse(url).hostname or "Web").lstrip("www.")
                if title and _match_text_similarity(cand_text, f"{title} {snippet}"):
                    _add_match(src, title)

    if context_lines:
        existing_summary = (getattr(article, "original_summary", "") or "").strip()
        if _CORROBORATING_MARKER not in existing_summary:
            block = f"{_CORROBORATING_MARKER}\n" + "\n".join(context_lines[:max_results])
            merged = f"{existing_summary}\n\n{block}".strip() if existing_summary else block
            article.original_summary = merged[:1500]
            logger.info(
                "X context enriched for '%s' (%s): %d corroborating source(s)",
                (getattr(article, "original_title", "") or "")[:50],
                getattr(article, "source_name", "X"),
                len(context_lines),
            )

    return corroboration_records
