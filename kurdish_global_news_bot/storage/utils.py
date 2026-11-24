"""
Kurdish Global News Bot — Shared Storage Utilities

Common URL hashing, UTC timestamp parsing, recent-headline filtering,
and 24h digest prioritization helpers shared between SQLite and Supabase backends.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Optional

from storage.models import Article


def compute_url_hash(url: str) -> str:
    """Compute deterministic SHA-256 hash of a canonicalized URL."""
    canonical = Article.canonicalize_url(url) if url else ""
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_utc_timestamp(ts_str: Optional[str]) -> Optional[datetime]:
    """Parse an ISO/RFC timestamp string into a UTC-aware datetime, or None."""
    if not ts_str:
        return None
    try:
        from dateutil import parser as dateparser
        dt = dateparser.parse(ts_str)
        if not dt:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def filter_recent_headlines(rows: list[dict[str, Any]], cutoff: datetime) -> list[dict[str, Any]]:
    """Filter row dicts to those published/discovered at or after `cutoff`."""
    recent: list[dict[str, Any]] = []
    for row in rows:
        dt = parse_utc_timestamp(row.get("published_at") or row.get("discovered_at"))
        if dt is not None and dt >= cutoff:
            recent.append({
                "id": row["id"],
                "original_title": row.get("original_title") or "",
                "translated_title": row.get("translated_title") or "",
                "category": row.get("category") or "",
                "published_at": row.get("published_at") or "",
            })
    return recent


def prioritize_digest_articles(articles: list[Article], limit: int = 10) -> list[Article]:
    """Sort recent 24h articles by Kurdistan boost + importance score descending."""
    def _sort_key(a: Article) -> tuple[float, int, float, str]:
        is_kurd = 1 if (a.category or "").strip().lower() == "kurdistan" else 0
        boosted = (a.importance_score or 0.0) + (0.20 if is_kurd else 0.0)
        return (boosted, is_kurd, a.importance_score or 0.0, a.published_at or "")

    sorted_articles = sorted(articles, key=_sort_key, reverse=True)
    return sorted_articles[:limit]
