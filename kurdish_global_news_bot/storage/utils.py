"""
Kurdish Global News Bot — Shared Storage Utilities

Common URL hashing, UTC timestamp parsing, recent-headline filtering,
and 24h digest prioritization helpers shared between SQLite and Supabase backends.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Optional

from storage.models import Article


def normalize_event_key(raw_key: Any) -> str:
    """Normalize a canonical English snake_case event key safely."""
    text = str(raw_key or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:120]


def normalize_event_entities(raw_entities: Any) -> list[str]:
    """Normalize 1-N lowercase English entity/topic tokens safely from list or JSON string."""
    if not raw_entities:
        return []
    items: list[Any]
    if isinstance(raw_entities, str):
        stripped = raw_entities.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                items = parsed if isinstance(parsed, list) else stripped.split(",")
            except Exception:
                items = stripped.split(",")
        else:
            items = re.split(r"[,;|]+", stripped)
    elif isinstance(raw_entities, (list, tuple, set)):
        items = list(raw_entities)
    else:
        return []

    clean: list[str] = []
    seen: set[str] = set()
    for item in items:
        tok = re.sub(r"[^a-z0-9_\- ]+", "", str(item or "").lower()).strip()
        if not tok:
            continue
        for sub in tok.replace("_", " ").split():
            if len(sub) >= 2 and sub not in seen:
                seen.add(sub)
                clean.append(sub)
    return clean[:12]


def pack_event_metadata(
    event_key: str = "",
    event_entities: Optional[list[str]] = None,
) -> str:
    """Serialize `(event_key, event_entities)` cleanly into JSON for `named_entities` persistence."""
    key = normalize_event_key(event_key)
    ents = normalize_event_entities(event_entities or [])
    if not key and not ents:
        return ""
    return json.dumps({"event_key": key, "event_entities": ents}, ensure_ascii=True)


def unpack_event_metadata(
    named_entities: Any = "",
    event_key: Any = "",
    event_entities: Any = None,
) -> tuple[str, list[str]]:
    """
    Extract `(event_key, event_entities)` from explicit columns or JSON-encoded `named_entities`,
    tolerating missing columns in older Supabase/SQLite schemas.
    """
    resolved_key = normalize_event_key(event_key)
    resolved_ents = normalize_event_entities(event_entities)

    if isinstance(named_entities, str) and named_entities.strip().startswith("{"):
        try:
            payload = json.loads(named_entities)
            if isinstance(payload, dict):
                if not resolved_key and payload.get("event_key"):
                    resolved_key = normalize_event_key(payload.get("event_key"))
                if not resolved_ents and payload.get("event_entities"):
                    resolved_ents = normalize_event_entities(payload.get("event_entities"))
        except Exception:
            pass
    elif isinstance(named_entities, dict):
        if not resolved_key and named_entities.get("event_key"):
            resolved_key = normalize_event_key(named_entities.get("event_key"))
        if not resolved_ents and named_entities.get("event_entities"):
            resolved_ents = normalize_event_entities(named_entities.get("event_entities"))

    return resolved_key, resolved_ents


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
            ev_key, ev_entities = unpack_event_metadata(
                row.get("named_entities"),
                row.get("event_key"),
                row.get("event_entities"),
            )
            raw_msg_id = row.get("telegram_message_id")
            raw_root_id = row.get("thread_root_message_id")
            recent.append({
                "id": row["id"],
                "original_title": row.get("original_title") or "",
                "translated_title": row.get("translated_title") or "",
                "translated_summary": row.get("translated_summary") or "",
                "original_summary": row.get("original_summary") or "",
                "category": row.get("category") or "",
                "published_at": row.get("published_at") or "",
                "telegram_message_id": int(raw_msg_id) if raw_msg_id is not None else None,
                "thread_root_message_id": int(raw_root_id) if raw_root_id is not None else None,
                "event_key": ev_key,
                "event_entities": ev_entities,
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
