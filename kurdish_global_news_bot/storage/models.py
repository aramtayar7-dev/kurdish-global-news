"""
Kurdish Global News Bot — Data Models

Dataclass-based models for articles, sources, events, and system logs.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode


@dataclass
class Article:
    """Represents a news article through its lifecycle."""
    id: Optional[int] = None
    canonical_url: str = ""
    original_url: str = ""
    original_title: str = ""
    normalized_title: str = ""
    translated_title: str = ""
    original_summary: str = ""
    translated_summary: str = ""
    source_name: str = ""
    category: str = ""
    priority: str = "NORMAL"
    importance_score: float = 0.0
    publication_date: Optional[str] = None
    discovered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    processed_at: Optional[str] = None
    published_at: Optional[str] = None
    status: str = "DISCOVERED"
    content_hash: str = ""
    event_fingerprint: str = ""
    retry_count: int = 0
    error_message: str = ""
    telegram_message_id: Optional[int] = None
    thread_root_message_id: Optional[int] = None
    image_url: str = ""
    video_url: Optional[str] = None
    country: str = ""
    region: str = ""
    named_entities: str = ""  # JSON string
    event_key: str = ""
    event_entities: list[str] = field(default_factory=list)
    is_breaking: bool = False
    is_thread_update: bool = False

    def compute_content_hash(self) -> str:
        """Compute a hash of the article's core content for deduplication."""
        if not self.normalized_title and self.original_title:
            self.normalized_title = self.normalize_title(self.original_title)
        content = f"{self.normalized_title}|{self.canonical_url}"
        self.content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]
        return self.content_hash

    def compute_event_fingerprint(self) -> str:
        """Compute a fingerprint for event-level deduplication."""
        if not self.normalized_title and self.original_title:
            self.normalized_title = self.normalize_title(self.original_title)
        words = self.normalized_title.split()
        significant = [w for w in words if len(w) > 3][:8]
        if not significant:
            significant = words[:8]
        fp_text = " ".join(sorted(significant))
        self.event_fingerprint = hashlib.sha256(fp_text.encode("utf-8")).hexdigest()[:24]
        return self.event_fingerprint

    @staticmethod
    def normalize_title(title: str) -> str:
        """Normalize a title for comparison: lowercase, strip punctuation, collapse whitespace."""
        if not title:
            return ""
        t = title.lower().strip()
        t = re.sub(r"[^\w\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @staticmethod
    def canonicalize_url(url: str) -> str:
        """Normalize a URL by removing tracking parameters, fragments, and trailing slashes."""
        if not url:
            return ""
        try:
            parsed = urlparse(url.strip())
            # Remove fragments
            parsed = parsed._replace(fragment="")
            # Remove tracking parameters (exact match or known prefixes like utm_* / mc_*)
            tracking_params = {
                "ref", "ref_src", "ref_url", "fbclid", "gclid", "yclid", "msclkid",
                "dclid", "twclid", "igshid", "si", "source", "mkt_tok",
                "s_cid", "soc_trk", "feature", "share", "sprefix", "amp",
            }
            tracking_prefixes = ("utm_", "mc_")
            if parsed.query:
                params = parse_qs(parsed.query, keep_blank_values=True)
                filtered = {
                    k: v
                    for k, v in params.items()
                    if k.lower() not in tracking_params
                    and not k.lower().startswith(tracking_prefixes)
                }
                new_query = urlencode(sorted(filtered.items()), doseq=True)
                parsed = parsed._replace(query=new_query)
            # Normalize scheme
            scheme = parsed.scheme.lower() or "https"
            # Normalize hostname (strip trailing dot, www., and mobile prefixes like m./mobile.)
            netloc = parsed.netloc.lower().rstrip(".")
            for sub_prefix in ("www.", "mobile.", "m."):
                if netloc.startswith(sub_prefix) and len(netloc) > len(sub_prefix) + 3:
                    netloc = netloc[len(sub_prefix):]
            # Normalize path
            path = parsed.path.rstrip("/") or "/"
            return urlunparse((scheme, netloc, path, parsed.params, parsed.query, ""))
        except Exception:
            return url.strip()


@dataclass
class Source:
    """Represents a news source/feed."""
    id: Optional[int] = None
    name: str = ""
    source_type: str = ""  # "rss", "ddg", "gdelt", "official"
    url: str = ""
    category: str = ""
    enabled: bool = True
    last_checked_at: Optional[str] = None
    last_success_at: Optional[str] = None
    last_error: str = ""
    consecutive_failures: int = 0


@dataclass
class Event:
    """Represents a news event (groups related articles)."""
    id: Optional[int] = None
    event_fingerprint: str = ""
    event_title: str = ""
    first_seen_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    article_count: int = 1


@dataclass
class SystemLog:
    """A system log entry."""
    id: Optional[int] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    level: str = "INFO"
    component: str = ""
    message: str = ""
