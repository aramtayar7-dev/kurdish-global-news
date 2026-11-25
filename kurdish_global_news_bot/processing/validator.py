"""
Kurdish Global News Bot — Article Validator

Validates article quality, completeness, and URL safety.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from urllib.parse import urlparse
from typing import Optional

from storage.models import Article
import config
from config import ENABLE_TELEGRAM, MAX_ARTICLE_AGE_HOURS, MAX_ARTICLE_AGE_MINUTES, TELEGRAM_SOURCE_CHANNELS

logger = logging.getLogger("processing.validator")

_TELEGRAM_POST_PATH_RE = re.compile(r"^/([A-Za-z0-9_]{3,})/(\d+)/?$")
_X_STATUS_PATH_RE = re.compile(r"^/([A-Za-z0-9_]{1,30})/status/(\d+)/?$")

# Domains to reject
BLOCKED_DOMAINS: set[str] = {
    "facebook.com", "twitter.com", "x.com", "instagram.com",
    "tiktok.com", "youtube.com", "reddit.com", "pinterest.com",
    "linkedin.com", "t.me", "telegram.org",
    "bit.ly", "tinyurl.com", "goo.gl",
    "amazon.com", "ebay.com", "aliexpress.com",
}

# Title patterns to reject
SPAM_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?i)click\s*here"),
    re.compile(r"(?i)you\s*won't\s*believe"),
    re.compile(r"(?i)shocking\s*video"),
    re.compile(r"(?i)earn\s*money\s*fast"),
    re.compile(r"(?i)sponsored\s*content"),
    re.compile(r"(?i)advertisement"),
    re.compile(r"(?i)^\s*$"),
    re.compile(r"(?i)subscribe\s*now"),
    re.compile(r"(?i)buy\s*now"),
    re.compile(r"(?i)limited\s*time\s*offer"),
]


class ArticleValidator:
    """Validates articles for quality, safety, and completeness."""

    def __init__(
        self,
        max_age_hours: Optional[float] = None,
        max_age_minutes: Optional[float] = None,
        cold_start: bool = False,
    ) -> None:
        self.max_age_hours = max_age_hours if max_age_hours is not None else MAX_ARTICLE_AGE_HOURS
        self.max_age_minutes = max_age_minutes if max_age_minutes is not None else MAX_ARTICLE_AGE_MINUTES
        self.cold_start = cold_start

    def validate(self, article: Article, is_cold_start: Optional[bool] = None) -> tuple[bool, str]:
        """
        Validate an article.
        Returns (is_valid, rejection_reason).
        """
        # Check required fields
        if not article.original_url:
            return False, "Missing URL"
        if not article.original_title:
            return False, "Missing title"
        if len(article.original_title) < 10:
            return False, f"Title too short: {len(article.original_title)} chars"
        if len(article.original_title) > 500:
            return False, f"Title too long: {len(article.original_title)} chars"

        # Check recency (cold start 30m guard or normal 3h limit)
        cold = self.cold_start if is_cold_start is None else is_cold_start
        if article.publication_date:
            if cold:
                is_recent, recency_reason = self.is_recent(
                    article.publication_date,
                    max_age_minutes=self.max_age_minutes,
                )
                if not is_recent:
                    return False, f"Cold start recency limit: {recency_reason}"
            elif self.max_age_hours is not None:
                eff_max_hours = float(self.max_age_hours)
                parsed_for_age = urlparse(article.original_url)
                dom_for_age = (parsed_for_age.netloc or "").lower().split(":")[0]
                if self._is_configured_x_status_url(dom_for_age, parsed_for_age.path):
                    eff_max_hours = max(eff_max_hours, float(getattr(config, "X_MAX_VIDEO_AGE_HOURS", 6.0)))
                is_recent, recency_reason = self.is_recent(
                    article.publication_date,
                    max_age_hours=eff_max_hours,
                )
                if not is_recent:
                    return False, recency_reason

        # Validate URL
        url_valid, url_reason = self._validate_url(article.original_url)
        if not url_valid:
            return False, f"Invalid URL: {url_reason}"

        # Check for spam/clickbait
        for pattern in SPAM_PATTERNS:
            if pattern.search(article.original_title):
                return False, f"Spam pattern detected in title"

        # Check canonical URL
        if not article.canonical_url:
            return False, "Failed to canonicalize URL"

        # Basic content check — at least title or summary should be meaningful
        total_content = len(article.original_title) + len(article.original_summary)
        if total_content < 20:
            return False, "Insufficient content"

        return True, ""

    def _validate_url(self, url: str) -> tuple[bool, str]:
        """Validate a URL for safety and format."""
        try:
            parsed = urlparse(url)
        except Exception:
            return False, "URL parse error"

        if parsed.scheme not in ("http", "https"):
            return False, f"Invalid scheme: {parsed.scheme}"

        if not parsed.netloc:
            return False, "Missing domain"

        domain = parsed.netloc.lower()
        # Remove port for comparison
        domain_no_port = domain.split(":")[0]

        # Allow canonical public-channel post URLs configured for Telegram or X discovery
        if not self._is_configured_telegram_post_url(domain_no_port, parsed.path) and not self._is_configured_x_status_url(
            domain_no_port, parsed.path
        ):
            for blocked in BLOCKED_DOMAINS:
                if domain_no_port == blocked or domain_no_port.endswith(f".{blocked}"):
                    return False, f"Blocked domain: {blocked}"

        # Reject IPs and localhost
        if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", domain_no_port):
            return False, "IP address not allowed"
        if domain_no_port in ("localhost", "127.0.0.1", "0.0.0.0"):
            return False, "Localhost not allowed"

        # Check for overly long URLs
        if len(url) > 2048:
            return False, "URL too long"

        return True, ""

    @staticmethod
    def _is_configured_telegram_post_url(domain_no_port: str, path: str) -> bool:
        """Allow canonical `t.me/{configured_username}/{post_id}` URLs when Telegram source is enabled."""
        if not ENABLE_TELEGRAM or domain_no_port not in ("t.me", "telegram.me"):
            return False
        m = _TELEGRAM_POST_PATH_RE.match(path or "")
        if not m:
            return False
        username = m.group(1).lower()
        allowed = {
            (ch.get("username") or "").strip().lstrip("@").lower()
            for ch in TELEGRAM_SOURCE_CHANNELS
            if ch.get("username")
        }
        return username in allowed

    @staticmethod
    def _is_configured_x_status_url(domain_no_port: str, path: str) -> bool:
        """Allow canonical `x.com/{configured_handle}/status/{id}` URLs when X video source is enabled."""
        if not getattr(config, "ENABLE_X_VIDEO_SOURCE", True):
            return False
        dom = (domain_no_port or "").lower().removeprefix("www.")
        if dom not in ("x.com", "twitter.com"):
            return False
        m = _X_STATUS_PATH_RE.match(path or "")
        if not m:
            return False
        handle = m.group(1).lower()
        allowed = {
            str(acc.get("handle") or "").strip().lstrip("@").lower()
            for acc in getattr(config, "X_VIDEO_SOURCE_ACCOUNTS", [])
            if acc.get("handle")
        }
        return handle in allowed

    def is_quality_summary(self, summary: str) -> bool:
        """Check if a summary has sufficient quality for publishing."""
        if not summary:
            return False
        if len(summary) < 30:
            return False
        # Check it's not just the title repeated or garbage
        word_count = len(summary.split())
        return word_count >= 5

    @staticmethod
    def is_recent(
        pub_date_str: str,
        max_age_hours: float = 3.0,
        max_age_minutes: Optional[float] = None,
    ) -> tuple[bool, str]:
        """Check if article publication date is within max_age_hours or max_age_minutes."""
        if not pub_date_str or not pub_date_str.strip():
            return True, ""

        try:
            from dateutil import parser as dt_parser
            pub_dt = dt_parser.parse(pub_date_str)
        except Exception:
            try:
                from email.utils import parsedate_to_datetime
                pub_dt = parsedate_to_datetime(pub_date_str)
            except Exception:
                return True, ""

        now = datetime.now(timezone.utc)
        if pub_dt.tzinfo is None:
            pub_dt = pub_dt.replace(tzinfo=timezone.utc)
        else:
            pub_dt = pub_dt.astimezone(timezone.utc)

        age_seconds = (now - pub_dt).total_seconds()
        # Allow small future skew (e.g. up to 1 hour ahead)
        if age_seconds < -3600:
            return True, ""

        if max_age_minutes is not None:
            age_mins = age_seconds / 60.0
            if age_mins > max_age_minutes:
                return False, f"Article too old ({age_mins:.1f}m > {max_age_minutes:.1f}m limit)"
        else:
            age_hours = age_seconds / 3600.0
            if age_hours > max_age_hours:
                return False, f"Article too old ({age_hours:.1f}h > {max_age_hours:.1f}h limit)"

        return True, ""
