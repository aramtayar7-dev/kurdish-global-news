"""
Kurdish Global News Bot — Native Telegram Inline Source URL Button Builder

Builds validated single-row InlineKeyboardMarkup source buttons for published
posts while protecting subscriber retention on external `t.me/` channels and
guarding against malformed/unsafe URLs. Zero Gemini API calls.
"""

from __future__ import annotations

import ipaddress
import re
from typing import Optional
from urllib.parse import urlparse

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

import config
from storage.models import Article

DEFAULT_BUTTON_LABEL = "🔗 بابەتی ئەسڵی"
SOURCE_BUTTON_PREFIX = "🔗 سەرچاوە — "

_TELEGRAM_DOMAINS = {"t.me", "telegram.me", "telegram.dog"}
_X_DOMAINS = {
    "x.com",
    "twitter.com",
    "mobile.x.com",
    "mobile.twitter.com",
    "fxtwitter.com",
    "vxtwitter.com",
}
_SUFFIX_STRIP_RE = re.compile(
    r"\s+(?:RSS|Feed|Sorani|Arabic|English|EN|Digital|TV)\s*$",
    re.IGNORECASE,
)


def _matches_domain(host: str, domains: set[str]) -> bool:
    return host in domains or any(host.endswith(f".{dom}") for dom in domains)


def is_valid_button_url(
    url: Optional[str],
    allow_telegram_links: bool = False,
    allow_x_links: bool = False,
) -> bool:
    """
    Validate that `url` is a safe, public HTTP(S) URL suitable for a Telegram URL button.
    Rejects empty strings, whitespace/control chars, unsafe schemes (javascript:, mailto:, tg:),
    localhost/private IPs, overly long URLs (>2048 chars), external Telegram links unless
    explicitly permitted, and X/Twitter links unless `ENABLE_X_SOURCE_LINK_BUTTON` is True.
    """
    if not url or not isinstance(url, str):
        return False

    cleaned = url.strip()
    if not cleaned or len(cleaned) > 2048:
        return False
    if any(ch.isspace() or ord(ch) < 32 for ch in cleaned):
        return False

    try:
        parsed = urlparse(cleaned)
    except Exception:
        return False

    if parsed.scheme.lower() not in ("http", "https"):
        return False

    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host:
        return False
    if host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"} or host.endswith(".local"):
        return False

    # Reject private/loopback/reserved IP literals
    try:
        ip_obj = ipaddress.ip_address(host)
        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_reserved or ip_obj.is_unspecified:
            return False
    except ValueError:
        # Domain name: require at least one dot (e.g., example.com)
        if "." not in host:
            return False

    if not allow_telegram_links and _matches_domain(host, _TELEGRAM_DOMAINS):
        return False

    if not allow_x_links and _matches_domain(host, _X_DOMAINS):
        return False

    return True


def format_source_button_label(
    source_name: Optional[str],
    max_len: Optional[int] = None,
) -> str:
    """
    Format compact Sorani button text (`🔗 سەرچاوە — <Source>`), falling back to
    `🔗 بابەتی ئەسڵی` if source_name is missing or exceeds `max_len`.
    """
    cap = (
        int(getattr(config, "INLINE_SOURCE_BUTTON_MAX_LABEL_LEN", 36))
        if max_len is None
        else max(16, int(max_len))
    )
    raw = re.sub(r"\s+", " ", (source_name or "").strip())
    if not raw:
        return DEFAULT_BUTTON_LABEL

    # Clean noisy feed/channel suffixes iteratively (e.g. "Kurdistan24 Sorani" -> "Kurdistan24")
    cleaned = raw
    for _ in range(2):
        next_val = _SUFFIX_STRIP_RE.sub("", cleaned).strip(" -–—|")
        if not next_val or next_val == cleaned:
            break
        cleaned = next_val

    candidate = f"{SOURCE_BUTTON_PREFIX}{cleaned}"
    if len(candidate) <= cap:
        return candidate
    return DEFAULT_BUTTON_LABEL


def resolve_article_button_url(
    article: Optional[Article] = None,
    url: Optional[str] = None,
    allow_telegram_links: bool = False,
    allow_x_links: bool = False,
) -> Optional[str]:
    """Return the first eligible source URL from explicit `url` or Article fields, else None."""
    candidates: list[Optional[str]] = [url]
    if article is not None:
        candidates.extend([
            getattr(article, "original_url", None),
            getattr(article, "canonical_url", None),
            getattr(article, "source_url", None),
            getattr(article, "url", None),
        ])

    for cand in candidates:
        if is_valid_button_url(
            cand,
            allow_telegram_links=allow_telegram_links,
            allow_x_links=allow_x_links,
        ):
            return str(cand).strip()
    return None


def build_source_button_markup(
    url: Optional[str] = None,
    source_name: Optional[str] = None,
    *,
    article: Optional[Article] = None,
    enabled: Optional[bool] = None,
    allow_telegram_links: Optional[bool] = None,
    allow_x_links: Optional[bool] = None,
    max_label_len: Optional[int] = None,
) -> Optional[InlineKeyboardMarkup]:
    """
    Build a single-row Telegram `InlineKeyboardMarkup` linking to the article's source URL,
    or return `None` when disabled, missing/invalid, or blocked (`t.me/` / X retention protection).
    """
    is_enabled = (
        bool(getattr(config, "ENABLE_INLINE_SOURCE_BUTTON", True))
        if enabled is None
        else bool(enabled)
    )
    if not is_enabled:
        return None

    allow_tg = (
        bool(getattr(config, "ENABLE_TELEGRAM_SOURCE_LINK_BUTTON", False))
        if allow_telegram_links is None
        else bool(allow_telegram_links)
    )
    allow_x = (
        bool(getattr(config, "ENABLE_X_SOURCE_LINK_BUTTON", False))
        if allow_x_links is None
        else bool(allow_x_links)
    )
    resolved_url = resolve_article_button_url(
        article=article,
        url=url,
        allow_telegram_links=allow_tg,
        allow_x_links=allow_x,
    )
    if not resolved_url:
        return None

    resolved_source = source_name if source_name is not None else (
        getattr(article, "source_name", "") if article is not None else ""
    )
    label = format_source_button_label(resolved_source, max_len=max_label_len)
    return InlineKeyboardMarkup([[InlineKeyboardButton(text=label, url=resolved_url)]])
