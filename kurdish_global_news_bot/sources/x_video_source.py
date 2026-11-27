"""
Kurdish Global News Bot — Free-Tier X (Twitter) Video Discovery Source

Zero-cost X video discovery using:
1. Rotating public Nitter-compatible RSS mirrors (with failover on 429/403/5xx/timeout)
2. Free DuckDuckGo `site:x.com` fallback when all RSS mirrors fail in a cycle
3. Keyless `api.fxtwitter.com` / `api.vxtwitter.com` metadata resolution for direct MP4 URLs

Applies strict pre-download filters: native video only (no GIFs), duration bounds,
minimum width >= 480px, freshness <= 6h, retweet skip, fake/recycled/game clip rejection,
and `X_MAX_VIDEOS_PER_CYCLE` cap. ZERO Gemini API calls and ZERO paid X API keys.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import email.utils
import json
import logging
import random
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from urllib.parse import quote_plus, urlparse

import aiohttp
import feedparser
from bs4 import BeautifulSoup

import config
from processing.video_watermark_detector import has_third_party_repost_marker
from sources.base import BaseSource, RawArticle

logger = logging.getLogger(__name__)

_STATUS_URL_RE = re.compile(
    r"https?://(?:(?:www|mobile)\.)?(?:x\.com|twitter\.com|fxtwitter\.com|vxtwitter\.com|fixupx\.com|twittpr\.com|[^/]+)/"
    r"([A-Za-z0-9_]{1,30})/status/(\d+)",
    re.IGNORECASE,
)
_URL_STRIP_RE = re.compile(r"https?://\S+")
_RETWEET_PREFIX_RE = re.compile(r"^\s*(?:RT\s+by\b|RT\s+@|Reposted\s+by\b)", re.IGNORECASE)
_FAKE_OR_RECYCLED_RE = re.compile(
    r"\b(?:arma\s*3|dcs\s*world|war\s*thunder|gameplay|simulation|flight\s*sim|"
    r"throwback|archive\s+footage|old\s+video|from\s+(?:201\d|202[0-5])|"
    r"tiktok|ai[- ]?generated|deepfake)\b",
    re.IGNORECASE,
)

CHROME_DESKTOP_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
SYNDICATION_HEADERS: dict[str, str] = {
    "User-Agent": CHROME_DESKTOP_USER_AGENT,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}
GOOGLE_NEWS_RSS_TEMPLATE = (
    "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
)
_PRIORITY_REGIONS: frozenset[str] = frozenset({"kurdistan_bashur", "middle_east"})


def normalize_canonical_x_url(raw_url: str) -> Optional[str]:
    """
    Normalize x.com / twitter.com / fxtwitter.com / vxtwitter.com / mirror status URLs
    into canonical `https://x.com/{handle}/status/{status_id}`.
    """
    if not raw_url or not isinstance(raw_url, str):
        return None
    m = _STATUS_URL_RE.search(raw_url.strip())
    if not m:
        return None
    handle, status_id = m.group(1), m.group(2)
    return f"https://x.com/{handle}/status/{status_id}"


def _parse_retry_after_seconds(raw_header: Optional[str]) -> Optional[float]:
    """Parse HTTP Retry-After header (integer seconds or HTTP-date) safely."""
    if not raw_header or not isinstance(raw_header, str):
        return None
    val = raw_header.strip()
    if not val:
        return None
    try:
        secs = float(val)
        return max(0.0, secs)
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(val)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = (dt.astimezone(timezone.utc) - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, delta)
    except Exception:
        return None


def _format_mirror_rss_url(mirror: str, handle: str) -> str:
    """Support `{}` or `{handle}` templates as well as plain base mirror URLs."""
    clean = mirror.strip().rstrip("/")
    if "{handle}" in clean:
        return clean.format(handle=handle)
    if "{}" in clean:
        return clean.format(handle)
    return f"{clean}/{handle}/rss"


def _strip_html_and_urls(raw_html: str) -> str:
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "lxml")
    text = soup.get_text(" ", strip=True)
    text = _URL_STRIP_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_fake_or_recycled_clip(text: str) -> bool:
    if not text:
        return False
    if _FAKE_OR_RECYCLED_RE.search(text):
        return True
    return False


def _parse_datetime(raw_date: Any, struct_time: Any = None) -> Optional[datetime]:
    if struct_time is not None:
        try:
            import calendar

            ts = calendar.timegm(struct_time)
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            pass
    if isinstance(raw_date, (int, float)):
        try:
            return datetime.fromtimestamp(float(raw_date), tz=timezone.utc)
        except Exception:
            return None
    if isinstance(raw_date, str) and raw_date.strip():
        try:
            dt = email.utils.parsedate_to_datetime(raw_date.strip())
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass
        try:
            iso = raw_date.strip().replace("Z", "+00:00")
            dt = datetime.fromisoformat(iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None
    return None


def _build_ddg_discovery_queries(accounts: list[dict[str, Any]]) -> list[str]:
    handles = [str(a.get("handle") or "").strip() for a in accounts if a.get("handle")]
    queries: list[str] = []
    for i in range(0, len(handles), 4):
        chunk = handles[i : i + 4]
        site_terms = " OR ".join(f"site:x.com/{h}/status" for h in chunk)
        queries.append(f"({site_terms})")
    return queries


class XVideoSource(BaseSource):
    """
    Free-tier curated X/Twitter discovery adapter with:
      - Lock-protected circular account rotation preserving Middle East / Kurdistan priority
        (in-memory cursor starts at index 0 on process restart so priority accounts run first)
      - Sequential Tier 1 syndication fetching with random jitter and HTTP 429 circuit breaker
      - Tier 2 RSS bridges (RSSHub / Nitter mirrors) bounded per account
      - Tier 3 Google News RSS + DDG search fallback bounded per cycle + fx/vx media verification
    """

    def __init__(
        self,
        accounts: Optional[list[dict[str, Any]]] = None,
        mirrors: Optional[list[str]] = None,
        request_timeout: Optional[int] = None,
        max_age_hours: Optional[float] = None,
        max_videos_per_cycle: Optional[int] = None,
        batch_size: Optional[int] = None,
        jitter_min: Optional[float] = None,
        jitter_max: Optional[float] = None,
        cooldown_seconds: Optional[float] = None,
        max_rss_items_per_account: Optional[int] = None,
        max_fallback_results_per_cycle: Optional[int] = None,
        max_requests_per_cycle: Optional[int] = None,
        sleep_fn: Optional[Callable[[float], Awaitable[None]]] = None,
        rand_uniform_fn: Optional[Callable[[float, float], float]] = None,
        monotonic_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        super().__init__(
            name="x_video",
            source_type="x_video",
            enabled=bool(getattr(config, "ENABLE_X_VIDEO_SOURCE", True)),
        )
        raw_accounts = list(
            accounts if accounts is not None else getattr(config, "X_VIDEO_SOURCE_ACCOUNTS", [])
        )
        # Preserve stable config ordering while ensuring Middle East / Kurdistan priority accounts come first
        self._accounts: list[dict[str, Any]] = self._prioritize_accounts(raw_accounts)
        self._mirrors: list[str] = [
            m.strip().rstrip("/")
            for m in (mirrors if mirrors is not None else getattr(config, "X_NITTER_RSS_MIRRORS", []))
            if m and m.strip()
        ]
        cfg_timeout = int(getattr(config, "X_REQUEST_TIMEOUT_S", 15))
        self._request_timeout: int = max(3, int(request_timeout if request_timeout is not None else cfg_timeout))
        self._max_age_hours: float = float(
            max_age_hours if max_age_hours is not None else getattr(config, "X_MAX_VIDEO_AGE_HOURS", 6.0)
        )
        if max_videos_per_cycle is not None:
            default_cap = int(max_videos_per_cycle)
        elif getattr(config, "ENABLE_BATCH_PACING_ENGINE", True):
            default_cap = int(getattr(config, "X_MAX_CANDIDATE_VIDEOS_PER_CYCLE", 6))
        else:
            default_cap = int(getattr(config, "X_MAX_VIDEOS_PER_CYCLE", 4))
        self._max_videos_per_cycle: int = max(1, default_cap)

        self._batch_size: int = max(
            1,
            int(batch_size if batch_size is not None else getattr(config, "X_BATCH_SIZE", 5)),
        )
        self._jitter_min: float = max(
            0.0,
            float(jitter_min if jitter_min is not None else getattr(config, "X_JITTER_MIN", 1.5)),
        )
        raw_jmax = float(jitter_max if jitter_max is not None else getattr(config, "X_JITTER_MAX", 2.5))
        self._jitter_max: float = max(self._jitter_min, raw_jmax)
        self._syndication_cooldown_s: float = max(
            1.0,
            float(cooldown_seconds if cooldown_seconds is not None else getattr(config, "X_SYNDICATION_COOLDOWN_S", 600)),
        )
        self._max_rss_items_per_account: int = max(
            1,
            int(max_rss_items_per_account if max_rss_items_per_account is not None else getattr(config, "X_MAX_RSS_ITEMS_PER_ACCOUNT", 5)),
        )
        self._max_fallback_results_per_cycle: int = max(
            1,
            int(max_fallback_results_per_cycle if max_fallback_results_per_cycle is not None else getattr(config, "X_MAX_FALLBACK_RESULTS_PER_CYCLE", 20)),
        )
        default_budget = max(self._batch_size * max(2, len(self._mirrors) + 3), self._max_fallback_results_per_cycle)
        self._max_requests_per_cycle: int = max(
            1,
            int(max_requests_per_cycle if max_requests_per_cycle is not None else default_budget),
        )

        # Injectable timing/random hooks for deterministic mocked tests
        self._sleep_fn = sleep_fn
        self._rand_uniform_fn = rand_uniform_fn
        self._monotonic_fn = monotonic_fn

        # In-memory cursor (starts at 0 on process restart) & 429 breaker protected by lightweight locks
        self._rotation_lock = threading.Lock()
        self._rotation_cursor: int = 0
        self._breaker_lock = threading.Lock()
        self._syndication_degraded: bool = False
        self._syndication_cooldown_until: float = 0.0
        self._last_http_status: Optional[int] = None
        self._last_http_headers: dict[str, str] = {}
        self._cycle_requests_used: int = 0

        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(4)
        self._account_by_handle: dict[str, dict[str, Any]] = {
            str(acc.get("handle") or "").strip().lower(): acc
            for acc in self._accounts
            if acc.get("handle")
        }

    @staticmethod
    def _is_priority_account(account_cfg: dict[str, Any]) -> bool:
        region = str(account_cfg.get("region") or "").strip().lower()
        return region in _PRIORITY_REGIONS or int(account_cfg.get("credibility_tier", 3) or 3) == 1

    @classmethod
    def _prioritize_accounts(cls, accounts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Stable ordering putting Middle East / Kurdistan & tier-1 accounts ahead of global accounts."""
        return sorted(
            accounts,
            key=lambda a: (
                0 if cls._is_priority_account(a) else 1,
                0 if bool(a.get("is_official_media", False)) else 1,
            ),
        )

    def _now_monotonic(self) -> float:
        return self._monotonic_fn() if self._monotonic_fn is not None else time.monotonic()

    async def _sleep(self, seconds: float) -> None:
        if seconds <= 0:
            return
        fn = self._sleep_fn if self._sleep_fn is not None else asyncio.sleep
        await fn(seconds)

    def _next_jitter(self) -> float:
        fn = self._rand_uniform_fn if self._rand_uniform_fn is not None else random.uniform
        return max(0.0, float(fn(self._jitter_min, self._jitter_max)))

    def select_account_batch(self, batch_size: Optional[int] = None) -> list[dict[str, Any]]:
        """
        Deterministically select the next circular subset of accounts under a lock.
        Avoids repeating subsets across consecutive cycles when total accounts exceed batch size,
        wraps cleanly without duplicates, and preserves priority ordering within mixed batches.
        Resets cleanly to index 0 on process restart.
        """
        if not self._accounts:
            return []
        size = max(1, int(batch_size if batch_size is not None else self._batch_size))
        total = len(self._accounts)

        with self._rotation_lock:
            start = self._rotation_cursor % total
            if size >= total:
                selected = [self._accounts[(start + i) % total] for i in range(total)]
                self._rotation_cursor = (start + total) % total
                return self._prioritize_accounts(selected)

            selected = [self._accounts[(start + i) % total] for i in range(size)]
            self._rotation_cursor = (start + size) % total

        logger.info(
            "X account batch selected (%d/%d): %s",
            len(selected),
            total,
            ", ".join(str(a.get("handle") or "") for a in selected),
        )
        return self._prioritize_accounts(selected)

    def is_syndication_cooldown_active(self, now: Optional[float] = None) -> bool:
        """Return True while syndication circuit breaker cooldown is active."""
        cur = now if now is not None else self._now_monotonic()
        with self._breaker_lock:
            if not self._syndication_degraded:
                return False
            if cur >= self._syndication_cooldown_until:
                self._syndication_degraded = False
                self._syndication_cooldown_until = 0.0
                logger.info("X syndication circuit breaker cooldown expired; resuming syndication tier.")
                return False
            return True

    def trip_syndication_breaker(
        self,
        *,
        retry_after_seconds: Optional[float] = None,
        now: Optional[float] = None,
    ) -> float:
        """
        Activate HTTP 429 circuit breaker for syndication requests.
        Never resumes earlier than `Retry-After` or `X_SYNDICATION_COOLDOWN_S`.
        Returns the monotonic timestamp when cooldown expires.
        """
        cur = now if now is not None else self._now_monotonic()
        cooldown = float(self._syndication_cooldown_s)
        if retry_after_seconds is not None and retry_after_seconds > 0:
            cooldown = max(cooldown, float(retry_after_seconds))

        target_until = cur + cooldown
        with self._breaker_lock:
            self._syndication_degraded = True
            if target_until > self._syndication_cooldown_until:
                self._syndication_cooldown_until = target_until
            active_until = self._syndication_cooldown_until

        logger.warning(
            "X syndication circuit breaker TRIPPED after HTTP 429; cooling down for %.1fs.",
            max(0.0, active_until - cur),
        )
        return active_until

    def _consume_budget(self) -> bool:
        """Consume one unit of per-cycle HTTP budget; returns False when budget is exhausted."""
        if self._cycle_requests_used >= self._max_requests_per_cycle:
            return False
        self._cycle_requests_used += 1
        return True

    @property
    def source_name(self) -> str:
        return "X (Twitter) Video"

    async def initialize(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._request_timeout),
                headers={
                    "User-Agent": CHROME_DESKTOP_USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )

    async def cleanup(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _http_get_text(
        self,
        url: str,
        headers: Optional[dict[str, str]] = None,
    ) -> Optional[str]:
        self._last_http_status = None
        self._last_http_headers = {}
        await self.initialize()
        assert self._session is not None
        timeout = aiohttp.ClientTimeout(total=self._request_timeout)
        try:
            async with self._session.get(
                url,
                headers=headers,
                allow_redirects=True,
                timeout=timeout,
            ) as resp:
                self._last_http_status = int(resp.status)
                raw_hdrs = getattr(resp, "headers", None)
                if raw_hdrs is not None:
                    self._last_http_headers = {str(k): str(v) for k, v in raw_hdrs.items()}
                if resp.status in (403, 404, 429) or resp.status >= 500:
                    logger.debug("X mirror HTTP %s for %s", resp.status, url)
                    return None
                if resp.status != 200:
                    return None
                return await resp.text(errors="replace")
        except Exception as exc:
            logger.debug("X mirror fetch error (%s): %s", url, exc)
            return None

    async def _http_get_with_status(
        self,
        url: str,
        headers: Optional[dict[str, str]] = None,
    ) -> tuple[int, Optional[str], dict[str, str]]:
        """
        Perform HTTP GET returning `(status, body, headers)` while seamlessly honoring
        instance/class `_http_get_text` monkeypatches in existing unit tests.
        """
        has_custom_get_text = (
            "_http_get_text" in self.__dict__
            or type(self)._http_get_text is not XVideoSource._http_get_text
        )
        if has_custom_get_text:
            body = await self._http_get_text(url)
            status = self._last_http_status if self._last_http_status is not None else (200 if body is not None else 0)
            return status, body, dict(self._last_http_headers)

        body = await self._http_get_text(url, headers=headers)
        status = self._last_http_status if self._last_http_status is not None else 0
        return status, body, dict(self._last_http_headers)

    async def _http_get_json(self, url: str) -> Optional[dict[str, Any]]:
        await self.initialize()
        assert self._session is not None
        try:
            async with self._session.get(url, allow_redirects=True) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json(content_type=None)
                return data if isinstance(data, dict) else None
        except Exception as exc:
            logger.debug("X metadata JSON error (%s): %s", url, exc)
            return None

    async def fetch_channel_rss(self, account_cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
        """
        Try configured RSS bridges (RSSHub / Nitter mirrors) sequentially until one returns
        valid RSS/Atom XML with verifiable items. Stops retrying on budget exhaustion.
        """
        handle = str(account_cfg.get("handle") or "").strip()
        if not handle:
            return [], False

        for mirror in self._mirrors:
            if not self._consume_budget():
                logger.info("X discovery request budget exhausted during RSS fallback.")
                return [], False
            rss_url = _format_mirror_rss_url(mirror, handle)
            status, body, _ = await self._http_get_with_status(rss_url)
            if status == 429 or not body:
                continue
            low = body.lstrip()[:250].lower()
            if "<rss" not in low and "<feed" not in low and "<?xml" not in low:
                continue

            parsed = feedparser.parse(body)
            entries = list(getattr(parsed, "entries", []) or [])
            results: list[dict[str, Any]] = []
            for entry in entries:
                if len(results) >= self._max_rss_items_per_account:
                    break
                link = str(entry.get("link") or entry.get("id") or "").strip()
                m = _STATUS_URL_RE.search(link)
                if not m:
                    continue
                url_handle, status_id = m.group(1), m.group(2)
                raw_title = str(entry.get("title") or "")
                raw_desc = str(entry.get("summary") or entry.get("description") or "")
                creator = str(entry.get("author") or entry.get("dc_creator") or "").lstrip("@").strip()
                if _RETWEET_PREFIX_RE.search(raw_title) or (
                    creator and creator.lower() != handle.lower()
                ):
                    continue
                if url_handle.lower() != handle.lower():
                    continue
                pub_dt = _parse_datetime(
                    entry.get("published") or entry.get("updated"),
                    entry.get("published_parsed") or entry.get("updated_parsed"),
                )
                if pub_dt is None:
                    continue
                caption = _strip_html_and_urls(raw_desc) or _strip_html_and_urls(raw_title)
                canonical = normalize_canonical_x_url(link) or f"https://x.com/{handle}/status/{status_id}"
                results.append(
                    {
                        "handle": handle,
                        "status_id": status_id,
                        "canonical_url": canonical,
                        "caption": caption,
                        "published_at": pub_dt,
                        "account": account_cfg,
                        "discovery_tier": "rss",
                    }
                )
            if results:
                logger.info("X RSS fallback mirror=%s returned %d item(s) for @%s", mirror, len(results), handle)
            return results, True

        return [], False

    @staticmethod
    def _parse_syndication_html(html_content: str, account_cfg: dict[str, Any]) -> list[dict[str, Any]]:
        """Parse Twitter syndication `__NEXT_DATA__` timeline JSON into candidate dicts."""
        handle = str(account_cfg.get("handle") or "").strip()
        if not handle or not html_content or "__NEXT_DATA__" not in html_content:
            return []
        soup = BeautifulSoup(html_content, "html.parser")
        nd = soup.find("script", id="__NEXT_DATA__")
        if not nd or not nd.string:
            return []
        try:
            data = json.loads(nd.string)
        except Exception:
            return []

        entries = (
            data.get("props", {})
            .get("pageProps", {})
            .get("timeline", {})
            .get("entries", [])
        )
        if not isinstance(entries, list):
            return []

        results: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            tweet = (entry.get("content") or {}).get("tweet")
            if not isinstance(tweet, dict):
                continue
            if tweet.get("retweeted_status") or tweet.get("retweeted_status_result"):
                continue
            status_id = str(tweet.get("id_str") or tweet.get("id") or "").strip()
            if not status_id.isdigit():
                continue
            user_screen = str((tweet.get("user") or {}).get("screen_name") or "").strip()
            if user_screen and user_screen.lower() != handle.lower():
                continue
            raw_text = str(tweet.get("full_text") or tweet.get("text") or "").strip()
            if _RETWEET_PREFIX_RE.search(raw_text):
                continue
            caption = _strip_html_and_urls(raw_text)
            pub_dt = _parse_datetime(tweet.get("created_at"))
            results.append(
                {
                    "handle": handle,
                    "status_id": status_id,
                    "canonical_url": f"https://x.com/{handle}/status/{status_id}",
                    "caption": caption,
                    "published_at": pub_dt,
                    "account": account_cfg,
                }
            )
        return results

    async def fetch_channel_syndication(self, account_cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
        """
        Fetch recent tweets via Twitter's public keyless syndication endpoint.
        Respects circuit-breaker cooldown, uses Chrome + JSON Accept headers,
        trips breaker immediately on HTTP 429 without retrying, and avoids empty bypasses.
        """
        handle = str(account_cfg.get("handle") or "").strip()
        if not handle:
            return [], False
        if self.is_syndication_cooldown_active():
            return [], False
        if not self._consume_budget():
            return [], False

        url = f"https://syndication.twitter.com/srv/timeline-profile/screen-name/{handle}"
        status, body, hdrs = await self._http_get_with_status(url, headers=SYNDICATION_HEADERS)
        if status == 429:
            retry_after = _parse_retry_after_seconds(
                hdrs.get("Retry-After") or hdrs.get("retry-after")
            )
            self.trip_syndication_breaker(retry_after_seconds=retry_after)
            return [], False
        if not body:
            return [], False
        items = self._parse_syndication_html(body, account_cfg)
        return items, bool(items)

    async def fetch_google_news_rss_fallback(
        self,
        accounts: Optional[list[dict[str, Any]]] = None,
    ) -> list[dict[str, Any]]:
        """
        Tier 3 public search fallback using Google News RSS `site:x.com from:{account}`.
        IMPORTANT: Google News pubDate reflects indexing time, NOT tweet creation time;
        therefore `published_at` stays `None` until fxtwitter/vxtwitter verifies `created_at`.
        """
        target_accounts = accounts if accounts is not None else self._accounts
        results: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for acc in target_accounts:
            if len(results) >= self._max_fallback_results_per_cycle:
                break
            handle = str(acc.get("handle") or "").strip()
            if not handle:
                continue
            if not self._consume_budget():
                break

            query = f"site:x.com from:{handle}"
            url = GOOGLE_NEWS_RSS_TEMPLATE.format(query=quote_plus(query))
            status, body, _ = await self._http_get_with_status(url)
            if status == 429:
                logger.info("Google News RSS fallback rate-limited; stopping search queries.")
                break
            if not body:
                continue
            low = body.lstrip()[:250].lower()
            if "<rss" not in low and "<feed" not in low and "<?xml" not in low:
                continue

            parsed = feedparser.parse(body)
            for entry in getattr(parsed, "entries", []) or []:
                if len(results) >= self._max_fallback_results_per_cycle:
                    break
                candidate_sources = [
                    str(entry.get("link") or ""),
                    str(entry.get("id") or ""),
                    str(entry.get("summary") or entry.get("description") or ""),
                ]
                found_url = ""
                for blob in candidate_sources:
                    norm = normalize_canonical_x_url(blob)
                    if norm:
                        found_url = norm
                        break
                if not found_url:
                    continue
                m = _STATUS_URL_RE.search(found_url)
                if not m:
                    continue
                url_handle, status_id = m.group(1), m.group(2)
                if url_handle.lower() != handle.lower() or status_id in seen_ids:
                    continue
                seen_ids.add(status_id)
                caption = _strip_html_and_urls(str(entry.get("title") or ""))
                results.append(
                    {
                        "handle": handle,
                        "status_id": status_id,
                        "canonical_url": f"https://x.com/{handle}/status/{status_id}",
                        "caption": caption,
                        "published_at": None,  # Do NOT trust search index time as tweet time
                        "account": acc,
                        "discovery_tier": "google_news_rss",
                    }
                )
        return results

    async def _ddg_fallback_discover(self) -> list[dict[str, Any]]:
        """Discover recent X status URLs via DuckDuckGo when all RSS mirrors fail."""
        queries = _build_ddg_discovery_queries(self._accounts)

        def _search_sync() -> list[dict[str, str]]:
            from duckduckgo_search import DDGS

            found: list[dict[str, str]] = []
            with DDGS() as ddgs:
                for q in queries:
                    try:
                        hits = list(ddgs.text(keywords=q, region="wt-wt", safesearch="off", max_results=6) or [])
                    except Exception:
                        hits = []
                    for hit in hits:
                        url = str(hit.get("href") or hit.get("url") or "").strip()
                        if url:
                            found.append(
                                {
                                    "url": url,
                                    "title": str(hit.get("title") or ""),
                                    "body": str(hit.get("body") or ""),
                                }
                            )
            return found

        try:
            hits = await asyncio.wait_for(asyncio.to_thread(_search_sync), timeout=10.0)
        except Exception as exc:
            logger.debug("DDG X fallback discovery failed cleanly: %s", exc)
            return []

        results: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for hit in hits:
            m = _STATUS_URL_RE.search(hit.get("url") or "")
            if not m:
                continue
            handle, status_id = m.group(1), m.group(2)
            if status_id in seen_ids:
                continue
            seen_ids.add(status_id)
            known = self._account_by_handle.get(handle.lower())
            account_cfg = known or {
                "handle": handle,
                "display_name": handle,
                "region": "",
                "lang": "en",
                "credibility_tier": 3,
                "source_group": handle.lower(),
                "is_official_media": False,
            }
            caption = _strip_html_and_urls(hit.get("body") or hit.get("title") or "")
            results.append(
                {
                    "handle": handle,
                    "status_id": status_id,
                    "canonical_url": f"https://x.com/{handle}/status/{status_id}",
                    "caption": caption,
                    "published_at": None,
                    "account": account_cfg,
                }
            )
        return results

    @staticmethod
    def _extract_fxtwitter_video(data: dict[str, Any]) -> Optional[dict[str, Any]]:
        tweet = data.get("tweet")
        if not isinstance(tweet, dict):
            return None
        media = tweet.get("media") or {}
        if not isinstance(media, dict):
            return None
        videos = list(media.get("videos") or [])
        if not videos:
            for item in media.get("all") or []:
                if isinstance(item, dict) and str(item.get("type") or "").lower() == "video":
                    videos.append(item)
        if not videos:
            return None
        vid = videos[0]
        if not isinstance(vid, dict):
            return None
        if str(vid.get("type") or "video").lower() != "video":
            return None
        video_url = str(vid.get("url") or "").strip()
        if not video_url.startswith(("http://", "https://")):
            return None
        return {
            "video_url": video_url,
            "thumbnail_url": str(vid.get("thumbnail_url") or "").strip(),
            "duration": vid.get("duration"),
            "width": vid.get("width"),
            "text": str(tweet.get("text") or "").strip(),
            "created_at": tweet.get("created_timestamp") or tweet.get("created_at"),
        }

    @staticmethod
    def _extract_vxtwitter_video(data: dict[str, Any]) -> Optional[dict[str, Any]]:
        media_ext = data.get("media_extended") or []
        if not isinstance(media_ext, list):
            return None
        for item in media_ext:
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "").lower() != "video":
                continue
            video_url = str(item.get("url") or "").strip()
            if not video_url.startswith(("http://", "https://")):
                continue
            dur = item.get("duration")
            if dur is None and item.get("duration_millis") is not None:
                try:
                    dur = float(item["duration_millis"]) / 1000.0
                except Exception:
                    dur = None
            size = item.get("size") or {}
            width = item.get("width") or (size.get("width") if isinstance(size, dict) else None)
            return {
                "video_url": video_url,
                "thumbnail_url": str(item.get("thumbnail_url") or "").strip(),
                "duration": dur,
                "width": width,
                "text": str(data.get("text") or "").strip(),
                "created_at": data.get("date_epoch") or data.get("date"),
            }
        return None

    @staticmethod
    def _extract_fxtwitter_media(data: dict[str, Any], allow_photo: bool = False) -> Optional[dict[str, Any]]:
        vid = XVideoSource._extract_fxtwitter_video(data)
        if vid is not None:
            vid["media_type"] = "video"
            vid.setdefault("image_url", vid.get("thumbnail_url") or "")
            return vid
        if not allow_photo or not isinstance(data, dict):
            return None
        tweet = data.get("tweet")
        if not isinstance(tweet, dict):
            return None
        media = tweet.get("media") or {}
        if not isinstance(media, dict):
            return None
        photos = list(media.get("photos") or [])
        if not photos:
            for item in media.get("all") or []:
                if isinstance(item, dict) and str(item.get("type") or "").lower() == "photo":
                    photos.append(item)
        if not photos or not isinstance(photos[0], dict):
            return None
        photo_url = str(photos[0].get("url") or "").strip()
        if not photo_url.startswith(("http://", "https://")):
            return None
        return {
            "media_type": "photo",
            "video_url": "",
            "image_url": photo_url,
            "thumbnail_url": photo_url,
            "duration": None,
            "width": photos[0].get("width"),
            "text": str(tweet.get("text") or "").strip(),
            "created_at": tweet.get("created_timestamp") or tweet.get("created_at"),
        }

    @staticmethod
    def _extract_vxtwitter_media(data: dict[str, Any], allow_photo: bool = False) -> Optional[dict[str, Any]]:
        vid = XVideoSource._extract_vxtwitter_video(data)
        if vid is not None:
            vid["media_type"] = "video"
            vid.setdefault("image_url", vid.get("thumbnail_url") or "")
            return vid
        if not allow_photo or not isinstance(data, dict):
            return None
        media_ext = data.get("media_extended") or []
        if not isinstance(media_ext, list):
            return None
        for item in media_ext:
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "").lower() != "image":
                continue
            photo_url = str(item.get("url") or "").strip()
            if not photo_url.startswith(("http://", "https://")):
                continue
            size = item.get("size") or {}
            width = item.get("width") or (size.get("width") if isinstance(size, dict) else None)
            return {
                "media_type": "photo",
                "video_url": "",
                "image_url": photo_url,
                "thumbnail_url": photo_url,
                "duration": None,
                "width": width,
                "text": str(data.get("text") or "").strip(),
                "created_at": data.get("date_epoch") or data.get("date"),
            }
        return None

    async def _resolve_video_metadata(self, handle: str, status_id: str) -> Optional[dict[str, Any]]:
        """
        Resolve direct `.mp4` metadata for a tweet via `fxtwitter` then fallback `vxtwitter`.
        Enforces native video (`type == "video"`, not GIF), duration, and min width gates.
        """
        return await self._resolve_media_metadata(handle, status_id, allow_photo=False)

    async def _resolve_media_metadata(
        self, handle: str, status_id: str, allow_photo: bool = False
    ) -> Optional[dict[str, Any]]:
        """Resolve native `.mp4` video (or high-priority photo if `allow_photo=True`)."""
        async with self._semaphore:
            fx_data = await self._http_get_json(f"https://api.fxtwitter.com/{handle}/status/{status_id}")
        meta = self._extract_fxtwitter_media(fx_data, allow_photo=allow_photo) if fx_data else None

        if meta is None:
            async with self._semaphore:
                vx_data = await self._http_get_json(f"https://api.vxtwitter.com/{handle}/status/{status_id}")
            meta = self._extract_vxtwitter_media(vx_data, allow_photo=allow_photo) if vx_data else None

        if meta is None:
            return None

        if meta.get("media_type") == "photo":
            return meta

        min_dur = int(getattr(config, "X_MIN_VIDEO_DURATION_SEC", 4))
        max_dur = int(getattr(config, "X_MAX_VIDEO_DURATION_SEC", 120))
        min_width = int(getattr(config, "X_MIN_VIDEO_WIDTH", 480))

        dur_val = meta.get("duration")
        if dur_val is not None:
            try:
                dur_f = float(dur_val)
                if dur_f < min_dur or dur_f > max_dur:
                    logger.debug(
                        "X video %s/%s skipped: duration %.1fs outside [%d, %d]",
                        handle, status_id, dur_f, min_dur, max_dur,
                    )
                    return None
            except (TypeError, ValueError):
                pass

        width_val = meta.get("width")
        if width_val is not None:
            try:
                width_i = int(width_val)
                if 0 < width_i < min_width:
                    logger.debug(
                        "X video %s/%s skipped: width %d < %d",
                        handle, status_id, width_i, min_width,
                    )
                    return None
            except (TypeError, ValueError):
                pass

        return meta

    async def fetch_articles(self) -> list[RawArticle]:
        if not getattr(config, "ENABLE_X_VIDEO_SOURCE", True) or not self._accounts:
            return []

        now = datetime.now(timezone.utc)
        max_age = timedelta(hours=self._max_age_hours)
        self._cycle_requests_used = 0

        batch_accounts = self.select_account_batch()
        candidates: list[dict[str, Any]] = []
        syndication_ok_handles: set[str] = set()

        # TIER 1 — Sequential polite X syndication across the rotating batch
        if self.is_syndication_cooldown_active():
            logger.info("X syndication skipped: circuit breaker cooldown active.")
        else:
            seen_batch_handles: set[str] = set()
            syn_requests = 0
            for acc in batch_accounts:
                handle_key = str(acc.get("handle") or "").strip().lower()
                if not handle_key or handle_key in seen_batch_handles:
                    continue
                seen_batch_handles.add(handle_key)

                if self.is_syndication_cooldown_active():
                    logger.info("Stopping remaining syndication requests for cycle (breaker tripped).")
                    break

                if syn_requests > 0:
                    await self._sleep(self._next_jitter())

                syn_requests += 1
                entries, ok = await self.fetch_channel_syndication(acc)
                if ok and entries:
                    syndication_ok_handles.add(handle_key)
                    candidates.extend(entries)

                if self.is_syndication_cooldown_active():
                    logger.info("Syndication HTTP 429 cooldown tripped mid-cycle; stopping remaining syndication calls.")
                    break

        # TIER 2 — RSS bridges for batch accounts not satisfied by Tier 1
        rss_targets = [
            acc
            for acc in batch_accounts
            if str(acc.get("handle") or "").strip().lower() not in syndication_ok_handles
        ]
        if rss_targets and self._mirrors:
            for acc in rss_targets:
                if self._cycle_requests_used >= self._max_requests_per_cycle:
                    break
                entries, _ = await self.fetch_channel_rss(acc)
                candidates.extend(entries)

        # TIER 3 — Public Search Fallback (Google News RSS first, then DuckDuckGo secondary fallback)
        if not candidates and self._cycle_requests_used < self._max_requests_per_cycle:
            logger.info("X Tier 1/2 yielded 0 candidates; trying Tier 3 Google News RSS search fallback.")
            gnews_hits = await self.fetch_google_news_rss_fallback(batch_accounts)
            candidates.extend(gnews_hits)

        if not candidates:
            logger.info("Trying DuckDuckGo public X status fallback discovery.")
            ddg_hits = await self._ddg_fallback_discover()
            candidates.extend(ddg_hits[: self._max_fallback_results_per_cycle])

        # Cross-tier canonical URL + status_id deduplication and early freshness check
        seen_status_ids: set[str] = set()
        seen_canonical_urls: set[str] = set()
        fresh_candidates: list[dict[str, Any]] = []
        for cand in candidates:
            sid = str(cand.get("status_id") or "").strip()
            raw_canon = str(cand.get("canonical_url") or "")
            canon_url = normalize_canonical_x_url(raw_canon) or (
                f"https://x.com/{cand.get('handle')}/status/{sid}" if sid else ""
            )
            if not sid or not canon_url:
                continue
            if sid in seen_status_ids or canon_url.lower() in seen_canonical_urls:
                logger.debug("Deduplicating duplicate X candidate %s across tiers.", canon_url)
                continue
            seen_status_ids.add(sid)
            seen_canonical_urls.add(canon_url.lower())
            cand["canonical_url"] = canon_url

            pub_dt = cand.get("published_at")
            if isinstance(pub_dt, datetime) and (now - pub_dt) > max_age:
                logger.debug("Rejected stale X candidate %s (%s).", canon_url, pub_dt.isoformat())
                continue

            caption = str(cand.get("caption") or "").strip()
            if _is_fake_or_recycled_clip(caption):
                continue

            acct = cand.get("account") or {}
            if not bool(acct.get("is_official_media", False)) and has_third_party_repost_marker(caption):
                continue

            fresh_candidates.append(cand)

        # Resolve priority / official accounts first, then newer items
        fresh_candidates.sort(
            key=lambda c: (
                0 if self._is_priority_account(c.get("account") or {}) else 1,
                0 if bool((c.get("account") or {}).get("is_official_media", False)) else 1,
                -(c["published_at"].timestamp() if isinstance(c.get("published_at"), datetime) else 0.0),
            )
        )

        probe_limit = min(self._max_fallback_results_per_cycle, max(self._max_videos_per_cycle * 2, 16))
        probe_pool = fresh_candidates[:probe_limit]

        def _allow_photo_for_candidate(cand: dict[str, Any]) -> bool:
            acct = cand.get("account") or {}
            return bool(
                acct.get("is_official_media", False)
                or acct.get("region") in ("kurdistan_bashur", "middle_east")
                or int(acct.get("credibility_tier", 3) or 3) <= 2
            )

        async def _resolve_candidate(cand: dict[str, Any]) -> Optional[dict[str, Any]]:
            # Always invoke 2-arg `_resolve_video_metadata` first so tests mocking it stay compatible.
            vid_meta = await self._resolve_video_metadata(cand["handle"], cand["status_id"])
            if isinstance(vid_meta, dict):
                return vid_meta
            if _allow_photo_for_candidate(cand):
                return await self._resolve_media_metadata(cand["handle"], cand["status_id"], allow_photo=True)
            return None

        meta_results = await asyncio.gather(
            *[_resolve_candidate(c) for c in probe_pool],
            return_exceptions=True,
        )

        articles: list[RawArticle] = []
        for cand, meta in zip(probe_pool, meta_results):
            if not isinstance(meta, dict):
                continue

            video_url = str(meta.get("video_url") or "").strip()
            image_url = str(meta.get("thumbnail_url") or meta.get("image_url") or "").strip()
            if not video_url and not image_url:
                continue

            acct = cand.get("account") or {}
            full_text = _strip_html_and_urls(meta.get("text") or cand.get("caption") or "")
            if not full_text:
                continue
            if _is_fake_or_recycled_clip(full_text):
                continue
            if not bool(acct.get("is_official_media", False)) and has_third_party_repost_marker(full_text):
                continue

            # Require verifiable publication timestamp (from syndication/RSS or fxtwitter/vxtwitter)
            pub_dt = cand.get("published_at") or _parse_datetime(meta.get("created_at"))
            if not isinstance(pub_dt, datetime):
                logger.debug("Rejected X candidate without verifiable timestamp: %s", cand["canonical_url"])
                continue
            if (now - pub_dt) > max_age:
                logger.debug("Rejected stale X candidate after metadata check: %s", cand["canonical_url"])
                continue

            articles.append(
                RawArticle(
                    url=cand["canonical_url"],
                    title=full_text[:180],
                    summary=full_text,
                    source_name=str(acct.get("display_name") or cand["handle"]),
                    category="breaking" if bool(acct.get("is_official_media", False)) else "world",
                    publication_date=pub_dt.isoformat(),
                    image_url=image_url,
                    video_url=video_url,
                    country="",
                    region="",  # Content-based classifier determines kurdistan_bashur strictly from text
                )
            )
            if len(articles) >= self._max_videos_per_cycle:
                break

        logger.info("XVideoSource discovered %d verified X candidate(s).", len(articles))
        return articles
