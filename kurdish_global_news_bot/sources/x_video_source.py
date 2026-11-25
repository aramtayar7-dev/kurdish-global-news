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
import email.utils
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import aiohttp
import feedparser
from bs4 import BeautifulSoup

import config
from processing.video_watermark_detector import has_third_party_repost_marker
from sources.base import BaseSource, RawArticle

logger = logging.getLogger(__name__)

_STATUS_URL_RE = re.compile(r"https?://[^/]+/([A-Za-z0-9_]{1,30})/status/(\d+)")
_URL_STRIP_RE = re.compile(r"https?://\S+")
_RETWEET_PREFIX_RE = re.compile(r"^\s*(?:RT\s+by\b|RT\s+@|Reposted\s+by\b)", re.IGNORECASE)
_FAKE_OR_RECYCLED_RE = re.compile(
    r"\b(?:arma\s*3|dcs\s*world|war\s*thunder|gameplay|simulation|flight\s*sim|"
    r"throwback|archive\s+footage|old\s+video|from\s+(?:201\d|202[0-5])|"
    r"tiktok|ai[- ]?generated|deepfake)\b",
    re.IGNORECASE,
)


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
    """Free-tier curated X/Twitter video discovery adapter."""

    def __init__(
        self,
        accounts: Optional[list[dict[str, Any]]] = None,
        mirrors: Optional[list[str]] = None,
        request_timeout: int = 8,
        max_age_hours: Optional[float] = None,
        max_videos_per_cycle: Optional[int] = None,
    ) -> None:
        super().__init__(
            name="x_video",
            source_type="x_video",
            enabled=bool(getattr(config, "ENABLE_X_VIDEO_SOURCE", True)),
        )
        self._accounts: list[dict[str, Any]] = list(
            accounts if accounts is not None else getattr(config, "X_VIDEO_SOURCE_ACCOUNTS", [])
        )
        self._mirrors: list[str] = [
            m.rstrip("/")
            for m in (mirrors if mirrors is not None else getattr(config, "X_NITTER_RSS_MIRRORS", []))
            if m and m.strip()
        ]
        self._request_timeout: int = max(3, int(request_timeout))
        self._max_age_hours: float = float(
            max_age_hours if max_age_hours is not None else getattr(config, "X_MAX_VIDEO_AGE_HOURS", 6.0)
        )
        self._max_videos_per_cycle: int = max(
            1,
            int(
                max_videos_per_cycle
                if max_videos_per_cycle is not None
                else getattr(config, "X_MAX_VIDEOS_PER_CYCLE", 4)
            ),
        )
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(4)
        self._account_by_handle: dict[str, dict[str, Any]] = {
            str(acc.get("handle") or "").strip().lower(): acc
            for acc in self._accounts
            if acc.get("handle")
        }

    @property
    def source_name(self) -> str:
        return "X (Twitter) Video"

    async def initialize(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._request_timeout),
                headers={"User-Agent": "Mozilla/5.0 (compatible; KGNBot/1.0)"},
            )

    async def cleanup(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _http_get_text(self, url: str) -> Optional[str]:
        await self.initialize()
        assert self._session is not None
        try:
            async with self._session.get(url, allow_redirects=True) as resp:
                if resp.status in (403, 404, 429) or resp.status >= 500:
                    logger.debug("X mirror HTTP %s for %s", resp.status, url)
                    return None
                if resp.status != 200:
                    return None
                return await resp.text(errors="replace")
        except Exception as exc:
            logger.debug("X mirror fetch error (%s): %s", url, exc)
            return None

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
        Try configured Nitter-compatible RSS mirrors sequentially until one returns valid RSS/Atom XML.
        Returns `(entries, mirror_ok)`.
        """
        handle = str(account_cfg.get("handle") or "").strip()
        if not handle:
            return [], False

        for mirror in self._mirrors:
            rss_url = f"{mirror}/{handle}/rss"
            async with self._semaphore:
                body = await self._http_get_text(rss_url)
            if not body:
                continue
            low = body.lstrip()[:200].lower()
            if "<rss" not in low and "<feed" not in low and "<?xml" not in low:
                continue

            parsed = feedparser.parse(body)
            entries = list(getattr(parsed, "entries", []) or [])
            results: list[dict[str, Any]] = []
            for entry in entries:
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
                caption = _strip_html_and_urls(raw_desc) or _strip_html_and_urls(raw_title)
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
            return results, True

        return [], False

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

    async def _resolve_video_metadata(self, handle: str, status_id: str) -> Optional[dict[str, Any]]:
        """
        Resolve direct `.mp4` metadata for a tweet via `fxtwitter` then fallback `vxtwitter`.
        Enforces native video (`type == "video"`, not GIF), duration, and min width gates.
        """
        async with self._semaphore:
            fx_data = await self._http_get_json(f"https://api.fxtwitter.com/{handle}/status/{status_id}")
        meta = self._extract_fxtwitter_video(fx_data) if fx_data else None

        if meta is None:
            async with self._semaphore:
                vx_data = await self._http_get_json(f"https://api.vxtwitter.com/{handle}/status/{status_id}")
            meta = self._extract_vxtwitter_video(vx_data) if vx_data else None

        if meta is None:
            return None

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

        rss_tasks = [self.fetch_channel_rss(acc) for acc in self._accounts]
        rss_results = await asyncio.gather(*rss_tasks, return_exceptions=True)

        candidates: list[dict[str, Any]] = []
        any_mirror_ok = False
        for res in rss_results:
            if isinstance(res, tuple):
                entries, ok = res
                if ok:
                    any_mirror_ok = True
                candidates.extend(entries)

        if not any_mirror_ok and not candidates:
            logger.info("All X RSS mirrors unavailable; trying free DuckDuckGo X discovery fallback.")
            candidates = await self._ddg_fallback_discover()

        seen_status_ids: set[str] = set()
        fresh_candidates: list[dict[str, Any]] = []
        for cand in candidates:
            sid = str(cand.get("status_id") or "")
            if not sid or sid in seen_status_ids:
                continue
            seen_status_ids.add(sid)

            pub_dt = cand.get("published_at")
            if isinstance(pub_dt, datetime) and (now - pub_dt) > max_age:
                continue

            caption = str(cand.get("caption") or "").strip()
            if _is_fake_or_recycled_clip(caption):
                continue

            acct = cand.get("account") or {}
            if not bool(acct.get("is_official_media", False)) and has_third_party_repost_marker(caption):
                continue

            fresh_candidates.append(cand)

        # Resolve official accounts first, then newer items
        fresh_candidates.sort(
            key=lambda c: (
                0 if bool((c.get("account") or {}).get("is_official_media", False)) else 1,
                -(c["published_at"].timestamp() if isinstance(c.get("published_at"), datetime) else 0.0),
            )
        )

        # Limit metadata lookups while allowing enough headroom to hit cycle cap
        probe_limit = max(self._max_videos_per_cycle * 3, 12)
        probe_pool = fresh_candidates[:probe_limit]
        meta_results = await asyncio.gather(
            *[self._resolve_video_metadata(c["handle"], c["status_id"]) for c in probe_pool],
            return_exceptions=True,
        )

        articles: list[RawArticle] = []
        for cand, meta in zip(probe_pool, meta_results):
            if not isinstance(meta, dict):
                continue

            acct = cand.get("account") or {}
            full_text = _strip_html_and_urls(meta.get("text") or cand.get("caption") or "")
            if not full_text:
                continue
            if _is_fake_or_recycled_clip(full_text):
                continue
            if not bool(acct.get("is_official_media", False)) and has_third_party_repost_marker(full_text):
                continue

            pub_dt = cand.get("published_at") or _parse_datetime(meta.get("created_at")) or now
            if (now - pub_dt) > max_age:
                continue

            articles.append(
                RawArticle(
                    url=cand["canonical_url"],
                    title=full_text[:180],
                    summary=full_text,
                    source_name=str(acct.get("display_name") or cand["handle"]),
                    category="breaking" if bool(acct.get("is_official_media", False)) else "world",
                    publication_date=pub_dt,
                    image_url=str(meta.get("thumbnail_url") or ""),
                    video_url=str(meta.get("video_url") or ""),
                    country="",
                    region="",  # Content-based classifier determines kurdistan_bashur strictly from text
                )
            )
            if len(articles) >= self._max_videos_per_cycle:
                break

        logger.info("XVideoSource discovered %d verified X video candidate(s).", len(articles))
        return articles
