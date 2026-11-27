"""
Kurdish Global News Bot — RSS/Atom Feed Source

Discovers news articles from configured RSS and Atom feeds.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import aiohttp
import feedparser

from sources.base import BaseSource, RawArticle
from config import RSS_FEEDS, REQUEST_TIMEOUT, USER_AGENT, is_blocked_western_source
from processing.content_formatter import strip_rss_boilerplate

logger = logging.getLogger("sources.rss")


def parse_rss_entry(entry: object, feed_name: str, category: str) -> Optional[RawArticle]:
    """Parse a feedparser entry into a RawArticle cleanly across RSS adapters."""
    link = getattr(entry, "link", "").strip()
    title = getattr(entry, "title", "").strip()

    if not link or not title or not link.startswith(("http://", "https://")):
        return None

    # Extract summary
    summary = ""
    if hasattr(entry, "summary") and isinstance(entry.summary, str):
        summary = entry.summary.strip()
    elif hasattr(entry, "description") and isinstance(entry.description, str):
        summary = entry.description.strip()

    # Strip HTML tags and RSS footer boilerplate from summary
    if summary:
        from bs4 import BeautifulSoup
        summary = BeautifulSoup(summary, "html.parser").get_text(separator=" ").strip()
        summary = strip_rss_boilerplate(summary)
        summary = summary[:1500]

    # Extract publication date
    pub_date = ""
    if hasattr(entry, "published") and isinstance(entry.published, str):
        pub_date = entry.published
    elif hasattr(entry, "updated") and isinstance(entry.updated, str):
        pub_date = entry.updated

    # Extract image
    image_url = ""
    if hasattr(entry, "media_content") and entry.media_content:
        for media in entry.media_content:
            if isinstance(media, dict) and media.get("url"):
                image_url = str(media["url"]).strip()
                break
    if not image_url and hasattr(entry, "media_thumbnail") and entry.media_thumbnail:
        for thumb in entry.media_thumbnail:
            if isinstance(thumb, dict) and thumb.get("url"):
                image_url = str(thumb["url"]).strip()
                break
    if not image_url and hasattr(entry, "links"):
        for link_entry in entry.links:
            if isinstance(link_entry, dict):
                link_type = str(link_entry.get("type", ""))
                if "image" in link_type and link_entry.get("href"):
                    image_url = str(link_entry["href"]).strip()
                    break

    return RawArticle(
        url=link,
        title=title,
        summary=summary,
        source_name=feed_name,
        category=category,
        publication_date=pub_date,
        image_url=image_url,
    )


async def fetch_rss_feed(
    session: aiohttp.ClientSession,
    feed_config: dict[str, str],
    max_entries: int = 20,
    region: str = "",
) -> list[RawArticle]:
    """Fetch and parse a single RSS/Atom feed into RawArticle entries."""
    url = feed_config["url"]
    feed_name = feed_config["name"]
    category = feed_config.get("category", "")
    articles: list[RawArticle] = []

    try:
        async with session.get(url) as response:
            if response.status != 200:
                logger.warning("RSS feed '%s' returned status %d.", feed_name, response.status)
                return []
            content = await response.text(errors="replace")

        parsed = await asyncio.to_thread(feedparser.parse, content)

        for entry in parsed.entries[:max_entries]:
            raw = parse_rss_entry(entry, feed_name, category)
            if raw:
                raw.region = region or feed_config.get("region", "")
                articles.append(raw)
    except asyncio.TimeoutError:
        logger.warning("RSS feed '%s' timed out.", feed_name)
    except Exception as e:
        logger.warning("RSS feed '%s' error: %s", feed_name, str(e)[:200])

    return articles


class RSSSource(BaseSource):
    """Fetches news articles from configured RSS/Atom feeds."""

    def __init__(self) -> None:
        super().__init__(name="RSS", source_type="rss", enabled=True)
        self._session: Optional[aiohttp.ClientSession] = None

    async def initialize(self) -> None:
        """Create aiohttp session."""
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        self._session = aiohttp.ClientSession(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
        )

    async def cleanup(self) -> None:
        """Close aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch_articles(self) -> list[RawArticle]:
        """Fetch articles from all configured RSS feeds concurrently."""
        if not self._session:
            await self.initialize()

        articles: list[RawArticle] = []
        active_feeds = [
            feed
            for feed in RSS_FEEDS
            if feed.get("enabled", True)
            and not is_blocked_western_source(
                str(feed.get("url", "")),
                str(feed.get("name", "")),
            )
        ]
        tasks = [self._fetch_feed(feed) for feed in active_feeds]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            feed_name = active_feeds[i]["name"]
            if isinstance(result, Exception):
                logger.warning("RSS feed '%s' failed: %s", feed_name, str(result)[:200])
                continue
            if isinstance(result, list):
                articles.extend(result)

        logger.info("RSS discovered %d articles from %d feeds.", len(articles), len(active_feeds))
        return articles

    async def _fetch_feed(self, feed_config: dict[str, str]) -> list[RawArticle]:
        """Fetch and parse a single RSS feed."""
        assert self._session is not None
        return await fetch_rss_feed(self._session, feed_config, max_entries=20)

    def _parse_entry(self, entry: object, feed_name: str, category: str) -> Optional[RawArticle]:
        """Parse a feedparser entry into a RawArticle."""
        return parse_rss_entry(entry, feed_name, category)
