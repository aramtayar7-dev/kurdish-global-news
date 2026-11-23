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
from config import RSS_FEEDS, REQUEST_TIMEOUT, USER_AGENT

logger = logging.getLogger("sources.rss")


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
        tasks = [self._fetch_feed(feed) for feed in RSS_FEEDS]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            feed_name = RSS_FEEDS[i]["name"]
            if isinstance(result, Exception):
                logger.warning("RSS feed '%s' failed: %s", feed_name, str(result)[:200])
                continue
            if isinstance(result, list):
                articles.extend(result)

        logger.info("RSS discovered %d articles from %d feeds.", len(articles), len(RSS_FEEDS))
        return articles

    async def _fetch_feed(self, feed_config: dict[str, str]) -> list[RawArticle]:
        """Fetch and parse a single RSS feed."""
        url = feed_config["url"]
        feed_name = feed_config["name"]
        category = feed_config.get("category", "")
        articles: list[RawArticle] = []

        try:
            assert self._session is not None
            async with self._session.get(url) as response:
                if response.status != 200:
                    logger.warning("RSS feed '%s' returned status %d.", feed_name, response.status)
                    return []
                content = await response.text()

            # Parse feed in a thread to avoid blocking
            parsed = await asyncio.to_thread(feedparser.parse, content)

            for entry in parsed.entries[:20]:  # Limit entries per feed
                raw = self._parse_entry(entry, feed_name, category)
                if raw:
                    articles.append(raw)

        except asyncio.TimeoutError:
            logger.warning("RSS feed '%s' timed out.", feed_name)
        except Exception as e:
            logger.warning("RSS feed '%s' error: %s", feed_name, str(e)[:200])

        return articles

    def _parse_entry(self, entry: object, feed_name: str, category: str) -> Optional[RawArticle]:
        """Parse a feedparser entry into a RawArticle."""
        link = getattr(entry, "link", "").strip()
        title = getattr(entry, "title", "").strip()

        if not link or not title:
            return None

        if not link.startswith(("http://", "https://")):
            return None

        # Extract summary
        summary = ""
        if hasattr(entry, "summary"):
            summary = entry.summary.strip()
        elif hasattr(entry, "description"):
            summary = entry.description.strip()

        # Strip HTML tags from summary
        if summary:
            from bs4 import BeautifulSoup
            summary = BeautifulSoup(summary, "html.parser").get_text(separator=" ").strip()
            summary = summary[:1000]

        # Extract publication date
        pub_date = ""
        if hasattr(entry, "published"):
            pub_date = entry.published
        elif hasattr(entry, "updated"):
            pub_date = entry.updated

        # Extract image
        image_url = ""
        if hasattr(entry, "media_content") and entry.media_content:
            for media in entry.media_content:
                if isinstance(media, dict) and "url" in media:
                    image_url = media["url"]
                    break
        if not image_url and hasattr(entry, "media_thumbnail") and entry.media_thumbnail:
            for thumb in entry.media_thumbnail:
                if isinstance(thumb, dict) and "url" in thumb:
                    image_url = thumb["url"]
                    break
        if not image_url and hasattr(entry, "links"):
            for link_entry in entry.links:
                if isinstance(link_entry, dict):
                    link_type = link_entry.get("type", "")
                    if "image" in link_type:
                        image_url = link_entry.get("href", "")
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
