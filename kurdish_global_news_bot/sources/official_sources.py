"""
Kurdish Global News Bot — Official Sources Adapter

Discovers news from official government and international organization feeds.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import aiohttp
import feedparser

from sources.base import BaseSource, RawArticle
from config import REQUEST_TIMEOUT, USER_AGENT

logger = logging.getLogger("sources.official")

# Official feeds from international organizations
OFFICIAL_FEEDS: list[dict[str, str]] = [
    {
        "name": "UN News",
        "url": "https://news.un.org/feed/subscribe/en/news/all/rss.xml",
        "category": "politics_world",
    },
    {
        "name": "WHO News",
        "url": "https://www.who.int/rss-feeds/news-english.xml",
        "category": "politics_world",
    },
    {
        "name": "NATO News",
        "url": "https://www.nato.int/cps/en/natolive/news.xml",
        "category": "military_conflict",
    },
    {
        "name": "EU News",
        "url": "https://ec.europa.eu/commission/presscorner/api/rss",
        "category": "politics_world",
    },
    {
        "name": "IAEA News",
        "url": "https://www.iaea.org/feeds/topnews",
        "category": "economy_markets",
    },
]


class OfficialSourcesAdapter(BaseSource):
    """Fetches news from official governmental and international organization feeds."""

    def __init__(self) -> None:
        super().__init__(name="OfficialSources", source_type="official", enabled=True)
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
        """Fetch articles from all official feeds."""
        if not self._session:
            await self.initialize()

        articles: list[RawArticle] = []
        tasks = [self._fetch_feed(feed) for feed in OFFICIAL_FEEDS]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                feed_name = OFFICIAL_FEEDS[i]["name"]
                logger.warning("Official feed '%s' failed: %s", feed_name, str(result)[:200])
                continue
            if isinstance(result, list):
                articles.extend(result)

        logger.info("Official sources discovered %d articles.", len(articles))
        return articles

    async def _fetch_feed(self, feed_config: dict[str, str]) -> list[RawArticle]:
        """Fetch and parse a single official feed."""
        url = feed_config["url"]
        feed_name = feed_config["name"]
        category = feed_config.get("category", "International Organizations")
        articles: list[RawArticle] = []

        try:
            assert self._session is not None
            async with self._session.get(url) as response:
                if response.status != 200:
                    logger.warning("Official feed '%s' status %d.", feed_name, response.status)
                    return []
                content = await response.text()

            parsed = await asyncio.to_thread(feedparser.parse, content)

            for entry in parsed.entries[:15]:
                raw = self._parse_entry(entry, feed_name, category)
                if raw:
                    articles.append(raw)

        except asyncio.TimeoutError:
            logger.warning("Official feed '%s' timed out.", feed_name)
        except Exception as e:
            logger.warning("Official feed '%s' error: %s", feed_name, str(e)[:200])

        return articles

    def _parse_entry(self, entry: object, feed_name: str, category: str) -> Optional[RawArticle]:
        """Parse a feed entry into a RawArticle."""
        link = getattr(entry, "link", "").strip()
        title = getattr(entry, "title", "").strip()

        if not link or not title or not link.startswith(("http://", "https://")):
            return None

        summary = ""
        if hasattr(entry, "summary"):
            summary = entry.summary.strip()
        elif hasattr(entry, "description"):
            summary = entry.description.strip()
        if summary:
            from bs4 import BeautifulSoup
            summary = BeautifulSoup(summary, "html.parser").get_text(separator=" ").strip()[:800]

        pub_date = getattr(entry, "published", "") or getattr(entry, "updated", "")

        return RawArticle(
            url=link,
            title=title,
            summary=summary,
            source_name=feed_name,
            category=category,
            publication_date=pub_date,
        )
