"""
Kurdish Global News Bot — Four Parts of Kurdistan News Source

Discovers regional breaking and political/human-rights news across all four
parts of Kurdistan:
  - South (باشوور): Erbil, Sulaymaniyah, Duhok, Kirkuk, Kurdistan Region
  - West (ڕۆژئاوا): Rojava, North and East Syria, Qamishli, Kobani, Hasakah, SDF
  - North (باکوور): Diyarbakir (Amed), Van, Kurdish political developments in Turkey
  - East (ڕۆژهەڵات): Rojhelat, Sanandaj (Sine), Mahabad, Urmia, Kermanshah, Iran
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import aiohttp

from config import REQUEST_TIMEOUT, USER_AGENT
from sources.base import BaseSource, RawArticle
from sources.rss_source import fetch_rss_feed

logger = logging.getLogger("sources.kurdistan")

KURDISTAN_FEEDS: list[dict[str, str]] = [
    # ── South Kurdistan (باشوور) ──
    {
        "name": "Kurdistan24 EN",
        "url": "https://www.kurdistan24.net/en/rss.xml",
        "category": "kurdistan",
        "region": "South Kurdistan",
    },
    {
        "name": "Kurdistan24 Sorani",
        "url": "https://www.kurdistan24.net/ckb/rss.xml",
        "category": "kurdistan",
        "region": "South Kurdistan",
    },
    {
        "name": "PUKMedia EN",
        "url": "https://www.pukmedia.com/EN/rss",
        "category": "kurdistan",
        "region": "South Kurdistan",
    },
    {
        "name": "PUKMedia Sorani",
        "url": "https://www.pukmedia.com/KS/rss",
        "category": "kurdistan",
        "region": "South Kurdistan",
    },
    {
        "name": "Google News - South Kurdistan",
        "url": (
            "https://news.google.com/rss/search?"
            "q=site:rudaw.net+OR+Erbil+OR+Sulaymaniyah+OR+Duhok+OR+Kirkuk+OR+%22Kurdistan+Region%22"
            "&hl=en-US&gl=US&ceid=US:en"
        ),
        "category": "kurdistan",
        "region": "South Kurdistan",
    },
    {
        "name": "Google News - Kurdistan Sorani",
        "url": (
            "https://news.google.com/rss/search?"
            "q=%D9%87%DB%95%D9%88%D9%84%DB%8E%D8%B1+OR+%D8%B3%D9%84%DB%8E%D9%85%D8%A7%D9%86%DB%8C+OR+"
            "%DA%A9%DB%95%D8%B1%DA%A9%D9%88%DA%A9+OR+%DA%A9%D9%88%D8%B1%D8%AF%D8%B3%D8%AA%D8%A7%D9%86"
            "&hl=ar&gl=IQ&ceid=IQ:ar"
        ),
        "category": "kurdistan",
        "region": "South Kurdistan",
    },
    # ── West Kurdistan (ڕۆژئاوا) ──
    {
        "name": "North Press Agency",
        "url": "https://npasyria.com/en/feed/",
        "category": "kurdistan",
        "region": "West Kurdistan",
    },
    {
        "name": "Rojava Information Center",
        "url": "https://rojavainformationcenter.org/feed/",
        "category": "kurdistan",
        "region": "West Kurdistan",
    },
    {
        "name": "Google News - West Kurdistan",
        "url": (
            "https://news.google.com/rss/search?"
            "q=Rojava+OR+Qamishli+OR+Kobani+OR+Hasakah+OR+%22Syrian+Democratic+Forces%22"
            "&hl=en-US&gl=US&ceid=US:en"
        ),
        "category": "kurdistan",
        "region": "West Kurdistan",
    },
    # ── North Kurdistan (باکوور) ──
    {
        "name": "Bianet EN",
        "url": "https://bianet.org/english/rss",
        "category": "kurdistan",
        "region": "North Kurdistan",
    },
    {
        "name": "Google News - North Kurdistan",
        "url": (
            "https://news.google.com/rss/search?"
            "q=Diyarbakir+OR+Amed+OR+%22Van+Turkey%22+OR+%22DEM+Party%22+OR+Kurdish+Turkey"
            "&hl=en-US&gl=US&ceid=US:en"
        ),
        "category": "kurdistan",
        "region": "North Kurdistan",
    },
    # ── East Kurdistan (ڕۆژهەڵات) ──
    {
        "name": "KHRN",
        "url": "https://kurdistanhumanrights.org/en/feed/",
        "category": "kurdistan",
        "region": "East Kurdistan",
    },
    {
        "name": "Google News - East Kurdistan",
        "url": (
            "https://news.google.com/rss/search?"
            "q=Rojhelat+OR+Sanandaj+OR+Mahabad+OR+Urmia+OR+Kermanshah+OR+Kurds+Iran"
            "&hl=en-US&gl=US&ceid=US:en"
        ),
        "category": "kurdistan",
        "region": "East Kurdistan",
    },
]


class KurdistanNewsSource(BaseSource):
    """Fetches news across all four parts of Kurdistan (South, West, North, East)."""

    def __init__(self) -> None:
        super().__init__(name="KurdistanRegions", source_type="kurdistan_rss", enabled=True)
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
        """Fetch articles across all four Kurdistan regional feed groups."""
        if not self._session:
            await self.initialize()

        assert self._session is not None
        tasks = [
            fetch_rss_feed(
                self._session,
                feed,
                max_entries=12,
                region=feed.get("region", ""),
            )
            for feed in KURDISTAN_FEEDS
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        articles: list[RawArticle] = []
        for i, res in enumerate(results):
            feed_name = KURDISTAN_FEEDS[i]["name"]
            if isinstance(res, Exception):
                logger.warning("Kurdistan feed '%s' failed: %s", feed_name, str(res)[:150])
                continue
            if isinstance(res, list):
                articles.extend(res)

        logger.info(
            "Kurdistan regional adapter discovered %d articles across %d feeds.",
            len(articles),
            len(KURDISTAN_FEEDS),
        )
        return articles
