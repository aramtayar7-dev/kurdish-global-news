"""
Kurdish Global News Bot — GDELT News Source

Discovers news articles from the GDELT Project's events database.
Uses the GDELT DOC API (public, no key required).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
from urllib.parse import quote_plus

import aiohttp

from sources.base import BaseSource, RawArticle
from config import REQUEST_TIMEOUT, USER_AGENT, is_blocked_western_source

logger = logging.getLogger("sources.gdelt")

# GDELT DOC 2.0 API endpoint (public access)
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"


class GDELTSource(BaseSource):
    """Fetches news articles from the GDELT DOC API."""

    def __init__(self, request_delay: float = 1.5) -> None:
        super().__init__(name="GDELT", source_type="gdelt", enabled=True)
        self._session: Optional[aiohttp.ClientSession] = None
        self._query_index = 0
        self._request_delay = request_delay
        # 60/40 weighted queries covering Kurdistan + Global Primary (60%) & Secondary (40%)
        self._queries = [
            # Primary Focus (~60%: Kurdistan 4 parts, Wars, Defense, World Leaders)
            "Kurdistan Erbil Sulaymaniyah Kirkuk",
            "Rojava Qamishli SDF Kurdish Syria",
            "Diyarbakir Amed Rojhelat Sanandaj Kurdish",
            "war armed conflict airstrike missile",
            "world leaders president prime minister UN NATO",
            "ceasefire geopolitical sanctions conflict",
            # Secondary Focus (~40%: AI, Cyber, Economy, Science)
            "artificial intelligence AI model breakthrough",
            "cyberattack cybersecurity ransomware breach",
            "global economy inflation financial crisis",
            "science space exploration natural disaster",
        ]

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
        """Fetch articles from GDELT DOC API using rotating queries."""
        if not self._session:
            await self.initialize()

        articles: list[RawArticle] = []

        # Use 2 queries per cycle
        batch_size = 2
        start = self._query_index
        queries = self._queries[start:start + batch_size]
        self._query_index = (start + batch_size) % len(self._queries)
        if not queries:
            queries = self._queries[:batch_size]

        for i, query in enumerate(queries):
            if i > 0 and self._request_delay > 0:
                await asyncio.sleep(self._request_delay)
            try:
                results = await self._search_gdelt(query)
                articles.extend(results)
            except Exception as e:
                logger.warning("GDELT query '%s' failed: %s", query, str(e)[:200])

        logger.info("GDELT discovered %d articles.", len(articles))
        return articles

    async def _search_gdelt(self, query: str) -> list[RawArticle]:
        """Search GDELT DOC API."""
        assert self._session is not None

        params = {
            "query": query,
            "mode": "ArtList",
            "maxrecords": "15",
            "format": "json",
            "sort": "DateDesc",
            "timespan": "60min",
        }

        try:
            async with self._session.get(GDELT_DOC_API, params=params) as response:
                if response.status != 200:
                    logger.warning("GDELT API returned status %d", response.status)
                    return []
                data = await response.json(content_type=None)
        except asyncio.TimeoutError:
            logger.warning("GDELT API timed out for query '%s'", query)
            return []
        except Exception as e:
            logger.warning("GDELT API error: %s", str(e)[:200])
            return []

        articles: list[RawArticle] = []
        article_list = data.get("articles", [])

        for item in article_list:
            raw = self._parse_article(item)
            if raw:
                articles.append(raw)

        return articles

    def _parse_article(self, item: dict) -> Optional[RawArticle]:
        """Parse a GDELT article record."""
        url = item.get("url", "").strip()
        title = item.get("title", "").strip()

        if not url or not title:
            return None

        if not url.startswith(("http://", "https://")):
            return None

        source_domain = item.get("domain", "GDELT").strip()
        if is_blocked_western_source(url=url, source_name=source_domain):
            return None

        # GDELT provides seendate in format YYYYMMDDTHHMMSS
        seen_date = item.get("seendate", "")
        pub_date = ""
        if seen_date:
            try:
                dt = datetime.strptime(seen_date, "%Y%m%dT%H%M%S")
                pub_date = dt.replace(tzinfo=timezone.utc).isoformat()
            except (ValueError, TypeError):
                pub_date = seen_date

        return RawArticle(
            url=url,
            title=title,
            summary=item.get("title", "")[:500],
            source_name=source_domain,
            category="",
            publication_date=pub_date,
            image_url=item.get("socialimage", ""),
            country=item.get("sourcecountry", ""),
        )
