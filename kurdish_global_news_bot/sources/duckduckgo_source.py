"""
Kurdish Global News Bot — DuckDuckGo News Source

Discovers news articles via the DuckDuckGo search news API.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from sources.base import BaseSource, RawArticle
from config import SEARCH_KEYWORDS, REQUEST_TIMEOUT, is_blocked_western_source

logger = logging.getLogger("sources.duckduckgo")


class DuckDuckGoSource(BaseSource):
    """Fetches news articles using duckduckgo-search library."""

    def __init__(self, request_delay: float = 1.5) -> None:
        super().__init__(name="DuckDuckGo", source_type="ddg", enabled=True)
        self._keyword_index = 0
        self._request_delay = request_delay

    async def fetch_articles(self) -> list[RawArticle]:
        """Fetch news from DuckDuckGo using rotating keywords."""
        articles: list[RawArticle] = []

        # Use a subset of keywords per cycle to avoid rate limits
        batch_size = 3
        start = self._keyword_index
        keywords = SEARCH_KEYWORDS[start:start + batch_size]
        self._keyword_index = (start + batch_size) % len(SEARCH_KEYWORDS)

        if not keywords:
            keywords = SEARCH_KEYWORDS[:batch_size]

        for i, keyword in enumerate(keywords):
            if i > 0 and self._request_delay > 0:
                await asyncio.sleep(self._request_delay)
            try:
                results = await self._search_news(keyword)
                for result in results:
                    raw = self._parse_result(result)
                    if raw:
                        articles.append(raw)
            except Exception as e:
                logger.warning("DDG search failed for '%s': %s", keyword, str(e)[:200])
                continue

        logger.info("DDG discovered %d articles from %d keywords.", len(articles), len(keywords))
        return articles

    async def _search_news(self, keyword: str) -> list[dict]:
        """Run a DuckDuckGo news search in a thread to avoid blocking."""
        def _do_search() -> list[dict]:
            try:
                from duckduckgo_search import DDGS
                with DDGS(timeout=REQUEST_TIMEOUT) as ddgs:
                    results = list(ddgs.news(
                        keywords=keyword,
                        region="wt-wt",  # Worldwide
                        safesearch="off",
                        max_results=10,
                    ))
                    return results
            except Exception as e:
                logger.warning("DDG search error for '%s': %s", keyword, str(e)[:200])
                return []

        return await asyncio.to_thread(_do_search)

    def _parse_result(self, result: dict) -> Optional[RawArticle]:
        """Parse a DuckDuckGo news result into a RawArticle."""
        url = result.get("url", "").strip()
        title = result.get("title", "").strip()

        if not url or not title:
            return None

        # Basic URL validation
        if not url.startswith(("http://", "https://")):
            return None

        source_name = result.get("source", "DuckDuckGo").strip()
        if is_blocked_western_source(url=url, source_name=source_name):
            return None

        return RawArticle(
            url=url,
            title=title,
            summary=result.get("body", "").strip()[:1000],
            source_name=source_name,
            category="",  # Will be classified later
            publication_date=result.get("date", ""),
            image_url=result.get("image", ""),
        )
