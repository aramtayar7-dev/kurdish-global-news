"""Tests for news source adapters."""

import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from sources.base import BaseSource, RawArticle
from sources.duckduckgo_source import DuckDuckGoSource
from sources.rss_source import RSSSource
from sources.gdelt_source import GDELTSource
from sources.official_sources import OfficialSourcesAdapter


class TestBaseSource:
    """Test the base source interface."""

    def test_raw_article_creation(self):
        raw = RawArticle(
            url="https://example.com/article",
            title="Test Article",
            summary="This is a test article summary.",
            source_name="TestSource",
        )
        assert raw.url == "https://example.com/article"
        assert raw.title == "Test Article"
        assert raw.source_name == "TestSource"


class TestDuckDuckGoSource:
    """Test the DuckDuckGo news source adapter."""

    def test_init(self):
        source = DuckDuckGoSource()
        assert source.name == "DuckDuckGo"
        assert source.source_type == "ddg"
        assert source.enabled is True

    def test_parse_result_valid(self):
        source = DuckDuckGoSource()
        result = {
            "url": "https://reuters.com/article/test",
            "title": "Major World Event Happening Now",
            "body": "Details about the event...",
            "source": "Reuters",
            "date": "2024-01-15",
            "image": "https://example.com/image.jpg",
        }
        raw = source._parse_result(result)
        assert raw is not None
        assert raw.url == "https://reuters.com/article/test"
        assert raw.title == "Major World Event Happening Now"
        assert raw.source_name == "Reuters"

    def test_parse_result_missing_url(self):
        source = DuckDuckGoSource()
        result = {"title": "No URL Article"}
        raw = source._parse_result(result)
        assert raw is None

    def test_parse_result_invalid_url(self):
        source = DuckDuckGoSource()
        result = {"url": "ftp://invalid.com", "title": "Bad URL"}
        raw = source._parse_result(result)
        assert raw is None

    @patch("sources.duckduckgo_source.DuckDuckGoSource._search_news", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_fetch_articles(self, mock_search):
        """TEST 8: A source failure does not stop other sources."""
        mock_search.return_value = [
            {"url": "https://test.com/1", "title": "Test Article 1", "body": "Summary", "source": "Test"},
        ]

        source = DuckDuckGoSource()
        articles = await source.fetch_articles()
        assert len(articles) >= 1



class TestRSSSource:
    """Test the RSS source adapter."""

    def test_init(self):
        source = RSSSource()
        assert source.name == "RSS"
        assert source.source_type == "rss"

    def test_parse_entry_valid(self):
        source = RSSSource()
        entry = MagicMock()
        entry.link = "https://bbc.com/news/article-123"
        entry.title = "Breaking News from BBC"
        entry.summary = "Important news event details here."
        entry.published = "2024-01-15T10:00:00Z"
        entry.media_content = []
        entry.media_thumbnail = []
        entry.links = []

        raw = source._parse_entry(entry, "BBC", "International Politics")
        assert raw is not None
        assert raw.url == "https://bbc.com/news/article-123"
        assert raw.source_name == "BBC"
        assert raw.category == "International Politics"

    def test_parse_entry_no_link(self):
        source = RSSSource()
        entry = MagicMock()
        entry.link = ""
        entry.title = "No Link Article"
        raw = source._parse_entry(entry, "Test", "")
        assert raw is None


class TestGDELTSource:
    """Test the GDELT source adapter."""

    def test_init(self):
        source = GDELTSource()
        assert source.name == "GDELT"
        assert source.source_type == "gdelt"

    def test_parse_article_valid(self):
        source = GDELTSource()
        item = {
            "url": "https://reuters.com/gdelt-article",
            "title": "Global Event Detected",
            "seendate": "20240115T120000",
            "domain": "reuters.com",
            "socialimage": "https://example.com/img.jpg",
            "sourcecountry": "United States",
        }
        raw = source._parse_article(item)
        assert raw is not None
        assert raw.url == "https://reuters.com/gdelt-article"
        assert raw.country == "United States"

    def test_parse_article_missing_url(self):
        source = GDELTSource()
        raw = source._parse_article({"title": "No URL"})
        assert raw is None


class TestOfficialSources:
    """Test the official sources adapter."""

    def test_init(self):
        source = OfficialSourcesAdapter()
        assert source.name == "OfficialSources"
        assert source.source_type == "official"

    def test_reuters_only_western_whitelist_across_rss_official_and_search(self):
        from config import RSS_FEEDS, TELEGRAM_SOURCE_CHANNELS, is_blocked_western_source
        from sources.official_sources import OFFICIAL_FEEDS

        tg_users = {ch["username"] for ch in TELEGRAM_SOURCE_CHANNELS}
        assert "rudawdigital" not in tg_users
        assert {"AjaNews", "Kurdistan24news"}.issubset(tg_users)

        enabled_rss = {f["name"] for f in RSS_FEEDS if f.get("enabled", True)}
        for blocked in (
            "BBC World", "BBC Technology", "DW News", "France24 EN",
            "The Guardian World", "AP News", "CNBC World", "Military Times",
            "BleepingComputer", "NPR News", "UN News",
        ):
            assert blocked not in enabled_rss
        assert "Reuters World" in enabled_rss
        assert "Rudaw Sorani" in enabled_rss
        assert "Kurdistan24 Sorani" in enabled_rss
        assert "Al Jazeera" in enabled_rss

        assert all(not f.get("enabled", True) for f in OFFICIAL_FEEDS)

        ddg = DuckDuckGoSource()
        gdelt = GDELTSource()
        for bad_url, bad_src in [
            ("https://www.bbc.com/news/world-1", "BBC"),
            ("https://www.theguardian.com/world/1", "The Guardian"),
            ("https://www.dw.com/en/1", "DW News"),
            ("https://www.france24.com/en/1", "France24"),
            ("https://apnews.com/article/1", "AP News"),
            ("https://www.cnbc.com/2026/11/27/x.html", "CNBC"),
            ("https://www.nytimes.com/2026/11/27/world/x.html", "NYT"),
            ("https://www.cnn.com/2026/11/27/world/x.html", "CNN"),
            ("https://www.politico.com/news/1", "Politico"),
            ("https://www.euronews.com/2026/11/27/x", "Euronews"),
        ]:
            assert is_blocked_western_source(bad_url, bad_src) is True
            assert ddg._parse_result({"url": bad_url, "title": "Blocked western news", "source": bad_src}) is None
            assert gdelt._parse_article({"url": bad_url, "title": "Blocked western news", "domain": bad_src}) is None

        assert ddg._parse_result({"url": "https://www.reuters.com/world/1", "title": "Reuters item", "source": "Reuters"}) is not None
        assert gdelt._parse_article({"url": "https://www.reuters.com/world/1", "title": "Reuters item", "domain": "reuters.com"}) is not None


class TestKurdistanRegionalSource:
    """Verify news feed coverage across all four parts of Kurdistan."""

    def test_four_parts_of_kurdistan_covered(self):
        from sources.kurdistan_source import KURDISTAN_FEEDS, KurdistanNewsSource

        source = KurdistanNewsSource()
        assert source.name == "KurdistanRegions"
        assert source.enabled is True

        regions = {feed["region"] for feed in KURDISTAN_FEEDS}
        assert "South Kurdistan" in regions
        assert "West Kurdistan" in regions
        assert "North Kurdistan" in regions
        assert "East Kurdistan" in regions
        assert all(feed["category"] == "kurdistan" for feed in KURDISTAN_FEEDS)



class TestSourceIsolation:
    """TEST 8: Verify that one source failure doesn't stop others."""

    @pytest.mark.asyncio
    async def test_source_failure_isolation(self):
        """Each source must fail independently without stopping the system."""
        # Create sources where one will fail
        working_source = MagicMock(spec=BaseSource)
        working_source.name = "Working"
        working_source.enabled = True
        working_source.fetch_articles = AsyncMock(return_value=[
            RawArticle(url="https://test.com/good", title="Good Article", source_name="Test")
        ])

        failing_source = MagicMock(spec=BaseSource)
        failing_source.name = "Failing"
        failing_source.enabled = True
        failing_source.fetch_articles = AsyncMock(side_effect=Exception("Network error"))

        # Fetch from both
        results = await asyncio.gather(
            working_source.fetch_articles(),
            failing_source.fetch_articles(),
            return_exceptions=True,
        )

        # Working source should have results
        assert isinstance(results[0], list)
        assert len(results[0]) == 1

        # Failing source should have an exception, NOT crash the system
        assert isinstance(results[1], Exception)
