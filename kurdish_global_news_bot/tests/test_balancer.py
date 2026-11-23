"""
Tests for 60/40 News Prioritization, Balancer, and Fallback Window.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from config import (
    PRIMARY_CATEGORY_RATIO,
    SECONDARY_CATEGORY_RATIO,
    FALLBACK_WINDOW_MINUTES,
    PRIMARY_CATEGORIES,
    SECONDARY_CATEGORIES,
    is_primary_category,
    SEARCH_KEYWORDS,
    RSS_FEEDS,
    ArticleStatus,
    Priority,
)
from storage.database import Database
from storage.models import Article
from core.importance_ranker import ImportanceRanker
from core.pipeline import ArticlePipeline
from tg_bot.publisher import TelegramPublisher
from monitoring.health import HealthMonitor
from monitoring.metrics import Metrics
from sources.gdelt_source import GDELTSource


@pytest.fixture
async def temp_db(tmp_path):
    """Provide an isolated temporary SQLite database for testing."""
    db_file = tmp_path / "test_balancer.db"
    db = Database(str(db_file))
    await db.connect()
    yield db
    await db.close()


class TestCategoryClassification:
    """Verify category classification and default ratio configurations."""

    def test_default_ratio_configs(self):
        assert PRIMARY_CATEGORY_RATIO == 0.60
        assert SECONDARY_CATEGORY_RATIO == 0.40
        assert FALLBACK_WINDOW_MINUTES == 20

    def test_primary_categories_identified(self):
        primary_examples = [
            "Wars & Armed Conflicts",
            "Military & Defense",
            "World Leaders & Governments",
            "International Politics",
            "Diplomacy & International Relations",
        ]
        for cat in primary_examples:
            assert is_primary_category(cat) is True

    def test_secondary_categories_identified(self):
        secondary_examples = [
            "AI & Technology",
            "Cybersecurity",
            "Global Economy & Financial Markets",
            "Natural Disasters & Emergencies",
            "Science & Research",
            "Space & Astronomy",
            "Health & Medical",
            "Climate & Environment",
        ]
        for cat in secondary_examples:
            assert is_primary_category(cat) is False
        assert is_primary_category(None) is False
        assert is_primary_category("") is False


class TestFeedAndKeywordDistributions:
    """Verify 60% Primary / 40% Secondary distribution across sources."""

    def test_search_keywords_ratio(self):
        assert len(SEARCH_KEYWORDS) == 20
        # First 12 are Primary (60%), last 8 are Secondary (40%)
        primary_kws = SEARCH_KEYWORDS[:12]
        secondary_kws = SEARCH_KEYWORDS[12:]
        assert len(primary_kws) == 12
        assert len(secondary_kws) == 8
        assert len(primary_kws) / len(SEARCH_KEYWORDS) == 0.60
        assert len(secondary_kws) / len(SEARCH_KEYWORDS) == 0.40

    def test_rss_feeds_ratio(self):
        assert len(RSS_FEEDS) == 20
        primary_feeds = [f for f in RSS_FEEDS if is_primary_category(f.get("category"))]
        secondary_feeds = [f for f in RSS_FEEDS if not is_primary_category(f.get("category"))]
        assert len(primary_feeds) == 12
        assert len(secondary_feeds) == 8
        assert len(primary_feeds) / len(RSS_FEEDS) == 0.60
        assert len(secondary_feeds) / len(RSS_FEEDS) == 0.40

    def test_gdelt_queries_ratio(self):
        source = GDELTSource()
        assert len(source._queries) == 10
        # 6 Primary (60%), 4 Secondary (40%)
        primary_queries = source._queries[:6]
        secondary_queries = source._queries[6:]
        assert len(primary_queries) == 6
        assert len(secondary_queries) == 4
        assert len(primary_queries) / len(source._queries) == 0.60
        for q in primary_queries:
            assert any(term in q for term in ("war", "military", "leader", "Security Council", "NATO", "ceasefire"))
        for q in secondary_queries:
            assert any(term in q for term in ("intelligence", "cyber", "economy", "disaster"))


class TestImportanceRankerBoosts:
    """Verify +0.20 Primary and +0.10 Secondary priority boosts in ImportanceRanker."""

    def test_primary_category_boost(self):
        ranker = ImportanceRanker()
        art_primary = Article(
            original_title="President signs new international agreement",
            original_summary="Details about the diplomatic signing between heads of state.",
            source_name="Reuters",
            category="World Leaders & Governments",
            publication_date=datetime.now(timezone.utc).isoformat(),
        )
        art_generic = Article(
            original_title="Local museum exhibition opens downtown",
            original_summary="Various paintings and artifacts displayed for the public.",
            source_name="Reuters",
            category="Other Significant Events",
            publication_date=datetime.now(timezone.utc).isoformat(),
        )

        boost_primary = ranker._calculate_focus_boost(art_primary)
        boost_generic = ranker._calculate_focus_boost(art_generic)

        assert boost_primary == 0.20
        assert boost_generic == 0.0

    def test_primary_keyword_boost_without_category(self):
        ranker = ImportanceRanker()
        art_keyword = Article(
            original_title="Airstrike hit military base during escalation",
            original_summary="Armed troops responded to drone attack across border.",
            source_name="Reuters",
            category="Other Significant Events",
            publication_date=datetime.now(timezone.utc).isoformat(),
        )
        boost = ranker._calculate_focus_boost(art_keyword)
        assert boost == 0.20

    def test_secondary_category_boost(self):
        ranker = ImportanceRanker()
        art_ai = Article(
            original_title="New neural model released with enhanced reasoning",
            original_summary="Researchers published findings on model architecture efficiency.",
            source_name="Reuters",
            category="AI & Technology",
            publication_date=datetime.now(timezone.utc).isoformat(),
        )
        boost_ai = ranker._calculate_focus_boost(art_ai)
        assert boost_ai == 0.10

    def test_secondary_keyword_boost(self):
        ranker = ImportanceRanker()
        art_cyber = Article(
            original_title="Hospital systems hit by ransomware attack",
            original_summary="Hackers demanded extortion payment after encrypting servers.",
            source_name="Reuters",
            category="Health & Medical",
            publication_date=datetime.now(timezone.utc).isoformat(),
        )
        boost_cyber = ranker._calculate_focus_boost(art_cyber)
        assert boost_cyber == 0.10

    def test_critical_breaking_news_from_any_category(self):
        ranker = ImportanceRanker()
        # Natural disaster with critical keywords (earthquake magnitude, emergency declared, killed)
        art_disaster = Article(
            original_title="BREAKING: Earthquake magnitude 7.8 killed hundreds, emergency declared",
            original_summary="Urgent rescue operations underway following massive tsunami warning.",
            source_name="Reuters",
            category="Natural Disasters & Emergencies",
            publication_date=datetime.now(timezone.utc).isoformat(),
        )
        score, priority = ranker.score_article(art_disaster)
        assert priority == Priority.CRITICAL


class TestBalancerPublishingLogic:
    """Verify soft rolling window balancer and automatic fallback logic."""

    @pytest.mark.asyncio
    async def test_critical_article_always_publishes_immediately(self, temp_db):
        pipeline = ArticlePipeline(temp_db, MagicMock(spec=TelegramPublisher), HealthMonitor(), Metrics())
        art = Article(
            category="Science & Research",
            priority=Priority.CRITICAL,
            original_title="Critical discovery",
        )
        publish_now, reason = await pipeline.should_publish_now(art)
        assert publish_now is True
        assert "CRITICAL" in reason

    @pytest.mark.asyncio
    async def test_primary_article_always_publishes_immediately(self, temp_db):
        pipeline = ArticlePipeline(temp_db, MagicMock(spec=TelegramPublisher), HealthMonitor(), Metrics())
        art = Article(
            category="Wars & Armed Conflicts",
            priority=Priority.HIGH,
            original_title="Peace treaty negotiations",
        )
        publish_now, reason = await pipeline.should_publish_now(art)
        assert publish_now is True
        assert "PRIMARY" in reason

    @pytest.mark.asyncio
    async def test_secondary_article_publishes_when_sample_small(self, temp_db):
        pipeline = ArticlePipeline(temp_db, MagicMock(spec=TelegramPublisher), HealthMonitor(), Metrics())
        art = Article(
            category="AI & Technology",
            priority=Priority.NORMAL,
            original_title="AI breakthrough",
        )
        # Empty DB (< 5 articles)
        publish_now, reason = await pipeline.should_publish_now(art)
        assert publish_now is True

    @pytest.mark.asyncio
    async def test_secondary_article_publishes_when_ratio_within_40_percent(self, temp_db):
        pipeline = ArticlePipeline(temp_db, MagicMock(spec=TelegramPublisher), HealthMonitor(), Metrics())
        now = datetime.now(timezone.utc)

        # Seed 10 published articles: 8 Primary (80%), 2 Secondary (20%)
        for i in range(8):
            art = Article(
                canonical_url=f"https://reuters.com/primary-{i}",
                original_url=f"https://reuters.com/primary-{i}",
                original_title=f"Primary News {i}",
                source_name="Reuters",
                category="Wars & Armed Conflicts",
                status=ArticleStatus.PUBLISHED,
                published_at=(now - timedelta(minutes=5 + i)).isoformat(),
            )
            await temp_db.insert_article(art)

        for i in range(2):
            art = Article(
                canonical_url=f"https://reuters.com/secondary-{i}",
                original_url=f"https://reuters.com/secondary-{i}",
                original_title=f"Secondary News {i}",
                source_name="Reuters",
                category="AI & Technology",
                status=ArticleStatus.PUBLISHED,
                published_at=(now - timedelta(minutes=2 + i)).isoformat(),
            )
            await temp_db.insert_article(art)

        # New secondary article should publish because 2/10 = 20% <= 40%
        new_secondary = Article(
            category="Cybersecurity",
            priority=Priority.NORMAL,
            original_title="New cybersecurity threat analyzed",
        )
        publish_now, reason = await pipeline.should_publish_now(new_secondary)
        assert publish_now is True
        assert "<=" in reason

    @pytest.mark.asyncio
    async def test_secondary_article_held_when_ratio_exceeded_and_primary_recent(self, temp_db):
        pipeline = ArticlePipeline(temp_db, MagicMock(spec=TelegramPublisher), HealthMonitor(), Metrics())
        now = datetime.now(timezone.utc)

        # Seed 10 published articles: 5 Primary (50%), 5 Secondary (50% > 40%)
        # Last primary was published 5 minutes ago (< FALLBACK_WINDOW_MINUTES of 20 min)
        for i in range(5):
            art = Article(
                canonical_url=f"https://reuters.com/primary-seed-{i}",
                original_url=f"https://reuters.com/primary-seed-{i}",
                original_title=f"Primary Article {i}",
                source_name="Reuters",
                category="Military & Defense",
                status=ArticleStatus.PUBLISHED,
                published_at=(now - timedelta(minutes=5 + i)).isoformat(),
            )
            await temp_db.insert_article(art)

        for i in range(5):
            art = Article(
                canonical_url=f"https://reuters.com/secondary-seed-{i}",
                original_url=f"https://reuters.com/secondary-seed-{i}",
                original_title=f"Secondary Article {i}",
                source_name="Reuters",
                category="AI & Technology",
                status=ArticleStatus.PUBLISHED,
                published_at=(now - timedelta(minutes=1 + i)).isoformat(),
            )
            await temp_db.insert_article(art)

        new_secondary = Article(
            category="Global Economy & Financial Markets",
            priority=Priority.NORMAL,
            original_title="Market inflation overview",
        )
        publish_now, reason = await pipeline.should_publish_now(new_secondary)
        assert publish_now is False
        assert "RATIO HELD" in reason

    @pytest.mark.asyncio
    async def test_automatic_fallback_window_relaxes_ratio_when_primary_quiet(self, temp_db):
        pipeline = ArticlePipeline(temp_db, MagicMock(spec=TelegramPublisher), HealthMonitor(), Metrics())
        now = datetime.now(timezone.utc)

        # Seed 10 published articles where secondary share is 80%, BUT last primary was 25 minutes ago (> 20 min fallback)
        art_primary = Article(
            canonical_url="https://reuters.com/old-primary",
            original_url="https://reuters.com/old-primary",
            original_title="Old Primary Article",
            source_name="Reuters",
            category="International Politics",
            status=ArticleStatus.PUBLISHED,
            published_at=(now - timedelta(minutes=25)).isoformat(),
        )
        await temp_db.insert_article(art_primary)

        for i in range(9):
            art = Article(
                canonical_url=f"https://reuters.com/secondary-burst-{i}",
                original_url=f"https://reuters.com/secondary-burst-{i}",
                original_title=f"Secondary Burst {i}",
                source_name="Reuters",
                category="AI & Technology",
                status=ArticleStatus.PUBLISHED,
                published_at=(now - timedelta(minutes=2 + i)).isoformat(),
            )
            await temp_db.insert_article(art)

        # Even though secondary share is 9/10 = 90%, fallback window activates because last primary was 25m ago!
        new_secondary = Article(
            category="Space & Astronomy",
            priority=Priority.NORMAL,
            original_title="Telescope discovers exoplanet",
        )
        publish_now, reason = await pipeline.should_publish_now(new_secondary)
        assert publish_now is True
        assert "FALLBACK ACTIVE" in reason

    @pytest.mark.asyncio
    async def test_database_helper_last_published_primary_time(self, temp_db):
        now = datetime.now(timezone.utc)
        primary_dt = now - timedelta(minutes=14)

        art1 = Article(
            canonical_url="https://reuters.com/db-test-primary",
            original_url="https://reuters.com/db-test-primary",
            original_title="UN Security Council Meeting",
            source_name="Reuters",
            category="Diplomacy & International Relations",
            status=ArticleStatus.PUBLISHED,
            published_at=primary_dt.isoformat(),
        )
        await temp_db.insert_article(art1)

        art2 = Article(
            canonical_url="https://reuters.com/db-test-secondary",
            original_url="https://reuters.com/db-test-secondary",
            original_title="AI Chip Announcement",
            source_name="Reuters",
            category="AI & Technology",
            status=ArticleStatus.PUBLISHED,
            published_at=now.isoformat(),
        )
        await temp_db.insert_article(art2)

        last_primary = await temp_db.get_last_published_primary_time(PRIMARY_CATEGORIES)
        assert last_primary is not None
        # Difference should be under 1 second
        diff = abs((last_primary - primary_dt).total_seconds())
        assert diff < 1.0
