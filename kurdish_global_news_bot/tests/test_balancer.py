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
    BATCH_PUBLISH_LIMIT,
    BASHUR_QUOTA_PER_8,
    GLOBAL_QUOTA_PER_8,
    BASHUR_TARGET_RATIO,
    GLOBAL_TARGET_RATIO,
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
from core.balancer import QuotaBalancer
from core.importance_ranker import ImportanceRanker, RelevanceEvaluation, is_bashur_article
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
            assert any(
                term in q
                for term in (
                    "Kurdistan", "Rojava", "Diyarbakir", "Rojhelat",
                    "war", "military", "leader", "Security Council", "NATO", "ceasefire",
                )
            )
        for q in secondary_queries:
            assert any(term in q for term in ("intelligence", "cyber", "economy", "science", "disaster"))


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


def _make_quota_article(
    idx: int,
    title: str,
    summary: str = "",
    category: str = "politics_world",
    region: str = "",
    source_name: str = "Reuters",
    score: float = 8.0,
    is_breaking: bool = False,
) -> Article:
    url = f"https://example.com/quota-{idx}"
    art = Article(
        id=idx,
        original_url=url,
        canonical_url=url,
        original_title=title,
        normalized_title=Article.normalize_title(title),
        original_summary=summary or (
            f"{title}. Comprehensive international dispatch covering geopolitical developments, "
            "government announcements, and strategic regional impacts across major capitals."
        ),
        source_name=source_name,
        category=category,
        region=region,
        importance_score=score,
        is_breaking=is_breaking,
        publication_date=datetime.now(timezone.utc).isoformat(),
    )
    art.compute_content_hash()
    art.compute_event_fingerprint()
    return art


class TestFiveOfEightBashurQuotaBalancer:
    """Verify 5-of-8 Southern Kurdistan quota, Priority 1A/1B/2 ordering, and elastic global backfill."""

    def test_quota_ratio_constants(self):
        assert BATCH_PUBLISH_LIMIT == 8
        assert BASHUR_QUOTA_PER_8 == 5
        assert GLOBAL_QUOTA_PER_8 == 3
        assert BASHUR_TARGET_RATIO == 0.625
        assert GLOBAL_TARGET_RATIO == 0.375

    def test_scenario_1_full_8plus_bashur_and_global_picks_5_bashur_3_global_ordered(self):
        balancer = QuotaBalancer(batch_limit=8, bashur_per_8=5)

        # 4 Bashur 1B (salaries/oil/local, high raw score 8.9) + 5 Bashur 1A (politics & war, score 8.0-8.5)
        bashur_1b = [
            _make_quota_article(
                100 + i,
                title=f"KRG announces Sulaymaniyah electricity & salary schedule {i}",
                category="kurdistan",
                region="South Kurdistan",
                source_name="Kurdistan24 EN",
                score=8.9,
            )
            for i in range(4)
        ]
        bashur_1a = [
            _make_quota_article(
                200 + i,
                title=f"Erbil and Baghdad hold political talks on Peshmerga security {i}",
                category="kurdistan",
                region="South Kurdistan",
                source_name="Kurdistan24 EN",
                score=8.2,
            )
            for i in range(5)
        ]
        # Global mix: 4 non-political (tech/lifestyle, high score 8.7) + 4 regional politics/war (score 7.5-8.5)
        global_non_pol = [
            _make_quota_article(
                300 + i,
                title=f"New consumer smartphone chip benchmark released {i}",
                category="ai_tech",
                score=8.7,
            )
            for i in range(4)
        ]
        global_pol_war = [
            _make_quota_article(
                400 + i,
                title=f"Iran and US diplomats clash over Middle East military strike {i}",
                category="military_conflict",
                score=8.1,
            )
            for i in range(4)
        ]

        candidates = global_non_pol + bashur_1b + global_pol_war + bashur_1a
        res = balancer.select_batch(candidates, limit=8)

        assert len(res.selected) == 8
        assert res.bashur_selected == 5
        assert res.global_selected == 3

        # First 5 published MUST be Southern Kurdistan (Bashur) and all 5 must be Priority 1A Politics & War!
        selected_bashur = res.selected[:5]
        selected_global = res.selected[5:]
        assert all(is_bashur_article(a) for a in selected_bashur)
        assert all(a.id >= 200 and a.id < 300 for a in selected_bashur)
        # Routine Bashur political stories stay non-breaking
        assert all(a.is_breaking is False for a in selected_bashur)

        # Last 3 published MUST be Global Politics/War ahead of consumer tech
        assert all(not is_bashur_article(a) for a in selected_global)
        assert all(a.id >= 400 for a in selected_global)

    def test_scenario_2_only_2_bashur_backfills_to_2_bashur_and_6_global(self):
        balancer = QuotaBalancer(batch_limit=8, bashur_per_8=5)
        bashur = [
            _make_quota_article(1, "Erbil parliament convenes urgent political session", category="kurdistan", region="South Kurdistan", score=8.5),
            _make_quota_article(2, "Peshmerga intercepts hostile drone near Kirkuk", category="kurdistan", region="South Kurdistan", score=8.8),
        ]
        globals_list = [
            _make_quota_article(10 + i, f"Middle East ceasefire diplomatic talks {i}", category="politics_world", score=8.5 - i * 0.1)
            for i in range(8)
        ]

        res = balancer.select_batch(globals_list + bashur, limit=8)
        assert len(res.selected) == 8
        assert res.bashur_selected == 2
        assert res.global_selected == 6
        assert all(is_bashur_article(a) for a in res.selected[:2])
        assert all(not is_bashur_article(a) for a in res.selected[2:])

    def test_scenario_3_only_1_bashur_backfills_to_1_bashur_and_7_global(self):
        balancer = QuotaBalancer(batch_limit=8, bashur_per_8=5)
        bashur = [
            _make_quota_article(1, "Duhok governor discusses KRG infrastructure budget", category="kurdistan", region="South Kurdistan", score=8.2),
        ]
        globals_list = [
            _make_quota_article(10 + i, f"Iran missile escalation triggers NATO emergency summit {i}", category="military_conflict", score=8.6 - i * 0.1)
            for i in range(9)
        ]

        res = balancer.select_batch(globals_list + bashur, limit=8)
        assert len(res.selected) == 8
        assert res.bashur_selected == 1
        assert res.global_selected == 7
        assert is_bashur_article(res.selected[0])
        assert all(not is_bashur_article(a) for a in res.selected[1:])

    def test_scenario_4_zero_bashur_picks_8_top_global(self):
        balancer = QuotaBalancer(batch_limit=8, bashur_per_8=5)
        globals_list = [
            _make_quota_article(10 + i, f"Turkey and Syria border military buildup update {i}", category="military_conflict", score=8.9 - i * 0.1)
            for i in range(10)
        ]

        res = balancer.select_batch(globals_list, limit=8)
        assert len(res.selected) == 8
        assert res.bashur_selected == 0
        assert res.global_selected == 8
        assert all(not is_bashur_article(a) for a in res.selected)

    def test_rolling_8_window_when_batch_limit_differs_from_8_and_breaking_ordered_first(self):
        balancer = QuotaBalancer(batch_limit=8, bashur_per_8=5)
        # Recent 4 published posts were all Bashur -> over rolling window of 8 (4 recent + 4 batch),
        # only 1 Bashur budget remains (5 - 4 = 1), leaving 3 for Global when limit=4.
        recent_4_bashur = [
            _make_quota_article(i, f"Erbil political summit part {i}", category="kurdistan", region="South Kurdistan")
            for i in range(4)
        ]
        candidates = [
            _make_quota_article(10, "Routine KRG diplomacy update", category="kurdistan", region="South Kurdistan", score=8.5, is_breaking=False),
            _make_quota_article(11, "Halabja political statement", category="kurdistan", region="South Kurdistan", score=8.4, is_breaking=False),
            _make_quota_article(20, "Major Tehran refinery missile explosion", category="military_conflict", score=9.5, is_breaking=True),
            _make_quota_article(21, "Baghdad parliament votes on defense bill", category="politics_world", score=8.2, is_breaking=False),
            _make_quota_article(22, "US envoy arrives in Ankara for NATO talks", category="politics_world", score=8.0, is_breaking=False),
        ]

        res = balancer.select_batch(candidates, limit=4, recent_published=recent_4_bashur)
        assert len(res.selected) == 4
        assert res.bashur_selected == 1
        assert res.global_selected == 3
        # Genuine breaking global emergency publishes FIRST ahead of Bashur non-breaking
        assert res.selected[0].id == 20
        assert res.selected[0].is_breaking is True
        assert is_bashur_article(res.selected[1])
        assert res.selected[1].is_breaking is False

    @pytest.mark.asyncio
    async def test_pipeline_process_batch_dedup_prevents_same_bashur_event_consuming_two_slots(self, temp_db):
        """
        Verify process_batch runs cross-source dedup BEFORE slot allocation so two Kurdish
        sources reporting the same Bashur event only consume 1 of the 5 Bashur slots,
        and routine Bashur politics publish non-breaking in Bashur-first order.
        """
        publisher = MagicMock(spec=TelegramPublisher)
        published_titles: list[tuple[str, bool]] = []

        async def fake_publish(art: Article) -> int:
            published_titles.append((art.original_title, art.is_breaking))
            return 1000 + len(published_titles)

        publisher.publish_article = AsyncMock(side_effect=fake_publish)
        pipeline = ArticlePipeline(temp_db, publisher, HealthMonitor(), Metrics())

        summary_bashur = (
            "Detailed official report from Erbil confirmed high-level talks between Kurdish leaders "
            "and Iraqi federal representatives over constitutional governance and regional security coordination."
        )
        summary_global = (
            "Foreign ministers of Iran and Turkey met in Ankara to discuss border stability, "
            "diplomatic corridors, regional security treaties, and shifting alliances across the broader Middle East."
        )
        # Two duplicate Bashur outlets covering the same Erbil political talks event
        dup1 = Article(
            original_url="https://kurdistan24.net/erbil-event-a",
            canonical_url="https://kurdistan24.net/erbil-event-a",
            original_title="Erbil coalition political talks open between KRG and Baghdad",
            normalized_title=Article.normalize_title("Erbil coalition political talks open between KRG and Baghdad"),
            original_summary=summary_bashur,
            source_name="Kurdistan24 EN",
            category="kurdistan",
            region="South Kurdistan",
            publication_date=datetime.now(timezone.utc).isoformat(),
        )
        dup2 = Article(
            original_url="https://pukmedia.com/erbil-event-b",
            canonical_url="https://pukmedia.com/erbil-event-b",
            original_title="KRG and Baghdad open coalition political talks in Erbil",
            normalized_title=Article.normalize_title("KRG and Baghdad open coalition political talks in Erbil"),
            original_summary=summary_bashur,
            source_name="PUKMedia EN",
            category="kurdistan",
            region="South Kurdistan",
            publication_date=datetime.now(timezone.utc).isoformat(),
        )
        global_story = Article(
            original_url="https://reuters.com/world/tehran-summit",
            canonical_url="https://reuters.com/world/tehran-summit",
            original_title="Iran and Turkey foreign ministers meet over regional security crisis",
            normalized_title=Article.normalize_title("Iran and Turkey foreign ministers meet over regional security crisis"),
            original_summary=summary_global,
            source_name="Reuters",
            category="politics_world",
            publication_date=datetime.now(timezone.utc).isoformat(),
        )

        for a in (dup1, dup2, global_story):
            a.compute_content_hash()
            a.compute_event_fingerprint()
            a.id = await temp_db.insert_article(a)

        async def mock_relevance(art: Article) -> RelevanceEvaluation:
            if "Erbil" in art.original_title or "KRG" in art.original_title:
                return RelevanceEvaluation(score=8, is_breaking=False, category="kurdistan", reason="Bashur politics", passed=True)
            return RelevanceEvaluation(score=8, is_breaking=False, category="politics_world", reason="Global politics", passed=True)

        # Ensure Stage B never hits real network if Stage A passes
        pipeline._semantic_dedup._gemini._api_keys = []

        with patch.object(pipeline._ranker, "evaluate_global_relevance", side_effect=mock_relevance), \
             patch.object(pipeline._extractor, "fetch_article_content_and_image", new=AsyncMock(return_value=(None, None))), \
             patch.object(pipeline._translator, "translate_article", side_effect=lambda title, summary="": (f"کوردی: {title}", f"پوختە: {summary}")):
            res = await pipeline.process_batch([global_story, dup1, dup2], max_publish=8)

        # Duplicate Bashur event eliminated -> only 1 Bashur + 1 Global published (never 2 Bashur)
        assert res["published"] == 2
        assert len(published_titles) == 2
        # Bashur Politics published FIRST, Global SECOND, and routine politics NOT breaking
        assert "Kurdistan" in published_titles[0][0] or "Erbil" in published_titles[0][0] or "KRG" in published_titles[0][0]
        assert published_titles[0][1] is False
        assert "Iran and Turkey" in published_titles[1][0]
