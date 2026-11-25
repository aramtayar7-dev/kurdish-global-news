"""
Kurdish Global News Bot — Master Engineering Specification Tests
Strict Editorial Intelligence, Military & High-Level Politics Filtering,
and 90/10 Middle East-Global Geographic Control.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import config
from config import (
    ALLOWED_EDITORIAL_CATEGORIES,
    DISABLED_EDITORIAL_CATEGORIES,
    MAX_GLOBAL_POSTS_PER_DAY,
    MAX_GLOBAL_POSTS_PER_BATCH,
    MIDDLE_EAST_TARGET_RATIO,
    GLOBAL_MAX_RATIO,
    REGION_KURDISTAN_BASHUR,
    REGION_MIDDLE_EAST,
    REGION_GLOBAL,
    ArticleStatus,
)
from core.balancer import (
    BalanceSelectionResult,
    QuotaBalancer,
    clear_global_selection_reservations,
    get_source_bucket,
    select_paced_batch,
)
from core.importance_ranker import (
    ImportanceRanker,
    RelevanceEvaluation,
    check_pre_filter_off_topic,
    classify_geographic_region,
    is_bashur_article,
    is_genuine_kurdistan_event,
    is_strictly_military_or_political,
)
from processing.validator import ArticleValidator
from storage.database import Database
from storage.models import Article
from storage.supabase_db import SupabaseDatabase


def _make_test_article(
    id: int,
    title: str,
    summary: str = "",
    category: str = "politics_world",
    source_name: str = "Reuters",
    region: str = "",
    source_type: str = "search",
    score: float = 8.5,
    url: str = "",
) -> Article:
    art = Article(
        id=id,
        original_title=title,
        original_summary=summary or title,
        translated_title=title,
        translated_summary=summary or title,
        category=category,
        source_name=source_name,
        original_url=url or f"https://example.com/article/{id}",
        canonical_url=url or f"https://example.com/article/{id}",
        publication_date=datetime.now(timezone.utc).isoformat(),
        discovered_at=datetime.now(timezone.utc).isoformat(),
        importance_score=score,
        status=ArticleStatus.DISCOVERED,
        region=region,
    )
    setattr(art, "source_type", source_type)
    return art


# ============================================================
# 6.1 BANNED TOPIC TESTS
# ============================================================

class TestBannedTopics:
    """Articles on banned topics must return False deterministically with zero LLM calls."""

    BANNED_HEADLINES = [
        "Billionaire signs $10B Lukoil oil deal",
        "Ransomware cyberattack hits bank",
        "New AI model launched",
        "OpenAI releases a new model",
        "Stock markets fall following inflation report",
        "Football tournament begins",
        "New smartphone announced",
        "Oil prices increase",
    ]

    @pytest.mark.parametrize("headline", BANNED_HEADLINES)
    def test_banned_topics_deterministic_rejection(self, headline: str):
        art = _make_test_article(1, headline)
        is_ok, reason = is_strictly_military_or_political(art)
        assert is_ok is False, f"Expected '{headline}' to be rejected, but passed."
        assert len(reason) > 0

        # Pre-filter off topic check
        pre_rejected = check_pre_filter_off_topic(art)
        assert pre_rejected is not None
        assert pre_rejected != ""

        # Validator check
        validator = ArticleValidator()
        v_ok, v_reason = validator.validate_editorial(art)
        assert v_ok is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("headline", BANNED_HEADLINES)
    async def test_banned_topics_zero_llm_calls(self, headline: str):
        art = _make_test_article(1, headline)
        mock_gemini = AsyncMock()
        ranker = ImportanceRanker(api_key="fake-test-key")
        ranker._gemini = mock_gemini

        eval_res = await ranker.evaluate_global_relevance(art)
        assert eval_res.passed is False
        assert mock_gemini.generate_json.await_count == 0

        batch_evals = await ranker.evaluate_batch_relevance([art])
        assert len(batch_evals) == 1
        assert batch_evals[0].passed is False
        assert mock_gemini.generate_json.await_count == 0


# ============================================================
# 6.2 PROTOCOL / FLUFF TESTS
# ============================================================

class TestProtocolFluff:
    """Protocol, ceremonial, courtesy visits, and PR fluff must be deterministically rejected."""

    FLUFF_HEADLINES = [
        "Annual cultural festival opens in Erbil",
        "Barzani Charity Foundation met with a European delegation",
        "Company welcomes foreign delegation",
        "Routine ambassador courtesy visit",
        "Official sends congratulatory message",
        "کۆبوونەوەی دامەزراوەیەک لەگەڵ شاندێکی بیانی",
        "Annual exhibition opens in Baghdad",
        "Routine diplomatic condolence message",
    ]

    @pytest.mark.parametrize("headline", FLUFF_HEADLINES)
    def test_protocol_fluff_deterministic_rejection(self, headline: str):
        art = _make_test_article(1, headline)
        is_ok, reason = is_strictly_military_or_political(art)
        assert is_ok is False, f"Expected fluff '{headline}' to be rejected, but passed."
        assert "fluff" in reason.lower() or "protocol" in reason.lower() or "ceremonial" in reason.lower() or "insufficient" in reason.lower()


# ============================================================
# 6.3 ALLOWED MILITARY TESTS
# ============================================================

class TestAllowedMilitary:
    """Substantive military and security events must be accepted."""

    ALLOWED_MILITARY_STORIES = [
        ("Military airstrikes hit strategic command center", "Air force launched precision airstrikes"),
        ("Drone attacks target military airbase", "Hostile explosive drone attacks strike runway facilities"),
        ("Ballistic missile attacks strike border fortification", "Heavy missile attacks reported by defense command"),
        ("Major military clashes erupt along border demarcation", "Armed troops in intense military clashes"),
        ("Troop mobilization ordered across regional defense perimeter", "Military command initiates troop mobilization"),
        ("Cross-border military operations launched against armed militant headquarters", "Army conducts cross-border operations"),
        ("Armed factions agree to ceasefire agreement and immediate weapons surrender", "Historic ceasefire signed after peace talks"),
        ("Major military withdrawals begin from fortified conflict sector", "Ground force military withdrawal underway"),
        ("Significant national-security operations neutralize armed sabotage cell", "Counter-terrorism security operation launched"),
    ]

    @pytest.mark.parametrize("title, summary", ALLOWED_MILITARY_STORIES)
    def test_substantive_military_accepted(self, title: str, summary: str):
        art = _make_test_article(1, title, summary=summary, category="military_conflict")
        is_ok, reason = is_strictly_military_or_political(art)
        assert is_ok is True, f"Expected '{title}' to be accepted, but got: {reason}"


# ============================================================
# 6.4 ALLOWED HIGH-LEVEL POLITICS TESTS
# ============================================================

class TestAllowedHighLevelPolitics:
    """Substantive high-level political developments must be accepted, while routine fluff is rejected."""

    ALLOWED_POLITICAL_STORIES = [
        ("Government crisis erupts as ruling coalition collapses", "Prime minister steps down amidst political crisis"),
        ("Parliament passes constitutional decision to dissolve assembly", "Parliamentary decision triggers snap elections"),
        ("Foreign ministry shifts foreign-policy stance and cuts diplomatic ties", "Major foreign policy redirection announced"),
        ("UN Security Council imposes sanctions targeting military officials", "New geopolitical sanctions enacted"),
        ("State-level defense treaty ratified between allied nations", "Presidents sign mutual state treaty on security"),
        ("Diplomatic crisis escalates after ambassador expelled", "Envoy recalled during acute diplomatic crisis"),
        ("Election crisis intensifies as supreme court invalidates presidential vote", "Constitutional deadlock in election crisis"),
        ("KRG and Baghdad reach landmark budget agreement resolving federal disputes", "Prime ministers sign historic KRG Baghdad revenue pact"),
    ]

    @pytest.mark.parametrize("title, summary", ALLOWED_POLITICAL_STORIES)
    def test_substantive_politics_accepted(self, title: str, summary: str):
        art = _make_test_article(1, title, summary=summary, category="politics_world")
        is_ok, reason = is_strictly_military_or_political(art)
        assert is_ok is True, f"Expected '{title}' to be accepted, but got: {reason}"


# ============================================================
# 6.5 GEOGRAPHIC CLASSIFICATION TESTS
# ============================================================

class TestGeographicClassification:
    """Test geographic classification into kurdistan_bashur, middle_east, and global."""

    CASES = [
        ("US airstrikes hit militia targets in Iraq", "middle_east"),
        ("EU announces new sanctions targeting Iran", "middle_east"),
        ("Pentagon discusses Erbil security", "middle_east"),
        ("Ukraine-Russia battlefield clashes", "global"),
        ("US domestic political crisis", "global"),
        ("China-Taiwan military confrontation", "global"),
        ("Major Peshmerga security development", "kurdistan_bashur"),
        ("Syrian military developments in Tartous", "middle_east"),
    ]

    @pytest.mark.parametrize("headline, expected_region", CASES)
    def test_geographic_regions(self, headline: str, expected_region: str):
        art = _make_test_article(1, headline)
        classified = classify_geographic_region(art)
        assert classified == expected_region, f"For '{headline}', expected '{expected_region}', got '{classified}'"

    def test_unrelated_syrian_and_iraqi_stories_not_tagged_as_kurdistan(self):
        tartous_art = _make_test_article(1, "Syrian military developments in Tartous", source_name="North Press Agency")
        assert classify_geographic_region(tartous_art) == "middle_east"
        assert is_bashur_article(tartous_art) is False
        assert is_genuine_kurdistan_event(tartous_art) is False

        baghdad_art = _make_test_article(2, "Baghdad parliament passes civil service law", source_name="Shafaq News")
        assert classify_geographic_region(baghdad_art) == "middle_east"
        assert is_bashur_article(baghdad_art) is False


# ============================================================
# 6.6 GLOBAL CAP TESTS
# ============================================================

class TestGlobalCap:
    """Verify strict daily and batch Global caps (max 1 Global per batch, max 4 per day)."""

    def setup_method(self):
        clear_global_selection_reservations()

    def test_scenario_a_zero_published_selects_max_1_global(self):
        """TEST A: global_posts_last_24h = 0, 5 Global + 5 Middle East candidates -> max 1 Global, <= 5 total."""
        global_cands = [
            _make_test_article(i, f"Ukraine-Russia conflict update {i}", region="global", source_type="x", score=9.0 - i * 0.1)
            for i in range(1, 6)
        ]
        me_cands = [
            _make_test_article(10 + i, f"Middle East security operation {i}", region="middle_east", source_type="telegram", score=8.5 - i * 0.1)
            for i in range(1, 6)
        ]

        result = select_paced_batch(
            candidates=global_cands + me_cands,
            target_posts=5,
            global_posts_last_24h=0,
        )

        assert len(result.selected) <= 5
        assert result.global_selected <= 1
        assert result.middle_east_selected >= 4
        assert result.global_remaining_today >= 3

    def test_scenario_b_four_published_allows_zero_global(self):
        """TEST B: global_posts_last_24h = 4, 5 Global + 5 Middle East candidates -> exactly 0 Global."""
        global_cands = [
            _make_test_article(i, f"Ukraine-Russia conflict update {i}", region="global", source_type="x", score=9.0)
            for i in range(1, 6)
        ]
        me_cands = [
            _make_test_article(10 + i, f"Middle East security operation {i}", region="middle_east", source_type="telegram", score=8.5)
            for i in range(1, 6)
        ]

        result = select_paced_batch(
            candidates=global_cands + me_cands,
            target_posts=5,
            global_posts_last_24h=4,
        )

        assert result.global_selected == 0
        assert all(classify_geographic_region(a) != "global" for a in result.selected)
        assert len(result.selected) == 5

    def test_scenario_c_three_published_allows_max_1_global(self):
        """TEST C: global_posts_last_24h = 3 -> exactly 1 global remaining today, max 1 in batch."""
        global_cands = [
            _make_test_article(i, f"China-Taiwan military standoff {i}", region="global", source_type="x", score=9.0)
            for i in range(1, 6)
        ]
        me_cands = [
            _make_test_article(10 + i, f"Syrian security forces deploy in Homs {i}", region="middle_east", source_type="search", score=8.5)
            for i in range(1, 6)
        ]

        result = select_paced_batch(
            candidates=global_cands + me_cands,
            target_posts=5,
            global_posts_last_24h=3,
        )

        assert result.global_selected <= 1
        assert result.global_remaining_today <= 1

    def test_scenario_d_five_published_allows_zero_global(self):
        """TEST D: global_posts_last_24h = 5 (over daily limit) -> 0 Global."""
        global_cands = [
            _make_test_article(i, f"Global political update {i}", region="global", source_type="x", score=9.5)
            for i in range(1, 6)
        ]
        me_cands = [
            _make_test_article(10 + i, f"Iraq security forces clash with militants {i}", region="middle_east", source_type="telegram", score=8.0)
            for i in range(1, 6)
        ]

        result = select_paced_batch(
            candidates=global_cands + me_cands,
            target_posts=5,
            global_posts_last_24h=5,
        )

        assert result.global_selected == 0

    def test_scenario_e_no_me_candidates_does_not_backfill_global(self):
        """TEST E: Only Global candidates available -> max 1 Global selected, never backfills to 5."""
        global_cands = [
            _make_test_article(i, f"Ukraine battlefield frontline report {i}", region="global", source_type="search", score=9.0)
            for i in range(1, 6)
        ]

        result = select_paced_batch(
            candidates=global_cands,
            target_posts=5,
            global_posts_last_24h=0,
        )

        # Quality > Quantity: never backfill regional slots with Global news!
        assert result.global_selected == 1
        assert len(result.selected) == 1

    @pytest.mark.asyncio
    async def test_scenario_f_database_count_failure_fails_closed(self, tmp_path):
        """TEST F: DB count failure fails closed (returns MAX_GLOBAL_POSTS_PER_DAY=4 -> 0 Global allowed)."""
        db = Database(str(tmp_path / "fail_closed.db"))
        await db.connect()
        # Drop articles table to force database query error
        async with db.conn.execute("DROP TABLE IF EXISTS articles"):
            pass

        count = await db.get_global_posts_count_last_24h()
        assert count == 4, "DB error must fail-closed to 4 (MAX_GLOBAL_POSTS_PER_DAY)"

        # SupabaseDatabase fail-closed check
        sdb = SupabaseDatabase(url="https://fake.supabase.co", key="fake-key")
        sdb._request = AsyncMock(side_effect=Exception("Connection refused"))
        s_count = await sdb.get_global_posts_count_last_24h()
        assert s_count == 4

        await db.close()

    @pytest.mark.asyncio
    async def test_scenario_g_concurrent_batch_selection_safety(self):
        """TEST G: Two concurrent batch selections cannot exceed the remaining daily Global allowance."""
        clear_global_selection_reservations()

        global_cands = [
            _make_test_article(i, f"Ukraine-Russia conflict report {i}", region="global", source_type="x", score=9.0)
            for i in range(1, 4)
        ]
        me_cands = [
            _make_test_article(10 + i, f"Erbil-Baghdad security pact {i}", region="kurdistan_bashur", source_type="telegram", score=8.5)
            for i in range(1, 6)
        ]

        # Only 1 global post remains today (3 already published)
        async def run_selection():
            return select_paced_batch(
                candidates=global_cands + me_cands,
                target_posts=5,
                global_posts_last_24h=3,
            )

        res1, res2 = await asyncio.gather(run_selection(), run_selection())
        total_global = res1.global_selected + res2.global_selected
        assert total_global <= 1, f"Expected total concurrent global posts <= 1, got {total_global}"


# ============================================================
# 6.7 PRE-LLM ZERO-CALL TESTS
# ============================================================

class TestPreLLMZeroCalls:
    """Verify that mocked Gemini and OpenRouter are NEVER called for non-military/non-political items."""

    @pytest.mark.asyncio
    async def test_zero_llm_calls_for_all_banned_categories(self):
        banned_samples = [
            "Billionaire signs $10B Lukoil oil deal",
            "Ransomware cyberattack hits bank",
            "New AI model launched",
            "Football tournament begins",
            "Annual cultural festival opens in Erbil",
            "Routine ambassador courtesy visit",
            "Company announces annual financial earnings",
        ]

        mock_gemini = AsyncMock()
        mock_gemini.has_keys = True
        mock_gemini.all_keys_in_cooldown = MagicMock(return_value=False)

        ranker = ImportanceRanker(api_key="test-key")
        ranker._gemini = mock_gemini

        articles = [_make_test_article(i, title) for i, title in enumerate(banned_samples, start=1)]

        # Batch evaluation
        results = await ranker.evaluate_batch_relevance(articles)
        assert len(results) == len(banned_samples)
        assert all(r.passed is False for r in results)
        assert mock_gemini.generate_json.await_count == 0

        # Single evaluation
        for art in articles:
            single_res = await ranker.evaluate_global_relevance(art)
            assert single_res.passed is False
        assert mock_gemini.generate_json.await_count == 0


# ============================================================
# 6.8 REGRESSION & DATABASE METRICS TESTS
# ============================================================

class TestRegressionAndDatabaseMetrics:
    """Verify database count query against actual published records."""

    @pytest.mark.asyncio
    async def test_sqlite_global_posts_count_last_24h(self, tmp_path):
        db = Database(str(tmp_path / "test_geo.db"))
        await db.connect()
        try:
            now = datetime.now(timezone.utc)
            # Insert 2 published Global articles in last 24h
            for i in range(1, 3):
                art = _make_test_article(i, f"Ukraine clash {i}", region="global")
                art.status = ArticleStatus.PUBLISHED
                art.published_at = now.isoformat()
                await db.insert_article(art)
                await db.update_article_status(i, ArticleStatus.PUBLISHED, published_at=now.isoformat())

            # Insert 1 published Middle East article in last 24h
            me_art = _make_test_article(3, "Peshmerga security", region="kurdistan_bashur")
            me_art.status = ArticleStatus.PUBLISHED
            me_art.published_at = now.isoformat()
            await db.insert_article(me_art)
            await db.update_article_status(3, ArticleStatus.PUBLISHED, published_at=now.isoformat())

            # Insert 1 published Global article older than 24h
            old_art = _make_test_article(4, "Old global story", region="global")
            old_art.status = ArticleStatus.PUBLISHED
            old_pub = (now - timedelta(hours=30)).isoformat()
            old_art.published_at = old_pub
            await db.insert_article(old_art)
            await db.update_article_status(4, ArticleStatus.PUBLISHED, published_at=old_pub)

            count = await db.get_global_posts_count_last_24h()
            assert count == 2
        finally:
            await db.close()
