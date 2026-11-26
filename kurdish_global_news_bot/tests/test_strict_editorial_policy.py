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
    can_admit_global_post,
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


# ============================================================
# PHASE 1 — DATABASE COUNTING PRECISION & ROLLING 24H TESTS
# ============================================================

class TestPhase1GlobalCountingAccuracy:
    """Verify unbounded query, UTC cutoff handling, and duplicate exclusion."""

    @pytest.mark.asyncio
    async def test_unbounded_counting_beyond_200_limit(self, tmp_path):
        """Articles beyond 200 rows must be accurately counted without query truncation."""
        db = Database(str(tmp_path / "unbounded.db"))
        await db.connect()
        try:
            now = datetime.now(timezone.utc)
            # Insert 220 published articles (210 ME + 10 Global)
            for i in range(1, 221):
                is_g = (i <= 10)
                art = _make_test_article(
                    i,
                    f"Headline {i}",
                    region="global" if is_g else "middle_east",
                    url=f"https://example.com/item/{i}",
                )
                art.status = ArticleStatus.PUBLISHED
                art.published_at = (now - timedelta(minutes=i)).isoformat()
                await db.insert_article(art)
                await db.update_article_status(i, ArticleStatus.PUBLISHED, published_at=art.published_at)

            g_count, t_count = await db.get_published_counts_last_24h()
            assert t_count == 220, f"Expected 220 total published, got {t_count}"
            assert g_count == 10, f"Expected 10 global published, got {g_count}"
            assert await db.get_global_posts_count_last_24h() == 10
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_published_at_used_strictly_never_discovered_at(self, tmp_path):
        """Only published_at determines 24h inclusion, never discovered_at."""
        db = Database(str(tmp_path / "pub_vs_disc.db"))
        await db.connect()
        try:
            now = datetime.now(timezone.utc)
            # Item A: discovered 1 hour ago, but published 30 hours ago -> MUST BE EXCLUDED
            art_a = _make_test_article(1, "Old published article", region="global", url="https://ex.com/a")
            art_a.status = ArticleStatus.PUBLISHED
            art_a.discovered_at = (now - timedelta(hours=1)).isoformat()
            art_a.published_at = (now - timedelta(hours=30)).isoformat()
            await db.insert_article(art_a)
            await db.update_article_status(1, ArticleStatus.PUBLISHED, published_at=art_a.published_at)

            # Item B: discovered 30 hours ago, but published 1 hour ago -> MUST BE INCLUDED
            art_b = _make_test_article(2, "Fresh published article", region="global", url="https://ex.com/b")
            art_b.status = ArticleStatus.PUBLISHED
            art_b.discovered_at = (now - timedelta(hours=30)).isoformat()
            art_b.published_at = (now - timedelta(hours=1)).isoformat()
            await db.insert_article(art_b)
            await db.update_article_status(2, ArticleStatus.PUBLISHED, published_at=art_b.published_at)

            g_count, t_count = await db.get_published_counts_last_24h()
            assert g_count == 1
            assert t_count == 1
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_duplicate_canonical_url_counted_once(self, tmp_path):
        """Articles sharing the same canonical URL are counted only once in 24h window."""
        db = Database(str(tmp_path / "dup_canonical.db"))
        await db.connect()
        try:
            now = datetime.now(timezone.utc)
            for i in (1, 2):
                art = _make_test_article(i, f"Duplicate story {i}", region="global", url="https://ex.com/same-canonical")
                art.status = ArticleStatus.PUBLISHED
                art.published_at = now.isoformat()
                await db.insert_article(art)
                await db.update_article_status(i, ArticleStatus.PUBLISHED, published_at=now.isoformat())

            g_count, t_count = await db.get_published_counts_last_24h()
            assert g_count == 1
            assert t_count == 1
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_fail_closed_on_database_error(self, tmp_path):
        """Database error fails closed to (MAX_GLOBAL_POSTS_PER_DAY=4, 0)."""
        db = Database(str(tmp_path / "error_fail_closed.db"))
        await db.connect()
        async with db.conn.execute("DROP TABLE IF EXISTS articles"):
            pass

        g_count, t_count = await db.get_published_counts_last_24h()
        assert g_count == 4
        assert t_count == 0
        await db.close()


# ============================================================
# PHASE 2 — CROSS-PROCESS CONCURRENCY & RESERVATIONS TESTS
# ============================================================

class TestPhase2CrossProcessReservations:
    """Verify system_logs backed reservations, TTL expiration, release, and commit."""

    @pytest.mark.asyncio
    async def test_reserve_release_and_commit_lifecycle(self, tmp_path):
        db = Database(str(tmp_path / "reservation_lifecycle.db"))
        await db.connect()
        try:
            url = "https://reuters.com/world/ukraine-standoff"
            # Initial active reservations: 0
            assert await db.get_active_global_reservations_count() == 0

            # Reserve slot
            success = await db.reserve_global_admission(url)
            assert success is True
            assert await db.get_active_global_reservations_count() == 1
            assert url in await db.get_active_global_reservations()

            # Release slot (e.g. publication failed or article dropped)
            await db.release_global_admission(url)
            assert await db.get_active_global_reservations_count() == 0

            # Re-reserve and commit upon publication
            assert await db.reserve_global_admission(url) is True
            assert await db.get_active_global_reservations_count() == 1
            await db.commit_global_admission(url)
            # Committed reservation is no longer an in-flight reservation
            assert await db.get_active_global_reservations_count() == 0
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_stale_reservation_ttl_expiry(self, tmp_path):
        """Reservations older than ttl_minutes are treated as expired."""
        db = Database(str(tmp_path / "stale_res.db"))
        await db.connect()
        try:
            url = "https://reuters.com/world/old-res"
            stale_dt = datetime.now(timezone.utc) - timedelta(minutes=20)
            await db.log_system_event(
                level="RESERVED",
                component="GLOBAL_RESERVATION",
                message=url,
                timestamp=stale_dt.isoformat(),
            )
            # With default 15-minute TTL, the 20-minute-old reservation has expired
            assert await db.get_active_global_reservations_count(ttl_minutes=15) == 0
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_concurrent_reservation_competition_at_daily_cap(self, tmp_path):
        """When only 1 global slot remains, concurrent reservations cannot both succeed."""
        db = Database(str(tmp_path / "concurrent_res.db"))
        await db.connect()
        try:
            now = datetime.now(timezone.utc)
            # Insert 3 published global articles in last 24h
            for i in range(1, 4):
                art = _make_test_article(i, f"Published global {i}", region="global")
                art.status = ArticleStatus.PUBLISHED
                art.published_at = now.isoformat()
                await db.insert_article(art)
                await db.update_article_status(i, ArticleStatus.PUBLISHED, published_at=now.isoformat())

            # Now exactly 1 global slot remains (4 - 3 = 1)
            async def try_reserve(url: str):
                return await db.reserve_global_admission(url)

            url1 = "https://example.com/global-candidate-1"
            url2 = "https://example.com/global-candidate-2"

            res1, res2 = await asyncio.gather(try_reserve(url1), try_reserve(url2))
            assert (res1, res2) in ((True, False), (False, True)), "Exactly one candidate must win the remaining slot"
            assert await db.get_active_global_reservations_count() == 1
        finally:
            await db.close()


# ============================================================
# PHASE 3 — ROLLING 24H 90/10 RATIO FORMULA TESTS
# ============================================================

class TestPhase3Rolling9010RatioFormula:
    """Verify strict adherence to 10 * (G + R + 1) <= (T + R + 1)."""

    def test_admission_formula_matrix(self):
        # 1. 0 published ME, 0 Global -> candidate 1 Global -> 10*1 <= 1 -> False
        assert can_admit_global_post(published_global_24h=0, total_published_24h=0) is False

        # 2. 9 published ME, 0 Global -> candidate 1 Global -> 10*1 <= 10 -> True
        assert can_admit_global_post(published_global_24h=0, total_published_24h=9) is True

        # 3. 9 published ME, 1 Global (total 10) -> candidate 1 Global -> 10*2 <= 11 -> False
        assert can_admit_global_post(published_global_24h=1, total_published_24h=10) is False

        # 4. 18 published ME, 1 Global (total 19) -> candidate 1 Global -> 10*2 <= 20 -> True
        assert can_admit_global_post(published_global_24h=1, total_published_24h=19) is True

        # 5. 4 Global already published -> candidate 1 Global -> exceeds MAX_GLOBAL_POSTS_PER_DAY -> False
        assert can_admit_global_post(published_global_24h=4, total_published_24h=50) is False

        # 6. Already selected 1 Global in current batch -> candidate 2nd Global -> False
        assert can_admit_global_post(published_global_24h=0, total_published_24h=20, batch_global_selected=1) is False

        # 7. In-flight reservation holds the slot -> candidate 2nd Global -> False
        assert can_admit_global_post(published_global_24h=0, total_published_24h=9, active_reservations=1) is False


# ============================================================
# PHASE 4 — EDITORIAL POLICY INTEGRITY TESTS
# ============================================================

class TestPhase4EditorialPolicyIntegrity:
    """Verify that local municipal fluff is rejected while state-level defense/politics passes."""

    @pytest.mark.parametrize(
        "fluff_title",
        [
            "Sulaymaniyah governor hosts constitutional federalism dialogue forum",
            "Duhok customs authorities expand cross-border trade customs screening",
            "Halabja provincial council inaugurates reconstruction funding committee",
        ],
    )
    def test_local_municipal_and_customs_administration_rejected(self, fluff_title: str):
        art = _make_test_article(1, fluff_title, region="South Kurdistan", category="kurdistan")
        reason = check_pre_filter_off_topic(art)
        assert reason is not None
        assert "insufficient_political_or_military_importance" in reason

    @pytest.mark.parametrize(
        "defense_title",
        [
            "Sulaymaniyah security forces launch joint counterterrorism operation along border",
            "Duhok border command intercepts armed militia drone targeting defense checkpoint",
            "Halabja leaders and Baghdad reach strategic border security agreement",
            "Erbil KRG cabinet approves civil servant salary distribution mechanism",
            "Peshmerga commanders inspect Kirkuk frontier defense fortifications",
        ],
    )
    def test_state_level_defense_and_governance_accepted(self, defense_title: str):
        art = _make_test_article(1, defense_title, region="South Kurdistan", category="kurdistan")
        reason = check_pre_filter_off_topic(art)
        assert reason is None, f"Expected {defense_title} to pass editorial filter"


# ============================================================
# PHASE 9 — DATABASE-AUTHORITATIVE GLOBAL LEASE LIFECYCLE
# & MULTIPROCESSING CONTENTION TESTS
# ============================================================


def _mp_sqlite_compete_worker(
    project_root: str,
    db_path: str,
    url: str,
    owner_token: str,
    barrier,
    out_queue,
) -> None:
    """Spawned child-process worker competing for SQLite Global admission under BEGIN IMMEDIATE."""
    import asyncio
    import sys

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from storage.database import Database

    async def _run() -> dict:
        db = Database(db_path)
        await db.connect()
        try:
            return await db.reserve_global_slot(
                canonical_url=url,
                owner_token=owner_token,
                lease_seconds=180,
            )
        finally:
            await db.close()

    barrier.wait(timeout=15)
    res = asyncio.run(_run())
    out_queue.put((owner_token, bool(res.get("admitted")), res.get("reason")))


class TestPhase9AuthoritativeGlobalLifecycle:
    """End-to-end verification of atomic leases, ambiguous quarantine, and multi-process contention."""

    @staticmethod
    async def _fetch_article_row(db: Database, article_id: int) -> tuple[str, Optional[int]]:
        async with db.conn.execute(
            "SELECT status, telegram_message_id FROM articles WHERE id = ?",
            (article_id,),
        ) as cur:
            row = await cur.fetchone()
        assert row is not None, f"Article {article_id} not found in DB"
        return str(row["status"]), row["telegram_message_id"]

    @pytest.mark.asyncio
    async def test_full_lifecycle_confirmed_publish_atomic_finalize(self, tmp_path):
        from core.pipeline import ArticlePipeline
        from monitoring.health import HealthMonitor
        from monitoring.metrics import Metrics

        db = Database(str(tmp_path / "phase9_commit.db"))
        await db.connect()
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            for i in range(1, 10):
                me = _make_test_article(i, f"Middle East defense update {i}", region="middle_east")
                me.status = ArticleStatus.PUBLISHED
                me.published_at = now_iso
                await db.insert_article(me)
                await db.update_article_status(i, ArticleStatus.PUBLISHED, published_at=now_iso, telegram_message_id=100 + i)

            g_art = _make_test_article(10, "NATO defense summit in Washington", region="global")
            g_art.translated_title = "کۆنگرەی بەرگری ناتۆ لە واشنتن بەڕێوەچوو"
            g_art.translated_summary = "سەرکردەکانی ناتۆ لە واشنتن کۆبوونەوە و بڕیاری بەرگرییان دا."
            g_art._require_global_reservation = True
            g_art.id = await db.insert_article(g_art)

            publisher = MagicMock()
            publisher.publish_article = AsyncMock(return_value=777)
            pipeline = ArticlePipeline(db=db, publisher=publisher, health=HealthMonitor(), metrics=Metrics())

            ok = await pipeline._publish_immediately(g_art)
            assert ok is True
            publisher.publish_article.assert_awaited_once()

            stored_status, stored_msg_id = await self._fetch_article_row(db, g_art.id)
            assert stored_status == ArticleStatus.PUBLISHED
            assert stored_msg_id == 777
            assert await db.get_active_global_reservations_count() == 0
            assert await db.get_published_counts_last_24h() == (1, 10)
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_definitive_telegram_failure_releases_reservation_for_reuse(self, tmp_path):
        from telegram.error import BadRequest
        from core.pipeline import ArticlePipeline
        from monitoring.health import HealthMonitor
        from monitoring.metrics import Metrics

        db = Database(str(tmp_path / "phase9_definitive_fail.db"))
        await db.connect()
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            for i in range(1, 10):
                me = _make_test_article(i, f"Baghdad border defense {i}", region="middle_east")
                me.status = ArticleStatus.PUBLISHED
                me.published_at = now_iso
                await db.insert_article(me)
                await db.update_article_status(i, ArticleStatus.PUBLISHED, published_at=now_iso, telegram_message_id=200 + i)

            g_art = _make_test_article(10, "Washington NATO missile shield plan", region="global")
            g_art.translated_title = "پلانی قەڵغانی مووشەکی ناتۆ لە واشنتن ڕاگەیاندرا"
            g_art.translated_summary = "بەرپرسانی ناتۆ پلانی نوێی بەرگری مووشەکیان ڕاگەیاند."
            g_art._require_global_reservation = True
            g_art.id = await db.insert_article(g_art)

            publisher = MagicMock()
            publisher.publish_article = AsyncMock(side_effect=BadRequest("Invalid caption entities"))
            pipeline = ArticlePipeline(db=db, publisher=publisher, health=HealthMonitor(), metrics=Metrics())

            ok = await pipeline._publish_immediately(g_art)
            assert ok is False

            stored_status, _ = await self._fetch_article_row(db, g_art.id)
            assert stored_status == ArticleStatus.RETRY_PENDING
            assert await db.get_active_global_reservations_count() == 0

            # Slot is free again for another valid Global candidate
            r2 = await db.reserve_global_slot("https://reuters.com/world/other-global", "worker:other")
            assert r2["admitted"] is True
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_ambiguous_timeout_quarantines_without_retry_or_card_fallback(self, tmp_path):
        from telegram.error import TimedOut
        from tg_bot.publisher import TelegramPublisher
        from core.pipeline import ArticlePipeline
        from monitoring.health import HealthMonitor
        from monitoring.metrics import Metrics

        g_sample = _make_test_article(1, "Global NATO pact", region="global")
        me_sample = _make_test_article(2, "Erbil Peshmerga briefing", region="kurdistan_bashur")
        assert TelegramPublisher._allow_ambiguous_network_retry(g_sample) is False
        assert TelegramPublisher._allow_ambiguous_network_retry(me_sample) is True

        # Video TimedOut for Global MUST set _telegram_outcome='ambiguous' and MUST NOT fall back to card
        dummy_vid = tmp_path / "global_clip.mp4"
        dummy_vid.write_bytes(b"00" * 64)
        real_pub = TelegramPublisher()
        real_pub._bot = MagicMock()
        real_pub._bot.send_video = AsyncMock(side_effect=TimedOut("Upload stalled after body sent"))
        real_pub._publish_with_card = AsyncMock(return_value=999)
        real_pub._send_with_optional_reply = AsyncMock(side_effect=TimedOut("Upload stalled after body sent"))
        g_vid = _make_test_article(99, "Pentagon Arctic exercise", region="global")
        g_vid.video_url = "https://video.twimg.com/ext_tw_video/99/pu/vid/720x1280/clip.mp4"
        g_vid._prepared_branded_video_path = str(dummy_vid)
        g_vid.translated_title = "ڕاهێنانی سەربازی پێنتاگۆن لە جەمسەری باکوور"
        g_vid.translated_summary = "هێزەکان ڕاهێنانی بەرگرییان ئەنجامدا."
        vid_res = await real_pub.publish_article(g_vid)
        assert vid_res is None
        assert getattr(g_vid, "_telegram_outcome", None) == "ambiguous"
        real_pub._publish_with_card.assert_not_awaited()

        db = Database(str(tmp_path / "phase9_ambiguous.db"))
        await db.connect()
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            for i in range(1, 10):
                me = _make_test_article(i, f"Regional event {i}", region="middle_east")
                me.status = ArticleStatus.PUBLISHED
                me.published_at = now_iso
                await db.insert_article(me)
                await db.update_article_status(i, ArticleStatus.PUBLISHED, published_at=now_iso, telegram_message_id=300 + i)

            g_art = _make_test_article(10, "Pacific naval exercise", region="global")
            g_art.translated_title = "ڕاهێنانی دەریایی لە زەریای هێمن ئەنجامدرا"
            g_art.translated_summary = "کەشتییە جەنگییەکان ڕاهێنانی هاوبەشیان ئەنجامدا."
            g_art._require_global_reservation = True
            g_art.id = await db.insert_article(g_art)

            publisher = MagicMock()
            publisher.publish_article = AsyncMock(side_effect=TimedOut("Telegram read timeout"))
            pipeline = ArticlePipeline(db=db, publisher=publisher, health=HealthMonitor(), metrics=Metrics())

            ok = await pipeline._publish_immediately(g_art)
            assert ok is False

            stored_status, _ = await self._fetch_article_row(db, g_art.id)
            assert stored_status == ArticleStatus.PUBLISHING

            quota = await db.get_global_quota_status()
            assert quota["active_reservations"] == 1
            assert quota["ambiguous_reservations"] == 1

            # Second Global must be blocked while AMBIGUOUS occupies the quota slot
            blocked = await db.reserve_global_slot("https://bbc.com/world/second-global", "worker:two")
            assert blocked["admitted"] is False

            # Middle East publishing continues normally
            me_next = _make_test_article(11, "Erbil defense announcement", region="kurdistan_bashur")
            me_next.translated_title = "وەزارەتی پێشمەرگە ئامادەیی بەرگری لە هەولێر ڕاگەیاند"
            me_next.translated_summary = "هێزەکانی پێشمەرگە سنوورەکان دەپارێزن."
            me_next.id = await db.insert_article(me_next)
            publisher.publish_article = AsyncMock(return_value=888)
            assert await pipeline._publish_immediately(me_next) is True
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_owner_mismatch_and_ambiguous_state_lock_guards(self, tmp_path):
        db = Database(str(tmp_path / "phase9_owner.db"))
        await db.connect()
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            for i in range(1, 10):
                me = _make_test_article(i, f"ME item {i}", region="middle_east")
                me.status = ArticleStatus.PUBLISHED
                me.published_at = now_iso
                await db.insert_article(me)
                await db.update_article_status(i, ArticleStatus.PUBLISHED, published_at=now_iso, telegram_message_id=400 + i)

            res = await db.reserve_global_slot("https://reuters.com/world/item-owner", "worker:owner-A")
            assert res["admitted"] is True
            rid = res["reservation_id"]

            assert (await db.renew_global_reservation(rid, "worker:owner-B"))["reason"] == "owner_mismatch"
            assert (await db.mark_global_publishing(rid, "worker:owner-B"))["reason"] == "owner_mismatch"
            assert (await db.release_global_reservation(rid, "worker:owner-B"))["reason"] == "owner_mismatch"
            assert (await db.finalize_global_publication(rid, "worker:owner-B", 1, 900))["reason"] == "owner_mismatch"

            assert (await db.mark_global_publishing(rid, "worker:owner-A"))["admitted"] is True
            amb = await db.mark_global_ambiguous(rid, "worker:owner-A", reason="timeout_mid_send")
            assert amb["state"] == "AMBIGUOUS"

            rel_locked = await db.release_global_reservation(rid, "worker:owner-A")
            assert rel_locked["ok"] is False
            assert rel_locked["reason"] == "cannot_release_ambiguous"
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_stale_publishing_transitions_to_ambiguous_and_recovery_quarantines(self, tmp_path):
        from core.recovery import RecoveryManager

        db = Database(str(tmp_path / "phase9_reconcile.db"))
        await db.connect()
        try:
            stale_ts = (datetime.now(timezone.utc) - timedelta(minutes=25)).isoformat()
            expired_ts = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()

            g_interrupted = _make_test_article(50, "Interrupted Global NATO item", region="global")
            g_interrupted.status = ArticleStatus.PUBLISHING
            me_interrupted = _make_test_article(51, "Interrupted Erbil item", region="kurdistan_bashur")
            me_interrupted.status = ArticleStatus.PUBLISHING
            g_id = await db.insert_article(g_interrupted)
            await db.update_article_status(g_id, ArticleStatus.PUBLISHING)
            me_id = await db.insert_article(me_interrupted)
            await db.update_article_status(me_id, ArticleStatus.PUBLISHING)

            await db.log_system_event(
                level="RESERVED",
                component="GLOBAL_RESERVATION",
                message=db._build_reservation_payload(
                    reservation_id="res-stale-unused",
                    owner_token="worker:dead-1",
                    canonical_url="https://reuters.com/world/stale-unused",
                    url_hash="h1",
                    article_id=None,
                    batch_id="b1",
                    state="RESERVED",
                    lease_seconds=180,
                    expires_at=datetime.fromisoformat(expired_ts),
                    reason="reserved",
                ),
                timestamp=stale_ts,
            )
            await db.log_system_event(
                level="PUBLISHING",
                component="GLOBAL_RESERVATION",
                message=db._build_reservation_payload(
                    reservation_id="res-stale-publishing",
                    owner_token="worker:dead-2",
                    canonical_url="https://reuters.com/world/stale-publishing",
                    url_hash="h2",
                    article_id=g_id,
                    batch_id="b2",
                    state="PUBLISHING",
                    lease_seconds=180,
                    expires_at=datetime.fromisoformat(expired_ts),
                    reason="publishing",
                ),
                timestamp=stale_ts,
            )

            recovery = RecoveryManager(db)
            await recovery.recover_interrupted_jobs()

            # Unused stale RESERVED -> EXPIRED; interrupted PUBLISHING -> AMBIGUOUS (still counts for quota!)
            quota = await db.get_global_quota_status()
            assert quota["active_reservations"] == 1
            assert quota["ambiguous_reservations"] == 1
            assert "https://reuters.com/world/stale-publishing" in quota["active_urls"]
            assert "https://reuters.com/world/stale-unused" not in quota["active_urls"]

            g_after_status, _ = await self._fetch_article_row(db, g_id)
            me_after_status, _ = await self._fetch_article_row(db, me_id)
            assert g_after_status == ArticleStatus.FAILED
            assert me_after_status == ArticleStatus.RETRY_PENDING
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_pre_publish_90_10_revalidation_defers_global_without_telegram_call(self, tmp_path):
        from core.pipeline import ArticlePipeline
        from monitoring.health import HealthMonitor
        from monitoring.metrics import Metrics

        db = Database(str(tmp_path / "phase9_prepublish_reval.db"))
        await db.connect()
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            for i in range(1, 10):
                me = _make_test_article(i, f"Baghdad update {i}", region="middle_east")
                me.status = ArticleStatus.PUBLISHED
                me.published_at = now_iso
                await db.insert_article(me)
                await db.update_article_status(i, ArticleStatus.PUBLISHED, published_at=now_iso, telegram_message_id=500 + i)

            # Insert candidate first so its DB ID is known and distinct
            g_cand = _make_test_article(10, "Deferred global summit", region="global")
            g_cand.canonical_url = "https://bbc.com/world/candidate-g"
            g_cand.original_url = "https://bbc.com/world/candidate-g"
            g_cand.translated_title = "لووتکەی نێودەوڵەتی دواخرا"
            g_cand.translated_summary = "سەرکردەکان لە لووتکەدا بەشداربوون."
            g_cand.id = await db.insert_article(g_cand)

            # Reserve Global slot when 9 ME published (allowed: 10*1 <= 10)
            res = await db.reserve_global_slot("https://bbc.com/world/candidate-g", "worker:main", article_id=g_cand.id)
            assert res["admitted"] is True
            g_cand._global_reservation_id = res["reservation_id"]
            g_cand._global_owner_token = "worker:main"
            g_cand._require_global_reservation = True

            # Concurrent worker/digest shifts rolling counts: insert 1 distinct published Global (G=1, T=10 -> 10*2 > 11)
            g_already = _make_test_article(99, "Earlier global headline", region="global")
            g_already.canonical_url = "https://reuters.com/world/earlier-g"
            g_already.original_url = "https://reuters.com/world/earlier-g"
            g_already.status = ArticleStatus.PUBLISHED
            g_already.published_at = now_iso
            g_already.id = await db.insert_article(g_already)
            await db.update_article_status(g_already.id, ArticleStatus.PUBLISHED, published_at=now_iso, telegram_message_id=599)

            publisher = MagicMock()
            publisher.publish_article = AsyncMock(return_value=999)
            pipeline = ArticlePipeline(db=db, publisher=publisher, health=HealthMonitor(), metrics=Metrics())

            ok = await pipeline._publish_immediately(g_cand)
            assert ok is False
            publisher.publish_article.assert_not_awaited()
            assert await db.get_active_global_reservations_count() == 0
            stored_status, _ = await self._fetch_article_row(db, g_cand.id)
            assert stored_status == ArticleStatus.QUEUED
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_sqlite_canonical_url_dedup_in_published_counts(self, tmp_path):
        db = Database(str(tmp_path / "phase9_url_dedup.db"))
        await db.connect()
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            a1 = _make_test_article(1, "Global NATO event", region="global")
            a1.canonical_url = "https://bbc.com/news/world-1?utm_source=fb"
            a1.original_url = a1.canonical_url
            a2 = _make_test_article(2, "Global NATO event copy", region="global")
            a2.canonical_url = "https://bbc.com/news/world-1/"
            a2.original_url = a2.canonical_url
            await db.insert_article(a1)
            await db.insert_article(a2)
            await db.update_article_status(a1.id, ArticleStatus.PUBLISHED, published_at=now_iso, telegram_message_id=601)
            await db.update_article_status(a2.id, ArticleStatus.PUBLISHED, published_at=now_iso, telegram_message_id=602)

            assert await db.get_published_counts_last_24h() == (1, 1)
        finally:
            await db.close()

    def test_real_sqlite_multiprocessing_two_processes_only_one_global_wins(self, tmp_path):
        import multiprocessing
        from pathlib import Path

        db_file = str(tmp_path / "phase9_mp_race.db")
        project_root = str(Path(__file__).resolve().parent.parent)

        async def _seed_9_me() -> None:
            db = Database(db_file)
            await db.connect()
            try:
                now_iso = datetime.now(timezone.utc).isoformat()
                for i in range(1, 10):
                    me = _make_test_article(i, f"Seeded ME security {i}", region="middle_east")
                    me.status = ArticleStatus.PUBLISHED
                    me.published_at = now_iso
                    await db.insert_article(me)
                    await db.update_article_status(i, ArticleStatus.PUBLISHED, published_at=now_iso, telegram_message_id=700 + i)
            finally:
                await db.close()

        asyncio.run(_seed_9_me())

        ctx = multiprocessing.get_context("spawn")
        barrier = ctx.Barrier(2)
        q = ctx.Queue()

        p1 = ctx.Process(
            target=_mp_sqlite_compete_worker,
            args=(project_root, db_file, "https://reuters.com/world/mp-race-1", "worker:proc-1", barrier, q),
        )
        p2 = ctx.Process(
            target=_mp_sqlite_compete_worker,
            args=(project_root, db_file, "https://reuters.com/world/mp-race-2", "worker:proc-2", barrier, q),
        )
        p1.start()
        p2.start()
        try:
            p1.join(timeout=20)
            p2.join(timeout=20)
            assert p1.exitcode == 0
            assert p2.exitcode == 0
        finally:
            for p in (p1, p2):
                if p.is_alive():
                    p.terminate()

        results = [q.get(timeout=5), q.get(timeout=5)]
        admitted_flags = [admitted for _, admitted, _ in results]
        assert sum(1 for x in admitted_flags if x) == 1, f"Expected exactly 1 MP winner, got {results}"

        async def _check_active() -> int:
            db = Database(db_file)
            await db.connect()
            try:
                return await db.get_active_global_reservations_count()
            finally:
                await db.close()

        assert asyncio.run(_check_active()) == 1

