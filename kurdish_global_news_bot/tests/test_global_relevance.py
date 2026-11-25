"""
Unit tests for Gemini Global Importance & Category Gatekeeper,
four-part Kurdistan prioritization, and SQLite skipped-URL hash persistence.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import MIN_GLOBAL_RELEVANCE_SCORE, ArticleStatus, Priority
from core.engine import NewsEngine
from core.importance_ranker import (
    GLOBAL_RELEVANCE_SYSTEM_PROMPT,
    PRIORITY_TIER_1A,
    PRIORITY_TIER_1B,
    PRIORITY_TIER_2,
    PRIORITY_TIER_3,
    ImportanceRanker,
    RelevanceEvaluation,
    check_blocked_topic,
    classify_priority_tier,
    is_bashur_article,
    is_genuine_kurdistan_event,
)
from core.pipeline import ArticlePipeline
from monitoring.health import HealthMonitor
from monitoring.metrics import Metrics
from processing.content_formatter import ContentFormatter
from processing.gemini_client import GEMINI_API_BASE_URL
from processing.translator import (
    GEMINI_KURDISH_REWRITE_SYSTEM_PROMPT,
    GEMINI_TRANSLATION_SYSTEM_PROMPT,
    strip_news_dateline,
)
from processing.importance_ranker import (
    ImportanceRanker as ProcessingImportanceRanker,
    RelevanceEvaluation as ProcessingRelevanceEvaluation,
)
from sources.base import RawArticle
from storage.database import Database
from tg_bot.publisher import TelegramPublisher


class _MockResponseContext:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _make_gemini_session(status: int, json_payload: dict | None = None, text_data: str = ""):
    mock_resp = AsyncMock()
    mock_resp.status = status
    if json_payload is not None:
        mock_resp.json = AsyncMock(return_value={
            "candidates": [{"content": {"parts": [{"text": json.dumps(json_payload)}]}}]
        })
    mock_resp.text = AsyncMock(return_value=text_data)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=_MockResponseContext(mock_resp))
    mock_session.closed = False
    return mock_session


def Article_factory(
    title: str,
    summary: str,
    source_name: str = "Reuters",
    category: str = "politics_world",
    url: str = "https://reuters.com/world/sample-story",
    region: str = "",
):
    from storage.models import Article

    art = Article(
        original_url=url,
        canonical_url=Article.canonicalize_url(url),
        original_title=title,
        original_summary=summary,
        source_name=source_name,
        category=category,
        region=region,
        publication_date=datetime.now(timezone.utc).isoformat(),
    )
    art.compute_content_hash()
    art.compute_event_fingerprint()
    return art


class TestGeminiRelevanceConfigAndExports:
    def test_default_threshold(self):
        assert MIN_GLOBAL_RELEVANCE_SCORE == 7

    def test_processing_reexports(self):
        assert ProcessingImportanceRanker is ImportanceRanker
        assert ProcessingRelevanceEvaluation is RelevanceEvaluation


class TestGeminiRelevanceAndCategoryClassifier:
    """Verify Gemini classification JSON, strict 7 categories, 4-part Kurdistan boost, and fallback."""

    @pytest.mark.asyncio
    async def test_kurdistan_news_four_parts_prioritized(self):
        for region, title, summary in [
            ("South Kurdistan", "Erbil and Baghdad Reach Agreement on KRG Civil Servant Salaries", "Kurdistan Regional Government delegation confirmed budget disbursement."),
            ("West Kurdistan", "SDF Reinforces Defense Positions Around Qamishli and Kobani", "Developments in Rojava and North and East Syria amid regional tension."),
            ("North Kurdistan", "Large Municipal Summit Convenes in Diyarbakir Over Kurdish Language Rights", "Elected mayors across Amed and Van discussed local governance."),
            ("East Kurdistan", "Human Rights Monitors Report Protests in Sanandaj and Mahabad", "Developments in Rojhelat across western Iran monitored closely."),
        ]:
            ranker = ImportanceRanker(api_key="gem_test_key", min_relevance_score=7)
            mock_session = _make_gemini_session(
                200,
                json_payload={
                    "score": 9,
                    "is_breaking": False,
                    "category": "kurdistan",
                    "reason": f"Major development in {region}.",
                },
            )
            ranker._gemini._get_session = AsyncMock(return_value=mock_session)

            art = Article_factory(title=title, summary=summary, source_name="Kurdistan24 EN", region=region)
            result = await ranker.evaluate_global_relevance(art)

            assert result.score >= 8
            assert result.category == "kurdistan"
            assert result.passed is True

            call_args, call_kwargs = mock_session.post.call_args
            assert GEMINI_API_BASE_URL in call_args[0]
            assert call_kwargs["json"]["generationConfig"]["temperature"] == 0.0
            assert GLOBAL_RELEVANCE_SYSTEM_PROMPT == call_kwargs["json"]["systemInstruction"]["parts"][0]["text"]

    @pytest.mark.asyncio
    async def test_al_jazeera_nvidia_stock_classified_economy_not_war(self):
        """Under strict editorial policy, Nvidia stock / earnings stories are rejected before Gemini as banned economy."""
        ranker = ImportanceRanker(api_key="gem_test_key", min_relevance_score=7)
        mock_session = _make_gemini_session(
            200,
            json_payload={
                "score": 8,
                "is_breaking": False,
                "category": "economy_markets",
                "reason": "Global semiconductor market movement led by Nvidia earnings.",
            },
        )
        ranker._gemini._get_session = AsyncMock(return_value=mock_session)

        art = Article_factory(
            title="Nvidia Stock Surges Following Record Quarterly Semiconductor Demand",
            summary="Global chipmaker reported steep earnings beat driving technology indexes higher.",
            source_name="Al Jazeera",
            category="politics_world",
        )
        result = await ranker.evaluate_global_relevance(art)

        # Under strict editorial policy (Section 1.1 & 6.8), economy/stock articles are rejected with 0 LLM calls
        assert result.passed is False
        assert "banned_economy" in result.reason or "insufficient" in result.reason

    @pytest.mark.asyncio
    async def test_hyperlocal_foreign_lawsuit_rejected(self):
        ranker = ImportanceRanker(api_key="gem_test_key", min_relevance_score=7)
        mock_session = _make_gemini_session(
            200,
            json_payload={
                "score": 2,
                "is_breaking": False,
                "category": "politics_world",
                "reason": "Hyper-local US state licensing dispute with no global significance.",
            },
        )
        ranker._gemini._get_session = AsyncMock(return_value=mock_session)

        art = Article_factory(
            title="Oregon Plumbing Licensing Board Faces Lawsuit Over Contractor Permit Fees",
            summary="Two local contractors filed suit in Marion County court disputing renewal fees.",
            source_name="Local News Oregon",
        )
        result = await ranker.evaluate_global_relevance(art)

        assert result.score == 2
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_gemini_key_rotation_and_fail_open_fallback(self):
        ranker = ImportanceRanker(api_key="gem_k1,gem_k2", min_relevance_score=7, base_backoff=0.0)

        resp_429 = AsyncMock(status=429, text=AsyncMock(return_value="Rate limited"))
        resp_200 = AsyncMock(
            status=200,
            json=AsyncMock(return_value={
                "candidates": [{"content": {"parts": [{"text": json.dumps({
                    "score": 9, "is_breaking": True, "category": "military_conflict",
                    "reason": "Regional airstrikes escalation."
                })}]}}]
            }),
        )
        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=[
            _MockResponseContext(resp_429),
            _MockResponseContext(resp_200),
        ])
        ranker._gemini._get_session = AsyncMock(return_value=mock_session)

        art = Article_factory(
            title="Coalition Conducts Precision Strikes Against Military Targets",
            summary="Defense ministry confirmed targeted overnight operations.",
            source_name="Reuters",
        )
        res = await ranker.evaluate_global_relevance(art)
        assert res.passed is True
        assert res.category == "military_conflict"
        assert res.is_breaking is True

        # Fail-open check on API crash
        broken_session = MagicMock()
        broken_session.post = MagicMock(side_effect=RuntimeError("Network failure"))
        ranker._gemini._get_session = AsyncMock(return_value=broken_session)

        fallback_res = await ranker.evaluate_global_relevance(art)
        assert fallback_res.used_fallback is True
        assert fallback_res.passed is True
        assert "Rule-based fallback" in fallback_res.reason

    @pytest.mark.asyncio
    async def test_kurdistan_boost_and_interviews_never_flip_breaking_true(self):
        """High-priority Kurdistan political statements/interviews must never become breaking."""
        ranker = ImportanceRanker(api_key="gem_test_key", min_relevance_score=7, base_backoff=0.0)

        # Prompt explicitly distinguishes high importance from breaking news
        assert "is_breaking" in GLOBAL_RELEVANCE_SYSTEM_PROMPT
        assert "interviews" in GLOBAL_RELEVANCE_SYSTEM_PROMPT.lower()
        assert "political statements" in GLOBAL_RELEVANCE_SYSTEM_PROMPT.lower()

        # Case 1: Score bumped to 8 for Kurdistan + model erroneously said is_breaking=True on score=8
        session_score8 = _make_gemini_session(
            200,
            json_payload={
                "score": 8,
                "is_breaking": True,
                "category": "kurdistan",
                "reason": "High-priority Kurdistan parliamentary remarks.",
            },
        )
        ranker._gemini._get_session = AsyncMock(return_value=session_score8)
        interview = Article_factory(
            title="KRG Official Gives Extensive TV Interview on Erbil Budget Talks",
            summary="Senior party leader spoke in a broadcast interview outlining negotiation priorities.",
            source_name="Rudaw",
            region="South Kurdistan",
        )
        res_gemini = await ranker.evaluate_global_relevance(interview)
        assert res_gemini.score == 8
        assert res_gemini.passed is True
        assert res_gemini.is_breaking is False

        # Case 2: Fallback without API key must NOT flip is_breaking due to Kurdistan boost
        ranker_no_key = ImportanceRanker(api_key="", min_relevance_score=7)
        res_fallback = await ranker_no_key.evaluate_global_relevance(interview)
        assert res_fallback.used_fallback is True
        assert res_fallback.category == "kurdistan"
        assert res_fallback.score >= 8
        assert res_fallback.is_breaking is False

    def test_bashur_politics_war_1a_vs_1b_vs_global_tiers(self):
        """Verify 1A Bashur Politics/War > 1B Bashur Economy/Salaries > 2 Global Politics/War > 3 Tech."""
        bashur_war = Article_factory(
            title="Drone strike targets security compound near Erbil airport",
            summary="Peshmerga and regional authorities opened an urgent investigation into the military attack.",
            category="kurdistan",
            region="South Kurdistan",
        )
        bashur_salary = Article_factory(
            title="KRG finance ministry begins monthly public salary distribution",
            summary="Officials in Sulaymaniyah and Duhok confirmed civil servant payroll transfers started.",
            category="kurdistan",
            region="South Kurdistan",
        )
        west_kurdistan = Article_factory(
            title="SDF reinforces defensive positions in Qamishli countryside",
            summary="Rojava authorities monitored deployments along the northern Syrian corridor.",
            category="kurdistan",
            region="West Kurdistan",
        )
        global_pol = Article_factory(
            title="Iraq and Iran officials hold diplomatic border crisis summit",
            summary="Baghdad hosted high-level foreign ministry talks.",
            category="politics_world",
        )
        global_tech = Article_factory(
            title="New consumer graphics processor released by chipmaker",
            summary="Benchmark tests showed faster gaming performance.",
            category="ai_tech",
        )

        assert is_bashur_article(bashur_war) is True
        assert is_bashur_article(bashur_salary) is True
        assert is_bashur_article(west_kurdistan) is False
        assert is_bashur_article(global_pol) is False

        assert classify_priority_tier(bashur_war) == PRIORITY_TIER_1A
        assert classify_priority_tier(bashur_salary) == PRIORITY_TIER_1B
        assert classify_priority_tier(global_pol) == PRIORITY_TIER_2
        assert classify_priority_tier(global_tech) == PRIORITY_TIER_3


class TestPipelineAndSkippedUrlPersistence:
    @pytest.mark.asyncio
    async def test_pipeline_rejects_local_lawsuit_and_persists_hash(self, tmp_path, caplog):
        db = Database(str(tmp_path / "test_rel.db"))
        await db.connect()

        publisher = MagicMock(spec=TelegramPublisher)
        publisher.publish_article = AsyncMock(return_value=9999)
        pipeline = ArticlePipeline(db, publisher, HealthMonitor(), Metrics())

        pipeline._ranker.evaluate_global_relevance = AsyncMock(
            return_value=RelevanceEvaluation(
                score=2,
                is_breaking=False,
                category="politics_world",
                reason="Routine US state contractor licensing lawsuit with no global impact.",
                passed=False,
                used_fallback=False,
            )
        )
        pipeline._extractor.fetch_article_content_and_image = AsyncMock()
        pipeline._translator.translate_article = AsyncMock()

        art = Article_factory(
            url="https://oregonlive.example.com/local-licensing-lawsuit?utm_source=rss",
            title="Oregon Contractors Sue State Board Over Routine Permit Licensing Delay",
            summary="A local civil lawsuit was filed in state court over municipal contractor license renewals.",
            source_name="Reuters",
        )
        art.id = await db.insert_article(art)

        with caplog.at_level(logging.INFO, logger="core.pipeline"):
            published = await pipeline.process_article(art)

        assert published is False
        pipeline._extractor.fetch_article_content_and_image.assert_not_called()
        pipeline._translator.translate_article.assert_not_called()
        publisher.publish_article.assert_not_called()

        assert any(
            "SKIPPED [Low Global Relevance 2/10]: Oregon Contractors Sue State Board" in rec.message
            for rec in caplog.records
        )

        db_art = await db.get_article_by_canonical_url(art.canonical_url)
        assert db_art.status == ArticleStatus.SKIPPED
        assert await db.is_url_skipped(art.original_url) is True
        assert await db.is_url_skipped("https://oregonlive.example.com/local-licensing-lawsuit?ref=twitter") is True

        # Second discovery run skips before inserting
        engine = NewsEngine()
        engine._db = db
        raw = RawArticle(
            title=art.original_title,
            url="https://oregonlive.example.com/local-licensing-lawsuit?fbclid=12345",
            summary=art.original_summary,
            source_name="Reuters",
            publication_date=art.publication_date or "",
        )
        assert await engine._extract_and_insert([raw]) == []
        await db.close()

    @pytest.mark.asyncio
    async def test_pipeline_overrides_source_category_with_gemini_category(self, tmp_path):
        """Ensure an Al Jazeera stock article gets economy_markets stored in DB rather than politics_world."""
        db = Database(str(tmp_path / "test_cat_override.db"))
        await db.connect()

        publisher = MagicMock(spec=TelegramPublisher)
        publisher.publish_article = AsyncMock(return_value=7777)
        pipeline = ArticlePipeline(db, publisher, HealthMonitor(), Metrics())

        pipeline._ranker.evaluate_global_relevance = AsyncMock(
            return_value=RelevanceEvaluation(
                score=8,
                is_breaking=False,
                category="economy_markets",
                reason="Global market surge.",
                passed=True,
                used_fallback=False,
            )
        )
        pipeline._extractor.fetch_article_content_and_image = AsyncMock(return_value=(
            "Global equity markets rallied strongly after Nvidia posted record quarterly revenue exceeding analyst expectations. "
            "Data center chip demand jumped across North America, Europe, and Asia, driving semiconductor shares upward.",
            "https://example.com/nvidia.jpg",
        ))
        pipeline._translator.translate_article = AsyncMock(
            return_value=("پشکەکانی کۆمپانیای ئێنڤیدیا بەرزبوونەوەی پێوانەییان تۆمارکرد", "بازاڕە داراییەکانی جیهان بەرزبوونەوەی بەرچاویان بەخۆوە بینی.")
        )

        art = Article_factory(
            url="https://aljazeera.com/economy/nvidia-surge",
            title="Nvidia Stock Surges on Record Chip Revenue",
            summary="Strong chip demand pushed quarterly revenue above forecasts.",
            source_name="Al Jazeera",
            category="politics_world",
        )
        art.id = await db.insert_article(art)

        published = await pipeline.process_article(art)
        assert published is True

        db_art = await db.get_article_by_canonical_url(art.canonical_url)
        assert db_art.status == ArticleStatus.PUBLISHED
        assert db_art.category == "economy_markets"
        assert db_art.image_url == "https://example.com/nvidia.jpg"
        await db.close()

    @pytest.mark.asyncio
    async def test_north_press_houston_fbi_isis_court_case_rejected_and_not_kurdistan(self, tmp_path):
        """
        Regression: North Press article with Qamishli dateline reporting a US federal court
        sentencing a Houston man for ISIS propaganda / lying to the FBI must be rejected
        (eligible=False, skipped_urls recorded) and NEVER classified as Kurdistan news.
        """
        db = Database(str(tmp_path / "test_houston_fbi.db"))
        await db.connect()

        publisher = MagicMock(spec=TelegramPublisher)
        publisher.publish_article = AsyncMock(return_value=1111)
        pipeline = ArticlePipeline(db, publisher, HealthMonitor(), Metrics())

        art = Article_factory(
            url="https://npasyria.com/en/houston-fbi-isis-sentence",
            title="Houston Man Sentenced to 15 Years for Distributing ISIS Propaganda and Lying to FBI",
            summary=(
                "Qamishli, Syria (North Press) – A U.S. federal court in Houston sentenced a Texas man "
                "to 15 years in prison on Friday after he pleaded guilty to distributing ISIS propaganda "
                "and lying to the FBI during a domestic counterterrorism investigation."
            ),
            source_name="North Press Agency",
            category="kurdistan",
            region="West Kurdistan",
        )
        art.id = await db.insert_article(art)

        # Ranker checks
        ranker = ImportanceRanker(api_key="gem_test_key")
        # Even if mock Gemini wasn't reached or tried to mislabel, deterministic guard blocks it
        mock_session = _make_gemini_session(
            200, json_payload={"score": 9, "is_breaking": False, "category": "kurdistan", "reason": "Mock leak"}
        )
        ranker._gemini._get_session = AsyncMock(return_value=mock_session)
        eval_res = await ranker.evaluate_global_relevance(art)

        assert eval_res.passed is False
        assert eval_res.category != "kurdistan"
        assert is_genuine_kurdistan_event(art) is False
        assert is_bashur_article(art) is False
        assert check_blocked_topic(art) is not None
        assert ranker.is_eligible(0.95, 0.90, article=art) is False

        # Pipeline rejects cleanly and saves to skipped_urls
        pipeline._extractor.fetch_article_content_and_image = AsyncMock()
        pipeline._translator.translate_article = AsyncMock()
        published = await pipeline.process_article(art)

        assert published is False
        pipeline._extractor.fetch_article_content_and_image.assert_not_called()
        pipeline._translator.translate_article.assert_not_called()
        publisher.publish_article.assert_not_called()

        db_art = await db.get_article_by_canonical_url(art.canonical_url)
        assert db_art.status == ArticleStatus.SKIPPED
        assert await db.is_url_skipped(art.canonical_url) is True

        # Ensure genuine ICJ ruling & Bashur Peshmerga operation remain allowed
        icj_art = Article_factory(
            url="https://reuters.com/world/icj-ruling",
            title="International Court of Justice Issues Emergency Ruling on Gaza Ceasefire",
            summary="The ICJ issued binding provisional measures amid ongoing Middle East diplomatic crisis.",
            source_name="Reuters",
            category="politics_world",
        )
        assert check_blocked_topic(icj_art) is None
        await db.close()

    @pytest.mark.asyncio
    async def test_npr_morocco_afcon_sports_match_rejected_never_politics(self, tmp_path):
        """
        Regression: NPR News article covering Morocco / Africa Cup of Nations football match
        must be rejected (eligible=False, skipped_urls) and never slip into Politics or Kurdistan.
        """
        db = Database(str(tmp_path / "test_sports_block.db"))
        await db.connect()

        publisher = MagicMock(spec=TelegramPublisher)
        publisher.publish_article = AsyncMock(return_value=2222)
        pipeline = ArticlePipeline(db, publisher, HealthMonitor(), Metrics())

        art = Article_factory(
            url="https://npr.org/2026/morocco-afcon-final",
            title="Morocco Faces Senegal in Chaotic Africa Cup of Nations Football Final",
            summary=(
                "Rabat stadium erupted as the Morocco national football team battled in the AFCON "
                "tournament match with dramatic penalty drama and coach reactions."
            ),
            source_name="NPR News",
            category="politics_world",
        )
        art.id = await db.insert_article(art)

        ranker = ImportanceRanker(api_key="gem_test_key")
        eval_res = await ranker.evaluate_global_relevance(art)
        assert eval_res.passed is False
        assert ranker.is_eligible(0.90, 0.90, article=art) is False
        assert check_blocked_topic(art) is not None

        pipeline._extractor.fetch_article_content_and_image = AsyncMock()
        pipeline._translator.translate_article = AsyncMock()
        published = await pipeline.process_article(art)

        assert published is False
        pipeline._extractor.fetch_article_content_and_image.assert_not_called()
        pipeline._translator.translate_article.assert_not_called()
        publisher.publish_article.assert_not_called()

        db_art = await db.get_article_by_canonical_url(art.canonical_url)
        assert db_art.status == ArticleStatus.SKIPPED
        assert await db.is_url_skipped(art.canonical_url) is True
        await db.close()

    def test_leading_agency_datelines_stripped_from_kurdish_summaries(self):
        """Verify Gemini prompts forbid agency datelines and formatter/cleaner strips them cleanly."""
        assert "قامیشلی، سوریا (نۆرس پرێس)" in GEMINI_TRANSLATION_SYSTEM_PROMPT
        assert "dateline" in GEMINI_KURDISH_REWRITE_SYSTEM_PROMPT.lower()

        assert strip_news_dateline(
            "قامیشلی، سوریا (نۆرس پرێس) – شاندی هەرێمی کوردستان لە بەغدا کۆبوونەوە."
        ) == "شاندی هەرێمی کوردستان لە بەغدا کۆبوونەوە."
        assert strip_news_dateline(
            "هەولێر - وەزارەتی پێشمەرگە ڕاگەیەندراوێکی بڵاوکردەوە."
        ) == "وەزارەتی پێشمەرگە ڕاگەیەندراوێکی بڵاوکردەوە."
        assert strip_news_dateline(
            "واشنتۆن (ڕۆیتەرز) - کۆشکی سپی هەڵوێستی خۆی ڕاگەیاند."
        ) == "کۆشکی سپی هەڵوێستی خۆی ڕاگەیاند."
        # Hyphenated compound headlines are preserved intact
        assert strip_news_dateline(
            "هەولێر-بەغدا لەسەر بودجە ڕێککەوتن."
        ) == "هەولێر-بەغدا لەسەر بودجە ڕێککەوتن."

        formatter = ContentFormatter()
        art = Article_factory(
            title="Erbil Baghdad Accord",
            summary="Summary placeholder",
            category="kurdistan",
        )
        art.translated_title = "ڕێککەوتنی نوێ لە هەولێر واژۆ کرا"
        art.translated_summary = (
            "قامیشلی، سوریا (نۆرس پرێس) – لایەنە سیاسییەکان گەیشتنە لێکتێگەیشتنێکی هاوبەش. "
            "وردەکاری ڕێککەوتنەکە لە کۆنگرەیەکی ڕۆژنامەوانیدا بڵاوکرایەوە."
        )
        caption = formatter.format_caption(art)
        text_msg = formatter.format_telegram_message(art)
        assert "قامیشلی، سوریا (نۆرس پرێس)" not in caption
        assert "قامیشلی، سوریا (نۆرس پرێس)" not in text_msg
        assert "لایەنە سیاسییەکان گەیشتنە لێکتێگەیشتنێکی هاوبەش." in caption
        assert "لایەنە سیاسییەکان گەیشتنە لێکتێگەیشتنێکی هاوبەش." in text_msg
