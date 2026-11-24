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
    ImportanceRanker,
    RelevanceEvaluation,
)
from core.pipeline import ArticlePipeline
from monitoring.health import HealthMonitor
from monitoring.metrics import Metrics
from processing.gemini_client import GEMINI_API_BASE_URL
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
        """Ensure Al Jazeera business/tech stories get economy_markets/ai_tech rather than military_conflict."""
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

        assert result.passed is True
        assert result.category == "economy_markets"

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
                    "score": 8, "is_breaking": True, "category": "military_conflict",
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
