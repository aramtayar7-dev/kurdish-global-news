"""
Unit tests for the AI-powered Global Importance & Relevance Gatekeeper
and SQLite skipped-URL hash persistence.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import MIN_GLOBAL_RELEVANCE_SCORE, ArticleStatus, Priority
from core.engine import NewsEngine
from core.importance_ranker import (
    GLOBAL_RELEVANCE_SYSTEM_PROMPT,
    GROQ_API_ENDPOINT,
    ImportanceRanker,
    RelevanceEvaluation,
)
from core.pipeline import ArticlePipeline
from monitoring.health import HealthMonitor
from monitoring.metrics import Metrics
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


def _make_groq_session(status: int, json_data: dict | None = None, text_data: str = ""):
    mock_resp = AsyncMock()
    mock_resp.status = status
    if json_data is not None:
        mock_resp.json = AsyncMock(return_value=json_data)
    mock_resp.text = AsyncMock(return_value=text_data)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=_MockResponseContext(mock_resp))
    mock_session.closed = False
    return mock_session


class TestGlobalRelevanceConfigAndExports:
    """Verify default configuration and re-exports."""

    def test_default_min_global_relevance_score(self):
        assert MIN_GLOBAL_RELEVANCE_SCORE == 7

    def test_processing_importance_ranker_reexports(self):
        assert ProcessingImportanceRanker is ImportanceRanker
        assert ProcessingRelevanceEvaluation is RelevanceEvaluation


class TestGlobalRelevanceClassifier:
    """Verify Groq LLM classification logic, parameters, and fail-open fallback."""

    @pytest.mark.asyncio
    async def test_middle_east_military_news_passes(self):
        ranker = ImportanceRanker(api_key="gsk_test_key", min_relevance_score=7)
        mock_session = _make_groq_session(
            200,
            json_data={
                "choices": [{
                    "message": {
                        "content": (
                            '{"score": 9, "is_breaking": true, '
                            '"reason": "Major Middle East military escalation involving Iran and Israel."}'
                        )
                    }
                }]
            },
        )
        ranker._get_session = AsyncMock(return_value=mock_session)

        art = Article_factory(
            title="Airstrikes Reported Near Strategic Military Sites in Western Iran",
            summary="Regional defense monitors reported overnight military strikes amid heightened Middle East tensions.",
            source_name="Reuters",
            category="Wars & Armed Conflicts",
        )
        result = await ranker.evaluate_global_relevance(art)

        assert result.score >= 7
        assert result.score == 9
        assert result.is_breaking is True
        assert result.passed is True
        assert result.used_fallback is False

        # Verify Groq API payload parameters
        call_args, call_kwargs = mock_session.post.call_args
        assert call_args[0] == GROQ_API_ENDPOINT
        payload = call_kwargs["json"]
        assert payload["model"] == "llama-3.3-70b-versatile"
        assert payload["temperature"] == 0.0
        assert payload["max_tokens"] == 120
        assert payload["messages"][0]["content"] == GLOBAL_RELEVANCE_SYSTEM_PROMPT
        assert "Airstrikes Reported Near Strategic Military Sites" in payload["messages"][1]["content"]

    @pytest.mark.asyncio
    async def test_major_ai_release_passes(self):
        ranker = ImportanceRanker(api_key="gsk_test_key", min_relevance_score=7)
        mock_session = _make_groq_session(
            200,
            json_data={
                "choices": [{
                    "message": {
                        "content": (
                            '{"score": 8, "is_breaking": false, '
                            '"reason": "Major frontier AI model release with global technology impact."}'
                        )
                    }
                }]
            },
        )
        ranker._get_session = AsyncMock(return_value=mock_session)

        art = Article_factory(
            title="OpenAI Announces Next-Generation Reasoning Model With Major Benchmark Gains",
            summary="The new frontier AI system improves global developer coding, mathematics, and multilingual support.",
            source_name="The Verge",
            category="AI & Technology",
        )
        result = await ranker.evaluate_global_relevance(art)

        assert result.score == 8
        assert result.passed is True
        assert result.used_fallback is False

    @pytest.mark.asyncio
    async def test_local_foreign_lawsuit_rejected(self):
        ranker = ImportanceRanker(api_key="gsk_test_key", min_relevance_score=7)
        mock_session = _make_groq_session(
            200,
            json_data={
                "choices": [{
                    "message": {
                        "content": (
                            '{"score": 2, "is_breaking": false, '
                            '"reason": "Hyper-local US state licensing dispute with no international significance."}'
                        )
                    }
                }]
            },
        )
        ranker._get_session = AsyncMock(return_value=mock_session)

        art = Article_factory(
            title="Oregon Plumbing Licensing Board Faces Lawsuit Over Contractor Permit Fees",
            summary="Two local contractors filed suit in Marion County court disputing municipal renewal fees.",
            source_name="Local News Oregon",
            category="Business & Corporate",
        )
        result = await ranker.evaluate_global_relevance(art)

        assert result.score < 7
        assert result.score == 2
        assert result.passed is False
        assert "licensing" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_key_rotation_on_429_and_fail_open_on_api_error(self):
        ranker = ImportanceRanker(api_key="gsk_k1,gsk_k2", min_relevance_score=7)

        # Case 1: Key 1 returns 429, Key 2 succeeds
        resp_429 = AsyncMock(status=429, text=AsyncMock(return_value="Rate limited"))
        resp_200 = AsyncMock(
            status=200,
            json=AsyncMock(return_value={
                "choices": [{
                    "message": {"content": '{"score": 8, "is_breaking": true, "reason": "Iraq oil pipeline agreement."}'}
                }]
            }),
        )
        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=[
            _MockResponseContext(resp_429),
            _MockResponseContext(resp_200),
        ])
        ranker._get_session = AsyncMock(return_value=mock_session)

        art = Article_factory(
            title="Baghdad and Erbil Agree on Crude Export Restart Through Ceyhan Pipeline",
            summary="Federal Iraqi officials and Kurdistan Regional Government negotiators reached a technical deal.",
            source_name="Reuters",
            category="Energy, Oil & Gas",
        )
        result = await ranker.evaluate_global_relevance(art)
        assert result.passed is True
        assert result.score == 8
        assert ranker._current_key_index == 1

        # Case 2: API throws network/server error -> fails open safely to rule-based score
        failing_session = MagicMock()
        failing_session.post = MagicMock(side_effect=RuntimeError("Connection reset"))
        ranker._get_session = AsyncMock(return_value=failing_session)

        fallback_result = await ranker.evaluate_global_relevance(art)
        assert fallback_result.used_fallback is True
        assert fallback_result.passed is True
        assert "Rule-based fallback" in fallback_result.reason


class TestPipelineAndSkippedUrlPersistence:
    """Verify rejected low-relevance articles skip translation and are saved in SQLite."""

    @pytest.mark.asyncio
    async def test_pipeline_rejects_local_lawsuit_logs_and_persists_hash(self, tmp_path, caplog):
        db = Database(str(tmp_path / "test_relevance.db"))
        await db.connect()

        publisher = MagicMock(spec=TelegramPublisher)
        publisher.publish_article = AsyncMock(return_value=9999)
        pipeline = ArticlePipeline(db, publisher, HealthMonitor(), Metrics())

        # Force Groq classifier rejection (score=2)
        pipeline._ranker.evaluate_global_relevance = AsyncMock(
            return_value=RelevanceEvaluation(
                score=2,
                is_breaking=False,
                reason="Routine US state contractor licensing lawsuit with no global impact.",
                passed=False,
                used_fallback=False,
            )
        )
        pipeline._extractor.fetch_full_article_text = AsyncMock()
        pipeline._translator.translate_article = AsyncMock()

        art = Article_factory(
            url="https://oregonlive.example.com/local-licensing-lawsuit?utm_source=rss",
            title="Oregon Contractors Sue State Board Over Routine Permit Licensing Delay",
            summary="A local civil lawsuit was filed in state court over municipal contractor license renewals and filing deadlines.",
            source_name="Reuters",  # High credibility so it passes rule-based check and tests the LLM gatekeeper!
            category="Business & Corporate",
        )
        art_id = await db.insert_article(art)
        art.id = art_id

        with caplog.at_level(logging.INFO, logger="core.pipeline"):
            published = await pipeline.process_article(art)

        assert published is False

        # Must NEVER scrape full text, translate, or publish!
        pipeline._extractor.fetch_full_article_text.assert_not_called()
        pipeline._translator.translate_article.assert_not_called()
        publisher.publish_article.assert_not_called()

        # Verify exact INFO log format
        assert any(
            "SKIPPED [Low Global Relevance 2/10]: Oregon Contractors Sue State Board" in rec.message
            and "Routine US state contractor licensing lawsuit" in rec.message
            for rec in caplog.records
        )

        # Verify DB status = SKIPPED & URL hash stored in skipped_urls
        db_art = await db.get_article_by_canonical_url(art.canonical_url)
        assert db_art is not None
        assert db_art.status == ArticleStatus.SKIPPED
        assert await db.is_url_skipped(art.original_url) is True
        # Tracking-param variation should also resolve to skipped via canonical URL hash!
        assert await db.is_url_skipped("https://oregonlive.example.com/local-licensing-lawsuit?ref=twitter") is True

        # Verify second discovery pass in NewsEngine skips it BEFORE DB insert / LLM re-check!
        engine = NewsEngine()
        engine._db = db
        raw = RawArticle(
            title=art.original_title,
            url="https://oregonlive.example.com/local-licensing-lawsuit?fbclid=12345",
            summary=art.original_summary,
            source_name="Reuters",
            publication_date=art.publication_date or "",
        )
        new_articles = await engine._extract_and_insert([raw])
        assert new_articles == []

        await db.close()

    @pytest.mark.asyncio
    async def test_pipeline_passes_high_impact_middle_east_article(self, tmp_path):
        db = Database(str(tmp_path / "test_relevance_pass.db"))
        await db.connect()

        publisher = MagicMock(spec=TelegramPublisher)
        publisher.publish_article = AsyncMock(return_value=7777)
        pipeline = ArticlePipeline(db, publisher, HealthMonitor(), Metrics())

        pipeline._ranker.evaluate_global_relevance = AsyncMock(
            return_value=RelevanceEvaluation(
                score=9,
                is_breaking=True,
                reason="Major regional security escalation in the Middle East.",
                passed=True,
                used_fallback=False,
            )
        )
        pipeline._extractor.fetch_full_article_text = AsyncMock(return_value=(
            "Regional officials confirmed emergency diplomatic consultations following escalating border incidents in northern Iraq and Syria. "
            "Security forces raised readiness levels across strategic energy corridors while UN envoys urged immediate de-escalation."
        ))
        pipeline._translator.translate_article = AsyncMock(
            return_value=("گرژییە ئەمنییەکان لە ناوچەکەدا زیادیان کرد", "بەرپرسانی ناوچەکە کۆبوونەوەی بەپەلەیان ئەنجامدا دوای پەرەسەندنی ڕووداوەکان.")
        )

        art = Article_factory(
            url="https://reuters.com/world/middle-east-security-escalation",
            title="Emergency Security Consultations Held Amid Middle East Border Escalation",
            summary="Officials convened urgent talks following security developments in Iraq and Syria.",
            source_name="Reuters",
            category="Wars & Armed Conflicts",
        )
        art_id = await db.insert_article(art)
        art.id = art_id

        published = await pipeline.process_article(art)
        assert published is True
        pipeline._translator.translate_article.assert_awaited_once()
        publisher.publish_article.assert_awaited_once()

        db_art = await db.get_article_by_canonical_url(art.canonical_url)
        assert db_art.status == ArticleStatus.PUBLISHED
        assert db_art.priority == Priority.CRITICAL
        assert await db.is_url_skipped(art.original_url) is False

        await db.close()


def Article_factory(
    title: str,
    summary: str,
    source_name: str = "Reuters",
    category: str = "International Politics",
    url: str = "https://reuters.com/world/sample-story",
):
    from storage.models import Article

    art = Article(
        original_url=url,
        canonical_url=Article.canonicalize_url(url),
        original_title=title,
        original_summary=summary,
        source_name=source_name,
        category=category,
        publication_date=datetime.now(timezone.utc).isoformat(),
    )
    art.compute_content_hash()
    art.compute_event_fingerprint()
    return art
