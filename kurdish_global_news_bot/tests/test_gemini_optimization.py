"""
Unit and integration tests for Gemini API quota optimization:
1. Comma-separated multi-key loading from GEMINI_API_KEYS and GEMINI_API_KEY
2. Immediate key rotation & per-key HTTP 429 cooldowns (prevents hammering exhausted keys)
3. Clean circuit-breaker & concise warning when ALL keys hit 429 cooldown
4. Zero-API local cross-source deduplication (`SemanticDeduplicator`)
5. Deterministic local pre-filtering (`check_pre_filter_off_topic`) & cycle cap
   (`select_top_candidates_for_gemini`) BEFORE Gemini calls
6. Single-request batched Gemini importance evaluation (`evaluate_batch_relevance`)
   integrated with `ArticlePipeline.process_batch` & `NewsEngine._discovery_cycle`
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import config
from config import ArticleStatus, Priority
from core.engine import NewsEngine
from core.importance_ranker import (
    ImportanceRanker,
    check_pre_filter_off_topic,
    is_bashur_article,
    select_top_candidates_for_gemini,
)
from core.pipeline import ArticlePipeline
from monitoring.health import HealthMonitor
from monitoring.metrics import Metrics
from processing.deduplicator import SemanticDeduplicator
from processing.gemini_client import (
    GeminiClient,
    GeminiQuotaExhaustedError,
    _parse_api_keys,
)
from sources.base import BaseSource, RawArticle
from storage.database import Database
from storage.models import Article
from tg_bot.publisher import TelegramPublisher


class _MockResponseContext:
    def __init__(self, response) -> None:
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


def _make_json_response(payload: dict) -> AsyncMock:
    resp = AsyncMock()
    resp.status = 200
    resp.json = AsyncMock(
        return_value={"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]}
    )
    resp.text = AsyncMock(return_value=json.dumps(payload))
    return resp


def _make_429_response(body: str = "Too Many Requests: rate limit") -> AsyncMock:
    resp = AsyncMock()
    resp.status = 429
    resp.text = AsyncMock(return_value=body)
    return resp


def _make_art(
    title: str,
    url: str,
    summary: str = "",
    category: str = "politics_world",
    region: str = "",
    hours_ago: float = 0.2,
    source_name: str = "Reuters",
) -> Article:
    pub_iso = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    full_sum = summary or (
        f"{title}. Official diplomatic and defense channels reported sustained regional security "
        "coordination, cabinet deliberations, and military readiness updates across key capitals."
    )
    art = Article(
        original_url=url,
        canonical_url=Article.canonicalize_url(url),
        original_title=title,
        normalized_title=Article.normalize_title(title),
        original_summary=full_sum,
        source_name=source_name,
        category=category,
        region=region,
        publication_date=pub_iso,
        discovered_at=pub_iso,
    )
    art.compute_content_hash()
    art.compute_event_fingerprint()
    return art


@pytest.fixture
async def temp_db(tmp_path):
    db = Database(str(tmp_path / "test_gemini_opt.db"))
    await db.connect()
    yield db
    await db.close()


class TestGeminiKeyCooldownAndRotation:
    def test_loads_and_deduplicates_gemini_api_keys_and_key(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEYS", "key_a, key_b , key_a")
        monkeypatch.setenv("GEMINI_API_KEY", "key_b, key_c")
        loaded = config._load_gemini_keys()
        assert loaded == ["key_a", "key_b", "key_c"]

        parsed = _parse_api_keys("k1, k2 , , k1 , k3")
        assert parsed == ["k1", "k2", "k3"]

    @pytest.mark.asyncio
    async def test_per_key_429_cooldown_skips_exhausted_key_on_subsequent_calls(self):
        cooldown_store: dict[str, float] = {}
        client = GeminiClient(
            api_key="k0,k1,k2",
            base_backoff=0.0,
            cooldown_store=cooldown_store,
            cooldown_seconds=60.0,
        )

        # Call 1: k0 hits 429 -> immediately rotates to k1 -> succeeds (NO sleep needed!)
        # Call 2: starts directly on k1 (k0 is still in 60s cooldown -> NEVER called again!)
        session = MagicMock()
        session.post = MagicMock(
            side_effect=[
                _MockResponseContext(_make_429_response()),
                _MockResponseContext(_make_json_response({"ok": 1})),
                _MockResponseContext(_make_json_response({"ok": 2})),
            ]
        )
        client._get_session = AsyncMock(return_value=session)

        res1 = await client.generate_json("sys", "req1")
        res2 = await client.generate_json("sys", "req2")

        assert res1 == {"ok": 1}
        assert res2 == {"ok": 2}
        assert session.post.call_count == 3
        keys_used = [c.kwargs["headers"]["x-goog-api-key"] for c in session.post.call_args_list]
        assert keys_used == ["k0", "k1", "k1"]
        assert client._is_key_cooling("k0") is True
        assert client._is_key_cooling("k1") is False

    @pytest.mark.asyncio
    async def test_all_keys_exhausted_circuit_breaks_without_looping_429s(self, caplog):
        cooldown_store: dict[str, float] = {}
        client = GeminiClient(
            api_key="k0,k1",
            base_backoff=0.0,
            cooldown_store=cooldown_store,
            cooldown_seconds=60.0,
            quota_cooldown_seconds=300.0,
        )

        session = MagicMock()
        session.post = MagicMock(
            side_effect=[
                _MockResponseContext(_make_429_response("Quota exceeded per day")),
                _MockResponseContext(_make_429_response("Rate limit exceeded")),
            ]
        )
        client._get_session = AsyncMock(return_value=session)

        with caplog.at_level(logging.WARNING, logger="processing.gemini"):
            with pytest.raises(GeminiQuotaExhaustedError):
                await client.generate_json("sys", "first")

        # k0 (quota/per day) gets 300s cooldown, k1 gets 60s cooldown
        assert client.all_keys_in_cooldown() is True
        assert session.post.call_count == 2

        # Second client sharing cooldown_store (e.g. Translator after Ranker) makes ZERO HTTP requests!
        second_client = GeminiClient(
            api_key="k0,k1",
            base_backoff=0.0,
            cooldown_store=cooldown_store,
        )
        second_session = MagicMock()
        second_client._get_session = AsyncMock(return_value=second_session)

        with pytest.raises(GeminiQuotaExhaustedError):
            await second_client.generate_json("sys", "second")

        second_session.post.assert_not_called()


class TestZeroApiLocalDedup:
    @pytest.mark.asyncio
    async def test_local_semantic_dedup_catches_cross_source_paraphrases_without_gemini(self, temp_db):
        pub = _make_art(
            title="Israeli airstrikes hit Iranian weapons depot near Beirut airport",
            url="https://reuters.com/world/beirut-1",
            category="military_conflict",
            hours_ago=1.0,
        )
        pub_id = await temp_db.insert_article(pub)
        await temp_db.update_article_status(pub_id, ArticleStatus.PUBLISHED, published_at=pub.published_at)

        dedup = SemanticDeduplicator(temp_db)
        assert dedup.uses_gemini is False

        paraphrase = _make_art(
            title="Explosions rock Lebanese capital airport after Israeli bombardment",
            url="https://t.me/SabrenNewss/500",
            category="military_conflict",
            source_name="Sabereen News",
        )
        res_dup = await dedup.check_duplicate(paraphrase)
        assert res_dup.is_duplicate is True
        assert res_dup.matched_id == pub_id
        assert res_dup.stage == "local_semantic"

        distinct = _make_art(
            title="UN Secretary General addresses Vienna climate summit on water scarcity",
            url="https://bbc.com/news/vienna-1",
            category="politics_world",
        )
        res_ok = await dedup.check_duplicate(distinct)
        assert res_ok.is_duplicate is False


class TestPreFilteringCappingAndBatchEvaluation:
    def test_pre_filter_rejects_noise_but_keeps_politics_war_and_kurdistan(self):
        # Obvious off-topic noise -> rejected locally before Gemini
        assert check_pre_filter_off_topic(
            _make_art("Manchester United wins Premier League match 3-1", "https://ex.com/s1")
        ) is not None
        assert check_pre_filter_off_topic(
            _make_art(
                "Wall Street S&P 500 stock rally continues as retail sales jump",
                "https://ex.com/e1",
                summary="Dow Jones and Nasdaq gained on quarterly earnings reports.",
                category="economy_markets",
            )
        ) is not None
        assert check_pre_filter_off_topic(
            _make_art(
                "New Apple iPhone smartphone camera benchmark released",
                "https://ex.com/t1",
                summary="Tech reviewers tested the consumer mobile device battery life.",
                category="ai_tech",
            )
        ) is not None
        assert check_pre_filter_off_topic(
            _make_art("Traffic police deploy speed radar cameras on ring road", "https://ex.com/tr1")
        ) is not None

        # Geopolitics, war, cyberwar, & Kurdistan -> KEPT for evaluation
        assert check_pre_filter_off_topic(
            _make_art("KRG officials discuss civil servant salaries and oil exports in Erbil", "https://ex.com/k1", region="South Kurdistan")
        ) is None
        assert check_pre_filter_off_topic(
            _make_art("Ballistic missiles intercepted near Baghdad military airbase", "https://ex.com/m1")
        ) is None
        assert check_pre_filter_off_topic(
            _make_art("State-sponsored cyberattack hits regional energy grid", "https://ex.com/c1", category="cybersecurity")
        ) is None

    def test_select_top_candidates_caps_to_freshest_most_relevant(self):
        sports = [_make_art(f"FIFA World Cup match update {i}", f"https://ex.com/sp/{i}") for i in range(10)]
        stocks = [
            _make_art(
                f"Dow Jones quarterly earnings stock surge {i}",
                f"https://ex.com/st/{i}",
                summary="Wall Street indices rose after corporate quarterly earnings.",
                category="economy_markets",
            )
            for i in range(10)
        ]
        politics = [
            _make_art(
                f"UN Security Council emergency sanctions summit {i}",
                f"https://ex.com/pol/{i}",
                hours_ago=0.1 * (i + 1),
            )
            for i in range(30)
        ]
        erbil = _make_art(
            "Peshmerga intercepts drone near Erbil airport",
            "https://ex.com/erbil/1",
            category="kurdistan",
            region="South Kurdistan",
            hours_ago=0.5,
        )

        selected, rejected = select_top_candidates_for_gemini(
            sports + stocks + politics + [erbil], max_candidates=15
        )
        assert len(selected) == 15
        # Kurdistan priority 1A placed right at the front despite 0.5h age
        assert selected[0].original_title == erbil.original_title
        # Total rejected = 10 sports + 10 stocks + 16 capped politics overflow = 36
        assert len(rejected) == 36

    @pytest.mark.asyncio
    async def test_evaluate_batch_relevance_cuts_api_calls_and_aborts_on_quota_exhaustion(self):
        ranker = ImportanceRanker(api_key="k1,k2", min_relevance_score=7, base_backoff=0.0)

        articles = [
            _make_art(f"Regional ceasefire diplomacy summit {i}", f"https://ex.com/pol/{i}")
            for i in range(12)
        ] + [
            _make_art(
                "Erbil political statement interview on budget talks",
                "https://ex.com/krg/1",
                category="kurdistan",
                region="South Kurdistan",
            )
        ]
        # 13 articles with batch_size=12 -> only 2 Gemini HTTP calls instead of 13!
        batch1_payload = {
            "items": [
                {"idx": i, "score": 8, "is_breaking": False, "category": "politics_world", "reason": "Diplomacy"}
                for i in range(12)
            ]
        }
        batch2_payload = {
            "items": [
                {"idx": 12, "score": 9, "is_breaking": True, "category": "kurdistan", "reason": "Interview statement"}
            ]
        }
        session = MagicMock()
        session.post = MagicMock(
            side_effect=[
                _MockResponseContext(_make_json_response(batch1_payload)),
                _MockResponseContext(_make_json_response(batch2_payload)),
            ]
        )
        ranker._gemini._get_session = AsyncMock(return_value=session)

        evals = await ranker.evaluate_batch_relevance(articles, batch_size=12)
        assert session.post.call_count == 2  # ~85% reduction even at 13 articles (12+1)
        assert len(evals) == 13
        assert all(e.score == 8 and e.passed is True for e in evals[:12])
        assert evals[12].category == "kurdistan"
        assert evals[12].is_breaking is True  # score 9 + Gemini True preserved

        # Now verify quota exhaustion stops further batch calls cleanly
        ranker2 = ImportanceRanker(api_key="k1", min_relevance_score=7, base_backoff=0.0)
        exhausted_session = MagicMock()
        exhausted_session.post = MagicMock(return_value=_MockResponseContext(_make_429_response("Rate limit")))
        ranker2._gemini._get_session = AsyncMock(return_value=exhausted_session)

        many_articles = [
            _make_art(f"Iraq military escalation crisis {i}", f"https://ex.com/m/{i}")
            for i in range(25)
        ]
        fallback_evals = await ranker2.evaluate_batch_relevance(many_articles, batch_size=10)
        assert len(fallback_evals) == 25
        assert all(e.used_fallback is True for e in fallback_evals)
        # Hit 429 ONCE on batch 1, aborted batches 2 and 3 with ZERO additional requests!
        assert exhausted_session.post.call_count == 1

    @pytest.mark.asyncio
    async def test_pipeline_process_batch_uses_batched_gemini_and_preserves_5_of_8_quota(self, temp_db):
        mock_bot = AsyncMock()
        msg_obj = MagicMock()
        msg_obj.message_id = 4444
        mock_bot.send_photo = AsyncMock(return_value=msg_obj)

        publisher = TelegramPublisher()
        publisher._channel_id = "@KurdishGlobalNews"
        publisher._bot = mock_bot
        publisher._image_handler.download_image = AsyncMock(return_value=None)

        pipeline = ArticlePipeline(temp_db, publisher, HealthMonitor(), Metrics())
        pipeline._ranker = ImportanceRanker(
            api_key="gem_test_key",
            min_relevance_score=7,
            base_backoff=0.0,
            cooldown_store=pipeline._gemini_cooldowns,
        )

        # Build 6 distinct Bashur Politics + 4 distinct Global War + 1 Paraphrase Duplicate + 2 Noise = 13 items
        bashur_titles = [
            "Erbil KRG cabinet approves civil servant salary distribution mechanism",
            "Peshmerga commanders inspect Kirkuk frontier defense fortifications",
            "Sulaymaniyah governor hosts constitutional federalism dialogue forum",
            "Duhok customs authorities expand cross-border trade customs screening",
            "Halabja provincial council inaugurates reconstruction funding committee",
            "KRG judicial council in Erbil reviews regional security oversight bill",
        ]
        global_titles = [
            "NATO and Tehran clash over Gulf naval deployment rules",
            "UN Security Council debates maritime shipping escorts in Red Sea",
            "Washington announces new European missile interceptor defense battery",
            "Geneva diplomatic envoys draft bilateral ceasefire verification framework",
        ]
        batch: list[Article] = []
        for i, b_title in enumerate(bashur_titles):
            batch.append(
                _make_art(
                    b_title,
                    f"https://rudaw.net/kurd/{i}",
                    summary=(
                        f"{b_title}. Regional authorities in Kurdistan published detailed operational measures "
                        f"for item {i} addressing provincial administration and constitutional coordination."
                    ),
                    category="kurdistan",
                    region="South Kurdistan",
                    source_name="Rudaw",
                )
            )
        for i, g_title in enumerate(global_titles):
            batch.append(
                _make_art(
                    g_title,
                    f"https://reuters.com/world/{i}",
                    summary=(
                        f"{g_title}. Ambassadors and defense spokespeople briefed reporters on specific "
                        f"deployable contingencies and international treaties under dossier {i}."
                    ),
                    category="military_conflict",
                )
            )
        # Paraphrase duplicate of Global #0 (different outlet & wording -> caught locally, ZERO Gemini calls!)
        batch.append(
            _make_art(
                "Gulf naval deployment tensions rise between Tehran and NATO forces",
                "https://bbc.com/news/dup-0",
                summary=(
                    "Gulf naval deployment tensions rise between Tehran and NATO forces as defense commands "
                    "review maritime patrol zones and allied rules of engagement."
                ),
                category="military_conflict",
                source_name="BBC",
            )
        )
        # Noise -> filtered BEFORE Gemini (with realistic sports & finance summaries)
        batch.append(
            _make_art(
                "Champions League football match ends 2-0",
                "https://ex.com/sport1",
                summary="Both European football squads played 90 minutes with two second-half goals.",
            )
        )
        batch.append(
            _make_art(
                "Nasdaq Wall Street stock surge on quarterly earnings",
                "https://ex.com/mkt1",
                summary="US equity indices climbed after retail tech firms beat quarterly revenue targets.",
                category="economy_markets",
            )
        )

        for a in batch:
            a.id = await temp_db.insert_article(a)

        # Gemini batch ranker will be called ONCE for the 10 surviving articles (NOT 13 times!)
        batch_rank_calls: list[list[dict]] = []

        async def fake_generate_json(system_prompt: str, user_content: str, **_kwargs):
            json_str = user_content.split("\n", 1)[1]
            items = json.loads(json_str)
            batch_rank_calls.append(items)
            bashur_markers = ("Erbil", "Peshmerga", "Sulaymaniyah", "Duhok", "Halabja", "Kurdistan")
            return {
                "items": [
                    {
                        "idx": item["idx"],
                        "score": 8,
                        "is_breaking": False,
                        "category": "kurdistan" if any(m in item["title"] for m in bashur_markers) else "military_conflict",
                        "reason": "Relevant war/politics news",
                    }
                    for item in items
                ]
            }

        pipeline._ranker._gemini.generate_json = AsyncMock(side_effect=fake_generate_json)
        pipeline._extractor.fetch_article_content_and_image = AsyncMock(return_value=("", None))
        bashur_markers = ("Erbil", "Peshmerga", "Sulaymaniyah", "Duhok", "Halabja", "Kurdistan")
        pipeline._translator.translate_article = AsyncMock(
            side_effect=lambda title, summary="": (
                (
                    "هەرێمی کوردستان و هەولێر بڕیاری نوێیان دەربارەی دۆخی سیاسی دەرکرد"
                    if any(m in title for m in bashur_markers)
                    else "پێشهاتی سەربازی و دیپلۆماسی لە ناوچەکەدا ڕاگەیەندرا"
                ),
                "ڕاپۆرتێکی وردی ڕۆژنامەوانی لەسەر پێشهاتە سیاسی و ئەمنییەکان. لایەنەکان لێدوانیان دا.",
            )
        )

        res = await pipeline.process_batch(batch, max_publish=8)

        assert res["published"] == 8
        # ONLY ONE batched Gemini ranking call for all 10 valid unique articles!
        assert len(batch_rank_calls) == 1
        assert len(batch_rank_calls[0]) == 10

        # Confirm 5-of-8 Bashur quota preserved: 5 Bashur first + 3 Global
        published_captions = [c.kwargs["caption"] for c in mock_bot.send_photo.call_args_list]
        assert len(published_captions) == 8
        bashur_sent = sum(1 for cap in published_captions if "هەرێمی کوردستان و هەولێر" in cap)
        assert bashur_sent == 5
