"""
Unit and integration tests for:
1. Smart silent vs. breaking Telegram notifications (`disable_notification`,
   `🔴 بەپەلە | ` prefix gated on `is_breaking=True` AND `score >= 9`,
   and `NOTIFY_DAILY_DIGEST`).
2. Red "بەپەلە" breaking visual badge on 1280x720 news cards.
3. Cross-source two-stage semantic event deduplication (12h DB lookup,
   Stage A fast similarity >= 0.60, Stage B Gemini semantic check, `skipped_urls`
   recording, `SKIPPED [Duplicate Event]` log format, and fail-open safeguard).
"""

from __future__ import annotations

import io
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from config import DEDUP_WINDOW_HOURS, NOTIFY_DAILY_DIGEST, ArticleStatus, Priority
from core.importance_ranker import RelevanceEvaluation
from core.pipeline import ArticlePipeline
from monitoring.health import HealthMonitor
from monitoring.metrics import Metrics
from processing.card_generator import NewsCardGenerator
from processing.content_formatter import ContentFormatter
from processing.deduplicator import SemanticDeduplicator
from storage.database import Database
from storage.models import Article
from tg_bot.publisher import TelegramPublisher


class _MockResponseContext:
    """Async context manager wrapping a mocked aiohttp response."""

    def __init__(self, response: AsyncMock) -> None:
        self._response = response

    async def __aenter__(self) -> AsyncMock:
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


def _make_json_session(status: int, payload: dict) -> MagicMock:
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value={
        "candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]
    })
    resp.text = AsyncMock(return_value=json.dumps(payload))
    session = MagicMock()
    session.post = MagicMock(return_value=_MockResponseContext(resp))
    return session


@pytest.fixture
async def temp_db(tmp_path):
    db = Database(str(tmp_path / "test_breaking_dedup.db"))
    await db.connect()
    yield db
    await db.close()


def _make_article(
    title: str = "Global powers hold emergency summit on Middle East security",
    url: str = "https://reuters.com/world/summit-1",
    category: str = "politics_world",
    translated_title: str = "",
    translated_summary: str = "",
    is_breaking: bool = False,
    published_at: str | None = None,
    status: str = ArticleStatus.DISCOVERED,
) -> Article:
    art = Article(
        original_url=url,
        canonical_url=Article.canonicalize_url(url),
        original_title=title,
        normalized_title=Article.normalize_title(title),
        translated_title=translated_title,
        original_summary=(
            f"{title}. Detailed international dispatch reporting official statements, "
            "diplomatic negotiations, and regional security developments across multiple capitals "
            "with immediate implications for allied defense forces and global energy corridors."
        ),
        translated_summary=translated_summary,
        source_name="Reuters",
        category=category,
        publication_date=datetime.now(timezone.utc).isoformat(),
        published_at=published_at,
        status=status,
        is_breaking=is_breaking,
    )
    art.compute_content_hash()
    art.compute_event_fingerprint()
    return art


class TestBreakingNotificationsAndBadge:
    def test_formatter_prefixes_breaking_headline_only(self):
        formatter = ContentFormatter()
        normal = _make_article(
            translated_title="هێزە جیهانییەکان کۆبوونەوەی بەپەلە ئەنجام دەدەن",
            translated_summary="لە کۆبوونەوەیەکدا بەرپرسانی باڵا تاوتوێی دۆخی ئاسایشی ناوچەکەیان کرد.",
            is_breaking=False,
        )
        breaking = _make_article(
            translated_title="هێزە جیهانییەکان کۆبوونەوەی بەپەلە ئەنجام دەدەن",
            translated_summary="لە کۆبوونەوەیەکدا بەرپرسانی باڵا تاوتوێی دۆخی ئاسایشی ناوچەکەیان کرد.",
            is_breaking=True,
        )

        assert "🔴 بەپەلە | " not in formatter.format_caption(normal)
        assert "🔴 بەپەلە | " not in formatter.format_telegram_message(normal)
        assert "🔴 بەپەلە | " in formatter.format_caption(breaking)
        assert "🔴 بەپەلە | " in formatter.format_telegram_message(breaking)

    @pytest.mark.asyncio
    async def test_publisher_silent_vs_breaking_photo_and_text(self, tmp_path):
        publisher = TelegramPublisher()
        mock_bot = MagicMock()
        mock_msg = MagicMock(message_id=404)
        mock_bot.send_photo = AsyncMock(return_value=mock_msg)
        mock_bot.send_message = AsyncMock(return_value=mock_msg)
        publisher._bot = mock_bot
        publisher._channel_id = "@KurdishGlobalNews"
        publisher._image_handler = MagicMock()
        publisher._image_handler.download_image = AsyncMock(return_value=None)

        fake_card = tmp_path / "card.jpg"
        fake_card.write_bytes(b"\xff\xd8\xff\xe0dummy")
        publisher._card_generator = MagicMock()
        publisher._card_generator.generate_card_file = AsyncMock(return_value=fake_card)

        normal_art = _make_article(
            translated_title="هێزە جیهانییەکان کۆبوونەوەی بەپەلە ئەنجام دەدەن",
            translated_summary="لە کۆبوونەوەیەکدا بەرپرسانی باڵا تاوتوێی دۆخی ئاسایشی ناوچەکەیان کرد.",
            is_breaking=False,
        )
        breaking_art = _make_article(
            translated_title="هێزە جیهانییەکان کۆبوونەوەی بەپەلە ئەنجام دەدەن",
            translated_summary="لە کۆبوونەوەیەکدا بەرپرسانی باڵا تاوتوێی دۆخی ئاسایشی ناوچەکەیان کرد.",
            is_breaking=True,
            url="https://reuters.com/world/brk",
        )

        with patch("tg_bot.publisher.ENABLE_IMAGES", True):
            await publisher.publish_article(normal_art)
            normal_photo_kwargs = mock_bot.send_photo.call_args.kwargs
            assert normal_photo_kwargs["disable_notification"] is True
            assert "🔴 بەپەلە | " not in normal_photo_kwargs["caption"]

            fake_card.write_bytes(b"\xff\xd8\xff\xe0dummy")
            await publisher.publish_article(breaking_art)
            brk_photo_kwargs = mock_bot.send_photo.call_args.kwargs
            assert brk_photo_kwargs["disable_notification"] is False
            assert brk_photo_kwargs["caption"].startswith("🌍 <b>🔴 بەپەلە | ")

        with patch("tg_bot.publisher.ENABLE_IMAGES", False):
            await publisher.publish_article(normal_art)
            assert mock_bot.send_message.call_args.kwargs["disable_notification"] is True
            assert "🔴 بەپەلە | " not in mock_bot.send_message.call_args.kwargs["text"]

            await publisher.publish_article(breaking_art)
            assert mock_bot.send_message.call_args.kwargs["disable_notification"] is False
            assert "🔴 بەپەلە | " in mock_bot.send_message.call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_daily_digest_respects_notify_config(self, tmp_path):
        assert NOTIFY_DAILY_DIGEST is False
        publisher = TelegramPublisher()
        mock_bot = MagicMock()
        mock_bot.send_photo = AsyncMock(return_value=MagicMock(message_id=101))
        mock_bot.send_message = AsyncMock(return_value=MagicMock(message_id=102))
        publisher._bot = mock_bot
        publisher._channel_id = "@KurdishGlobalNews"

        card = tmp_path / "digest.jpg"
        card.write_bytes(b"jpeg")
        await publisher.publish_daily_digest("🌙 <b>پوختەی شەوانە</b>", card)
        assert mock_bot.send_photo.call_args.kwargs["disable_notification"] is True

        card.write_bytes(b"jpeg")
        with patch("tg_bot.publisher.NOTIFY_DAILY_DIGEST", True):
            await publisher.publish_daily_digest("🌙 " + ("x" * 1100), card)
            assert mock_bot.send_photo.call_args.kwargs["disable_notification"] is False
            assert mock_bot.send_message.call_args.kwargs["disable_notification"] is False

    @pytest.mark.asyncio
    async def test_pipeline_gates_breaking_strictly_on_score_9(self, temp_db):
        publisher = MagicMock(spec=TelegramPublisher)
        publisher.publish_article = AsyncMock(return_value=777)
        pipeline = ArticlePipeline(temp_db, publisher, HealthMonitor(), Metrics())

        # Case 1: score=8 with is_breaking=True MUST NOT become breaking
        art8 = _make_article(
            title="Regional border tension escalates after overnight shelling incident",
            url="https://bbc.com/news/tension-8",
        )
        art8.id = await temp_db.insert_article(art8)

        with patch.object(
            pipeline._ranker,
            "evaluate_global_relevance",
            new=AsyncMock(return_value=RelevanceEvaluation(
                score=8, is_breaking=True, category="military_conflict",
                reason="Important but score 8", passed=True,
            )),
        ), patch.object(
            pipeline._extractor,
            "fetch_article_content_and_image",
            new=AsyncMock(return_value=(None, None)),
        ), patch.object(
            pipeline._translator,
            "translate_article",
            new=AsyncMock(return_value=("گرژی سنووری زیاد دەکات", "پوختەی هەواڵەکە بە کوردی سۆرانی.")),
        ):
            ok = await pipeline.process_article(art8)

        assert ok is True
        published_arg = publisher.publish_article.call_args.args[0]
        assert published_arg.is_breaking is False

        # Case 2: score=9 with is_breaking=True triggers breaking
        art9 = _make_article(
            title="Major air defense missile interception over Erbil international base",
            url="https://bbc.com/news/erbil-strike-9",
        )
        art9.id = await temp_db.insert_article(art9)

        with patch.object(
            pipeline._ranker,
            "evaluate_global_relevance",
            new=AsyncMock(return_value=RelevanceEvaluation(
                score=9, is_breaking=True, category="kurdistan",
                reason="Critical Kurdistan security event", passed=True,
            )),
        ), patch.object(
            pipeline._semantic_dedup,
            "check_duplicate",
            new=AsyncMock(return_value=MagicMock(is_duplicate=False)),
        ), patch.object(
            pipeline._extractor,
            "fetch_article_content_and_image",
            new=AsyncMock(return_value=(None, None)),
        ), patch.object(
            pipeline._translator,
            "translate_article",
            new=AsyncMock(return_value=("سیستەمی بەرگری لە هەولێر چالاک کرا", "وردەکاری ڕووداوە بەپەلەکە لە هەولێر.")),
        ):
            ok9 = await pipeline.process_article(art9)

        assert ok9 is True
        published_arg9 = publisher.publish_article.call_args.args[0]
        assert published_arg9.is_breaking is True
        assert published_arg9.priority == Priority.CRITICAL

    @pytest.mark.asyncio
    async def test_card_generator_renders_breaking_red_pill(self):
        gen = NewsCardGenerator()
        art_normal = _make_article(
            translated_title="هێزە جیهانییەکان کۆبوونەوەی بەپەلە ئەنجام دەدەن",
            category="economy_markets",
            is_breaking=False,
        )
        art_breaking = _make_article(
            translated_title="هێزە جیهانییەکان کۆبوونەوەی بەپەلە ئەنجام دەدەن",
            category="economy_markets",
            is_breaking=True,
        )

        normal_bytes = await gen.generate_card_bytes(art_normal)
        breaking_bytes = await gen.generate_card_bytes(art_breaking)

        img_normal = Image.open(io.BytesIO(normal_bytes))
        img_breaking = Image.open(io.BytesIO(breaking_bytes))
        assert img_normal.size == (1280, 720)
        assert img_breaking.size == (1280, 720)

        # Count crimson-red pixels (#dc2626) inside the mid-card pill strip (avoid top-right KGN badge)
        def _count_mid_crimson(img: Image.Image) -> int:
            px = img.load()
            hits = 0
            for y in range(250, 620, 2):
                for x in range(200, 1216, 4):
                    r, g, b = px[x, y]
                    if r > 185 and g < 70 and b < 70:
                        hits += 1
            return hits

        assert _count_mid_crimson(img_normal) == 0
        assert _count_mid_crimson(img_breaking) > 50


class TestSemanticEventDeduplication:
    @pytest.mark.asyncio
    async def test_db_get_recent_published_headlines_respects_12h_window(self, temp_db):
        assert DEDUP_WINDOW_HOURS == 12
        now = datetime.now(timezone.utc)

        art_recent = _make_article(
            title="Recent published summit",
            url="https://reuters.com/recent",
            category="politics_world",
            published_at=(now - timedelta(hours=2)).isoformat(),
            status=ArticleStatus.PUBLISHED,
        )
        art_old = _make_article(
            title="Stale published summit from yesterday",
            url="https://reuters.com/old",
            category="politics_world",
            published_at=(now - timedelta(hours=15)).isoformat(),
            status=ArticleStatus.PUBLISHED,
        )
        art_skipped = _make_article(
            title="Skipped recent story",
            url="https://reuters.com/skipped",
            category="politics_world",
            published_at=(now - timedelta(hours=1)).isoformat(),
            status=ArticleStatus.SKIPPED,
        )

        id_recent = await temp_db.insert_article(art_recent)
        await temp_db.update_article_status(
            id_recent, ArticleStatus.PUBLISHED, published_at=art_recent.published_at
        )
        id_old = await temp_db.insert_article(art_old)
        await temp_db.update_article_status(
            id_old, ArticleStatus.PUBLISHED, published_at=art_old.published_at
        )
        await temp_db.insert_article(art_skipped)

        recent = await temp_db.get_recent_published_headlines(hours=12)
        assert [item["id"] for item in recent] == [id_recent]
        assert recent[0]["original_title"] == "Recent published summit"
        assert recent[0]["category"] == "politics_world"

    @pytest.mark.asyncio
    async def test_stage_a_fast_similarity_skips_without_gemini_call(self, temp_db):
        now = datetime.now(timezone.utc)
        pub = _make_article(
            title="Iran launches ballistic missile strikes against regional military bases",
            url="https://reuters.com/world/iran-strikes-1",
            category="military_conflict",
            published_at=(now - timedelta(hours=1)).isoformat(),
            status=ArticleStatus.PUBLISHED,
        )
        pub_id = await temp_db.insert_article(pub)
        await temp_db.update_article_status(
            pub_id, ArticleStatus.PUBLISHED, published_at=pub.published_at
        )

        dedup = SemanticDeduplicator(temp_db, api_key="gem_test_key", base_backoff=0.0)
        mock_session = _make_json_session(200, {"is_duplicate": False, "matched_id": None, "reason": "N/A"})
        dedup._gemini._get_session = AsyncMock(return_value=mock_session)

        cand = _make_article(
            title="Iran launches ballistic missile strikes on regional military bases",
            url="https://apnews.com/world/iran-strikes-2",
            category="military_conflict",
        )
        res = await dedup.check_duplicate(cand)

        assert res.is_duplicate is True
        assert res.stage == "fast"
        assert res.matched_id == pub_id
        mock_session.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_stage_b_cross_source_gemini_dedup_in_pipeline_saves_skipped_url(
        self, temp_db, caplog
    ):
        now = datetime.now(timezone.utc)
        existing = _make_article(
            title="Israeli warplanes bomb weapon depots near Beirut airport",
            url="https://bbc.com/news/beirut-raids-1",
            category="military_conflict",
            published_at=(now - timedelta(hours=2)).isoformat(),
            status=ArticleStatus.PUBLISHED,
        )
        existing_id = await temp_db.insert_article(existing)
        await temp_db.update_article_status(
            existing_id, ArticleStatus.PUBLISHED, published_at=existing.published_at
        )

        publisher = MagicMock(spec=TelegramPublisher)
        publisher.publish_article = AsyncMock(return_value=999)
        pipeline = ArticlePipeline(temp_db, publisher, HealthMonitor(), Metrics())

        # Replace semantic dedup session with mocked Gemini semantic match
        pipeline._semantic_dedup = SemanticDeduplicator(
            temp_db, api_key="gem_k1,gem_k2", base_backoff=0.0
        )
        mock_session = _make_json_session(
            200,
            {
                "is_duplicate": True,
                "matched_id": existing_id,
                "reason": "Both report the same overnight Israeli airstrikes near Beirut airport.",
            },
        )
        pipeline._semantic_dedup._gemini._get_session = AsyncMock(return_value=mock_session)

        candidate = _make_article(
            title="Massive explosions shake southern Lebanese capital amid aerial bombardment",
            url="https://aljazeera.com/news/lebanon-explosions-99",
            category="military_conflict",
        )
        candidate.id = await temp_db.insert_article(candidate)

        with caplog.at_level(logging.INFO), patch.object(
            pipeline._ranker,
            "evaluate_global_relevance",
            new=AsyncMock(return_value=RelevanceEvaluation(
                score=8, is_breaking=False, category="military_conflict",
                reason="Major regional escalation", passed=True,
            )),
        ), patch.object(
            pipeline._extractor,
            "fetch_article_content_and_image",
            new=AsyncMock(return_value=("Should not scrape", None)),
        ) as mock_scrape, patch.object(
            pipeline._translator,
            "translate_article",
            new=AsyncMock(return_value=("Should not translate", "Body")),
        ) as mock_translate:
            published = await pipeline.process_article(candidate)

        assert published is False
        mock_scrape.assert_not_called()
        mock_translate.assert_not_called()
        publisher.publish_article.assert_not_called()

        db_cand = await temp_db.get_article_by_canonical_url(candidate.canonical_url)
        assert db_cand is not None
        assert db_cand.status == ArticleStatus.REJECTED_DUPLICATE
        assert await temp_db.is_url_skipped(candidate.canonical_url) is True
        assert (
            f"SKIPPED [Duplicate Event]: {candidate.original_title} — matches #{existing_id}"
            in caplog.text
        )

    @pytest.mark.asyncio
    async def test_stage_b_prioritizes_same_category_and_caps_at_25_with_fail_open(
        self, temp_db
    ):
        now = datetime.now(timezone.utc)
        for i in range(18):
            a = _make_article(
                title=f"Global Diplomacy Accord Update Alpha {i}",
                url=f"https://example.com/pol/{i}",
                category="politics_world",
                translated_title=f"نوێکاری دیپلۆماسی ژمارە {i}",
                published_at=(now - timedelta(minutes=i + 1)).isoformat(),
                status=ArticleStatus.PUBLISHED,
            )
            aid = await temp_db.insert_article(a)
            await temp_db.update_article_status(aid, ArticleStatus.PUBLISHED, published_at=a.published_at)

        for i in range(12):
            a = _make_article(
                title=f"Semiconductor AI Breakthrough Release Beta {i}",
                url=f"https://example.com/ai/{i}",
                category="ai_tech",
                translated_title=f"پێشکەوتنی زیرەکی دەستکرد ژمارە {i}",
                published_at=(now - timedelta(hours=3, minutes=i)).isoformat(),
                status=ArticleStatus.PUBLISHED,
            )
            aid = await temp_db.insert_article(a)
            await temp_db.update_article_status(aid, ArticleStatus.PUBLISHED, published_at=a.published_at)

        dedup = SemanticDeduplicator(temp_db, api_key="gem_k1", base_backoff=0.0)
        captured_payloads: list[dict] = []

        resp = AsyncMock()
        resp.status = 200
        resp.json = AsyncMock(return_value={
            "candidates": [{"content": {"parts": [{"text": json.dumps({
                "is_duplicate": False, "matched_id": None, "reason": "Distinct model launch."
            })}]}}]
        })
        session = MagicMock()

        def _post_side_effect(*args, **kwargs):
            captured_payloads.append(kwargs["json"])
            return _MockResponseContext(resp)

        session.post = MagicMock(side_effect=_post_side_effect)
        dedup._gemini._get_session = AsyncMock(return_value=session)

        cand = _make_article(
            title="New Open Source Robotics Foundation Unveils Hardware Chip",
            url="https://techcrunch.com/robotics-chip",
            category="ai_tech",
        )
        res = await dedup.check_duplicate(cand)
        assert res.is_duplicate is False
        assert res.stage == "gemini"

        prompt_text = captured_payloads[0]["contents"][0]["parts"][0]["text"]
        assert "RECENTLY_PUBLISHED (25 headlines):" in prompt_text
        json_part = prompt_text.split("headlines):\n", 1)[1]
        sent_items = json.loads(json_part)
        assert len(sent_items) == 25
        # All 12 same-category (ai_tech) items appear before politics_world items
        assert all(item["category"] == "ai_tech" for item in sent_items[:12])

        # Verify fail-open when Gemini throws network error
        broken_session = MagicMock()
        broken_session.post = MagicMock(side_effect=RuntimeError("Timeout"))
        dedup._gemini._get_session = AsyncMock(return_value=broken_session)

        fail_open_res = await dedup.check_duplicate(cand)
        assert fail_open_res.is_duplicate is False
        assert fail_open_res.used_fallback is True
        assert fail_open_res.stage == "fail_open"
