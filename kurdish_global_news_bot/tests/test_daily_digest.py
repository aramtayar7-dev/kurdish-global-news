"""
Unit tests for Automated Nightly Kurdish News Digest (--daily-digest),
24h article selection, duplicate date guard, Gemini digest synthesis,
1280x720 nightly cover card, CLI routing, and GitHub Actions workflow.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from config import CARD_HEIGHT, CARD_WIDTH, ArticleStatus
from core.engine import NewsEngine
from main import main, parse_args
from processing.card_generator import NewsCardGenerator
from processing.digest_generator import (
    DIGEST_HEADER_PREFIX,
    DailyDigestGenerator,
    format_kurdish_date,
)
from processing.gemini_client import GeminiAPIError
from storage.database import Database, get_kurdistan_date_str
from storage.models import Article
from tg_bot.publisher import TelegramPublisher


def _make_published_article(
    idx: int,
    hours_ago: float = 2.0,
    score: float = 0.75,
    category: str = "politics_world",
    status: str = ArticleStatus.PUBLISHED,
) -> Article:
    pub_dt = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    art = Article(
        original_url=f"https://example.com/story-{idx}",
        canonical_url=f"https://example.com/story-{idx}",
        original_title=f"Global Story Headline Number {idx}",
        translated_title=f"سەردێری هەواڵی کوردی ژمارە {idx}",
        translated_summary=f"ڕاپۆرتی تەواوی هەواڵی ژمارە {idx} بە وردەکارییەوە. هەموو لایەنەکان لێدوانیان دا.",
        source_name="Reuters",
        category=category,
        importance_score=score,
        status=status,
        published_at=pub_dt.isoformat(),
        discovered_at=pub_dt.isoformat(),
    )
    art.compute_content_hash()
    art.compute_event_fingerprint()
    return art


class TestDigestDatabaseAndTimezone:
    """Test 24h published queries, Kurdistan UTC+3 date, and daily_digests lock."""

    def test_kurdistan_utc_plus_3_date_boundary(self):
        # 2026-11-24 22:00 UTC == 2026-11-25 01:00 Erbil (UTC+3)
        utc_dt = datetime(2026, 11, 24, 22, 0, tzinfo=timezone.utc)
        assert get_kurdistan_date_str(utc_dt) == "2026-11-25"
        assert "تشرینی دووەم" in format_kurdish_date("2026-11-24")

    @pytest.mark.asyncio
    async def test_get_published_articles_last_24h_filters_and_prioritizes(self, tmp_path):
        db = Database(str(tmp_path / "digest_24h.db"))
        await db.connect()

        # Insert 12 recent published articles, 2 stale (>24h), and 1 skipped article
        await db.insert_article(_make_published_article(100, hours_ago=30.0, score=0.99, category="kurdistan"))
        await db.insert_article(_make_published_article(101, hours_ago=1.0, score=0.99, status=ArticleStatus.SKIPPED))

        # Kurdistan story with 0.78 score should outrank global 0.85 story due to Kurdistan priority boost
        await db.insert_article(_make_published_article(1, hours_ago=3.0, score=0.78, category="kurdistan"))
        await db.insert_article(_make_published_article(2, hours_ago=2.0, score=0.85, category="military_conflict"))
        for i in range(3, 13):
            await db.insert_article(
                _make_published_article(i, hours_ago=float(i), score=0.40 + i * 0.02, category="ai_tech")
            )

        candidates = await db.get_published_articles_last_24h(limit=10)
        assert len(candidates) == 10
        urls = {a.canonical_url for a in candidates}
        assert "https://example.com/story-100" not in urls  # Too old (>24h)
        assert "https://example.com/story-101" not in urls  # Status SKIPPED
        assert candidates[0].canonical_url == "https://example.com/story-1"  # Kurdistan prioritized first
        await db.close()

    @pytest.mark.asyncio
    async def test_duplicate_digest_prevented_unless_forced(self, tmp_path, caplog):
        db = Database(str(tmp_path / "digest_dup.db"))
        await db.connect()
        today_kurd = get_kurdistan_date_str()

        for i in range(4):
            await db.insert_article(_make_published_article(i, hours_ago=2.0, score=0.80))

        await db.mark_digest_published(today_kurd, story_count=4, digest_text="Test digest", telegram_message_id=101)
        assert await db.is_digest_published_for_date(today_kurd) is True
        await db.close()

        # Run without force -> skips immediately
        engine = NewsEngine()
        engine._db = Database(str(tmp_path / "digest_dup.db"))
        engine._publisher.initialize = AsyncMock()
        with caplog.at_level(logging.INFO, logger="core.engine"):
            result = await engine.run_daily_digest(force=False)

        assert result is False
        engine._publisher.initialize.assert_not_called()
        assert any("already published" in r.message for r in caplog.records)

        # Run with force=True -> overrides and publishes
        engine_forced = NewsEngine()
        engine_forced._db = Database(str(tmp_path / "digest_dup.db"))
        engine_forced._publisher.initialize = AsyncMock()
        engine_forced._publisher.test_connection = AsyncMock(return_value=True)
        engine_forced._publisher.cleanup = AsyncMock()
        engine_forced._publisher.publish_daily_digest = AsyncMock(return_value=5555)
        engine_forced._digest_generator = DailyDigestGenerator(api_key="")

        forced_result = await engine_forced.run_daily_digest(force=True)
        assert forced_result is True
        engine_forced._publisher.publish_daily_digest.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_engine_skips_when_fewer_than_3_articles(self, tmp_path, caplog):
        db = Database(str(tmp_path / "digest_few.db"))
        await db.connect()
        await db.insert_article(_make_published_article(1, hours_ago=1.0))
        await db.insert_article(_make_published_article(2, hours_ago=2.0))
        await db.close()

        engine = NewsEngine()
        engine._db = Database(str(tmp_path / "digest_few.db"))
        engine._publisher.initialize = AsyncMock()

        with caplog.at_level(logging.INFO, logger="core.engine"):
            published = await engine.run_daily_digest()

        assert published is False
        engine._publisher.initialize.assert_not_called()
        assert any("SKIPPED: fewer than 3 articles in the last 24h" in r.message for r in caplog.records)


class TestGeminiDigestWriterAndCoverCard:
    """Test Gemini digest formatting, fallback, and 1280x720 nightly cover card."""

    @pytest.mark.asyncio
    async def test_gemini_digest_synthesis_format(self):
        gen = DailyDigestGenerator(api_key="gem_test_key", base_backoff=0.0)
        articles = [
            _make_published_article(1, category="kurdistan"),
            _make_published_article(2, category="military_conflict"),
            _make_published_article(3, category="economy_markets"),
            _make_published_article(4, category="ai_tech"),
            _make_published_article(5, category="cybersecurity"),
        ]

        mock_stories = {
            "stories": [
                {"emoji": "☀️", "summary": "شاندێکی هەرێمی کوردستان لە بەغدا لەسەر دۆسیەی بودجە و نەوت گەیشتە ڕێککەوتنی سەرەتایی."},
                {"emoji": "⚔️", "summary": "هێرشە ئاسمانییەکان لە ناوچە سنوورییەکان گرژییە سەربازییەکانیان زیاد کرد و کۆبوونەوەی بەپەلە سازکرا."},
                {"emoji": "📈", "summary": "نرخی نەوتی خاو لە بازاڕەکانی جیهاندا بە ڕێژەی لە سەدا سێ بەرزبوونەوەی تۆمارکرد."},
                {"emoji": "🤖", "summary": "کۆمپانیا تەکنەلۆژییەکان نەوەیەکی نوێی مۆدێلی زیرەکیی دەستکردیان بۆ شیکاریی خێرا ڕاگەیاند."},
                {"emoji": "🛡️", "summary": "دەزگا ئەمنییەکان هۆشدارییان دا لەسەر شەپۆلێکی نوێی هێرشی سایبەری دژی دامەزراوە داراییەکان."},
            ]
        }

        with patch.object(gen._gemini, "generate_json", new_callable=AsyncMock, return_value=mock_stories):
            result = await gen.generate_digest(articles, digest_date="2026-11-24")

        assert result is not None
        assert result.story_count == 5
        assert result.used_fallback is False
        assert DIGEST_HEADER_PREFIX in result.digest_text
        assert "@KurdishGlobalNews" in result.digest_text
        assert "☀️" in result.digest_text
        assert len(result.digest_text) <= 1024

    @pytest.mark.asyncio
    async def test_digest_fallback_when_gemini_fails(self):
        gen = DailyDigestGenerator(api_key="gem_test_key", base_backoff=0.0)
        articles = [_make_published_article(i, category="kurdistan") for i in range(1, 5)]

        with patch.object(gen._gemini, "generate_json", new_callable=AsyncMock, side_effect=GeminiAPIError("429")):
            result = await gen.generate_digest(articles, digest_date="2026-11-24")

        assert result is not None
        assert result.used_fallback is True
        assert result.story_count == 4
        assert DIGEST_HEADER_PREFIX in result.digest_text

    @pytest.mark.asyncio
    async def test_generate_digest_cover_card_1280x720(self):
        card_gen = NewsCardGenerator()
        card_path = await card_gen.generate_digest_card(
            date_label="24ی تشرینی دووەمی 2026",
            story_count=6,
        )
        try:
            assert card_path.exists()
            with Image.open(card_path) as im:
                assert im.size == (CARD_WIDTH, CARD_HEIGHT)
                assert im.size == (1280, 720)
                assert im.format == "JPEG"
        finally:
            card_path.unlink(missing_ok=True)


class TestDigestPublisherCLISplitAndWorkflow:
    """Test publisher <=1024 vs >1024 behavior, CLI args, and daily_digest.yml."""

    @pytest.mark.asyncio
    async def test_publish_daily_digest_under_and_over_1024_chars(self, tmp_path):
        publisher = TelegramPublisher()
        mock_bot = MagicMock()
        mock_bot.send_photo = AsyncMock(return_value=MagicMock(message_id=111))
        mock_bot.send_message = AsyncMock(return_value=MagicMock(message_id=222))
        publisher._bot = mock_bot

        card1 = tmp_path / "card1.jpg"
        card1.write_bytes(b"fake-jpg-1")
        short_digest = "<b>🌙 پوختەی گرنگترین هەواڵەکانی ئەمڕۆ</b>\n☀️ هەواڵی کوردستان."
        msg_id1 = await publisher.publish_daily_digest(short_digest, card1)
        assert msg_id1 == 111
        mock_bot.send_photo.assert_awaited_once()
        mock_bot.send_message.assert_not_called()
        assert not card1.exists()

        # > 1024 chars triggers short photo header + send_message follow-up with preview disabled
        mock_bot.send_photo.reset_mock()
        card2 = tmp_path / "card2.jpg"
        card2.write_bytes(b"fake-jpg-2")
        long_digest = "🌙 پوختە\n" + ("☀️ ڕاپۆرتی درێژ. " * 90)
        assert len(long_digest) > 1024
        msg_id2 = await publisher.publish_daily_digest(long_digest, card2, header_caption="<b>🌙 پوختە</b>")
        assert msg_id2 == 222
        mock_bot.send_photo.assert_awaited_once()
        mock_bot.send_message.assert_awaited_once()
        assert mock_bot.send_message.call_args[1]["link_preview_options"].is_disabled is True
        assert not card2.exists()

    def test_cli_daily_digest_and_force_flags(self):
        args = parse_args(["--daily-digest", "--force"])
        assert args.daily_digest is True
        assert args.force is True
        assert args.once is False

        with patch("main.validate_config", return_value=[]), \
             patch("main.run_daily_digest", new_callable=AsyncMock) as mock_digest:
            rc = main(["--daily-digest", "--force"])
            assert rc == 0
            mock_digest.assert_awaited_once_with(force=True)

    def test_daily_digest_github_workflow(self):
        repo_root = Path(__file__).resolve().parents[2]
        wf_path = repo_root / ".github" / "workflows" / "daily_digest.yml"
        assert wf_path.exists(), f"Missing workflow at {wf_path}"

        content = wf_path.read_text(encoding="utf-8")
        assert "cron: '30 18 * * *'" in content
        assert "workflow_dispatch:" in content
        assert "force:" in content
        assert "path: kurdish_global_news_bot/data" in content
        assert "bot-db-${{ github.run_id }}" in content
        assert "bot-db-" in content
        assert "libfribidi0 libharfbuzz0b libraqm0" in content
        assert "net.ipv6.conf.all.disable_ipv6=1" in content
        assert "python main.py --daily-digest" in content
        assert "GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}" in content
        assert "GROQ_API_KEY" not in content
