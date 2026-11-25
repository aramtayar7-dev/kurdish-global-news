"""
Comprehensive automated tests for Smart Pacing & Persistent Publish Queue (`kurdish_global_news_bot`):
  1. Per-cycle burst cap (`MAX_NORMAL_POSTS_PER_CYCLE=3`): 3 publish (Bashur P&W prioritized), remaining 3 persist as QUEUED.
  2. Intra-cycle delay accuracy: `asyncio.sleep(INTRA_CYCLE_POST_DELAY_SECONDS)` called exactly N-1 times (never before 1st or after last).
  3. Zero-Gemini queue drain: next cycle drains QUEUED posts with 0 Gemini calls and updates status to PUBLISHED + telegram_message_id.
  4. Breaking fast-lane (`is_breaking=True`, `score >= 9.0`): publishes immediately with 0 sleep, bypassing burst cap and cooldown.
  5. Cross-cycle cooldown: `< MIN_MINUTES_BETWEEN_NORMAL_POSTS` defers normal posts cleanly to QUEUED.
  6. Stale queue expiration (> MAX_QUEUED_ARTICLE_AGE_HOURS) & backlog cap (<= MAX_QUEUE_SIZE).
  7. Pre-drain safety re-check: invalid Sorani or post-queue duplicate expires without publishing.
  8. Queued THREAD_UPDATE preserves `thread_root_message_id` and publishes as Telegram Reply (`reply_to_message_id`).
  9. Discovery deduplication of QUEUED URLs: skipped before evaluation/translation.
 10. Feature flag (`ENABLE_SMART_PACING=False`) & Supabase queue/pacing storage parity.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import config
from config import (
    ArticleStatus,
    ENABLE_SMART_PACING,
    INTRA_CYCLE_POST_DELAY_SECONDS,
    MAX_NORMAL_POSTS_PER_CYCLE,
    MAX_QUEUED_ARTICLE_AGE_HOURS,
    MAX_QUEUE_SIZE,
    MIN_MINUTES_BETWEEN_NORMAL_POSTS,
)
from core.engine import NewsEngine
from core.importance_ranker import RelevanceEvaluation
from core.pipeline import ArticlePipeline
from monitoring.health import HealthMonitor
from monitoring.metrics import Metrics
from sources.base import RawArticle
from storage.database import Database
from storage.models import Article
from storage.supabase_db import SupabaseDatabase
from tg_bot.publisher import TelegramPublisher


def _fresh_iso(minutes_ago: float = 2.0) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


def _make_raw(
    url: str,
    title: str,
    summary: str = "",
    source_name: str = "Reuters",
    category: str = "politics_world",
    region: str = "",
    minutes_ago: float = 2.0,
) -> RawArticle:
    return RawArticle(
        url=url,
        title=title,
        summary=summary or (
            f"{title}. Senior defense and diplomatic officials held emergency consultations today "
            "regarding regional security alignments, military posture, and cross-border coordination."
        ),
        source_name=source_name,
        category=category,
        region=region,
        publication_date=_fresh_iso(minutes_ago=minutes_ago),
    )


def _make_article(
    url: str,
    title: str,
    translated_title: str = "هەرێمی کوردستان پێشهاتێکی نوێی سیاسی ڕاگەیاند",
    translated_summary: str = "حکوومەتی هەرێمی کوردستان لەسەر دۆخی ئەمنی و سیاسی ناوچەکە بەیاننامەیەکی فەرمی بڵاوکردەوە.",
    category: str = "kurdistan",
    region: str = "South Kurdistan",
    source_name: str = "Kurdistan24",
    score: float = 8.2,
    is_breaking: bool = False,
    status: str = ArticleStatus.QUEUED,
    minutes_ago: float = 35.0,
    published_minutes_ago: float | None = None,
    telegram_message_id: int | None = None,
    thread_root_message_id: int | None = None,
    event_key: str = "",
    event_entities: list[str] | None = None,
) -> Article:
    ts = _fresh_iso(minutes_ago=minutes_ago)
    pub_ts = _fresh_iso(minutes_ago=published_minutes_ago) if published_minutes_ago is not None else None
    art = Article(
        original_url=url,
        canonical_url=Article.canonicalize_url(url),
        original_title=title,
        normalized_title=Article.normalize_title(title),
        original_summary=(
            f"{title}. Officials issued a detailed security briefing covering diplomatic talks, "
            "border operations, and strategic coordination."
        ),
        translated_title=translated_title,
        translated_summary=translated_summary,
        source_name=source_name,
        category=category,
        priority="CRITICAL" if is_breaking else ("HIGH" if category == "kurdistan" else "NORMAL"),
        importance_score=score,
        is_breaking=is_breaking,
        region=region,
        publication_date=ts,
        discovered_at=ts,
        processed_at=ts if status == ArticleStatus.QUEUED else None,
        published_at=pub_ts,
        status=status,
        telegram_message_id=telegram_message_id,
        thread_root_message_id=thread_root_message_id,
        is_thread_update=(thread_root_message_id is not None),
        event_key=event_key,
        event_entities=list(event_entities or []),
    )
    art.compute_content_hash()
    art.compute_event_fingerprint()
    return art


def _build_mock_publisher(tmp_path: Path, start_msg_id: int = 2000) -> tuple[TelegramPublisher, MagicMock, list[int]]:
    counter = {"id": start_msg_id}
    sent_ids: list[int] = []

    async def _fake_send_photo(*args, **kwargs):
        counter["id"] += 1
        sent_ids.append(counter["id"])
        msg = MagicMock()
        msg.message_id = counter["id"]
        return msg

    async def _fake_generate_card(*args, **kwargs) -> Path:
        card_idx = counter["id"] + len(sent_ids) + 1
        fake_card = tmp_path / f"dummy_card_{card_idx}_{id(args)}.jpg"
        fake_card.write_bytes(b"FAKE_JPEG")
        return fake_card

    publisher = TelegramPublisher()
    publisher._channel_id = "@KurdishGlobalNews"
    mock_bot = AsyncMock()
    mock_bot.send_photo = AsyncMock(side_effect=_fake_send_photo)
    mock_bot.send_message = AsyncMock(side_effect=_fake_send_photo)
    publisher._bot = mock_bot
    publisher._image_handler.download_image = AsyncMock(return_value=None)
    publisher._card_generator.generate_card_file = AsyncMock(side_effect=_fake_generate_card)
    return publisher, mock_bot, sent_ids


def _build_engine(db: Database, publisher: TelegramPublisher, sources: list | None = None) -> NewsEngine:
    engine = NewsEngine()
    engine._db = db
    engine._publisher = publisher
    engine._sources = list(sources or [])
    engine._pipeline = ArticlePipeline(
        db=db,
        publisher=publisher,
        health=engine._health,
        metrics=engine._metrics,
    )
    engine._pipeline._extractor.fetch_article_content_and_image = AsyncMock(return_value=("", None))
    return engine


_SORANI_MAP = {
    "Erbil": ("هەولێر و بەغدا رێککەوتنی ئەمنییان واژۆ کرد", "بەرپرسانی هەرێمی کوردستان و عێراق لە هەولێر کۆبوونەوەی هاوبەشیان ئەنجامدا."),
    "Peshmerga": ("پێشمەرگە ئامادەباشی لە سنووری کەرکووک زیاد دەکات", "هێزەکانی پێشمەرگە سەنگەرەکانی بەرگری لە باشووری کەرکووک قایم دەکەن."),
    "Sulaymaniyah": ("سلێمانی میوانداری دیداری دەستووری دەکات", "سەرکردە سیاسییەکان لە سلێمانی پرۆژەی یاسایی نوێ تاوتوێ دەکەن."),
    "Duhok": ("دهۆک هەنگاوی نوێی هاوئاهەنگی سنووری ڕاگەیاند", "بەرپرسانی پارێزگای دهۆک ڕێکاری نوێی ئەمنیان لە دەروازەکان پەسەند کرد."),
    "Ankara": ("ئەنقەرە و تاران گفتوگۆی دیپلۆماسی ئەنجام دەدەن", "وەزیرانی دەرەوەی تورکیا و ئێران دۆخی سنووری و ناوچەکەیان تاوتوێ کرد."),
    "Geneva": ("نەتەوە یەکگرتووەکان لە جنێڤ کۆبوونەوەی بەپەلە ساز دەکات", "ئەنجومەنی ئاسایش ڕەشنووسی ئاگربەست و رێڕەوە مرۆییەکان تاوتوێ دەکات."),
    "Tehran": ("پەرلەمانی ئێران بودجەی بەرگری تاوتوێ دەکات", "ئەندامانی پەرلەمان لە تاران دانیشتنی تایبەتیان بۆ پلانە ستراتیژییەکان ئەنجامدا."),
    "Moscow": ("مۆسکۆ هۆشداری لە گرژییەکانی ئەورووپا دەدات", "وەزارەتی بەرگری ڕووسیا بەیاننامەیەکی نوێی لەسەر دۆخی سنوور بڵاوکردەوە."),
}


def _sorani_for(title: str) -> tuple[str, str]:
    for key, pair in _SORANI_MAP.items():
        if key in title:
            return pair
    return (f"پێشهاتی سیاسی نوێ دەربارەی {title[:20]}", "ڕاپۆرتی ڕۆژنامەوانی سۆرانی تەواو و بێلایەن لەسەر دۆخەکە.")


@pytest.fixture
async def temp_db(tmp_path):
    db = Database(str(tmp_path / "test_smart_pacing.db"))
    await db.connect()
    yield db
    await db.close()


class TestSmartPacingConfig:
    def test_default_pacing_constants_and_queue_statuses(self):
        assert ENABLE_SMART_PACING is True
        assert MAX_NORMAL_POSTS_PER_CYCLE == 3
        assert INTRA_CYCLE_POST_DELAY_SECONDS == 25.0
        assert MIN_MINUTES_BETWEEN_NORMAL_POSTS == 5.0
        assert MAX_QUEUED_ARTICLE_AGE_HOURS == 4
        assert MAX_QUEUE_SIZE == 12
        assert ArticleStatus.QUEUED == "QUEUED"
        assert ArticleStatus.EXPIRED == "EXPIRED"


class TestSmartPacingAndQueueDrain:
    @pytest.fixture(autouse=True)
    def _legacy_smart_pacing_mode(self, monkeypatch):
        """Isolate Idea-2 3-post/25s legacy Smart Pacing tests from the 10-post 4:4:2 batch engine."""
        monkeypatch.setattr(config, "ENABLE_BATCH_PACING_ENGINE", False)

    @pytest.mark.asyncio
    async def test_burst_cap_intra_delay_and_zero_gemini_queue_drain(self, temp_db, tmp_path, monkeypatch):
        """
        Tests 1, 2 & 3 combined end-to-end across two cycles:
          - Cycle 1 with 6 eligible normal stories (4 Bashur + 2 Global) publishes exactly 3
            with asyncio.sleep(25) called exactly 2 times, persisting 3 as QUEUED.
          - Cycle 2 (with cooldown elapsed and no Gemini mocks allowed to be called) drains all 3
            QUEUED stories using stored Kurdish text -> status=PUBLISHED + telegram_message_id saved!
        """
        monkeypatch.setattr(config, "ENABLE_SMART_PACING", True)
        monkeypatch.setattr(config, "MAX_NORMAL_POSTS_PER_CYCLE", 3)
        monkeypatch.setattr(config, "INTRA_CYCLE_POST_DELAY_SECONDS", 25.0)
        monkeypatch.setattr(config, "MIN_MINUTES_BETWEEN_NORMAL_POSTS", 5.0)

        publisher, mock_bot, _ = _build_mock_publisher(tmp_path, start_msg_id=1000)

        raw_batch = [
            _make_raw("https://k24.net/erbil-security-1", "Erbil and Baghdad sign border security agreement", category="kurdistan", region="South Kurdistan", source_name="Kurdistan24"),
            _make_raw("https://rudaw.net/peshmerga-kirkuk-2", "Peshmerga strengthens defense trenches near Kirkuk", category="kurdistan", region="South Kurdistan", source_name="Rudaw"),
            _make_raw("https://pukmedia.com/sulaymaniyah-talks-3", "Sulaymaniyah hosts constitutional federalism summit", category="kurdistan", region="South Kurdistan", source_name="PUKMedia"),
            _make_raw("https://k24.net/duhok-customs-4", "Duhok governor announces joint border protocol", category="kurdistan", region="South Kurdistan", source_name="Kurdistan24"),
            _make_raw("https://reuters.com/ankara-tehran-5", "Ankara and Tehran diplomats hold border talks", category="politics_world", source_name="Reuters"),
            _make_raw("https://bbc.com/geneva-un-6", "Geneva UN Security Council holds ceasefire session", category="politics_world", source_name="BBC"),
        ]

        mock_source = MagicMock()
        mock_source.name = "MockRSS"
        mock_source.source_type = "rss"
        mock_source.enabled = True
        mock_source.fetch_articles = AsyncMock(return_value=raw_batch)

        engine = _build_engine(temp_db, publisher, sources=[mock_source])

        async def _fake_batch_eval(articles, batch_size=12):
            evals = []
            for idx, art in enumerate(articles):
                is_bash = any(k in art.original_title for k in ("Erbil", "Peshmerga", "Sulaymaniyah", "Duhok"))
                evals.append(
                    RelevanceEvaluation(
                        score=9 if is_bash else 8,
                        is_breaking=False,
                        category="kurdistan" if is_bash else "politics_world",
                        reason="Politics/security",
                        passed=True,
                        event_key=f"evt_{idx}_{art.original_title[:12].lower().replace(' ', '_')}",
                        event_entities=[art.original_title.split()[0].lower(), "security", "politics"],
                    )
                )
            return evals

        rank_mock = AsyncMock(side_effect=_fake_batch_eval)
        trans_mock = AsyncMock(side_effect=lambda title, summary="": _sorani_for(title))
        engine._pipeline._ranker.evaluate_batch_relevance = rank_mock
        engine._pipeline._translator.translate_article = trans_mock

        sleep_calls: list[float] = []

        async def _fake_sleep(secs: float):
            sleep_calls.append(secs)

        with patch("core.engine.asyncio.sleep", side_effect=_fake_sleep):
            await engine._discovery_cycle()

        # Test 1: Exactly 3 published (prioritizing Southern Kurdistan P&W), 3 queued
        assert mock_bot.send_photo.await_count == 3
        published_rows = await temp_db.get_recent_published_articles(limit=20)
        assert len(published_rows) == 3
        first_two_cats = [a.category for a in published_rows]
        assert first_two_cats.count("kurdistan") >= 2

        queued_rows = await temp_db.get_queued_articles()
        assert len(queued_rows) == 3
        for q in queued_rows:
            assert q.status == ArticleStatus.QUEUED
            assert q.translated_title != ""
            assert q.translated_summary != ""

        # Test 2: asyncio.sleep(25) called EXACTLY 2 times (between 1->2 and 2->3 only)
        assert sleep_calls == [25.0, 25.0]

        # Backdate published_at by 10m to pass MIN_MINUTES_BETWEEN_NORMAL_POSTS cooldown cleanly
        ten_min_ago = _fresh_iso(minutes_ago=10.0)
        for p in published_rows:
            await temp_db.update_article_status(
                p.id, ArticleStatus.PUBLISHED, published_at=ten_min_ago, telegram_message_id=p.telegram_message_id
            )

        # Cycle 2: 0 new articles from sources; Gemini MUST NOT be called at all!
        mock_source.fetch_articles = AsyncMock(return_value=[])
        rank_mock.reset_mock()
        trans_mock.reset_mock()
        mock_bot.send_photo.reset_mock()
        sleep_calls.clear()

        with patch("core.engine.asyncio.sleep", side_effect=_fake_sleep):
            await engine._discovery_cycle()

        # Test 3: All 3 QUEUED articles drained cleanly with ZERO Gemini calls
        assert rank_mock.await_count == 0
        assert trans_mock.await_count == 0
        assert mock_bot.send_photo.await_count == 3
        assert sleep_calls == [25.0, 25.0]

        remaining_queued = await temp_db.get_queued_articles()
        assert len(remaining_queued) == 0

        all_pub = await temp_db.get_recent_published_articles(limit=20)
        assert len(all_pub) == 6
        assert all(a.telegram_message_id is not None for a in all_pub)

    @pytest.mark.asyncio
    async def test_breaking_fast_lane_bypasses_cooldown_and_cycle_cap_with_zero_sleep(self, temp_db, tmp_path, monkeypatch):
        """Test 4: Genuine breaking news publishes immediately with 0 delay even during cooldown & full cap."""
        monkeypatch.setattr(config, "ENABLE_SMART_PACING", True)
        monkeypatch.setattr(config, "MAX_NORMAL_POSTS_PER_CYCLE", 1)
        monkeypatch.setattr(config, "INTRA_CYCLE_POST_DELAY_SECONDS", 25.0)
        monkeypatch.setattr(config, "MIN_MINUTES_BETWEEN_NORMAL_POSTS", 5.0)

        # Recent normal post 1 minute ago triggers cross-cycle cooldown
        recent_norm = _make_article(
            "https://reuters.com/recent-norm",
            "Previous normal diplomatic briefing",
            category="politics_world",
            is_breaking=False,
            status=ArticleStatus.PUBLISHED,
            published_minutes_ago=1.0,
            telegram_message_id=3001,
        )
        await temp_db.insert_article(recent_norm)

        queued_norm = _make_article(
            "https://reuters.com/queued-norm",
            "Ankara trade talks continue quietly",
            translated_title="گفتوگۆکانی ئەنقەرە بەردەوامن",
            translated_summary="بەرپرسان لە ئەنقەرە کۆبوونەوەی ئاساییان ئەنجامدا.",
            category="politics_world",
            is_breaking=False,
            status=ArticleStatus.QUEUED,
            minutes_ago=25.0,
            event_key="ankara_trade_talks",
            event_entities=["ankara", "trade", "talks"],
        )
        await temp_db.insert_article(queued_norm)

        # Two genuine breaking emergencies queued/arriving -> MUST bypass cooldown & cap with 0 sleep!
        brk_raw = _make_raw(
            "https://rudaw.net/erbil-missile-breaking",
            "Breaking airstrikes and explosions hit military depot near Erbil",
            category="kurdistan",
            region="South Kurdistan",
            source_name="Rudaw",
        )
        mock_source = MagicMock()
        mock_source.name = "Rudaw"
        mock_source.source_type = "rss"
        mock_source.enabled = True
        mock_source.fetch_articles = AsyncMock(return_value=[brk_raw])

        publisher, mock_bot, _ = _build_mock_publisher(tmp_path, start_msg_id=4000)
        engine = _build_engine(temp_db, publisher, sources=[mock_source])
        engine._pipeline._ranker.evaluate_batch_relevance = AsyncMock(
            return_value=[
                RelevanceEvaluation(
                    score=10,
                    is_breaking=True,
                    category="kurdistan",
                    reason="Active military strike emergency",
                    passed=True,
                    event_key="erbil_depot_breaking_strike",
                    event_entities=["erbil", "airstrike", "explosion"],
                )
            ]
        )
        engine._pipeline._translator.translate_article = AsyncMock(
            return_value=(
                "هێرشی مووشەکی و تەقینەوە کۆگایەکی سەربازی لە هەولێر کردە ئامانج",
                "هێرشی بەپەلەی مووشەکی و تەقینەوەی گەورە لە نزیک هەولێر تۆمارکرا.",
            )
        )

        sleep_calls: list[float] = []
        with patch("core.engine.asyncio.sleep", side_effect=lambda s: sleep_calls.append(s)):
            await engine._discovery_cycle()

        # Only the breaking story published; normal queued story waited cleanly; 0 sleep called!
        assert mock_bot.send_photo.await_count == 1
        assert sleep_calls == []
        saved_brk = await temp_db.get_article_by_canonical_url(brk_raw.url)
        assert saved_brk is not None
        assert saved_brk.status == ArticleStatus.PUBLISHED
        assert saved_brk.is_breaking is True

        # Queued normal remained QUEUED because cooldown was active (1m < 5m)
        still_queued = await temp_db.get_queued_articles()
        assert len(still_queued) == 1
        assert still_queued[0].canonical_url == queued_norm.canonical_url

    @pytest.mark.asyncio
    async def test_cross_cycle_cooldown_defers_normal_posts(self, temp_db, tmp_path, monkeypatch):
        """Test 5: When last normal post published 2 minutes ago (< 5m), normal posts defer to QUEUED."""
        monkeypatch.setattr(config, "ENABLE_SMART_PACING", True)
        monkeypatch.setattr(config, "MIN_MINUTES_BETWEEN_NORMAL_POSTS", 5.0)
        monkeypatch.setattr(config, "MAX_NORMAL_POSTS_PER_CYCLE", 3)

        last_post = _make_article(
            "https://bbc.com/last-normal",
            "NATO ministers conclude Brussels meeting",
            category="politics_world",
            is_breaking=False,
            status=ArticleStatus.PUBLISHED,
            published_minutes_ago=2.0,
            telegram_message_id=5001,
        )
        await temp_db.insert_article(last_post)

        raw_new = _make_raw(
            "https://reuters.com/tehran-budget",
            "Tehran parliament debates annual defense budget",
            category="politics_world",
            source_name="Reuters",
        )
        mock_source = MagicMock(name="MockRSS")
        mock_source.name = "MockRSS"
        mock_source.source_type = "rss"
        mock_source.enabled = True
        mock_source.fetch_articles = AsyncMock(return_value=[raw_new])

        publisher, mock_bot, _ = _build_mock_publisher(tmp_path)
        engine = _build_engine(temp_db, publisher, sources=[mock_source])
        engine._pipeline._ranker.evaluate_batch_relevance = AsyncMock(
            return_value=[
                RelevanceEvaluation(
                    score=8,
                    is_breaking=False,
                    category="politics_world",
                    reason="Parliament debate",
                    passed=True,
                    event_key="tehran_defense_budget",
                    event_entities=["tehran", "parliament", "budget"],
                )
            ]
        )
        engine._pipeline._translator.translate_article = AsyncMock(return_value=_sorani_for("Tehran"))

        await engine._discovery_cycle()

        assert mock_bot.send_photo.await_count == 0
        queued = await temp_db.get_queued_articles()
        assert len(queued) == 1
        assert queued[0].canonical_url == Article.canonicalize_url(raw_new.url)
        assert queued[0].translated_title == _sorani_for("Tehran")[0]

    @pytest.mark.asyncio
    async def test_stale_queue_expiration_and_max_queue_size_trim(self, temp_db, tmp_path, monkeypatch):
        """Test 6: Queued items older than MAX_QUEUED_ARTICLE_AGE_HOURS expire; queue size caps at MAX_QUEUE_SIZE."""
        monkeypatch.setattr(config, "ENABLE_SMART_PACING", True)
        monkeypatch.setattr(config, "MAX_QUEUED_ARTICLE_AGE_HOURS", 4)
        monkeypatch.setattr(config, "MAX_QUEUE_SIZE", 3)

        stale_old = _make_article(
            "https://reuters.com/stale-5h",
            "Stale five hour old diplomatic meeting",
            translated_title="کۆبوونەوەی دیپلۆماسی پێنج کاتژمێر لەمەوبەر",
            status=ArticleStatus.QUEUED,
            minutes_ago=310.0,  # 5h 10m > 4h!
        )
        await temp_db.insert_article(stale_old)

        # Insert 5 fresh QUEUED articles of varying priority/score
        for i, (cat, sc) in enumerate(
            [
                ("economy_markets", 7.6),
                ("politics_world", 8.0),
                ("kurdistan", 9.0),
                ("kurdistan", 8.5),
                ("politics_world", 7.8),
            ]
        ):
            fresh = _make_article(
                f"https://example.com/q-{i}",
                f"Queued story {i}",
                translated_title=f"هەواڵی ژمارە {i} لەسەر دۆخی سیاسی",
                category=cat,
                score=sc,
                status=ArticleStatus.QUEUED,
                minutes_ago=15.0 + i,
            )
            await temp_db.insert_article(fresh)

        expired = await temp_db.expire_stale_queued_articles(max_age_hours=4)
        assert expired == 1
        stale_row = await temp_db.get_article_by_canonical_url(stale_old.canonical_url)
        assert stale_row.status == ArticleStatus.EXPIRED

        trimmed = await temp_db.trim_queue(max_size=3)
        assert trimmed == 2
        surviving = await temp_db.get_queued_articles(max_age_hours=4)
        assert len(surviving) == 3
        # Lowest priority/score ones expired cleanly, Bashur 9.0 & 8.5 kept!
        kept_urls = {a.canonical_url for a in surviving}
        assert "https://example.com/q-2" in kept_urls
        assert "https://example.com/q-3" in kept_urls

    @pytest.mark.asyncio
    async def test_pre_drain_safety_expires_bad_sorani_and_recent_duplicate(self, temp_db, tmp_path, monkeypatch):
        """Test 7: Queued items failing Sorani gate or duplicating a newly published post expire without publishing."""
        monkeypatch.setattr(config, "ENABLE_SMART_PACING", True)

        published_while_queued = _make_article(
            "https://reuters.com/erbil-accord-root",
            "Erbil and Baghdad sign border security agreement",
            translated_title="هەولێر و بەغدا رێککەوتنی ئەمنییان واژۆ کرد",
            translated_summary="بەرپرسانی هەرێمی کوردستان و عێراق لە هەولێر رێککەوتنی هاوبەشی سنوورییان واژۆ کرد.",
            category="kurdistan",
            status=ArticleStatus.PUBLISHED,
            published_minutes_ago=30.0,
            telegram_message_id=6100,
            event_key="erbil_baghdad_border_accord",
            event_entities=["erbil", "baghdad", "border", "security"],
        )
        await temp_db.insert_article(published_while_queued)

        bad_arabic_queued = _make_article(
            "https://sabereen.example/arabic-queued",
            "Iraq political update",
            translated_title="خبر عاجل في بغداد اليوم بدون ترجمة",  # Arabic (fails Sorani gate!)
            translated_summary="تفاصيل الخبر السياسي العاجل.",
            category="politics_world",
            status=ArticleStatus.QUEUED,
            minutes_ago=20.0,
        )
        await temp_db.insert_article(bad_arabic_queued)

        dup_queued = _make_article(
            "https://kurdistan24.net/erbil-accord-dup",
            "Rudaw reports Erbil and Baghdad sealed border security deal",
            translated_title="هەولێر و بەغدا رێککەوتنی ئەمنی سنووریان واژۆ کرد",
            translated_summary="لە هەولێر بەرپرسانی کوردستان و عێراق رێککەوتنی ئەمنییان واژۆ کرد.",
            category="kurdistan",
            status=ArticleStatus.QUEUED,
            minutes_ago=15.0,
            event_key="erbil_baghdad_border_accord",
            event_entities=["erbil", "baghdad", "border", "security"],
        )
        await temp_db.insert_article(dup_queued)

        publisher, mock_bot, _ = _build_mock_publisher(tmp_path)
        engine = _build_engine(temp_db, publisher, sources=[])

        await engine._discovery_cycle()

        assert mock_bot.send_photo.await_count == 0
        row_bad = await temp_db.get_article_by_canonical_url(bad_arabic_queued.canonical_url)
        row_dup = await temp_db.get_article_by_canonical_url(dup_queued.canonical_url)
        assert row_bad.status == ArticleStatus.EXPIRED
        assert row_dup.status == ArticleStatus.EXPIRED

    @pytest.mark.asyncio
    async def test_queued_thread_update_drains_as_telegram_reply(self, temp_db, tmp_path, monkeypatch):
        """Test 8: A queued THREAD_UPDATE retains thread_root_message_id and publishes as a Telegram reply."""
        monkeypatch.setattr(config, "ENABLE_SMART_PACING", True)

        root_story = _make_article(
            "https://reuters.com/damascus-root-8",
            "Airstrikes target military depot near Damascus",
            translated_title="تەقینەوە لە کۆگایەکی سەربازی لە دیمەشق ڕوویدا",
            translated_summary="لە ئەنجامی هێرشێکی ئاسمانیدا دوو سەرباز لە نزیک دیمەشق برینداربوون.",
            category="military_conflict",
            status=ArticleStatus.PUBLISHED,
            published_minutes_ago=90.0,
            telegram_message_id=7500,
            event_key="damascus_depot_strike",
            event_entities=["damascus", "depot", "airstrike"],
        )
        await temp_db.insert_article(root_story)

        queued_thread_reply = _make_article(
            "https://bbc.com/damascus-update-8",
            "Damascus death toll rises to 18 killed after official statement",
            translated_title="ئاماری قوربانیانی هێرشەکەی دیمەشق بۆ ١٨ کوژراو بەرزبووەوە",
            translated_summary="وەزارەتی بەرگری لە بەیاننامەیەکدا پشتڕاستی کردەوە ژمارەی کوژراوان بۆ ١٨ کەس بەرزبووەتەوە.",
            category="military_conflict",
            status=ArticleStatus.QUEUED,
            minutes_ago=10.0,
            thread_root_message_id=7500,
            event_key="damascus_depot_strike",
            event_entities=["damascus", "depot", "airstrike", "casualties"],
        )
        await temp_db.insert_article(queued_thread_reply)

        publisher, mock_bot, _ = _build_mock_publisher(tmp_path, start_msg_id=7600)
        engine = _build_engine(temp_db, publisher, sources=[])

        await engine._discovery_cycle()

        mock_bot.send_photo.assert_awaited_once()
        send_kwargs = mock_bot.send_photo.call_args.kwargs
        assert send_kwargs["reply_to_message_id"] == 7500
        assert send_kwargs["allow_sending_without_reply"] is True
        assert "🔄 نوێکردنەوە | " in send_kwargs["caption"]

        saved = await temp_db.get_article_by_canonical_url(queued_thread_reply.canonical_url)
        assert saved.status == ArticleStatus.PUBLISHED
        assert saved.telegram_message_id == 7601
        assert saved.thread_root_message_id == 7500

    @pytest.mark.asyncio
    async def test_discovery_skips_urls_already_in_queued_status(self, temp_db, tmp_path, monkeypatch):
        """Test 9: Discovery skips URLs already present with QUEUED status so they are never re-translated."""
        monkeypatch.setattr(config, "ENABLE_SMART_PACING", True)

        already_queued = _make_article(
            "https://reuters.com/world/moscow-queued?utm_source=twitter",
            "Moscow warns on European security treaty",
            translated_title=_sorani_for("Moscow")[0],
            translated_summary=_sorani_for("Moscow")[1],
            category="politics_world",
            status=ArticleStatus.QUEUED,
            minutes_ago=30.0,
        )
        await temp_db.insert_article(already_queued)

        duplicate_raw = _make_raw(
            "https://www.reuters.com/world/moscow-queued/?fbclid=xyz123",
            "Moscow warns on European security treaty",
            category="politics_world",
        )

        publisher, _, _ = _build_mock_publisher(tmp_path)
        engine = _build_engine(temp_db, publisher)
        new_articles = await engine._extract_and_insert([duplicate_raw])
        assert new_articles == []

    @pytest.mark.asyncio
    async def test_feature_flag_false_preserves_immediate_publishing(self, temp_db, tmp_path, monkeypatch):
        """Test 10a: When ENABLE_SMART_PACING=False, all eligible articles publish immediately with 0 queue deferral."""
        monkeypatch.setattr(config, "ENABLE_SMART_PACING", False)
        monkeypatch.setattr(config, "MAX_NORMAL_POSTS_PER_CYCLE", 1)

        raw_list = [
            _make_raw("https://k24.net/imm-1", "Erbil security pact signed", category="kurdistan", region="South Kurdistan"),
            _make_raw("https://bbc.com/imm-2", "Ankara and Tehran border talks open", category="politics_world"),
            _make_raw("https://reuters.com/imm-3", "Geneva UN session discusses ceasefire", category="politics_world"),
        ]
        mock_source = MagicMock()
        mock_source.name = "MockRSS"
        mock_source.source_type = "rss"
        mock_source.enabled = True
        mock_source.fetch_articles = AsyncMock(return_value=raw_list)

        publisher, mock_bot, _ = _build_mock_publisher(tmp_path)
        engine = _build_engine(temp_db, publisher, sources=[mock_source])
        engine._pipeline._ranker.evaluate_batch_relevance = AsyncMock(
            return_value=[
                RelevanceEvaluation(score=8, is_breaking=False, category="kurdistan" if "Erbil" in a.title else "politics_world", reason="Politics", passed=True)
                for a in raw_list
            ]
        )
        engine._pipeline._translator.translate_article = AsyncMock(side_effect=lambda title, summary="": _sorani_for(title))

        sleep_calls: list[float] = []
        with patch("core.engine.asyncio.sleep", side_effect=lambda s: sleep_calls.append(s)):
            await engine._discovery_cycle()

        assert mock_bot.send_photo.await_count == 3
        assert sleep_calls == []
        assert await temp_db.get_queued_articles() == []

    @pytest.mark.asyncio
    async def test_supabase_queue_and_last_normal_methods(self):
        """Test 10b: SupabaseDatabase supports get_queued_articles, expire_stale_queued_articles, trim_queue, and get_last_normal_published_at."""
        now = datetime.now(timezone.utc)
        rows_queued = [
            {
                "id": 1,
                "canonical_url": "https://example.com/fresh-1",
                "original_url": "https://example.com/fresh-1",
                "original_title": "Fresh Kurdistan talk",
                "translated_title": "هەولێر کۆبوونەوەیەکی نوێی ئەنجامدا",
                "translated_summary": "پوختەی هەواڵ.",
                "category": "kurdistan",
                "importance_score": 8.8,
                "discovered_at": (now - timedelta(minutes=20)).isoformat(),
                "processed_at": (now - timedelta(minutes=15)).isoformat(),
                "status": "QUEUED",
                "is_breaking": False,
            },
            {
                "id": 2,
                "canonical_url": "https://example.com/stale-2",
                "original_url": "https://example.com/stale-2",
                "original_title": "Old stale news",
                "translated_title": "هەواڵی کۆن",
                "translated_summary": "پوختەی کۆن.",
                "category": "politics_world",
                "importance_score": 7.5,
                "discovered_at": (now - timedelta(hours=6)).isoformat(),
                "processed_at": (now - timedelta(hours=5)).isoformat(),
                "status": "QUEUED",
                "is_breaking": False,
            },
        ]
        last_normal_ts = (now - timedelta(minutes=8)).isoformat()
        patched_calls: list[tuple[str, dict]] = []

        def _handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if request.method == "GET" and path.endswith("/articles"):
                q = str(request.url.query)
                if "status=eq.QUEUED" in q:
                    return httpx.Response(200, json=rows_queued)
                if "status=eq.PUBLISHED" in q and "is_breaking=eq.false" in q:
                    return httpx.Response(200, json=[{"published_at": last_normal_ts, "is_breaking": False}])
            if request.method == "PATCH" and path.endswith("/articles"):
                patched_calls.append((str(request.url.query), json.loads(request.content.decode("utf-8"))))
                return httpx.Response(204)
            return httpx.Response(200, json=[])

        client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
        sb = SupabaseDatabase(url="https://example.supabase.co", key="anon-key", client=client)

        fresh = await sb.get_queued_articles(max_age_hours=4)
        assert len(fresh) == 1
        assert fresh[0].id == 1

        expired_cnt = await sb.expire_stale_queued_articles(max_age_hours=4)
        assert expired_cnt == 1
        assert patched_calls[-1][1]["status"] == ArticleStatus.EXPIRED

        last_dt = await sb.get_last_normal_published_at()
        assert last_dt is not None
        assert abs((now - last_dt).total_seconds() - 480.0) < 5.0

        await client.aclose()
