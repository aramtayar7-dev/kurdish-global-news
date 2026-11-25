"""
Comprehensive automated tests for Private Admin Watchdog & Daily Health Telemetry (Idea 5).
All Telegram, Supabase, and Gemini calls are mocked — ZERO live network requests.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import config
from config import ArticleStatus, WATCHDOG_DAILY_REPORT_HOUR_UTC3, WATCHDOG_SOURCE_FAILURE_THRESHOLD
from core.engine import NewsEngine
from monitoring.watchdog import (
    ALERT_GEMINI_EXHAUSTED,
    ALERT_PIPELINE_CRASH,
    ALERT_PUBLISH_DROUGHT,
    ALERT_SOURCE_OUTAGE,
    ALERT_SUPABASE_FALLBACK,
    AdminWatchdog,
    WatchdogCycleMetrics,
    is_valid_admin_target,
    redact_secrets,
)
from processing.gemini_client import GeminiClient
from storage.database import Database
from storage.models import Article, Source
from storage.supabase_db import SupabaseAPIError, SupabaseDatabase
from tg_bot.publisher import TelegramPublisher


@pytest.fixture
async def temp_db(tmp_path: Path):
    db = Database(str(tmp_path / "test_watchdog.db"))
    await db.connect()
    yield db
    await db.close()


def _make_publisher(channel_id: str = "@KurdishGlobalNews") -> tuple[TelegramPublisher, AsyncMock]:
    publisher = TelegramPublisher()
    publisher._channel_id = channel_id
    mock_bot = AsyncMock()
    mock_bot.send_message = AsyncMock(return_value=MagicMock(message_id=999))
    publisher._bot = mock_bot
    return publisher, mock_bot


def _make_article(
    url: str,
    title: str,
    category: str = "politics_world",
    status: str = ArticleStatus.PUBLISHED,
    published_at: str | None = None,
    discovered_at: str | None = None,
    is_breaking: bool = False,
    score: float = 8.4,
    thread_root_message_id: int | None = None,
) -> Article:
    now_iso = datetime.now(timezone.utc).isoformat()
    art = Article(
        original_url=url,
        canonical_url=Article.canonicalize_url(url),
        original_title=title,
        normalized_title=Article.normalize_title(title),
        translated_title="سەردێڕی هەواڵی سۆرانی",
        translated_summary="دەقی پوختەی هەواڵەکە دەربارەی پێشهاتە جیهانییەکان.",
        original_summary=f"{title}. Detailed international report and official briefing.",
        source_name="Reuters",
        category=category,
        importance_score=score,
        is_breaking=is_breaking,
        thread_root_message_id=thread_root_message_id,
        publication_date=published_at or discovered_at or now_iso,
        discovered_at=discovered_at or published_at or now_iso,
        published_at=published_at,
        status=status,
    )
    art.compute_content_hash()
    art.compute_event_fingerprint()
    return art


# ═══════════════════════════════════════════════════
# 1. Hard Public-Channel Isolation Guard
# ═══════════════════════════════════════════════════
class TestPublicChannelIsolationGuard:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "admin_target",
        [
            "",
            "   ",
            "@KurdishGlobalNews",
            "kurdishglobalnews",
            "  @KURDISHGLOBALNEWS  ",
            "your_admin_id_here",
        ],
    )
    async def test_watchdog_never_sends_to_empty_invalid_or_public_channel(self, temp_db, admin_target: str):
        publisher, mock_bot = _make_publisher(channel_id="@KurdishGlobalNews")
        assert is_valid_admin_target(admin_target, "@KurdishGlobalNews") is False

        wd = AdminWatchdog(
            db=temp_db,
            publisher=publisher,
            admin_chat_id=admin_target,
            public_channel_id="@KurdishGlobalNews",
        )
        sent_alert = await wd.dispatch_alert(
            ALERT_GEMINI_EXHAUSTED,
            title="Test",
            details="Should never reach public channel",
        )
        assert sent_alert is False
        assert await wd.maybe_send_daily_report(
            now=datetime(2026, 11, 25, 19, 30, tzinfo=timezone.utc)
        ) is False
        assert await publisher.send_admin_message(admin_target, "Test") is False
        mock_bot.send_message.assert_not_awaited()

    def test_valid_private_numeric_id_accepted(self):
        assert is_valid_admin_target("123456789", "@KurdishGlobalNews") is True
        assert is_valid_admin_target("-1009876543210", "@KurdishGlobalNews") is True


# ═══════════════════════════════════════════════════
# 2. Secret Redaction
# ═══════════════════════════════════════════════════
class TestSecretRedaction:
    def test_redact_secrets_scrubs_tokens_keys_and_jwts(self, monkeypatch):
        fake_bot = "7123456789:AAFakeTelegramBotTokenSecret123456"
        fake_aiza = "AIzaSyFakeGeminiKeyValue987654321"
        fake_aq = "AQ.AbcDefGhIjKlMnOpQrStUvWxYz.12345"
        fake_jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoic2VydmljZV9yb2xlIn0.FakeSignature123"

        monkeypatch.setattr(config, "TELEGRAM_BOT_TOKEN", fake_bot)
        monkeypatch.setattr(config, "GEMINI_API_KEY", f"{fake_aiza},{fake_aq}")
        monkeypatch.setattr(config, "GEMINI_API_KEYS", [fake_aiza, fake_aq])
        monkeypatch.setattr(config, "SUPABASE_KEY", fake_jwt)

        raw_err = (
            f"POST https://api.telegram.org/bot{fake_bot}/sendMessage failed; "
            f"Gemini keys {fake_aiza} and {fake_aq} hit 429; "
            f"Supabase bearer={fake_jwt}; extra key AIzaSyHardcodedPattern777 and bot999999:AbCdEfGhIjKlMnOpQrStUv"
        )
        cleaned = redact_secrets(raw_err)
        for secret in (fake_bot, fake_aiza, fake_aq, fake_jwt, "AIzaSyHardcodedPattern777", "999999:AbCdEfGhIjKlMnOpQrStUv"):
            assert secret not in cleaned
        assert "[REDACTED]" in cleaned


# ═══════════════════════════════════════════════════
# 3. Daily Health Report (Once Per Kurdistan UTC+3 Day)
# ═══════════════════════════════════════════════════
class TestDailyHealthReport:
    @pytest.mark.asyncio
    async def test_daily_report_respects_hour_and_once_per_day_dedup(self, temp_db):
        publisher, mock_bot = _make_publisher()
        # 2026-11-25 14:00 UTC+3 == 11:00 UTC
        early_utc = datetime(2026, 11, 25, 11, 0, tzinfo=timezone.utc)
        # 2026-11-25 22:15 UTC+3 == 19:15 UTC
        night_utc = datetime(2026, 11, 25, 19, 15, tzinfo=timezone.utc)
        night_later_utc = datetime(2026, 11, 25, 19, 45, tzinfo=timezone.utc)

        # Populate realistic 24h operational data
        pub_time = (night_utc - timedelta(hours=3)).isoformat()
        arts = [
            _make_article(
                "https://rudaw.net/bashur-1",
                "Erbil KRG parliament approves Peshmerga security budget",
                category="kurdistan",
                published_at=pub_time,
                is_breaking=True,
                score=9.3,
            ),
            _make_article(
                "https://reuters.com/global-pw-2",
                "NATO defense ministers hold summit on Baltic deterrence",
                category="politics_world",
                published_at=pub_time,
                thread_root_message_id=501,
            ),
            _make_article(
                "https://cnbc.com/econ-3",
                "OPEC cuts global crude oil supply amid inflation concerns",
                category="economy_markets",
                published_at=pub_time,
            ),
            _make_article(
                "https://bleepingcomputer.com/cyber-4",
                "CISA warns on critical zero-day ransomware campaign",
                category="cybersecurity",
                published_at=pub_time,
            ),
            _make_article(
                "https://example.com/dup-5",
                "Duplicate report",
                status=ArticleStatus.REJECTED_DUPLICATE,
                discovered_at=pub_time,
            ),
            _make_article(
                "https://example.com/skip-6",
                "Low relevance item",
                status=ArticleStatus.SKIPPED,
                discovered_at=pub_time,
            ),
            _make_article(
                "https://example.com/queue-7",
                "Queued diplomatic summit",
                status=ArticleStatus.QUEUED,
                discovered_at=pub_time,
            ),
        ]
        for a in arts:
            aid = await temp_db.insert_article(a)
            if a.status != ArticleStatus.DISCOVERED:
                await temp_db.update_article_status(
                    aid,
                    a.status,
                    published_at=a.published_at,
                    thread_root_message_id=a.thread_root_message_id,
                    is_breaking=a.is_breaking,
                )

        await temp_db.upsert_source(Source(name="Reuters", source_type="rss", enabled=True))
        await temp_db.upsert_source(Source(name="BrokenFeed", source_type="rss", enabled=True))
        for _ in range(3):
            await temp_db.update_source_status("BrokenFeed", success=False, error="Timeout")

        wd = AdminWatchdog(
            db=temp_db,
            publisher=publisher,
            admin_chat_id="123456789",
            public_channel_id="@KurdishGlobalNews",
        )

        # Persist 429 telemetry from earlier in the day
        await wd.record_cycle_telemetry(WatchdogCycleMetrics(gemini_429_hits=4), now=early_utc)

        # 14:00 Erbil time (< 22:00) -> must NOT send daily report
        assert WATCHDOG_DAILY_REPORT_HOUR_UTC3 == 22
        assert await wd.maybe_send_daily_report(now=early_utc) is False
        mock_bot.send_message.assert_not_awaited()

        # 22:15 Erbil time -> MUST send daily report silently
        sent = await wd.maybe_send_daily_report(
            metrics=WatchdogCycleMetrics(storage_backend="SQLite"),
            now=night_utc,
        )
        assert sent is True
        mock_bot.send_message.assert_awaited_once()
        kw = mock_bot.send_message.call_args.kwargs
        assert kw["chat_id"] == "123456789"
        assert kw["disable_notification"] is True
        msg = kw["text"]
        assert "ڕاپۆرتی ڕۆژانەی تەندروستیی بۆت — 2026-11-25" in msg
        assert "بڵاوکراوەکانی ٢٤ کاتژمێر:</b> 4" in msg
        assert "باشوور 1 | جیهان 3" in msg
        assert "سیاسەت و جەنگ 2 | ئابووری 1 | سایبەری 1" in msg
        assert "🔴 بەپەلە 1 | 🔄 نوێکردنەوەی زنجیرە 1" in msg
        assert "دووبارە ڕاگیراو: 1 | پشتگوێخراو: 1" in msg
        assert "QUEUED): 1" in msg
        assert "Gemini 429 (٢٤ کاتژمێر): 4" in msg
        assert "BrokenFeed (3)" in msg

        # 22:45 Erbil time same day -> MUST NOT send again
        mock_bot.send_message.reset_mock()
        assert await wd.maybe_send_daily_report(now=night_later_utc) is False
        mock_bot.send_message.assert_not_awaited()


# ═══════════════════════════════════════════════════
# 4. Alert Cooldown Deduplication
# ═══════════════════════════════════════════════════
class TestAlertCooldownDeduplication:
    @pytest.mark.asyncio
    async def test_gemini_exhausted_alert_deduplicated_within_cooldown_and_resends_after(self, temp_db):
        publisher, mock_bot = _make_publisher()
        wd = AdminWatchdog(
            db=temp_db,
            publisher=publisher,
            admin_chat_id="123456789",
            public_channel_id="@KurdishGlobalNews",
        )
        t0 = datetime(2026, 11, 25, 10, 0, tzinfo=timezone.utc)
        t15 = t0 + timedelta(minutes=15)
        t75 = t0 + timedelta(minutes=75)

        res1 = await wd.run_cycle_checks(
            WatchdogCycleMetrics(gemini_429_hits=3, gemini_exhausted=1, discovery_ran=False),
            now=t0,
        )
        assert ALERT_GEMINI_EXHAUSTED in res1["alerts_sent"]
        assert mock_bot.send_message.await_count == 1
        assert mock_bot.send_message.call_args.kwargs["disable_notification"] is False

        # 15m later (< 60m cooldown): suppressed
        res2 = await wd.run_cycle_checks(
            WatchdogCycleMetrics(gemini_429_hits=2, gemini_exhausted=1, discovery_ran=False),
            now=t15,
        )
        assert res2["alerts_sent"] == []
        assert mock_bot.send_message.await_count == 1

        # Simulate fresh stateless process (new AdminWatchdog reading persisted system_logs) at +75m (> 60m cooldown)
        wd_fresh = AdminWatchdog(
            db=temp_db,
            publisher=publisher,
            admin_chat_id="123456789",
            public_channel_id="@KurdishGlobalNews",
        )
        res3 = await wd_fresh.run_cycle_checks(
            WatchdogCycleMetrics(gemini_429_hits=1, gemini_exhausted=1, discovery_ran=False),
            now=t75,
        )
        assert ALERT_GEMINI_EXHAUSTED in res3["alerts_sent"]
        assert mock_bot.send_message.await_count == 2


# ═══════════════════════════════════════════════════
# 5. Publish Drought Alert
# ═══════════════════════════════════════════════════
class TestPublishDroughtAlert:
    @pytest.mark.asyncio
    async def test_publish_drought_behavior_on_empty_recent_and_stale_db(self, temp_db):
        publisher, mock_bot = _make_publisher()
        wd = AdminWatchdog(
            db=temp_db,
            publisher=publisher,
            admin_chat_id="123456789",
            public_channel_id="@KurdishGlobalNews",
        )
        now = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)

        # 1. Brand-new empty DB (0 rows) -> NO false alarm
        res_empty = await wd.run_cycle_checks(WatchdogCycleMetrics(discovery_ran=True), now=now)
        assert res_empty["alerts_sent"] == []

        # 2. Stale published article (7h old > 6h threshold) -> PUBLISH_DROUGHT fires
        stale_art = _make_article(
            "https://reuters.com/stale-1",
            "Older published story",
            published_at=(now - timedelta(hours=7)).isoformat(),
        )
        aid = await temp_db.insert_article(stale_art)
        await temp_db.update_article_status(aid, ArticleStatus.PUBLISHED, published_at=stale_art.published_at)

        res_stale = await wd.run_cycle_checks(WatchdogCycleMetrics(discovery_ran=True), now=now)
        assert ALERT_PUBLISH_DROUGHT in res_stale["alerts_sent"]
        assert mock_bot.send_message.await_count == 1

        # 3. Fresh published article (2h old < 6h) -> NO alert even on new watchdog after cooldown
        fresh_art = _make_article(
            "https://reuters.com/fresh-2",
            "Recent published story",
            published_at=(now - timedelta(hours=2)).isoformat(),
        )
        aid2 = await temp_db.insert_article(fresh_art)
        await temp_db.update_article_status(aid2, ArticleStatus.PUBLISHED, published_at=fresh_art.published_at)

        wd2 = AdminWatchdog(
            db=temp_db,
            publisher=publisher,
            admin_chat_id="123456789",
            public_channel_id="@KurdishGlobalNews",
        )
        res_fresh = await wd2.run_cycle_checks(
            WatchdogCycleMetrics(discovery_ran=True),
            now=now + timedelta(hours=2),  # fresh_art is 4h old (< 6h) and cooldown expired
        )
        assert ALERT_PUBLISH_DROUGHT not in res_fresh["alerts_sent"]


# ═══════════════════════════════════════════════════
# 6. Source Outage Alert
# ═══════════════════════════════════════════════════
class TestSourceOutageAlert:
    @pytest.mark.asyncio
    async def test_source_outage_triggers_at_threshold_with_redacted_error(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "GEMINI_API_KEY", "AIzaSySecretKeyLeak")
        publisher, mock_bot = _make_publisher()
        await temp_db.upsert_source(Source(name="HealthyFeed", source_type="rss", enabled=True))
        await temp_db.upsert_source(Source(name="DeadTelegram", source_type="telegram_web", enabled=True))
        for _ in range(WATCHDOG_SOURCE_FAILURE_THRESHOLD - 1):
            await temp_db.update_source_status("HealthyFeed", success=False, error="Minor blip")
        for _ in range(WATCHDOG_SOURCE_FAILURE_THRESHOLD):
            await temp_db.update_source_status("DeadTelegram", success=False, error="HTTP 502 with AIzaSySecretKeyLeak")

        wd = AdminWatchdog(
            db=temp_db,
            publisher=publisher,
            admin_chat_id="123456789",
            public_channel_id="@KurdishGlobalNews",
        )
        res = await wd.run_cycle_checks(
            WatchdogCycleMetrics(discovery_ran=False),
            now=datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc),
        )
        assert ALERT_SOURCE_OUTAGE in res["alerts_sent"]
        sent_text = mock_bot.send_message.call_args.kwargs["text"]
        assert "DeadTelegram" in sent_text
        assert "HealthyFeed" not in sent_text
        assert "AIzaSySecretKeyLeak" not in sent_text
        assert "[REDACTED]" in sent_text


# ═══════════════════════════════════════════════════
# 7 & 8. Crash Alert, Non-Blocking Delivery, Zero Gemini Calls & Supabase Fallback
# ═══════════════════════════════════════════════════
class TestCrashNonBlockingZeroGeminiAndFallback:
    @pytest.mark.asyncio
    async def test_fatal_cycle_crash_sends_redacted_alert_and_delivery_failure_never_breaks_bot(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(config, "ADMIN_CHAT_ID", "123456789")
        monkeypatch.setattr(config, "ENABLE_ADMIN_WATCHDOG", True)
        monkeypatch.setattr(config, "GEMINI_API_KEY", "AIzaSyCrashSecret99")

        engine = NewsEngine()
        engine._db.db_path = str(tmp_path / "crash_wd.db")
        engine._publisher.initialize = AsyncMock()
        engine._publisher.test_connection = AsyncMock(return_value=True)
        engine._publisher.cleanup = AsyncMock()
        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock(return_value=MagicMock(message_id=77))
        engine._publisher._bot = mock_bot
        engine._publisher._channel_id = "@KurdishGlobalNews"
        engine._init_sources = AsyncMock()

        async def _boom():
            raise RuntimeError("Fatal database error leaking AIzaSyCrashSecret99")

        engine._discovery_cycle = _boom  # type: ignore[method-assign]

        with patch.object(GeminiClient, "generate_json", new=AsyncMock()) as mock_gemini:
            with pytest.raises(RuntimeError, match="Fatal database error"):
                await engine.run_once()

        mock_gemini.assert_not_awaited()
        mock_bot.send_message.assert_awaited_once()
        alert_txt = mock_bot.send_message.call_args.kwargs["text"]
        assert ALERT_PIPELINE_CRASH in alert_txt
        assert "AIzaSyCrashSecret99" not in alert_txt
        assert "[REDACTED]" in alert_txt

        # Verify Telegram send_message raising does NOT crash watchdog or caller
        publisher_err, bot_err = _make_publisher()
        bot_err.send_message = AsyncMock(side_effect=RuntimeError("Bot blocked by admin"))
        wd_err = AdminWatchdog(
            db=None,
            publisher=publisher_err,
            admin_chat_id="123456789",
            public_channel_id="@KurdishGlobalNews",
        )
        ok = await wd_err.notify_pipeline_crash(ValueError("Something failed"))
        assert ok is False

    @pytest.mark.asyncio
    async def test_supabase_failure_falls_back_to_sqlite_and_fires_alert(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "DATABASE_PATH", str(tmp_path / "fallback.db"))
        monkeypatch.setattr(config, "ADMIN_CHAT_ID", "123456789")
        monkeypatch.setattr(config, "ENABLE_ADMIN_WATCHDOG", True)

        sb_db = SupabaseDatabase(url="https://test.supabase.co", key="eyJfakeSupabaseJwt.part2.part3", base_backoff=0.0)
        sb_db.connect = AsyncMock(side_effect=SupabaseAPIError("Supabase unreachable (eyJfakeSupabaseJwt.part2.part3)"))  # type: ignore[method-assign]
        sb_db.close = AsyncMock()  # type: ignore[method-assign]

        engine = NewsEngine()
        engine._db = sb_db
        engine._publisher.initialize = AsyncMock()
        engine._publisher.test_connection = AsyncMock(return_value=True)
        engine._publisher.cleanup = AsyncMock()
        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock(return_value=MagicMock(message_id=55))
        engine._publisher._bot = mock_bot
        engine._publisher._channel_id = "@KurdishGlobalNews"
        engine._init_sources = AsyncMock()
        engine._sources = []

        with patch.object(GeminiClient, "generate_json", new=AsyncMock()) as mock_gemini:
            await engine.run_once()

        mock_gemini.assert_not_awaited()
        assert isinstance(engine._db, Database)
        assert engine._supabase_fallback_active is True
        mock_bot.send_message.assert_awaited()
        all_sent_texts = [call.kwargs.get("text", "") for call in mock_bot.send_message.call_args_list]
        assert any(ALERT_SUPABASE_FALLBACK in t for t in all_sent_texts)
        assert all("eyJfakeSupabaseJwt.part2.part3" not in t for t in all_sent_texts)
