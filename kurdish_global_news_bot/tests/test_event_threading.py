"""
Comprehensive automated tests for Live Event Threading (`kurdish_global_news_bot`):
  1. sendPhoto and sendVideo return and persist telegram_message_id & thread_root_message_id.
  2. Reworded duplicate (no material delta) stays DUPLICATE (never replies).
  3. Material update after 20m cooldown becomes THREAD_UPDATE -> reply_to_message_id + allow_sending_without_reply=True.
  4. Both sendPhoto cards and sendVideo branded clips support threaded replies (+ deleted-root fallback).
  5. Cooldown (< 20m) blocks non-breaking updates while is_breaking=True (score >= 9.0) bypasses cooldown.
  6. Thread cap (MAX_UPDATES_PER_THREAD=3) blocks 4th update cleanly as DUPLICATE.
  7. Expired window (> EVENT_THREAD_WINDOW_HOURS=24h) is NOT threaded to old message.
  8. Post-translation Sorani safety check downgrades repeated Kurdish summaries to DUPLICATE before card generation.
  9. Supabase missing telegram_message_id/thread_root_message_id columns degrades gracefully without crashing.
 10. THREAD_UPDATE caption includes `🔄 نوێکردنەوە | ` (or `🔴 بەپەلە | ` if breaking) and stays <= 1024 chars.
 11. Zero extra Gemini API calls used for threading decisions.
 12. SQLite auto-migration & Supabase idempotent SQL migration script verified.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from telegram.error import BadRequest

from config import (
    ArticleStatus,
    ENABLE_LIVE_EVENT_THREADING,
    EVENT_THREAD_WINDOW_HOURS,
    MAX_UPDATES_PER_THREAD,
    THREAD_UPDATE_COOLDOWN_MINUTES,
)
from core.importance_ranker import RelevanceEvaluation
from core.pipeline import ArticlePipeline
from monitoring.health import HealthMonitor
from monitoring.metrics import Metrics
from processing.content_formatter import ContentFormatter
from processing.deduplicator import (
    OUTCOME_DUPLICATE,
    OUTCOME_NEW_STORY,
    OUTCOME_THREAD_UPDATE,
    SemanticDeduplicator,
)
from storage.database import Database
from storage.models import Article
from storage.supabase_db import SupabaseDatabase
from tg_bot.publisher import TelegramPublisher


def _make_article(
    title: str,
    url: str,
    summary: str = "",
    category: str = "military_conflict",
    region: str = "",
    source_name: str = "Reuters",
    translated_title: str = "",
    translated_summary: str = "",
    event_key: str = "",
    event_entities: list[str] | None = None,
    score: float = 8.2,
    is_breaking: bool = False,
    hours_ago: float = 0.05,
    status: str = ArticleStatus.DISCOVERED,
    telegram_message_id: int | None = None,
    thread_root_message_id: int | None = None,
    video_url: str | None = None,
) -> Article:
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    art = Article(
        original_url=url,
        canonical_url=Article.canonicalize_url(url),
        original_title=title,
        normalized_title=Article.normalize_title(title),
        original_summary=summary or (
            f"{title}. Authorities issued an operational briefing detailing field deployments, "
            "communications checks, and preliminary regional security assessments."
        ),
        translated_title=translated_title,
        translated_summary=translated_summary,
        source_name=source_name,
        category=category,
        region=region,
        publication_date=ts,
        discovered_at=ts,
        published_at=ts if status == ArticleStatus.PUBLISHED else None,
        status=status,
        importance_score=score,
        is_breaking=is_breaking,
        telegram_message_id=telegram_message_id,
        thread_root_message_id=thread_root_message_id,
        video_url=video_url,
        event_key=event_key,
        event_entities=list(event_entities or []),
    )
    art.compute_content_hash()
    art.compute_event_fingerprint()
    return art


@pytest.fixture
async def temp_db(tmp_path):
    db = Database(str(tmp_path / "test_event_threading.db"))
    await db.connect()
    yield db
    await db.close()


class TestThreadingConfigAndMigrations:
    def test_threading_config_defaults_and_idempotent_sql_migration(self):
        assert ENABLE_LIVE_EVENT_THREADING is True
        assert EVENT_THREAD_WINDOW_HOURS == 24
        assert THREAD_UPDATE_COOLDOWN_MINUTES == 20
        assert MAX_UPDATES_PER_THREAD == 3

        sql_path = Path(__file__).resolve().parents[1] / "storage" / "sql" / "add_event_threading_columns.sql"
        assert sql_path.exists()
        sql_text = sql_path.read_text(encoding="utf-8")
        assert "ALTER TABLE articles" in sql_text
        assert "ADD COLUMN IF NOT EXISTS telegram_message_id BIGINT" in sql_text
        assert "ADD COLUMN IF NOT EXISTS thread_root_message_id BIGINT" in sql_text

    @pytest.mark.asyncio
    async def test_persist_telegram_message_id_and_thread_root_in_sqlite(self, temp_db):
        root = _make_article(
            title="Drone strikes hit power substation near Kirkuk",
            url="https://bbc.com/news/kirkuk-root-1",
            category="kurdistan",
            region="South Kurdistan",
            translated_title="درۆنێک وێستگەی کارەبای لە نزیک کەرکووک کردە ئامانج",
            translated_summary="بەهۆی هێرشێکی درۆنییەوە زیان بە وێستگەی کارەبای نزیک کەرکووک گەیشت.",
            event_key="kirkuk_substation_drone_strike",
            event_entities=["kirkuk", "drone", "substation", "strike"],
            hours_ago=1.5,
            status=ArticleStatus.PUBLISHED,
        )
        root_id = await temp_db.insert_article(root)
        await temp_db.update_article_status(
            root_id, ArticleStatus.PUBLISHED, telegram_message_id=5011, thread_root_message_id=None
        )

        loaded_root = await temp_db.get_article_by_canonical_url(root.canonical_url)
        assert loaded_root is not None
        assert loaded_root.telegram_message_id == 5011
        assert loaded_root.thread_root_message_id is None

        history = await temp_db.get_recent_published_headlines(hours=72)
        assert len(history) == 1
        assert history[0]["telegram_message_id"] == 5011
        assert history[0]["thread_root_message_id"] is None


class TestThreeWayClassificationAndGuardrails:
    @pytest.mark.asyncio
    async def test_reworded_repeat_without_material_delta_stays_duplicate(self, temp_db):
        root = _make_article(
            title="Air defense intercepts two drones over Erbil airport",
            url="https://reuters.com/world/erbil-root",
            summary="Coalition air defenses downed two explosives-laden drones over Erbil international airport at dawn with no casualties reported.",
            translated_title="بەرگریی ئاسمانی دوو درۆنی لە ئاسمانی فڕۆکەخانەی هەولێر خستە خوارەوە",
            translated_summary="سیستەمی بەرگریی ئاسمانی دوو فڕۆکەی بێفڕۆکەوانی لە نزیک فڕۆکەخانەی هەولێر پووچەڵ کردەوە.",
            event_key="erbil_airport_drone_interception",
            event_entities=["erbil", "airport", "drone", "interception"],
            hours_ago=2.0,
            status=ArticleStatus.PUBLISHED,
            telegram_message_id=7701,
        )
        root_id = await temp_db.insert_article(root)
        await temp_db.update_article_status(root_id, ArticleStatus.PUBLISHED, telegram_message_id=7701)

        repeat = _make_article(
            title="Rudaw: Explosives-laden UAVs downed near Erbil aviation hub",
            url="https://rudaw.net/eng/erbil-dup",
            summary="Air defense batteries shot down two unmanned aerial vehicles near Erbil international airport early morning without injuries.",
            source_name="Rudaw",
            event_key="erbil_airport_drone_interception",
            event_entities=["erbil", "airport", "drone", "interception"],
        )

        dedup = SemanticDeduplicator(temp_db)
        res = await dedup.check_duplicate(repeat)
        assert res.classification == OUTCOME_DUPLICATE
        assert res.is_duplicate is True
        assert res.thread_root_message_id is None

    @pytest.mark.asyncio
    async def test_material_update_after_cooldown_classifies_thread_update_and_publishes_reply(self, temp_db, tmp_path):
        root = _make_article(
            title="Airstrikes target weapons depot near Damascus",
            url="https://reuters.com/world/damascus-root",
            summary="Initial reports cite explosions at a military depot south of Damascus overnight with 2 soldiers injured.",
            translated_title="تەقینەوە لە کۆگایەکی سەربازی لە باشووری دیمەشق ڕوویدا",
            translated_summary="لە ئەنجامی هێرشێکی ئاسمانیدا دوو سەرباز لە نزیک دیمەشق برینداربوون.",
            event_key="damascus_depot_airstrike",
            event_entities=["damascus", "depot", "airstrike", "syria"],
            hours_ago=1.2,
            status=ArticleStatus.PUBLISHED,
            telegram_message_id=8100,
        )
        root_id = await temp_db.insert_article(root)
        await temp_db.update_article_status(
            root_id,
            ArticleStatus.PUBLISHED,
            telegram_message_id=8100,
            published_at=root.published_at,
        )

        # Update 45m later with new casualty toll & official Defense Ministry statement!
        update_photo = _make_article(
            title="Damascus death toll rises to 14 killed after official Defense Ministry statement",
            url="https://bbc.com/news/damascus-update-photo",
            summary=(
                "The Defense Ministry issued an official response confirming the death toll rises to 14 killed "
                "and 28 wounded following forensic inspection of the targeted depot."
            ),
            source_name="BBC",
            category="military_conflict",
            hours_ago=0.05,
        )
        update_photo.id = await temp_db.insert_article(update_photo)

        mock_bot = AsyncMock()
        photo_reply_msg = MagicMock()
        photo_reply_msg.message_id = 8101
        mock_bot.send_photo = AsyncMock(return_value=photo_reply_msg)

        video_reply_msg = MagicMock()
        video_reply_msg.message_id = 8102
        mock_bot.send_video = AsyncMock(return_value=video_reply_msg)

        publisher = TelegramPublisher()
        publisher._channel_id = "@KurdishGlobalNews"
        publisher._bot = mock_bot
        publisher._image_handler.download_image = AsyncMock(return_value=None)

        pipeline = ArticlePipeline(temp_db, publisher, HealthMonitor(), Metrics())
        pipeline._semantic_dedup = SemanticDeduplicator(temp_db)

        with patch.object(
            pipeline._ranker,
            "evaluate_global_relevance",
            new=AsyncMock(
                return_value=RelevanceEvaluation(
                    score=8,
                    is_breaking=False,
                    category="military_conflict",
                    reason="Official casualties update",
                    passed=True,
                    event_key="damascus_depot_airstrike",
                    event_entities=["damascus", "depot", "airstrike", "casualties"],
                    update_type="casualties_update",
                )
            ),
        ), patch.object(
            pipeline._extractor,
            "fetch_article_content_and_image",
            new=AsyncMock(return_value=("", None)),
        ), patch.object(
            pipeline._translator,
            "translate_article",
            new=AsyncMock(
                return_value=(
                    "ئاماری قوربانیانی هێرشەکەی دیمەشق بۆ ١٤ کوژراو بەرزبووەوە",
                    (
                        "وەزارەتی بەرگری لە بەیاننامەیەکدا پشتڕاستی کردەوە ژمارەی کوژراوان بۆ ١٤ کەس "
                        "و برینداران بۆ ٢٨ کەس لە باشووری دیمەشق بەرزبووەتەوە."
                    ),
                )
            ),
        ):
            ok_photo = await pipeline.process_article(update_photo)

        assert ok_photo is True
        mock_bot.send_photo.assert_awaited_once()
        p_kwargs = mock_bot.send_photo.call_args.kwargs
        assert p_kwargs["reply_to_message_id"] == 8100
        assert p_kwargs["allow_sending_without_reply"] is True
        assert "🔄 نوێکردنەوە | " in p_kwargs["caption"]
        assert len(p_kwargs["caption"]) <= 1024

        saved_reply = await temp_db.get_article_by_canonical_url(update_photo.canonical_url)
        assert saved_reply is not None
        assert saved_reply.telegram_message_id == 8101
        assert saved_reply.thread_root_message_id == 8100

        # Now test branded KGN sendVideo threaded reply + deleted-root fallback on a second urgent event!
        branded_tmp = tmp_path / "kgn_thread_vid.mp4"
        branded_tmp.write_bytes(b"BRANDED_VIDEO")

        vid_update = _make_article(
            title="Video shows second wave retaliatory strikes near Damascus",
            url="https://t.me/SabrenNewss/8103",
            summary="Night video captures second wave missile retaliation near Damascus after initial depot blast, confirmed by official statement.",
            source_name="Sabereen News",
            category="military_conflict",
            translated_title="گرتەی ڤیدیۆیی شەپۆلی دووەمی هێرشەکان لە دیمەشق بڵاوکرایەوە",
            translated_summary="دیمەنی تەقینەوە و هێرشی مووشەکیی نوێ لە دیمەشق تۆمارکراوە.",
            event_key="damascus_depot_airstrike",
            event_entities=["damascus", "depot", "retaliation", "missiles"],
            score=9.5,
            is_breaking=True,
            thread_root_message_id=8100,
            video_url="https://cdn4.telegram-cdn.org/file/thread_vid.mp4",
        )
        vid_update.is_thread_update = True

        with patch("tg_bot.publisher.download_and_brand_video", new=AsyncMock(return_value=str(branded_tmp))):
            vid_msg_id = await publisher.publish_article(vid_update)

        assert vid_msg_id == 8102
        v_kwargs = mock_bot.send_video.call_args.kwargs
        assert v_kwargs["reply_to_message_id"] == 8100
        assert v_kwargs["allow_sending_without_reply"] is True
        # Breaking update preserves 🔴 بەپەلە | priority badge
        assert "🔴 بەپەلە | " in v_kwargs["caption"]
        assert len(v_kwargs["caption"]) <= 1024

        # Deleted root BadRequest fallback on send_photo -> retries cleanly without reply_to_message_id
        fallback_msg = MagicMock()
        fallback_msg.message_id = 8109
        mock_bot.send_photo = AsyncMock(
            side_effect=[BadRequest("Message to be replied not found"), fallback_msg]
        )
        deleted_root_reply = _make_article(
            title="Cleanup continues at blast site",
            url="https://example.com/del-root-test",
            translated_title="تیمەکانی فریاگوزاری لە دیمەشق بەردەوامن لە کارەکانیان",
            translated_summary="هەوڵەکان بۆ پاککردنەوەی شوێنی ڕووداوەکە بەردەوامە.",
            thread_root_message_id=999999,
            score=8.0,
        )
        deleted_root_reply.is_thread_update = True
        assert await publisher.publish_article(deleted_root_reply) == 8109
        assert mock_bot.send_photo.await_count == 2
        assert "reply_to_message_id" not in mock_bot.send_photo.call_args_list[1].kwargs

    @pytest.mark.asyncio
    async def test_cooldown_blocks_non_breaking_but_breaking_bypasses_cooldown(self, temp_db):
        root = _make_article(
            title="Missile alert sounded near oil port",
            url="https://reuters.com/world/port-root",
            summary="Air raid sirens sounded near oil port terminals.",
            translated_title="زەنگی ئاگادارکردنەوە لە بەندەرێکی نەوتی لێدرا",
            translated_summary="بەرپرسان دەڵێن سیستەمی ئاگادارکردنەوە لە نزیک بەندەرەکە چالاک بووە.",
            event_key="oil_port_missile_alert",
            event_entities=["oil", "port", "missile", "alert"],
            hours_ago=0.08,  # 5 minutes ago (< 20m cooldown!)
            status=ArticleStatus.PUBLISHED,
            telegram_message_id=6100,
        )
        root_id = await temp_db.insert_article(root)
        await temp_db.update_article_status(
            root_id,
            ArticleStatus.PUBLISHED,
            telegram_message_id=6100,
            published_at=root.published_at,
        )

        dedup = SemanticDeduplicator(temp_db)

        # 1. Non-breaking update 5m later -> blocked by 20m cooldown
        non_brk = _make_article(
            title="Official statement confirms 3 drones destroyed near oil port",
            url="https://apnews.com/world/port-stmt",
            summary="Port authorities issued a statement noting 3 drones were downed.",
            event_key="oil_port_missile_alert",
            event_entities=["oil", "port", "drones", "statement"],
            score=8.0,
            is_breaking=False,
        )
        res_cool = await dedup.check_duplicate(non_brk)
        assert res_cool.classification == OUTCOME_DUPLICATE
        assert res_cool.is_duplicate is True
        assert res_cool.stage == "thread_cooldown"

        # 2. Genuine breaking emergency (score >= 9.0 & is_breaking=True) 5m later -> bypasses cooldown!
        brk_update = _make_article(
            title="Second wave retaliatory strikes ignite 2 oil storage tanks, 9 killed",
            url="https://bbc.com/news/port-retaliation",
            summary="Retaliatory ballistic strikes ignited 2 storage tanks with 9 killed in a second wave attack.",
            event_key="oil_port_missile_alert",
            event_entities=["oil", "port", "retaliation", "killed"],
            score=9.5,
            is_breaking=True,
        )
        res_brk = await dedup.check_duplicate(brk_update)
        assert res_brk.classification == OUTCOME_THREAD_UPDATE
        assert res_brk.is_duplicate is False
        assert res_brk.thread_root_message_id == 6100

    @pytest.mark.asyncio
    async def test_thread_cap_max_3_updates_and_24h_window_expiry(self, temp_db):
        dedup = SemanticDeduplicator(temp_db)

        # Case A: Root with 3 updates already published -> 4th update MUST become DUPLICATE
        root_item = _make_article(
            title="Border talks open in Ankara",
            url="https://reuters.com/world/ankara-talks-root",
            translated_title="گفتوگۆکانی سنوور لە ئەنقەرە دەستیپێکرد",
            event_key="ankara_border_security_talks",
            event_entities=["ankara", "border", "security", "talks"],
            hours_ago=6.0,
            status=ArticleStatus.PUBLISHED,
            telegram_message_id=7200,
        )
        root_id = await temp_db.insert_article(root_item)
        await temp_db.update_article_status(
            root_id,
            ArticleStatus.PUBLISHED,
            telegram_message_id=7200,
            published_at=root_item.published_at,
        )

        for idx in range(3):
            rep = _make_article(
                title=f"Ankara talks update {idx + 1}",
                url=f"https://reuters.com/world/ankara-u{idx + 1}",
                translated_title=f"نوێکردنەوەی کۆبوونەوەکەی ئەنقەرە {idx + 1}",
                event_key="ankara_border_security_talks",
                event_entities=["ankara", "border", "security", "talks"],
                hours_ago=4.0 - idx,
                status=ArticleStatus.PUBLISHED,
                telegram_message_id=7201 + idx,
                thread_root_message_id=7200,
            )
            rep_id = await temp_db.insert_article(rep)
            await temp_db.update_article_status(
                rep_id,
                ArticleStatus.PUBLISHED,
                telegram_message_id=7201 + idx,
                thread_root_message_id=7200,
                published_at=rep.published_at,
            )

        fourth_update = _make_article(
            title="Ceasefire deal confirmed with 5 clauses at Ankara talks",
            url="https://bbc.com/news/ankara-u4",
            summary="Officials announced a ceasefire deal containing 5 security clauses after Ankara summit.",
            event_key="ankara_border_security_talks",
            event_entities=["ankara", "border", "ceasefire", "clauses"],
        )
        res_4th = await dedup.check_duplicate(fourth_update)
        assert res_4th.classification == OUTCOME_DUPLICATE
        assert res_4th.is_duplicate is True
        assert res_4th.stage == "thread_cap_reached"

        # Case B: Event older than 24h (e.g. 28h ago) -> NEVER threaded to the expired message!
        old_root = _make_article(
            title="Baghdad budget talks stall over oil export clause",
            url="https://reuters.com/world/baghdad-old",
            translated_title="گفتوگۆی بودجە لە بەغدا ڕاوەستا",
            event_key="baghdad_budget_export_talks",
            event_entities=["baghdad", "budget", "oil", "talks"],
            hours_ago=28.0,
            status=ArticleStatus.PUBLISHED,
            telegram_message_id=7300,
        )
        old_id = await temp_db.insert_article(old_root)
        await temp_db.update_article_status(
            old_id,
            ArticleStatus.PUBLISHED,
            telegram_message_id=7300,
            published_at=old_root.published_at,
        )

        late_dev = _make_article(
            title="Agreement reached on Baghdad budget law with 12 billion trillion allocation",
            url="https://rudaw.net/baghdad-budget-pass",
            summary="Parliament confirmed an agreement reached allocating 12 trillion dinars for exports.",
            event_key="baghdad_budget_export_talks",
            event_entities=["baghdad", "budget", "agreement", "dinars"],
        )
        res_expired = await dedup.check_duplicate(late_dev)
        assert res_expired.classification != OUTCOME_THREAD_UPDATE
        assert res_expired.thread_root_message_id is None
        assert res_expired.classification == OUTCOME_NEW_STORY

    @pytest.mark.asyncio
    async def test_post_translation_sorani_downgrades_repeated_kurdish_summary_before_card(self, temp_db):
        root = _make_article(
            title="Drone intercepted over Erbil base",
            url="https://bbc.com/news/erbil-base-root",
            summary="Initial report of a drone interception over a base in Erbil.",
            translated_title="درۆنێک لە ئاسمانی بنکەیەکی هەولێر تێکشکێندرا",
            translated_summary=(
                "سیستەمی بەرگریی ئاسمانی فڕۆکەیەکی بێفڕۆکەوانی لە نزیک بنکەیەکی سەربازی لە هەولێر خستە خوارەوە "
                "و هیچ زیانێکی لێ نەکەوتەوە."
            ),
            event_key="erbil_base_drone_strike",
            event_entities=["erbil", "base", "drone", "strike"],
            hours_ago=1.5,
            status=ArticleStatus.PUBLISHED,
            telegram_message_id=8500,
        )
        root_id = await temp_db.insert_article(root)
        await temp_db.update_article_status(
            root_id,
            ArticleStatus.PUBLISHED,
            telegram_message_id=8500,
            published_at=root.published_at,
        )

        # Candidate English title looks like an "official response", yet its translated Kurdish
        # text repeats the exact same Sorani facts as root -> post-translation gate MUST downgrade!
        deceptive = _make_article(
            title="Official statement confirms earlier Erbil base drone interception details",
            url="https://rudaw.net/eng/erbil-base-dup",
            summary=(
                "Officials released a formal security statement repeating that coalitional air defenses "
                "downed an unmanned aircraft near the Erbil military base with zero reported casualties or damage."
            ),
            source_name="Rudaw",
            category="kurdistan",
            region="South Kurdistan",
        )
        deceptive.id = await temp_db.insert_article(deceptive)

        mock_bot = AsyncMock()
        publisher = TelegramPublisher()
        publisher._channel_id = "@KurdishGlobalNews"
        publisher._bot = mock_bot

        pipeline = ArticlePipeline(temp_db, publisher, HealthMonitor(), Metrics())
        pipeline._semantic_dedup = SemanticDeduplicator(temp_db)

        with patch.object(
            pipeline._ranker,
            "evaluate_global_relevance",
            new=AsyncMock(
                return_value=RelevanceEvaluation(
                    score=8,
                    is_breaking=False,
                    category="kurdistan",
                    reason="Related coverage",
                    passed=True,
                    event_key="erbil_base_drone_strike",
                    event_entities=["erbil", "base", "drone", "strike"],
                )
            ),
        ), patch.object(
            pipeline._extractor,
            "fetch_article_content_and_image",
            new=AsyncMock(return_value=("", None)),
        ), patch.object(
            pipeline._translator,
            "translate_article",
            new=AsyncMock(
                return_value=(
                    "لە هەولێر درۆنێک لە ئاسمانی بنکە سەربازییەکە تێکشکێندرا",
                    (
                        "سیستەمی بەرگریی ئاسمانی فڕۆکەیەکی بێفڕۆکەوانی لە نزیک بنکەیەکی سەربازی لە هەولێر خستە خوارەوە "
                        "بێ ئەوەی هیچ زیانێکی لێ بکەوێتەوە."
                    ),
                )
            ),
        ):
            ok = await pipeline.process_article(deceptive)

        assert ok is False
        mock_bot.send_photo.assert_not_called()
        saved = await temp_db.get_article_by_canonical_url(deceptive.canonical_url)
        assert saved is not None and saved.status == ArticleStatus.REJECTED_DUPLICATE


class TestSupabaseThreadingFallbackAndFormatter:
    @pytest.mark.asyncio
    async def test_supabase_missing_threading_columns_logs_warning_and_publishes_cleanly(self, caplog):
        seen_patches: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST" and request.url.path == "/rest/v1/articles":
                body = json.loads(request.content.decode("utf-8"))
                if "thread_root_message_id" in body:
                    return httpx.Response(
                        400,
                        json={"code": "PGRST204", "message": "Could not find the 'thread_root_message_id' column of 'articles'"},
                    )
                return httpx.Response(201, json=[{"id": 91, **body}])

            if request.method == "PATCH" and request.url.path == "/rest/v1/articles":
                patch_body = json.loads(request.content.decode("utf-8"))
                seen_patches.append(patch_body)
                if "thread_root_message_id" in patch_body or "telegram_message_id" in patch_body:
                    return httpx.Response(
                        400,
                        json={"code": "42703", "message": "column articles.thread_root_message_id does not exist"},
                    )
                return httpx.Response(204)

            if request.method == "GET" and request.url.path == "/rest/v1/articles":
                select = request.url.params.get("select", "")
                if "thread_root_message_id" in select:
                    return httpx.Response(
                        400,
                        json={"code": "42703", "message": "column articles.thread_root_message_id does not exist"},
                    )
                return httpx.Response(200, json=[])

            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            sb = SupabaseDatabase(url="https://proj.supabase.co", key="sb_key", client=client, base_backoff=0.0)
            art = _make_article(
                title="Update story",
                url="https://reuters.com/world/upd-1",
                thread_root_message_id=500,
            )

            with caplog.at_level(logging.WARNING):
                art_id = await sb.insert_article(art)
                assert art_id == 91
                await sb.update_article_status(
                    91, ArticleStatus.PUBLISHED, telegram_message_id=505, thread_root_message_id=500
                )
                history = await sb.get_recent_published_headlines(hours=72)
                assert history == []

        assert any("add_event_threading_columns.sql" in rec.message for rec in caplog.records)
        assert len(seen_patches) >= 1

    def test_caption_thread_update_prefix_and_1024_limit(self):
        formatter = ContentFormatter()
        upd = _make_article(
            title="Update",
            url="https://reuters.com/world/u1",
            translated_title="ژمارەی قوربانیان بەرزبووەوە",
            translated_summary="وەزارەتی تەندروستی ئاماری نوێی بڵاوکردەوە. " * 20,
            is_breaking=False,
            thread_root_message_id=7700,
        )
        upd.is_thread_update = True
        cap_normal = formatter.format_caption(upd)
        assert cap_normal.startswith("⚔️ <b>🔄 نوێکردنەوە | ژمارەی قوربانیان بەرزبووەوە</b>")
        assert len(cap_normal) <= 1024

        # Breaking update keeps 🔴 بەپەلە | priority badge
        upd.is_breaking = True
        cap_brk = formatter.format_caption(upd)
        assert cap_brk.startswith("⚔️ <b>🔴 بەپەلە | ژمارەی قوربانیان بەرزبووەوە</b>")
        assert len(cap_brk) <= 1024

    def test_zero_extra_gemini_calls_for_threading_decision(self):
        # Verify SemanticDeduplicator exposes no Gemini client/calls whatsoever
        dedup = SemanticDeduplicator(MagicMock())
        assert dedup.uses_gemini is False
