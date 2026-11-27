"""
Comprehensive unit and integration tests for cross-source & cross-language deduplication:
1. Canonical URL & Arabic/Kurdish script normalization (tracking params, www./m.,
   diacritics, أ/إ/آ -> ا, ة -> ه, ي/ى -> ی, ك -> ک) blocking duplicate URLs and
   near-identical SabrenNewss / Sabren_News1 posts BEFORE Gemini.
2. Persistent 48-72h published history loaded across stateless runs (SQLite & Supabase).
3. Zero-extra-cost cross-language event_key + event_entities fingerprinting catching the
   exact same story reported by English BBC, Arabic Sabereen, and Kurdish Rudaw.
4. Post-translation Sorani Kurdish deduplication gate catching differently worded Sorani
   headlines both within the same cycle and against 48-72h Supabase/SQLite history.
5. Zero false positives for genuinely different events sharing the same city/country/leader
   (e.g., Erbil salary talks vs. Erbil drone strike).
6. Graceful fallback when Supabase schema lacks optional `event_key` / `event_entities`
   columns (no crash, seamless Kurdish headline/summary & named_entities fallback).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from config import ArticleStatus, DEDUP_HISTORY_HOURS
from core.importance_ranker import RelevanceEvaluation
from core.pipeline import ArticlePipeline
from monitoring.health import HealthMonitor
from monitoring.metrics import Metrics
from processing.deduplicator import (
    SemanticDeduplicator,
    normalize_arabic_kurdish_text,
    normalize_url,
)
from sources.base import BaseSource, RawArticle
from storage.database import Database
from storage.models import Article
from storage.supabase_db import SupabaseDatabase
from tg_bot.publisher import TelegramPublisher


def _make_article(
    title: str,
    url: str,
    summary: str = "",
    source_name: str = "Reuters",
    category: str = "military_conflict",
    region: str = "",
    translated_title: str = "",
    translated_summary: str = "",
    event_key: str = "",
    event_entities: list[str] | None = None,
    hours_ago: float = 0.2,
    status: str = ArticleStatus.DISCOVERED,
) -> Article:
    dt_iso = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    art = Article(
        original_url=url,
        canonical_url=Article.canonicalize_url(url),
        original_title=title,
        normalized_title=Article.normalize_title(title),
        original_summary=summary or (
            f"{title}. Official diplomatic and defense correspondents reported detailed facts "
            "regarding military movements, cabinet consultations, and security protocols."
        ),
        translated_title=translated_title,
        translated_summary=translated_summary,
        source_name=source_name,
        category=category,
        region=region,
        publication_date=dt_iso,
        discovered_at=dt_iso,
        published_at=dt_iso if status == ArticleStatus.PUBLISHED else None,
        status=status,
        event_key=event_key,
        event_entities=list(event_entities or []),
    )
    art.compute_content_hash()
    art.compute_event_fingerprint()
    return art


@pytest.fixture
async def temp_db(tmp_path):
    db = Database(str(tmp_path / "test_dedup_upgrade.db"))
    await db.connect()
    yield db
    await db.close()


class TestURLAndScriptNormalization:
    """Test canonical URL tracking/prefix stripping and Arabic/Kurdish script unification."""

    def test_canonicalize_url_strips_utm_fbclid_ref_fragments_slashes_www_and_mobile(self):
        base = "https://bbc.com/news/world-middle-east-9988"
        variants = [
            "https://www.bbc.com/news/world-middle-east-9988/?utm_source=x&utm_custom_tag=foo#comments",
            "https://m.bbc.com/news/world-middle-east-9988?fbclid=IwAR123&ref=tg",
            "https://mobile.bbc.com/news/world-middle-east-9988///?si=abc&share=1",
        ]
        for variant in variants:
            assert normalize_url(variant) == base
            assert Article.canonicalize_url(variant) == base

        # Keeps legitimate non-tracking query parameters
        assert "id=42" in normalize_url("https://m.reuters.com/article/?id=42&utm_campaign=fall")

    def test_arabic_kurdish_normalization_unifies_variants_and_strips_diacritics(self):
        a = "أَرْبِيلْ — إِعْلَانُ آليَةٍ جَدِيدَةٍ فِي كُورْدِسْتَانْ كَك"
        norm = normalize_arabic_kurdish_text(a)
        assert "ا" in norm and "أ" not in norm and "إ" not in norm and "آ" not in norm
        assert "ك" not in norm and "ک" in norm
        assert "ي" not in norm and "ی" in norm
        assert "ة" not in norm

        # Reposted near-identical Sabereen posts with different hamzas/diacritics/tashkeel unify cleanly
        p1 = "عَاجِلْ | انْفِجَارٌ يَهُزُّ مَطَارَ أَرْبِيلَ فِي إِقْلِيمِ كُورْدِسْتَانَ"
        p2 = "عاجل انفجار يهز مطار اربيل في اقليم كوردستان"
        assert normalize_arabic_kurdish_text(p1) == normalize_arabic_kurdish_text(p2)

    @pytest.mark.asyncio
    async def test_duplicate_url_with_tracking_and_repost_across_sabereen_blocked_before_gemini(self, temp_db):
        # Pre-seed skipped URL with tracking query param + published SabrenNewss item
        await temp_db.mark_url_skipped(
            "https://www.reuters.com/world/erbil-talk?id=55&utm_source=twitter",
            title="Skipped item",
            score=2.0,
            reason="Blocked",
        )
        assert await temp_db.is_url_skipped("https://m.reuters.com/world/erbil-talk/?id=55&fbclid=999#frag") is True

        sab_1 = _make_article(
            title="عَاجِلْ | الدِّفَاعُ الجَوِّيُّ يُسْقِطُ طَائِرَةً مُسَيَّرَةً قُرْبَ مَطَارِ أَرْبِيلَ",
            url="https://t.me/AjaNews/5001",
            source_name="Al Jazeera",
            summary=(
                "أفادت مصادر أمنية بإسقاط طائرة مسيرة قرب مطار أربيل دون تسجيل خسائر بشرية. "
                "وأضافت المصادر أن القوات الأمنية طوقت محيط المنطقة وفتحت تحقيقاً فورياً."
            ),
            translated_title="دژە تیرۆر فڕۆکەیەکی بێفڕۆکەوانی لە نزیک فڕۆکەخانەی هەولێر خستە خوارەوە",
            status=ArticleStatus.PUBLISHED,
            hours_ago=0.5,
        )
        sab_1_id = await temp_db.insert_article(sab_1)
        await temp_db.update_article_status(sab_1_id, ArticleStatus.PUBLISHED, published_at=sab_1.published_at)

        # Identical repost on Al Jazeera Web without diacritics + different URL & hamza spelling
        sab_2 = _make_article(
            title="الدفاع الجوي يسقط طائره مسيره قرب مطار اربيل",
            url="https://www.aljazeera.com/news/erbil-7009?utm_source=forward",
            source_name="Al Jazeera",
            summary=(
                "افادت مصادر امنيه باسقاط طائره مسيره قرب مطار اربيل دون خسائر بشريه "
                "فيما بدات الجهات المختصه تحقيقا في الحادث."
            ),
            hours_ago=0.1,
        )
        sab_2.id = await temp_db.insert_article(sab_2)

        publisher = MagicMock(spec=TelegramPublisher)
        publisher.publish_article = AsyncMock(return_value=111)
        pipeline = ArticlePipeline(temp_db, publisher, HealthMonitor(), Metrics())

        with patch.object(pipeline._ranker, "evaluate_global_relevance", new=AsyncMock()) as mock_gem:
            ok = await pipeline.process_article(sab_2)

        assert ok is False
        # Blocked locally BEFORE any Gemini call!
        mock_gem.assert_not_called()
        publisher.publish_article.assert_not_called()
        assert await temp_db.is_url_skipped(sab_2.canonical_url) is True


class TestCrossLanguageEventFingerprinting:
    """Test English BBC / Arabic Sabereen / Kurdish Rudaw cross-language dedup with 0 extra Gemini calls."""

    @pytest.mark.asyncio
    async def test_bbc_english_sabereen_arabic_rudaw_kurdish_same_event_only_first_publishes(self, temp_db):
        mock_bot = AsyncMock()
        msg_obj = MagicMock()
        msg_obj.message_id = 3301
        mock_bot.send_photo = AsyncMock(return_value=msg_obj)

        publisher = TelegramPublisher()
        publisher._channel_id = "@KurdishGlobalNews"
        publisher._bot = mock_bot
        publisher._image_handler.download_image = AsyncMock(return_value=None)

        pipeline = ArticlePipeline(temp_db, publisher, HealthMonitor(), Metrics())
        pipeline._semantic_dedup = SemanticDeduplicator(temp_db, base_backoff=0.0)

        # Phrased distinctly across 3 languages so lexical stage passes them to single batch Gemini call
        reuters_en = _make_article(
            title="Explosive UAV intercepted above northern aviation hub overnight",
            url="https://reuters.com/world/erbil-incident-1",
            source_name="Reuters",
            category="kurdistan",
            region="South Kurdistan",
            summary=(
                "Air defense batteries neutralized an unmanned aerial craft approaching a coalition compound "
                "while aviation authorities briefly paused departures and initiated a technical sweep."
            ),
        )
        sab_ar = _make_article(
            title="مصدر امني يؤكد تعامل المنظومة مع جسم طائر غامض شمال البلاد",
            url="https://t.me/AjaNews/8802",
            source_name="Al Jazeera",
            category="military_conflict",
            region="iraq_region",
            summary=(
                "تحدث شهود عيان عن اصوات مرتفعة في السماء عقب رصد هدف جوي ليلي وسط استنفار كامل "
                "للوحدات المكلفة بحماية القواعد والمنشآت المدنية."
            ),
        )
        rudaw_ku = _make_article(
            title="دەزگا ئەمنییەکان زانیاری لەسەر تێکشکاندنی ئامانجێکی ئاسمانی بڵاودەکەنەوە",
            url="https://rudaw.net/sorani/kurdistan/8803",
            source_name="Rudaw",
            category="kurdistan",
            region="South Kurdistan",
            summary=(
                "بەرپرسانی سەربازی ئاشکرایان کرد تیمەکانی بەرگریی ئاسمانی لە کاتی گونجاودا توانیویانە "
                "جموجۆڵێکی نائاسایی ئاسمانی ڕابگرن بێ هیچ زیانێک."
            ),
        )
        for art in (reuters_en, sab_ar, rudaw_ku):
            art.id = await temp_db.insert_article(art)

        # Single Gemini batch scoring call yields shared English event_key & event_entities!
        async def mock_batch_eval(articles, **_kwargs):
            return [
                RelevanceEvaluation(
                    score=9,
                    is_breaking=True,
                    category="kurdistan" if "rudaw" in art.original_url or "reuters" in art.original_url else "military_conflict",
                    reason="Erbil airport UAV interception",
                    passed=True,
                    event_key="erbil_airport_drone_attack",
                    event_entities=["erbil", "drone", "airport", "attack"],
                )
                for art in articles
            ]

        tr_calls = 0

        async def mock_translate(title: str, summary: str = ""):
            nonlocal tr_calls
            tr_calls += 1
            return (
                "فڕۆکەیەکی بێفڕۆکەوان لە ئاسمانی فڕۆکەخانەی هەولێر تێکشکێندرا",
                "سیستەمی بەرگریی ئاسمانی هێرشێکی درۆنی لە نزیک فڕۆکەخانەی هەولێر پووچەڵ کردەوە.",
            )

        with patch.object(pipeline._ranker, "evaluate_batch_relevance", side_effect=mock_batch_eval), \
             patch.object(pipeline._extractor, "fetch_article_content_and_image", new=AsyncMock(return_value=("", None))), \
             patch.object(pipeline._translator, "translate_article", side_effect=mock_translate):
            res = await pipeline.process_batch([reuters_en, sab_ar, rudaw_ku], max_publish=8)

        # Only ONE article publishes; duplicates blocked BEFORE calling translation/card generation!
        assert res["published"] == 1
        assert tr_calls == 1
        mock_bot.send_photo.assert_awaited_once()
        assert await temp_db.is_url_skipped(sab_ar.canonical_url) is True
        assert await temp_db.is_url_skipped(rudaw_ku.canonical_url) is True


class TestPostTranslationSoraniGateAndHistory:
    """Test post-translation Kurdish deduplication within cycle and across 48-72h history."""

    @pytest.mark.asyncio
    async def test_post_translation_sorani_catches_rephrased_kurdish_against_48h_history_and_same_cycle(
        self, temp_db, caplog
    ):
        # 1. Seed a story published 36 hours ago (inside 48-72h window, outside old 12h window!)
        old_published = _make_article(
            title="Agreement reached on Red Sea shipping escorts",
            url="https://reuters.com/world/red-sea-36h",
            translated_title="ڕێککەوتن لەسەر پاراستنی کەشتییە بازرگانییەکان لە دەریای سوور کرا",
            translated_summary=(
                "وڵاتانی هاوپەیمان بڕیاریان دا هێزی دەریایی بۆ پاراستنی ڕێڕەوی کەشتییە بازرگانییەکان "
                "لە دەریای سوور جێگیر بکەن."
            ),
            status=ArticleStatus.PUBLISHED,
            hours_ago=36.0,
        )
        old_id = await temp_db.insert_article(old_published)
        await temp_db.update_article_status(
            old_id,
            ArticleStatus.PUBLISHED,
            published_at=old_published.published_at,
        )

        # Verify 72h loader fetches the 36h-old story!
        assert DEDUP_HISTORY_HOURS >= 48
        history_72h = await temp_db.get_recent_published_headlines(hours=72)
        assert [h["id"] for h in history_72h] == [old_id]

        mock_bot = AsyncMock()
        msg_obj = MagicMock()
        msg_obj.message_id = 4402
        mock_bot.send_photo = AsyncMock(return_value=msg_obj)

        publisher = TelegramPublisher()
        publisher._channel_id = "@KurdishGlobalNews"
        publisher._bot = mock_bot
        publisher._image_handler.download_image = AsyncMock(return_value=None)

        pipeline = ArticlePipeline(temp_db, publisher, HealthMonitor(), Metrics())
        pipeline.set_cycle_published_history(history_72h)

        # Candidate A: rephrased Kurdish translation matching 36h-old published Red Sea story
        cand_vs_history = _make_article(
            title="Multinational coalition confirms maritime escort corridor deployment",
            url="https://apnews.com/world/corridor-naval-1",
        )
        # Candidate B1 & B2: two genuinely NEW distinct English headlines that translate into
        # differently worded Kurdish headlines describing the same Kirkuk oil pipeline blast in the SAME cycle!
        cand_same_cycle_1 = _make_article(
            title="Sabotage halts northern hydrocarbon flow near Baba Gurgur field",
            url="https://rudaw.net/energy/kirkuk-flow-1",
            category="kurdistan",
            region="South Kurdistan",
        )
        cand_same_cycle_2 = _make_article(
            title="Energy export line suffers damage following night incident in Iraq",
            url="https://k24.example/energy/export-line-2",
            category="politics_world",
        )
        for art in (cand_vs_history, cand_same_cycle_1, cand_same_cycle_2):
            art.id = await temp_db.insert_article(art)

        async def mock_eval(articles, **_kwargs):
            return [
                RelevanceEvaluation(score=8, is_breaking=False, category="politics_world", reason="Relevant", passed=True)
                for _ in articles
            ]

        translations = {
            cand_vs_history.original_title: (
                "هاوپەیمانان کەشتییە بازرگانییەکان لە دەریای سوور دەپارێزن و ڕێککەوتنەکە جێبەجێ دەکرێت",
                "هێزە دەریاییەکانی هاوپەیمانان پاراستنی ڕێڕەوی کەشتییە بازرگانییەکانی دەریای سووریان ڕاگەیاند.",
            ),
            cand_same_cycle_1.original_title: (
                "تەقینەوە لە هێڵی گواستنەوەی نەوتی کەرکووک هەناردەکردنی ڕاگرت",
                "بەهۆی تەقینەوەیەک لە بۆریی گواستنەوەی نەوت لە کەرکووک هەناردەی نەوت بە شێوەیەکی کاتی ڕاوەستا.",
            ),
            cand_same_cycle_2.original_title: (
                "هەناردەی نەوتی کەرکووک بەهۆی تەقینەوەی بۆرییەکەوە پەکی کەوت",
                "تەقینەوەیەک لە بۆریی گواستنەوەی نەوتی کەرکووک بووە هۆی وەستانی کاتیی هەناردەکردن.",
            ),
        }

        with caplog.at_level(logging.INFO), \
             patch.object(pipeline._ranker, "evaluate_batch_relevance", side_effect=mock_eval), \
             patch.object(pipeline._extractor, "fetch_article_content_and_image", new=AsyncMock(return_value=("", None))), \
             patch.object(
                 pipeline._translator,
                 "translate_article",
                 side_effect=lambda title, summary="": translations[title],
             ):
            res = await pipeline.process_batch(
                [cand_vs_history, cand_same_cycle_1, cand_same_cycle_2],
                max_publish=8,
            )

        # Exactly ONE Kirkuk pipeline story publishes; cand_vs_history (vs 36h DB) & the duplicate Kirkuk story are skipped!
        assert res["published"] == 1
        assert mock_bot.send_photo.await_count == 1
        assert "Duplicate story blocked:" in caplog.text
        assert await temp_db.is_url_skipped(cand_vs_history.canonical_url) is True
        kirkuk_skipped = [
            await temp_db.is_url_skipped(cand_same_cycle_1.canonical_url),
            await temp_db.is_url_skipped(cand_same_cycle_2.canonical_url),
        ]
        assert kirkuk_skipped.count(True) == 1

    @pytest.mark.asyncio
    async def test_distinct_events_in_same_city_or_leader_not_falsely_merged(self, temp_db):
        dedup = SemanticDeduplicator(temp_db)

        published_item = {
            "id": 10,
            "original_title": "KRG and Baghdad reach Erbil civil servant salary agreement",
            "translated_title": "هەولێر و بەغدا لەسەر مووچەی فەرمانبەرانی هەرێم ڕێککەوتن",
            "translated_summary": "شاندی حکوومەتی هەرێمی کوردستان لە هەولێر ڕێککەوتن لەسەر ناردنی مووچە و بودجە.",
            "event_key": "erbil_baghdad_krg_salary_agreement",
            "event_entities": ["erbil", "krg", "baghdad", "salary", "agreement"],
        }

        distinct_candidate = _make_article(
            title="Drone intercepted near military base in Erbil",
            url="https://reuters.com/world/erbil-drone-9",
            category="kurdistan",
            translated_title="درۆنێکی بۆمبڕێژکراو لە ئاسمانی هەولێر خرایە خوارەوە",
            translated_summary="بەرگریی ئاسمانی فڕۆکەیەکی بێفڕۆکەوانی لە نزیک بنکەیەکی سەربازی لە هەولێر پووچەڵ کردەوە.",
            event_key="erbil_military_drone_interception",
            event_entities=["erbil", "krg", "baghdad", "drone", "interception"],
        )

        # Neither event_key/event_entities check NOR post-translation Sorani gate may merge them!
        ev_res = await dedup.check_event_match(distinct_candidate, recent_history=[published_item])
        ku_res = await dedup.check_post_translation_duplicate(distinct_candidate, recent_history=[published_item])
        full_res = await dedup.check_duplicate(distinct_candidate, recent_history=[published_item])

        assert ev_res.is_duplicate is False
        assert ku_res.is_duplicate is False
        assert full_res.is_duplicate is False

        # Also verify two distinct Erbil stories WITHOUT structured event_key stay unmerged!
        published_no_key = {**published_item, "event_key": "", "event_entities": []}
        cand_no_key = _make_article(
            title="Drone intercepted near military base in Erbil",
            url="https://reuters.com/world/erbil-drone-no-key",
            category="kurdistan",
            translated_title="درۆنێکی بۆمبڕێژکراو لە ئاسمانی هەولێر خرایە خوارەوە",
            translated_summary="بەرگریی ئاسمانی فڕۆکەیەکی بێفڕۆکەوانی لە نزیک بنکەیەکی سەربازی لە هەولێر پووچەڵ کردەوە.",
        )
        assert (await dedup.check_duplicate(cand_no_key, recent_history=[published_no_key])).is_duplicate is False
        assert (await dedup.check_post_translation_duplicate(cand_no_key, recent_history=[published_no_key])).is_duplicate is False


class TestSupabaseSchemaBackwardCompatibility:
    """Ensure Supabase missing optional `event_key`/`event_entities` columns degrade cleanly without crashing."""

    @pytest.mark.asyncio
    async def test_supabase_falls_back_gracefully_when_event_columns_missing(self):
        now = datetime.now(timezone.utc)
        insert_select_attempts: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST" and request.url.path == "/rest/v1/articles":
                body = json.loads(request.content.decode("utf-8"))
                insert_select_attempts.append({"method": "POST", "keys": sorted(body.keys())})
                if "event_key" in body or "event_entities" in body:
                    return httpx.Response(
                        400,
                        json={"code": "PGRST204", "message": "Could not find the 'event_key' column of 'articles'"},
                    )
                return httpx.Response(201, json=[{"id": 71, **body}])

            if request.method == "GET" and request.url.path == "/rest/v1/articles":
                select = request.url.params.get("select", "")
                insert_select_attempts.append({"method": "GET", "select": select})
                if "event_key" in select:
                    return httpx.Response(
                        400,
                        json={"code": "42703", "message": "column articles.event_key does not exist"},
                    )
                return httpx.Response(
                    200,
                    json=[
                        {
                            "id": 71,
                            "original_title": "NATO summit opens in Brussels",
                            "translated_title": "لووتکەی ناتۆ لە برۆکسل دەستیپێکرد",
                            "translated_summary": "وەزیرانی بەرگریی ناتۆ لە برۆکسل کۆبوونەوەیان ئەنجامدا.",
                            "category": "politics_world",
                            "published_at": (now - timedelta(hours=30)).isoformat(),
                            "discovered_at": (now - timedelta(hours=30)).isoformat(),
                            "named_entities": json.dumps(
                                {"event_key": "brussels_nato_defense_summit", "event_entities": ["nato", "brussels", "summit"]}
                            ),
                        }
                    ],
                )

            raise AssertionError(f"Unexpected Supabase request: {request.method} {request.url}")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            sb = SupabaseDatabase(url="https://proj.supabase.co", key="sb_key", client=client, base_backoff=0.0)
            art = _make_article(
                title="NATO summit opens in Brussels",
                url="https://reuters.com/world/nato-brussels",
                translated_title="لووتکەی ناتۆ لە برۆکسل دەستیپێکرد",
                translated_summary="وەزیرانی بەرگریی ناتۆ لە برۆکسل کۆبوونەوەیان ئەنجامدا.",
                event_key="brussels_nato_defense_summit",
                event_entities=["nato", "brussels", "summit"],
                status=ArticleStatus.PUBLISHED,
            )

            # Insert auto-retries without event_key/event_entities columns (persisting JSON inside named_entities)
            inserted_id = await sb.insert_article(art)
            assert inserted_id == 71

            # Querying 72h published history also recovers gracefully & still unpacks Sorani text + event metadata!
            history = await sb.get_recent_published_headlines(hours=72)
            assert len(history) == 1
            assert history[0]["id"] == 71
            assert history[0]["translated_title"] == "لووتکەی ناتۆ لە برۆکسل دەستیپێکرد"
            assert history[0]["translated_summary"] != ""
            assert history[0]["event_key"] == "brussels_nato_defense_summit"
            assert "nato" in history[0]["event_entities"]

            # Post-translation Sorani gate blocks a rephrased duplicate against recovered Supabase history cleanly!
            dedup = SemanticDeduplicator(sb)
            rephrased = _make_article(
                title="Alliance ministers meet in Belgian capital",
                url="https://bbc.com/news/nato-brussels-2",
                translated_title="وەزیرانی بەرگریی ناتۆ لە لووتکەی برۆکسلدا کۆبوونەوە",
                translated_summary="کۆبوونەوەی هاوپەیمانانی ناتۆ لە برۆکسل بەردەوامە.",
            )
            res = await dedup.check_post_translation_duplicate(rephrased, recent_history=history)
            assert res.is_duplicate is True
            assert res.matched_id == 71
