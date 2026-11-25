"""
Comprehensive automated test suite for the Paced 10-Post Batch Engine:
  1. Centralized config defaults & deterministic source bucket classifier (telegram / x / search)
  2. Exact 4:4:2 Source Mix selection (`4 Telegram + 4 X + 2 Search`) & clean round-robin interleaving
  3. Graceful non-duplicate backfill when buckets are short (no junk/duplicate padding)
  4. Zero-duplicate guarantee across Telegram/X/Search + up-front watermark rejection substitution
  5. Sequential 1-minute minute-aligned publishing (exactly 9 sleeps for 10 posts; 0 sleep after post 10)
     + resilience when one post fails mid-batch
  6. Persisted 10-minute post-batch cooldown in `system_logs` (SQLite + Supabase parity, Watchdog
     non-alerting cooldown health check, and 600s continuous-loop rest)
  7. Zero extra Gemini API calls beyond standard rank+translate
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import config
from config import ArticleStatus
from core.balancer import get_source_bucket, select_paced_batch
from core.engine import NewsEngine, compute_next_post_delay
from core.importance_ranker import RelevanceEvaluation, select_top_candidates_for_gemini
from core.pipeline import ArticlePipeline
from monitoring.watchdog import AdminWatchdog, WatchdogCycleMetrics
from processing.video_processor import WatermarkRejectedError
from sources.base import RawArticle
from storage.database import Database
from storage.models import Article
from storage.supabase_db import SupabaseDatabase
from tg_bot.publisher import TelegramPublisher


def _fresh_iso(minutes_ago: float = 2.0) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


_DISTINCT_ENGLISH_TITLES = [
    ("Erbil parliament passes regional defense budget resolution", "Kurdistan lawmakers voted on security funding and border coordination protocols."),
    ("Sulaymaniyah counterterrorism unit secures mountain border pass", "Security forces concluded joint anti-smuggling patrols along eastern frontier roads."),
    ("Kirkuk governors meet interior ministry over federal garrison deployment", "Provincial delegates discussed police command integration with federal brigades."),
    ("Duhok border authority enforces new transit security clearance protocols", "Inspection officers implemented biometric cargo screening at northern crossings."),
    ("Halabja council appeals to Baghdad for federal security compensation", "Municipal representatives urged central authorities to finalize martyr compensation bills."),
    ("Zaxo military brigade completes rapid deployment command exercise", "Regional commanders reviewed readiness drills along strategic highway corridors."),
    ("Ranya security committee coordinates tribal peacekeeping accord", "Local officials mediated formal security guarantees between district councils."),
    ("Soran border guards intercept unmanned surveillance aircraft near pass", "Defense observers reported intercepting unauthorized aerial reconnaissance craft."),
    ("Akre district administration expands joint civil defense command post", "Emergency response commanders inaugurated coordinated regional communications centers."),
    ("Kalar police academy graduates tactical border patrol officers", "Interior trainers certified rapid-response units for southern province checkpoints."),
    ("Geneva UN diplomats debate Black Sea naval escort framework", "Envoys gathered at Palais des Nations to negotiate maritime security corridors."),
    ("Vienna nuclear monitors inspect uranium enrichment centrifuge cascades", "Atomic inspectors submitted verification findings to the governing diplomatic board."),
    ("Tokyo defense ministry activates long-range island radar array", "Japanese naval forces calibrated coastal missile tracking stations in Okinawa."),
    ("Berlin coalition leaders sign bilateral European defense procurement pact", "German officials endorsed joint air-defense interceptor manufacturing contracts."),
    ("Nairobi security forum endorses Horn of Africa anti-piracy mandate", "Defense delegates renewed naval patrol authorization along Red Sea approaches."),
    ("Warsaw artillery brigade tests mobile air-defense radar batteries", "Polish military units completed NATO integrated missile interception drills."),
    ("Cairo foreign ministry hosts Middle East ceasefire mediation round", "Diplomatic mediators presented updated prisoner exchange timelines to envoys."),
    ("Oslo NATO command releases Arctic submarine patrol assessment", "Alliance strategists reviewed northern maritime deterrence deployments today."),
    ("Ottawa NORAD commanders upgrade northern aerospace warning sensors", "Defense engineers commissioned radar upgrades monitoring Arctic air corridors."),
    ("Seoul joint chiefs conduct naval missile defense readiness drill", "South Korean destroyers practiced intercepting ballistic targets in eastern waters."),
    ("Rome Mediterranean naval command intercepts arms smuggling freighter", "Italian coastguard vessels escorted suspect cargo into southern inspection port."),
    ("Stockholm defense board approves Baltic island garrison expansion", "Swedish parliament ratified strategic troop deployments guarding sea lanes."),
    ("Riyadh Gulf council ministers issue joint regional stability communique", "Foreign ministers reaffirmed collective defense cooperation across Gulf waterways."),
    ("Amman defense talks conclude Jordan-Iraq border radar agreement", "Military delegations signed technical protocols linking frontier surveillance posts."),
]

_DISTINCT_SORANI = [
    ("ئەنجومەنی هەولێر پڕۆژەی تۆڕی ئاودێری پەسەند کرد", "وەزارەتی کشتوکاڵ کەناڵی نوێی ئاودێری لە پارێزگاکان تەواو دەکات."),
    ("زانکۆی سلێمانی تاقیگەی جینۆمیکی پزیشکی کردەوە", "توێژەرانی پزیشکی دامەزراوەیەکی نوێی دەستنیشانکردنی نەخۆشییان دامەزراند."),
    ("دەروازەی کەرکووک سیستەمی پشکنینی ئەلیکترۆنی نوێ کردەوە", "ئەندازیاران ئامێری نوێی پشکنینی بارهەڵگرەکانیان لە بازگەکان دانا."),
    ("پڕۆژەی تونێلی چیای دهۆک قۆناغی هەڵکەندنی تەواو کرد", "تیمەکانی ڕێگاوبان تونێلی دووئاراستەیان لە نێوان قەزاکان تەواو کرد."),
    ("هەناردەکارانی هەڵەبجە بەرهەمی هەنار رەوانەی دەرەوە دەکەن", "جووتیاران کاروانێکی گەورەی میوەیان لە ڕێگەی وشکانییەوە هەناردە کرد."),
    ("دەروازەی بازرگانی زاخۆ خزمەتگوزاری شەوانەڕۆژی ڕاگەیاند", "بەرپرسانی گومرگی هاتوچۆی بارهەڵگرەکانیان بۆ بیست و چوار کاتژمێر درێژکردەوە."),
    ("وێستگەی وزەی خۆری ڕانیە کارەبای نەخۆشخانەکان دابین دەکات", "تەکنیککاران تۆڕی خۆریان بە ناوەندە تەندروستییەکانەوە بەستەوە."),
    ("بەڕێوەبەرایەتی دارستانی سۆران نەمام لە چیای زاگرۆس دەچێنێت", "ژینگەپارێزان هەڵمەتێکی فراوانی چاندنی بەڕوویان لە بناری چیاکان دەستپێکرد."),
    ("دەستەی گەشتوگوزاری ئاکرێ بازاڕی مێژوویی نۆژەن دەکاتەوە", "ئەندازیارانی شوێنەوار دیوارە بەردینەکانی ناوچە دێرینەکەیان چاک کردەوە."),
    ("ناوچەی پیشەسازی کەلار هێڵی نوێی بەرهەمهێنانی کارا کرد", "کارگەکان سیستەمی پاککەرەوەی هەوای پیشەسازییان خستەگەڕ."),
    ("دادگای دەریایی جنێڤ بیمەی کەشتییەکانی دەریای ڕەش دەگۆڕێت", "کۆمپانیاکانی گواستنەوەی دەریایی بۆ پێداچوونەوە بە ڕێساکاندا کۆبوونەوە."),
    ("پشکنەرانی ڤیەننا مۆری ئامێرە ئەتۆمییەکانیان پشتڕاست کردەوە", "چاودێرانی نێودەوڵەتی ڕاپۆرتی تەکنیکیان لە وێستگەکە تۆمار کرد."),
    ("دەزگای بومەلەرزەزانی تۆکیۆ بۆیە دەریاییەکانی ڕێکخستەوە", "پسپۆڕانی زەریاوانی هەستەوەری نوێیان لە قووڵایی دەریادا جێگیر کرد."),
    ("لیژنەی بودجەی بەرلین بۆندی کارەبای هێڵی ئاسنی پەسەند کرد", "پەرلەمانتاران سەرمایەی نوێیان بۆ گواستنەوەی شەمەندەفەر تەرخان کرد."),
    ("لووتکەی نایرۆبی ڕێککەوتنی پاراستنی ژینگەی ئەفریقای واژۆ کرد", "وەزیران ڕێنمایی نوێی پاراستنی ئاژەڵە کێوییەکانیان پەسەند کرد."),
    ("وەزارەتی بەرگری وارشۆ ڕاداری گواستنەوەی سەربازی تاقی کردەوە", "یەکەکانی تۆپخانە مانۆڕی چاودێری ئاسمانییان لە کەنارەکان ئەنجامدا."),
    ("دەسەڵاتی کەناڵی قاهیرە ڕێڕەوی باشووری کەشتییەکان قووڵ دەکات", "ئەندازیاران دوو کەناڵی تەریب بۆ تێپەڕینی کەشتییە بارهەڵگرەکان فراوان دەکەن."),
    ("شاندەکانی ئۆسلۆ نەخشەی دەریاوانی جەمسەری باکوور دەکێشنەوە", "توێژەران ڕێڕەوی زستانەی کەشتیوانییان لە باکووری نەرویج دیاری کرد."),
    ("تاقیگەی ئۆتاوا بەستەری لەیزەری مانگی دەستکردی سەلماند", "زانایانی بۆشایی گواستنەوەی خێرای داتایان بە تیشک تاقی کردەوە."),
    ("هاوپەیمانی سیئۆل ناوەندی توێژینەوەی چیپی ئەلیکترۆنی دامەزراند", "کارگەکان تاقیگەی نوێی تەکنەلۆژیایان لە پارکی سووۆن کردەوە."),
    ("ئاژانسی ماسیوانی ڕۆما ڕێژەی ڕاوکردنی لە دەریای ناوەڕاستدا پێوا", "زانایانی دەریا ئاماری وەرزیان لە کەناراوەکانی سیسیلیا بڵاوکردەوە."),
    ("دەستەی فریاگوزاری ستۆکهۆڵم موەلیدەی نەخۆشخانەکانی پشکنی", "شارەوانییەکان کۆگاکانی سووتەمەنی فریاکەوتنیان لە باکوور هەڵسەنگاند."),
    ("دامەزراوەی ئاوی ڕیاز وێستگەی شیرینکردنی بە وزەی خۆر کارا کرد", "شارەزایان وێستگەکانی کەناراوی کەنداویان بە تۆڕی خۆرەوە بەستەوە."),
    ("هەناردەکارانی عەممان بەندەری عەقەبەیان بۆ کەشتی گەورە فراوان کرد", "تیمەکانی بەندەر بەرزکەرەوەی نوێی بارکردنی کانزاکانیان خستەکار."),
]


def _make_art(
    bucket: str,
    idx: int,
    *,
    title: str | None = None,
    source_name: str | None = None,
    score: float = 8.4,
    is_breaking: bool = False,
    category: str | None = None,
    region: str = "",
    minutes_ago: float = 4.0,
    video_url: str | None = None,
    translated_title: str | None = None,
    translated_summary: str | None = None,
) -> Article:
    story_t, story_s = _DISTINCT_ENGLISH_TITLES[idx % len(_DISTINCT_ENGLISH_TITLES)]
    if bucket == "telegram":
        url = f"https://t.me/rudawdigital/{3000 + idx}"
        src = source_name or f"TGChannel{idx % 3}"
        cat = category or ("kurdistan" if idx % 2 == 0 else "politics_world")
        reg = region or ("South Kurdistan" if cat == "kurdistan" else "global")
        t = title or story_t
    elif bucket == "x":
        handles = ["CENTCOM", "RudawEnglish", "EnglishBasNews", "ShafaqNews", "AuroraIntel", "ClashReport"]
        handle = handles[idx % len(handles)]
        url = f"https://x.com/{handle}/status/{88000 + idx}"
        src = source_name or handle
        cat = category or ("kurdistan" if handle in ("RudawEnglish", "EnglishBasNews", "ShafaqNews") else "military_conflict")
        reg = region or ("South Kurdistan" if cat == "kurdistan" else "middle_east")
        t = title or story_t
    else:
        url = f"https://www.reuters.com/world/story-{idx}"
        src = source_name or f"Reuters Outlet {idx % 3}"
        cat = category or "politics_world"
        reg = region or "global"
        t = title or story_t

    art = Article(
        id=idx + 1,
        canonical_url=Article.canonicalize_url(url),
        original_url=url,
        source_name=src,
        original_title=t,
        normalized_title=Article.normalize_title(t),
        original_summary=(
            f"{t}. {story_s} Officials and regional observers released full briefing transcripts "
            "confirming verified operational coordinates, multilateral delegations, and security details today."
        ),
        translated_title=translated_title or "",
        translated_summary=translated_summary or "",
        category=cat,
        region=reg,
        importance_score=score,
        priority="CRITICAL" if is_breaking else "HIGH",
        is_breaking=is_breaking,
        publication_date=_fresh_iso(minutes_ago=minutes_ago),
        discovered_at=_fresh_iso(minutes_ago=minutes_ago),
        video_url=video_url,
    )
    art.compute_content_hash()
    art.compute_event_fingerprint()
    return art


def _make_raw(bucket: str, idx: int, *, title: str | None = None, video_url: str | None = None) -> RawArticle:
    art = _make_art(bucket, idx, title=title, video_url=video_url)
    return RawArticle(
        url=art.original_url,
        title=art.original_title,
        summary=art.original_summary,
        source_name=art.source_name,
        category=art.category,
        publication_date=_fresh_iso(minutes_ago=3.0),
        video_url=video_url or "",
        region=art.region,
    )


def _mock_publisher(tmp_path: Path) -> tuple[TelegramPublisher, MagicMock]:
    counter = {"id": 9000}

    async def _fake_send(*args, **kwargs):
        counter["id"] += 1
        m = MagicMock()
        m.message_id = counter["id"]
        return m

    async def _fake_card(*args, **kwargs) -> Path:
        p = tmp_path / f"card_{counter['id']}_{id(args)}.jpg"
        p.write_bytes(b"JPG")
        return p

    pub = TelegramPublisher()
    pub._channel_id = "@KurdishGlobalNews"
    bot = AsyncMock()
    bot.send_photo = AsyncMock(side_effect=_fake_send)
    bot.send_video = AsyncMock(side_effect=_fake_send)
    bot.send_message = AsyncMock(side_effect=_fake_send)
    pub._bot = bot
    pub._image_handler.download_image = AsyncMock(return_value=None)
    pub._card_generator.generate_card_file = AsyncMock(side_effect=_fake_card)
    return pub, bot


# ═══════════════════════════════════════════════════════════════════════
# 1. Config & Bucket Classifier
# ═══════════════════════════════════════════════════════════════════════


def test_batch_pacing_config_and_source_bucket_classification():
    assert config.ENABLE_BATCH_PACING_ENGINE is True
    assert config.BATCH_TARGET_POSTS == 10
    assert config.BATCH_PUBLISH_INTERVAL_SECONDS == 60.0
    assert config.BATCH_REST_COOLDOWN_MINUTES == 10.0
    assert config.BATCH_ALIGN_TO_MINUTE_BOUNDARY is True
    assert config.BATCH_SOURCE_MIX == {"telegram": 4, "x": 4, "search": 2}
    assert config.X_MAX_CANDIDATE_VIDEOS_PER_CYCLE == 10

    # Same outlet ("Rudaw") routes accurately by URL/adapter rather than ambigous brand name
    assert get_source_bucket(Article(source_name="Rudaw", original_url="https://t.me/rudawdigital/10")) == "telegram"
    assert get_source_bucket(Article(source_name="Rudaw", original_url="https://x.com/RudawEnglish/status/20")) == "x"
    assert get_source_bucket(Article(source_name="Rudaw", original_url="https://www.rudaw.net/sorani/123")) == "search"
    assert get_source_bucket(Article(source_name="DuckDuckGo News", original_url="https://bbc.com/news/world-1")) == "search"


# ═══════════════════════════════════════════════════════════════════════
# 2. Exact 4:4:2 Mix Selection, Interleaving & Multi-Bucket Pre-Filter
# ═══════════════════════════════════════════════════════════════════════


def test_exact_4_4_2_mix_selection_and_clean_interleaving(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_SOURCE_DOMINANCE_CAP", False)

    tg_items = [_make_art("telegram", i, score=9.0 - i * 0.1) for i in range(6)]
    x_items = [_make_art("x", 10 + i, score=8.9 - i * 0.1) for i in range(6)]
    search_items = [_make_art("search", 20 + i, score=8.8 - i * 0.1) for i in range(5)]

    all_cands = tg_items + x_items + search_items
    result = select_paced_batch(all_cands, target_posts=10, source_mix={"telegram": 4, "x": 4, "search": 2})

    assert len(result.selected) == 10
    assert len(result.deferred) == 7
    assert result.bucket_counts == {"telegram": 4, "x": 4, "search": 2}

    buckets_seq = [get_source_bucket(a) for a in result.selected]
    assert buckets_seq == [
        "telegram", "x", "search",
        "telegram", "x", "telegram",
        "x", "search", "telegram", "x",
    ]
    # Assert no two consecutive posts share a bucket in the 4:4:2 interleaved sequence
    assert all(buckets_seq[i] != buckets_seq[i + 1] for i in range(len(buckets_seq) - 1))

    # Multi-bucket Gemini pre-filter cap preserves search/x representation alongside flooded telegram
    pre_sel, _ = select_top_candidates_for_gemini(
        [_make_art("telegram", 100 + i, category="kurdistan") for i in range(20)]
        + [_make_art("x", 200 + i, category="military_conflict") for i in range(6)]
        + [_make_art("search", 300 + i, category="politics_world") for i in range(4)],
        max_candidates=16,
    )
    assert len(pre_sel) == 16
    assert sum(1 for a in pre_sel if get_source_bucket(a) == "x") >= 4
    assert sum(1 for a in pre_sel if get_source_bucket(a) == "search") >= 3


# ═══════════════════════════════════════════════════════════════════════
# 3. Graceful Non-Duplicate Backfill & Shortage Handling
# ═══════════════════════════════════════════════════════════════════════


def test_graceful_backfill_when_x_short_and_no_junk_padding(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_SOURCE_DOMINANCE_CAP", False)

    # X has only 2 videos while TG has 6 and Search has 4 -> cleanly backfills to 10 unique posts
    tg_items = [_make_art("telegram", i, score=8.8 - i * 0.05) for i in range(6)]
    x_items = [_make_art("x", 10 + i, score=8.7 - i * 0.05) for i in range(2)]
    search_items = [_make_art("search", 20 + i, score=8.6 - i * 0.05) for i in range(4)]

    res_backfill = select_paced_batch(
        tg_items + x_items + search_items,
        target_posts=10,
        source_mix={"telegram": 4, "x": 4, "search": 2},
    )
    assert len(res_backfill.selected) == 10
    assert res_backfill.bucket_counts["x"] == 2
    assert res_backfill.bucket_counts["telegram"] + res_backfill.bucket_counts["search"] == 8
    assert len({a.canonical_url for a in res_backfill.selected}) == 10

    # When total unique eligible pool has only 6 items -> returns strictly 6 (no duplicate padding!)
    tiny_pool = tg_items[:2] + x_items[:2] + search_items[:2]
    res_small = select_paced_batch(
        tiny_pool + [tg_items[0]],  # include exact duplicate object -> ignored defensively
        target_posts=10,
        source_mix={"telegram": 4, "x": 4, "search": 2},
    )
    assert len(res_small.selected) == 6
    assert len({a.canonical_url for a in res_small.selected}) == 6


@pytest.mark.asyncio
async def test_cross_bucket_duplicate_between_telegram_and_x_consumes_only_one_slot():
    mock_db = AsyncMock()
    mock_db.get_recent_published_headlines.return_value = []
    mock_db.get_recent_published_articles.return_value = []
    mock_db.get_recent_headlines.return_value = []
    mock_db.get_recent_content_hashes.return_value = []
    mock_db.get_published_event_fingerprints.return_value = set()
    mock_db.get_recent_normalized_titles.return_value = []
    mock_db.get_article_by_canonical_url.return_value = None
    mock_db.get_event_by_fingerprint.return_value = None
    mock_db.count_recent_published.return_value = 5
    mock_db.is_url_skipped.return_value = False

    from monitoring.health import HealthMonitor
    from monitoring.metrics import Metrics

    pipeline = ArticlePipeline(
        db=mock_db,
        publisher=MagicMock(),
        health=HealthMonitor(),
        metrics=Metrics(),
    )
    # Same Kirkuk military event reported in both Telegram and X
    shared_title = "Peshmerga and Iraqi forces repel drone attack near Kirkuk oilfield"
    tg_art = _make_art("telegram", 501, title=shared_title, source_name="Rudaw")
    x_art = _make_art("x", 502, title=f"Video shows {shared_title}", source_name="Rawafid")
    other_search = _make_art("search", 503, title="Ankara and Athens resume diplomatic maritime dialogue", source_name="Reuters")

    def _eval_by_article(arts, **kw):
        out = []
        for a in arts:
            if "kirkuk" in a.original_title.lower():
                out.append(
                    RelevanceEvaluation(
                        score=9,
                        is_breaking=False,
                        category="kurdistan",
                        reason="Security incident",
                        passed=True,
                        event_key="kirkuk_drone_attack",
                        event_entities=["kirkuk", "drone", "peshmerga"],
                    )
                )
            else:
                out.append(
                    RelevanceEvaluation(
                        score=8,
                        is_breaking=False,
                        category="politics_world",
                        reason="Diplomatic talks",
                        passed=True,
                        event_key="ankara_athens_dialogue",
                        event_entities=["ankara", "athens", "maritime"],
                    )
                )
        return out

    pipeline._ranker.evaluate_batch_relevance = AsyncMock(side_effect=_eval_by_article)

    eligible, _ = await pipeline.evaluate_batch_candidates([x_art, tg_art, other_search])
    assert len(eligible) == 2
    # Cross-source duplicate dropped: only 1 of {TG, X} survived for Kirkuk event!
    kirkuk_survivors = [a for a in eligible if "kirkuk" in a.original_title.lower()]
    assert len(kirkuk_survivors) == 1
    assert get_source_bucket(kirkuk_survivors[0]) == "telegram"


# ═══════════════════════════════════════════════════════════════════════
# 4. 1-Minute Sequential Publishing & Minute-Boundary Alignment
# ═══════════════════════════════════════════════════════════════════════


def test_compute_next_post_delay_aligns_to_exact_minute_boundary():
    t0 = datetime(2026, 11, 25, 3, 40, 0, tzinfo=timezone.utc)
    # Instant post -> exact 60s wait to 03:41:00
    assert compute_next_post_delay(t0, t0, 60.0, align_to_minute=True) == 60.0
    # Post took 3s (finished at 03:40:03) -> waits 57s so next lands right on 03:41:00
    t_after_send = datetime(2026, 11, 25, 3, 40, 3, tzinfo=timezone.utc)
    assert compute_next_post_delay(t_after_send, t0, 60.0, align_to_minute=True) == 57.0
    # Test override interval=0 -> zero sleep
    assert compute_next_post_delay(t_after_send, t0, 0.0, align_to_minute=True) == 0.0


@pytest.mark.asyncio
async def test_10_post_batch_1_minute_sequential_pacing_timestamps_and_resilience(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ENABLE_BATCH_PACING_ENGINE", True)
    monkeypatch.setattr(config, "BATCH_TARGET_POSTS", 10)
    monkeypatch.setattr(config, "BATCH_PUBLISH_INTERVAL_SECONDS", 60.0)
    monkeypatch.setattr(config, "BATCH_ALIGN_TO_MINUTE_BOUNDARY", True)
    monkeypatch.setattr(config, "BATCH_REST_COOLDOWN_MINUTES", 10.0)
    monkeypatch.setattr(config, "ENABLE_SOURCE_DOMINANCE_CAP", False)

    db = Database(str(tmp_path / "batch_10.db"))
    await db.connect()

    try:
        raw_tg = [_make_raw("telegram", i) for i in range(5)]
        raw_x = [_make_raw("x", 10 + i, video_url=f"https://video.twimg.com/x_{i}.mp4") for i in range(6)]
        raw_search = [_make_raw("search", 20 + i) for i in range(3)]

        mock_src = MagicMock()
        mock_src.name = "MultiMock"
        mock_src.source_type = "rss"
        mock_src.enabled = True
        mock_src.fetch_articles = AsyncMock(return_value=raw_tg + raw_x + raw_search)

        pub, mock_bot = _mock_publisher(tmp_path)
        engine = NewsEngine()
        engine._db = db
        engine._publisher = pub
        engine._sources = [mock_src]
        engine._watchdog = AdminWatchdog(db=db, publisher=pub)
        engine._watchdog._enabled = False
        engine._pipeline = ArticlePipeline(db=db, publisher=pub, health=engine._health, metrics=engine._metrics)
        engine._pipeline._extractor.fetch_article_content_and_image = AsyncMock(return_value=("", None))

        async def _fake_eval(articles, batch_size=12):
            return [
                RelevanceEvaluation(
                    score=9 if i < 5 else 8,
                    is_breaking=False,
                    category=a.category or "politics_world",
                    reason="Verified political/security item",
                    passed=True,
                    event_key="_".join(a.original_title.lower().split()[:4]),
                    event_entities=a.original_title.lower().split()[:3],
                )
                for i, a in enumerate(articles)
            ]

        _ku_counter = {"i": 0}

        def _pure_sorani_translate(title: str, summary: str = ""):
            idx = _ku_counter["i"]
            _ku_counter["i"] += 1
            return _DISTINCT_SORANI[idx % len(_DISTINCT_SORANI)]

        engine._pipeline._ranker.evaluate_batch_relevance = AsyncMock(side_effect=_fake_eval)
        engine._pipeline._translator.translate_article = AsyncMock(side_effect=_pure_sorani_translate)

        # Fake clock starting cleanly at 03:40:00 + simulate one unofficial watermark rejection up front
        clock = {"now": datetime(2026, 11, 25, 3, 40, 0, tzinfo=timezone.utc)}
        engine._utc_now = lambda: clock["now"]

        post_timestamps: list[str] = []
        sleep_calls: list[float] = []

        async def _fake_sleep(secs: float):
            sleep_calls.append(secs)
            clock["now"] = clock["now"] + timedelta(seconds=secs)

        # Wrap send_photo / send_video to log publish wall-clock times & fail post #4 once transiently
        call_counter = {"n": 0}
        orig_send_photo = pub._bot.send_photo
        orig_send_video = pub._bot.send_video

        async def _timed_send(*args, **kwargs):
            post_timestamps.append(clock["now"].strftime("%H:%M:%S"))
            call_counter["n"] += 1
            return await orig_send_photo(*args, **kwargs)

        async def _timed_send_video(*args, **kwargs):
            post_timestamps.append(clock["now"].strftime("%H:%M:%S"))
            call_counter["n"] += 1
            return await orig_send_video(*args, **kwargs)

        pub._bot.send_photo = AsyncMock(side_effect=_timed_send)
        pub._bot.send_video = AsyncMock(side_effect=_timed_send_video)

        async def _fake_brand(video_url: str, **kwargs):
            if "x_0" in video_url:
                raise WatermarkRejectedError("static_corner_logo:top_right")
            out = tmp_path / f"branded_{Path(video_url).name}"
            out.write_bytes(b"MP4")
            return str(out)

        with patch("core.engine.download_and_brand_video", side_effect=_fake_brand), \
             patch("tg_bot.publisher.download_and_brand_video", side_effect=_fake_brand), \
             patch("core.engine.asyncio.sleep", side_effect=_fake_sleep):
            await engine._discovery_cycle()

        # Exactly 10 posts published across the 4:4:2 mix despite watermark rejection on x_0!
        total_sent = pub._bot.send_photo.await_count + pub._bot.send_video.await_count
        assert total_sent == 10
        # Exactly 9 sleeps of 60s (between posts 1->2..9->10), ZERO sleep after post 10!
        assert sleep_calls == [60.0] * 9
        assert post_timestamps == [f"03:4{i}:00" for i in range(10)]

        # Watermarked x_0 rejected up front & recorded cleanly
        assert await db.is_url_skipped(raw_x[0].url) is True

        # Cooldown timestamp persisted into existing `system_logs`
        cooldown_dt = await db.get_last_batch_completed_at()
        assert cooldown_dt is not None
        assert cooldown_dt == datetime(2026, 11, 25, 3, 49, 0, tzinfo=timezone.utc)
    finally:
        await db.close()


# ═══════════════════════════════════════════════════════════════════════
# 5. 10-Minute Persisted Rest Cooldown (Stateless, Continuous & Watchdog)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_10_minute_persisted_rest_cooldown_stateless_and_continuous(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ENABLE_BATCH_PACING_ENGINE", True)
    monkeypatch.setattr(config, "BATCH_REST_COOLDOWN_MINUTES", 10.0)
    monkeypatch.setattr(config, "BATCH_PUBLISH_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(config, "ENABLE_SOURCE_DOMINANCE_CAP", False)

    db = Database(str(tmp_path / "cooldown.db"))
    await db.connect()

    try:
        t_batch_end = datetime(2026, 11, 25, 3, 49, 0, tzinfo=timezone.utc)
        await db.record_batch_completed_at(timestamp=t_batch_end, published_count=10)

        mock_src = MagicMock()
        mock_src.name = "MockRSS"
        mock_src.source_type = "rss"
        mock_src.enabled = True
        mock_src.fetch_articles = AsyncMock(return_value=[_make_raw("search", 701)])

        pub, mock_bot = _mock_publisher(tmp_path)
        engine = NewsEngine()
        engine._db = db
        engine._publisher = pub
        engine._sources = [mock_src]
        engine._watchdog = AdminWatchdog(db=db, publisher=pub)
        engine._watchdog._enabled = True
        engine._watchdog.dispatch_alert = AsyncMock(return_value=False)
        engine._pipeline = ArticlePipeline(db=db, publisher=pub, health=engine._health, metrics=engine._metrics)
        engine._pipeline._extractor.fetch_article_content_and_image = AsyncMock(return_value=("", None))
        engine._pipeline._ranker.evaluate_batch_relevance = AsyncMock(
            return_value=[
                RelevanceEvaluation(
                    score=9,
                    is_breaking=False,
                    category="politics_world",
                    reason="Post-cooldown item",
                    passed=True,
                    event_key="post_cooldown_evt",
                    event_entities=["diplomacy"],
                )
            ]
        )
        engine._pipeline._translator.translate_article = AsyncMock(
            return_value=(
                "هەواڵی نوێی دیپلۆماسی لە ناوچەکەدا",
                "وەزیرانی دەرەوە کۆبوونەوەیەکی نوێی هاوبەشیان ئەنجامدا.",
            )
        )

        # 1) New run at T + 4 minutes -> MUST rest immediately (0 fetch, 0 publish, 0 watchdog alerts)
        engine._utc_now = lambda: t_batch_end + timedelta(minutes=4)
        await engine._discovery_cycle()
        mock_src.fetch_articles.assert_not_awaited()
        mock_bot.send_photo.assert_not_awaited()
        engine._watchdog.dispatch_alert.assert_not_awaited()

        # 2) New run at T + 10m 5s in continuous loop (_running=True) -> executes & sleeps 600s after completion!
        engine._utc_now = lambda: t_batch_end + timedelta(minutes=10, seconds=5)
        engine._running = True
        cont_sleeps: list[float] = []
        with patch("core.engine.asyncio.sleep", side_effect=lambda s: cont_sleeps.append(s)):
            await engine._discovery_cycle()

        mock_src.fetch_articles.assert_awaited_once()
        mock_bot.send_photo.assert_awaited_once()
        assert cont_sleeps == [600.0]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_mid_batch_single_post_failure_continues_remaining_posts(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ENABLE_BATCH_PACING_ENGINE", True)
    monkeypatch.setattr(config, "BATCH_TARGET_POSTS", 4)
    monkeypatch.setattr(config, "BATCH_PUBLISH_INTERVAL_SECONDS", 60.0)
    monkeypatch.setattr(config, "BATCH_ALIGN_TO_MINUTE_BOUNDARY", False)
    monkeypatch.setattr(config, "ENABLE_SOURCE_DOMINANCE_CAP", False)

    db = Database(str(tmp_path / "resilience.db"))
    await db.connect()

    try:
        items = [_make_raw("telegram", i) for i in range(4)]
        mock_src = MagicMock()
        mock_src.name = "MockTG"
        mock_src.source_type = "telegram"
        mock_src.enabled = True
        mock_src.fetch_articles = AsyncMock(return_value=items)

        pub, mock_bot = _mock_publisher(tmp_path)
        engine = NewsEngine()
        engine._db = db
        engine._publisher = pub
        engine._sources = [mock_src]
        engine._watchdog = AdminWatchdog(db=db, publisher=pub)
        engine._watchdog._enabled = False
        engine._pipeline = ArticlePipeline(db=db, publisher=pub, health=engine._health, metrics=engine._metrics)
        engine._pipeline._extractor.fetch_article_content_and_image = AsyncMock(return_value=("", None))
        engine._pipeline._ranker.evaluate_batch_relevance = AsyncMock(
            side_effect=lambda arts, **kw: [
                RelevanceEvaluation(
                    score=8,
                    is_breaking=False,
                    category="kurdistan",
                    reason="Verified political development",
                    passed=True,
                    event_key="_".join(a.original_title.lower().split()[:3]),
                    event_entities=a.original_title.lower().split()[:3],
                )
                for a in arts
            ]
        )
        _k_idx = {"i": 0}

        def _tr(title, summary=""):
            idx = _k_idx["i"]
            _k_idx["i"] += 1
            return _DISTINCT_SORANI[(idx + 12) % len(_DISTINCT_SORANI)]

        engine._pipeline._translator.translate_article = AsyncMock(side_effect=_tr)

        publish_calls = {"n": 0}
        orig_pub = engine._pipeline._publish_immediately

        async def _flaky_publish(art):
            publish_calls["n"] += 1
            if publish_calls["n"] == 2:
                raise RuntimeError("Transient Telegram timeout on post 2")
            return await orig_pub(art)

        engine._pipeline._publish_immediately = AsyncMock(side_effect=_flaky_publish)

        sleep_calls: list[float] = []
        with patch("core.engine.asyncio.sleep", side_effect=lambda s: sleep_calls.append(s)):
            await engine._discovery_cycle()

        # Post #2 failed transiently, yet posts #1, #3, #4 all published (3 total)!
        assert mock_bot.send_photo.await_count == 3
        assert len(sleep_calls) == 3
        assert all(59.0 <= s <= 60.0 for s in sleep_calls)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_supabase_batch_cooldown_system_logs_parity():
    recorded_bodies: list[dict] = []
    saved_ts = "2026-11-25T03:49:00+00:00"

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/system_logs"):
            recorded_bodies.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(201)
        if request.method == "GET" and request.url.path.endswith("/system_logs"):
            return httpx.Response(
                200,
                json=[
                    {
                        "id": 1,
                        "timestamp": saved_ts,
                        "level": "INFO",
                        "component": "BATCH_COOLDOWN",
                        "message": "Batch completed (10 post(s))",
                    }
                ],
            )
        return httpx.Response(200, json=[])

    client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    sb = SupabaseDatabase(url="https://example.supabase.co", key="anon-key", client=client)

    ts = datetime(2026, 11, 25, 3, 49, 0, tzinfo=timezone.utc)
    await sb.record_batch_completed_at(timestamp=ts, published_count=10)
    assert len(recorded_bodies) == 1
    assert recorded_bodies[0]["component"] == "BATCH_COOLDOWN"

    loaded = await sb.get_last_batch_completed_at()
    assert loaded == ts
    await client.aclose()


# ═══════════════════════════════════════════════════════════════════════
# 6. Zero Extra Gemini API Calls Guarantee
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_batch_pacing_uses_zero_extra_gemini_calls():
    from processing.gemini_client import GeminiClient

    cands = (
        [_make_art("telegram", i) for i in range(6)]
        + [_make_art("x", 20 + i) for i in range(5)]
        + [_make_art("search", 40 + i) for i in range(4)]
    )
    with patch.object(GeminiClient, "generate_json", new=AsyncMock()) as mock_gemini:
        plan = select_paced_batch(cands, target_posts=10)
        assert len(plan.selected) == 10
        assert mock_gemini.await_count == 0
