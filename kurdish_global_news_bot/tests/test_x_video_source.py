"""
Automated test suite for Free-Tier X (Twitter) Video Discovery, Tiered
Official-vs-Unofficial Watermark Policy, Strict Cross-Platform Deduplication,
Local + DuckDuckGo Context Enrichment, Smart KGN Corner Placement, and Zero
Extra Gemini API Calls.
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image, ImageDraw

import config
from core.importance_ranker import RelevanceEvaluation
from core.pipeline import ArticlePipeline
from core.source_verifier import (
    extract_x_handle_from_url,
    get_source_group,
    get_x_account_meta,
    is_official_media_source,
    is_x_source_url,
    apply_verification_guard,
    compute_corroboration,
)
from core.x_context_enricher import enrich_x_article_context, needs_context_enrichment
from monitoring.health import HealthMonitor
from monitoring.metrics import Metrics
from processing.content_formatter import X_EYEWITNESS_ATTRIBUTION_PREFIX, prepend_x_video_attribution
from processing.deduplicator import SemanticDeduplicator
from processing.video_processor import (
    WatermarkRejectedError,
    build_ffmpeg_watermark_command,
    download_and_brand_video,
)
from processing.video_watermark_detector import (
    choose_kgn_corner,
    has_image_corner_watermark,
    inspect_frames_for_watermarks,
)
from sources.x_video_source import XVideoSource, _is_fake_or_recycled_clip
from storage.database import Database
from storage.models import Article
from tg_bot.inline_buttons import build_source_button_markup
from tg_bot.publisher import TelegramPublisher


def _make_mock_db() -> AsyncMock:
    db = AsyncMock()
    db.get_recent_published_headlines.return_value = []
    db.get_recent_published_articles.return_value = []
    db.get_recent_headlines.return_value = []
    db.get_recent_content_hashes.return_value = []
    db.get_published_event_fingerprints.return_value = set()
    db.get_recent_normalized_titles.return_value = []
    db.get_article_by_canonical_url.return_value = None
    db.get_event_by_fingerprint.return_value = None
    db.count_recent_published.return_value = 5
    db.is_url_skipped.return_value = False
    return db


def _make_pipeline(mock_db: AsyncMock, mock_eval_side_effect_or_mock, mock_publisher=None) -> ArticlePipeline:
    pipeline = ArticlePipeline(
        db=mock_db,
        publisher=mock_publisher or MagicMock(),
        health=HealthMonitor(),
        metrics=Metrics(),
    )
    if callable(mock_eval_side_effect_or_mock) or hasattr(mock_eval_side_effect_or_mock, "await_count"):
        pipeline._ranker.evaluate_batch_relevance = mock_eval_side_effect_or_mock
    return pipeline


def _make_publisher(mock_bot=None) -> TelegramPublisher:
    pub = TelegramPublisher()
    pub._bot = mock_bot or AsyncMock()
    pub._channel_id = "@test_chan"
    return pub


def _make_moving_frames(
    corner_logo: str | None = None,
    width: int = 320,
    height: int = 180,
) -> list[Image.Image]:
    """Generate 3 synthetic RGB video frames with moving center content & optional corner logo."""
    frames: list[Image.Image] = []
    for idx in range(3):
        img = Image.new("RGB", (width, height), (28, 34, 46))
        draw = ImageDraw.Draw(img)
        x0 = 90 + idx * 35
        color = (220 - idx * 45, 70 + idx * 55, 45 + idx * 60)
        draw.rectangle([x0, 50, x0 + 60, 120], fill=color, outline=(255, 255, 255), width=2)

        if corner_logo:
            cw, ch = 56, 26
            if corner_logo == "top_right":
                box = (width - cw - 2, 2, width - 2, 2 + ch)
            elif corner_logo == "top_left":
                box = (2, 2, 2 + cw, 2 + ch)
            elif corner_logo == "bottom_right":
                box = (width - cw - 2, height - ch - 2, width - 2, height - 2)
            else:
                box = (2, height - ch - 2, 2 + cw, height - 2)
            draw.rectangle(box, fill=(245, 245, 245), outline=(10, 10, 10), width=2)
            bx0, by0, bx1, by1 = box
            for sx in range(bx0 + 4, bx1 - 4, 6):
                draw.line([(sx, by0 + 4), (sx, by1 - 4)], fill=(20, 20, 20), width=2)
        frames.append(img)
    return frames


def _make_static_image_bytes(with_corner_logo: str | None = None) -> bytes:
    """Create a PNG photo (smooth scene, optional high-contrast corner watermark)."""
    img = Image.new("RGB", (320, 180), (45, 65, 90))
    draw = ImageDraw.Draw(img)
    draw.rectangle([80, 45, 240, 135], fill=(60, 82, 110))
    if with_corner_logo == "top_right":
        box = (260, 2, 316, 28)
        draw.rectangle(box, fill=(250, 250, 250), outline=(5, 5, 5), width=2)
        for sx in range(264, 312, 5):
            draw.line([(sx, 4), (sx, 26)], fill=(15, 15, 15), width=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════
# Group 1: Source Registration & 16-Account Roster + Official Flags
# ═══════════════════════════════════════════════════════════════════════


def test_x_curated_16_accounts_roster_and_official_flags():
    accounts = config.X_VIDEO_SOURCE_ACCOUNTS
    assert len(accounts) == 16
    handles = {str(a["handle"]) for a in accounts}
    expected = {
        "CENTCOM",
        "Rudawkurdish",
        "RudawEnglish",
        "kurdistan24tv",
        "Kurdistan24_EN",
        "EnglishBasNews",
        "ShafaqNews",
        "Rawafiid",
        "AuroraIntel",
        "ClashReport",
        "Liveuamap",
        "OSINTtechnical",
        "Faytuks",
        "IntelCrab",
        "sentdefender",
        "ELINTNews",
    }
    assert handles == expected

    official_handles = {a["handle"] for a in accounts if a["is_official_media"]}
    assert official_handles == {
        "CENTCOM",
        "Rudawkurdish",
        "RudawEnglish",
        "kurdistan24tv",
        "Kurdistan24_EN",
        "EnglishBasNews",
        "ShafaqNews",
    }

    assert is_official_media_source("CENTCOM", "https://x.com/CENTCOM/status/12345") is True
    assert is_official_media_source("Rudaw", "https://x.com/RudawEnglish/status/12345") is True
    assert is_official_media_source("BasNews", "https://x.com/EnglishBasNews/status/12345") is True
    assert is_official_media_source("Rawafid", "https://x.com/Rawafiid/status/12345") is False
    assert is_official_media_source("Clash Report", "https://x.com/ClashReport/status/12345") is False
    assert is_official_media_source("UnknownOSINT", "https://x.com/random_user_99/status/12345") is False
    assert extract_x_handle_from_url("https://x.com/EnglishBasNews/status/99887766") == "englishbasnews"
    assert is_x_source_url("https://x.com/Rawafiid/status/112233") is True
    assert get_source_group("sentdefender") == "osintdefender"
    assert get_x_account_meta("EnglishBasNews")["source_group"] == "basnews"

    from processing.validator import ArticleValidator
    fresh_iso = (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat()
    validator = ArticleValidator(max_age_hours=3.0)
    ok_cfg, reason_cfg = validator.validate(
        Article(
            canonical_url="https://x.com/CENTCOM/status/123456",
            original_url="https://x.com/CENTCOM/status/123456",
            source_name="CENTCOM",
            original_title="CENTCOM forces conduct precision airstrikes",
            original_summary="Precision strikes targeted weapon storage sites in the region.",
            publication_date=fresh_iso,
        ),
        is_cold_start=False,
    )
    assert ok_cfg is True, reason_cfg
    recent_spam_iso = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    ok_spam, reason_spam = validator.validate(
        Article(
            canonical_url="https://x.com/random_spammer/status/999",
            original_url="https://x.com/random_spammer/status/999",
            source_name="random_spammer",
            original_title="Clickbait post on x domain",
            original_summary="Some unverified spam link that must remain blocked.",
            publication_date=recent_spam_iso,
        ),
        is_cold_start=False,
    )
    assert ok_spam is False and "Blocked domain: x.com" in reason_spam


@pytest.mark.asyncio
async def test_engine_registers_x_video_source_when_enabled(tmp_path):
    from core.engine import NewsEngine

    db = Database(db_path=str(tmp_path / "engine_x.db"))
    await db.connect()
    engine = NewsEngine()
    engine._db = db
    try:
        with patch.object(config, "ENABLE_X_VIDEO_SOURCE", True), \
             patch.object(config, "ENABLE_RSS", False), \
             patch.object(config, "ENABLE_DDG", False), \
             patch.object(config, "ENABLE_GDELT", False), \
             patch.object(config, "ENABLE_TELEGRAM", False):
            await engine._init_sources()
            names = [s.name for s in engine._sources]
            assert "x_video" in names
    finally:
        for s in getattr(engine, "_sources", []):
            await s.cleanup()
        await db.close()


# ═══════════════════════════════════════════════════════════════════════
# Group 2: Zero-Duplicate & Official-Over-Unofficial Priority
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_cross_language_rawafid_arabic_vs_basnews_english_deduplicates_official_winner():
    mock_db = _make_mock_db()
    dedup = SemanticDeduplicator(db=mock_db)
    fresh_iso = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()

    rawafid_ar = Article(
        id=1,
        canonical_url="https://x.com/Rawafiid/status/1001",
        original_url="https://x.com/Rawafiid/status/1001",
        source_name="Rawafid",
        original_title="تظاهرة حاشدة في القامشلي دعماً لإقليم كردستان",
        original_summary="شهدت مدينة القامشلي تظاهرة شعبية دعماً لإقليم كردستان وسط حضور واسع.",
        publication_date=fresh_iso,
        video_url="https://video.twimg.com/rawafid_clip.mp4",
    )
    basnews_en = Article(
        id=2,
        canonical_url="https://x.com/EnglishBasNews/status/1002",
        original_url="https://x.com/EnglishBasNews/status/1002",
        source_name="BasNews",
        original_title="Large protest in Qamishlo in support of Kurdistan",
        original_summary="Crowds gathered in Qamishlo for a protest in support of Kurdistan today.",
        publication_date=fresh_iso,
        video_url="https://video.twimg.com/basnews_clip.mp4",
    )

    ev_eval = RelevanceEvaluation(
        score=8,
        is_breaking=False,
        reason="Kurdistan protest rally in Qamishlo",
        passed=True,
        category="کوردستان",
        event_key="qamishlo_kurdistan_rally",
        event_entities=["Qamishlo", "Kurdistan"],
    )
    eval_mock = AsyncMock(side_effect=lambda arts, **kw: [ev_eval for _ in arts])

    pipeline = _make_pipeline(mock_db, eval_mock)
    pipeline._semantic_dedup = dedup

    eligible, counts = await pipeline.evaluate_batch_candidates([rawafid_ar, basnews_en])
    assert len(eligible) == 1
    assert eligible[0].source_name == "BasNews"
    assert eligible[0].original_url == "https://x.com/EnglishBasNews/status/1002"
    assert counts["rejected"] == 1


@pytest.mark.asyncio
async def test_x_video_vs_rss_same_event_picks_rss_official_and_blocks_late_duplicate():
    fresh_iso = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    reuters_rss = Article(
        id=10,
        canonical_url="https://www.reuters.com/world/middle-east/erbil-drone-intercept-2026",
        original_url="https://www.reuters.com/world/middle-east/erbil-drone-intercept-2026",
        source_name="Reuters World",
        original_title="Air defenses intercept armed drone near Erbil airport",
        original_summary="Air defense batteries intercepted an explosive drone over Erbil airbase.",
        publication_date=fresh_iso,
    )
    clash_x = Article(
        id=11,
        canonical_url="https://x.com/ClashReport/status/77777",
        original_url="https://x.com/ClashReport/status/77777",
        source_name="Clash Report",
        original_title="Video shows drone intercepted near Erbil airport tonight",
        original_summary="Air defense fire intercepted drone over Erbil airbase tonight.",
        publication_date=fresh_iso,
        video_url="https://video.twimg.com/clash_drone.mp4",
    )

    ev_eval = RelevanceEvaluation(
        score=9,
        is_breaking=True,
        reason="Security incident in Erbil",
        passed=True,
        category="جیهان",
        event_key="erbil_airport_drone_intercept",
        event_entities=["Erbil", "Drone"],
    )
    mock_db = _make_mock_db()
    eval_mock = AsyncMock(side_effect=lambda arts, **kw: [ev_eval for _ in arts])
    pipeline = _make_pipeline(mock_db, eval_mock)

    eligible, _ = await pipeline.evaluate_batch_candidates([clash_x, reuters_rss])
    assert len(eligible) == 1
    assert eligible[0].source_name == "Reuters World"

    # Late X duplicate arriving when Reuters story is already published in 24h DB -> blocked
    mock_db_late = _make_mock_db()
    mock_db_late.get_recent_published_headlines.return_value = [
        {
            "id": 10,
            "source_name": "Reuters World",
            "original_title": reuters_rss.original_title,
            "translated_title": "بەرگری ئاسمانی درۆنێکی لە نزیک فڕۆکەخانەی هەولێر خستەخوارەوە",
            "original_summary": reuters_rss.original_summary,
            "translated_summary": "",
            "event_key": "erbil_airport_drone_intercept",
            "event_entities": ["Erbil", "Drone"],
            "telegram_message_id": 500,
        }
    ]
    pipeline_late = _make_pipeline(mock_db_late, eval_mock)
    eligible_late, counts_late = await pipeline_late.evaluate_batch_candidates([clash_x])
    assert eligible_late == []
    assert counts_late["rejected"] == 1


# ═══════════════════════════════════════════════════════════════════════
# Group 3: Strict Watermark Policy (Unofficial Rejection vs Official Smart Corner)
# ═══════════════════════════════════════════════════════════════════════


def test_watermark_detector_rejects_unofficial_corner_logo_and_selects_smart_corner_for_official():
    frames_with_tr_logo = _make_moving_frames(corner_logo="top_right")
    report = inspect_frames_for_watermarks(frames_with_tr_logo)
    assert report.has_watermark is True
    assert "top_right" in report.occupied_corners

    assert choose_kgn_corner(report.occupied_corners) == "top_left"
    assert choose_kgn_corner(["top_right", "top_left"]) == "bottom_right"
    assert choose_kgn_corner([]) == "top_right"

    clean_frames = _make_moving_frames(corner_logo=None)
    clean_report = inspect_frames_for_watermarks(clean_frames)
    assert clean_report.has_watermark is False
    assert clean_report.occupied_corners == []

    dirty_png = _make_static_image_bytes(with_corner_logo="top_right")
    clean_png = _make_static_image_bytes(with_corner_logo=None)
    assert has_image_corner_watermark(dirty_png)[0] is True
    assert has_image_corner_watermark(clean_png)[0] is False

    cmd_tl = build_ffmpeg_watermark_command("in.mp4", "logo.png", "out.mp4", corner="top_left")
    assert any("x=W*0.035:y=W*0.035" in arg for arg in cmd_tl)


@pytest.mark.asyncio
async def test_unofficial_watermarked_video_raises_and_marks_skipped_without_fallback(tmp_path):
    dummy_in = tmp_path / "in.mp4"
    dummy_in.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"0" * 512)
    logo = tmp_path / "kgn_logo.png"
    logo.write_bytes(_make_static_image_bytes())

    report_dirty = SimpleNamespace(
        has_watermark=True,
        occupied_corners=["top_right"],
        has_third_party_repost_marker=False,
        reason="static_corner_logo:top_right",
    )

    with patch("processing.video_processor._download_video_to_temp", AsyncMock(return_value=dummy_in)), \
         patch("processing.video_processor.resolve_logo_path", return_value=logo), \
         patch("processing.video_processor.detect_video_watermarks", AsyncMock(return_value=report_dirty)):
        with pytest.raises(WatermarkRejectedError):
            await download_and_brand_video(
                "https://video.twimg.com/dirty.mp4",
                is_official_media=False,
                check_watermarks=True,
            )

    dummy_in2 = tmp_path / "in2.mp4"
    dummy_in2.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"0" * 512)
    dummy_out = tmp_path / "out_ok.mp4"
    dummy_out.write_bytes(b"branded")

    fake_proc = AsyncMock()
    fake_proc.returncode = 0
    fake_proc.communicate = AsyncMock(return_value=(b"", b""))

    captured_cmds: list[tuple] = []

    async def _fake_exec(*cmd, **kwargs):
        captured_cmds.append(cmd)
        Path(cmd[-1]).write_bytes(b"branded-mp4-data")
        return fake_proc

    # Official channel WITH own top_right bug -> ALLOWED and places KGN logo at top_left!
    with patch("processing.video_processor._download_video_to_temp", AsyncMock(return_value=dummy_in2)), \
         patch("processing.video_processor.resolve_logo_path", return_value=logo), \
         patch("processing.video_processor.detect_video_watermarks", AsyncMock(return_value=report_dirty)), \
         patch("processing.video_processor._create_temp_path", return_value=dummy_out), \
         patch("processing.video_processor.asyncio.create_subprocess_exec", side_effect=_fake_exec):
        branded = await download_and_brand_video(
            "https://video.twimg.com/rudaw_official.mp4",
            is_official_media=True,
            check_watermarks=True,
        )
        assert Path(branded) == dummy_out
        assert any("x=W*0.035:y=W*0.035" in str(arg) for arg in captured_cmds[0])


@pytest.mark.asyncio
async def test_publisher_and_pipeline_abort_unofficial_watermarked_video_and_strip_dirty_image():
    mock_bot = AsyncMock()
    pub = _make_publisher(mock_bot)
    unofficial_vid_art = Article(
        id=55,
        canonical_url="https://x.com/Rawafiid/status/5555",
        original_url="https://x.com/Rawafiid/status/5555",
        source_name="Rawafid",
        original_title="Video from Kirkuk",
        original_summary="Clip showing incident",
        translated_title="دیمەنی ڕووداوێک لە کەرکووک",
        translated_summary="دیمەنەکان ڕووداوێکی نوێ لە کەرکووک پیشان دەدەن و لێکۆڵینەوە بەردەوامە.",
        category="کوردستان",
        video_url="https://video.twimg.com/dirty_rawafid.mp4",
    )

    with patch(
        "tg_bot.publisher.download_and_brand_video",
        AsyncMock(side_effect=WatermarkRejectedError("static_corner_logo:top_right")),
    ):
        msg_id = await pub.publish_article(unofficial_vid_art)
        assert msg_id is None
        assert getattr(unofficial_vid_art, "_watermark_rejected_reason", None)
        mock_bot.send_photo.assert_not_called()
        mock_bot.send_message.assert_not_called()

    mock_db = _make_mock_db()
    pipeline = _make_pipeline(
        mock_db,
        AsyncMock(),
        mock_publisher=MagicMock(publish_article=AsyncMock(return_value=None)),
    )
    ok = await pipeline._publish_immediately(unofficial_vid_art)
    assert ok is False
    mock_db.mark_url_skipped.assert_awaited_once()
    assert mock_db.mark_url_skipped.call_args.kwargs["reason"] == "unofficial_third_party_watermark"

    # Unofficial watermarked image is stripped before card generation
    dirty_png = _make_static_image_bytes(with_corner_logo="top_right")
    img_art = Article(
        id=56,
        canonical_url="https://x.com/ClashReport/status/6666",
        original_url="https://x.com/ClashReport/status/6666",
        source_name="Clash Report",
        original_title="Explosion reported in region",
        translated_title="تەقینەوەیەک لە ناوچەکەدا تۆمار کرا",
        translated_summary="دەنگی تەقینەوەیەکی بەهێز لە ناوچەکەدا بیستراوە و زانیارییەکان بەردەستن.",
        category="جیهان",
        image_url="https://pbs.twimg.com/media/dirty.jpg",
    )
    pub2 = _make_publisher(AsyncMock())
    pub2._image_handler = AsyncMock()
    pub2._image_handler.download_image.return_value = dirty_png
    pub2._card_generator = MagicMock()
    pub2._card_generator.generate_card_file = AsyncMock(side_effect=RuntimeError("inspect_args_stop"))

    await pub2._publish_with_card(img_art)
    pub2._card_generator.generate_card_file.assert_awaited_once()
    assert pub2._card_generator.generate_card_file.call_args.kwargs["image_bytes"] is None
    assert img_art.image_url == ""


# ═══════════════════════════════════════════════════════════════════════
# Group 4: Free-Tier X Video Discovery, Failover & Quality Filters
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_x_video_source_mirror_failover_fx_vx_fallback_and_filters():
    now = datetime.now(timezone.utc)
    fresh_pub = (now - timedelta(minutes=20)).strftime("%a, %d %b %Y %H:%M:%S +0000")
    stale_pub = (now - timedelta(hours=10)).strftime("%a, %d %b %Y %H:%M:%S +0000")

    rss_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel><title>CENTCOM</title>
      <item>
        <title>CENTCOM precision strikes hit militant launchers</title>
        <link>https://xcancel.com/CENTCOM/status/9001</link>
        <pubDate>{fresh_pub}</pubDate>
        <description>CENTCOM forces conducted precision strikes on drone launchers.</description>
      </item>
      <item>
        <title>RT by @otheruser: Old repost</title>
        <link>https://xcancel.com/CENTCOM/status/9002</link>
        <pubDate>{fresh_pub}</pubDate>
        <description>RT by @otheruser: ignore</description>
      </item>
      <item>
        <title>Arma 3 simulation footage of strike</title>
        <link>https://xcancel.com/CENTCOM/status/9003</link>
        <pubDate>{fresh_pub}</pubDate>
        <description>Gameplay clip from Arma 3</description>
      </item>
      <item>
        <title>Stale clip from 10h ago</title>
        <link>https://xcancel.com/CENTCOM/status/9004</link>
        <pubDate>{stale_pub}</pubDate>
        <description>Old security clip</description>
      </item>
      <item>
        <title>GIF meme reaction</title>
        <link>https://xcancel.com/CENTCOM/status/9005</link>
        <pubDate>{fresh_pub}</pubDate>
        <description>Reaction animation</description>
      </item>
    </channel></rss>"""

    src = XVideoSource(
        accounts=[
            {
                "handle": "CENTCOM",
                "display_name": "CENTCOM",
                "region": "middle_east",
                "lang": "en",
                "credibility_tier": 1,
                "source_group": "centcom",
                "is_official_media": True,
            }
        ],
        mirrors=["https://mirror1.invalid", "https://mirror2.valid"],
        max_age_hours=6.0,
        max_videos_per_cycle=4,
    )

    async def _fake_get_text(url: str):
        if "mirror1.invalid" in url:
            return None
        return rss_xml

    async def _fake_get_json(url: str):
        if "fxtwitter" in url and "/9001" in url:
            return None
        if "vxtwitter" in url and "/9001" in url:
            return {
                "text": "CENTCOM precision strikes hit drone launchers.",
                "media_extended": [
                    {
                        "type": "video",
                        "url": "https://video.twimg.com/centcom_9001.mp4",
                        "thumbnail_url": "https://pbs.twimg.com/thumb_9001.jpg",
                        "duration_millis": 18500,
                        "size": {"width": 1280, "height": 720},
                    }
                ],
            }
        if "fxtwitter" in url and "/9005" in url:
            return {
                "tweet": {
                    "text": "GIF reaction",
                    "media": {"all": [{"type": "gif", "url": "https://video.twimg.com/g.mp4"}]},
                }
            }
        return None

    with patch.object(src, "_http_get_text", side_effect=_fake_get_text), \
         patch.object(src, "_http_get_json", side_effect=_fake_get_json):
        articles = await src.fetch_articles()

    assert len(articles) == 1
    art = articles[0]
    assert art.url == "https://x.com/CENTCOM/status/9001"
    assert art.video_url == "https://video.twimg.com/centcom_9001.mp4"
    assert art.source_name == "CENTCOM"
    assert art.region == ""
    assert _is_fake_or_recycled_clip("DCS World gameplay footage") is True


@pytest.mark.asyncio
async def test_x_video_source_ddg_fallback_when_all_rss_mirrors_fail_and_caps_4_videos():
    accounts = [
        {"handle": "CENTCOM", "display_name": "CENTCOM", "is_official_media": True},
        {"handle": "RudawEnglish", "display_name": "Rudaw", "is_official_media": True},
        {"handle": "EnglishBasNews", "display_name": "BasNews", "is_official_media": True},
        {"handle": "ShafaqNews", "display_name": "Shafaq News", "is_official_media": True},
        {"handle": "ClashReport", "display_name": "Clash Report", "is_official_media": False},
    ]
    src = XVideoSource(accounts=accounts, mirrors=["https://m1.invalid"], max_videos_per_cycle=4)

    ddg_hits = [
        {
            "handle": a["handle"],
            "status_id": str(8000 + i),
            "canonical_url": f"https://x.com/{a['handle']}/status/{8000 + i}",
            "caption": f"Breaking military briefing clip {i}",
            "published_at": datetime.now(timezone.utc),
            "account": a,
        }
        for i, a in enumerate(accounts)
    ]

    async def _ok_meta(handle: str, status_id: str):
        return {
            "video_url": f"https://video.twimg.com/{handle}_{status_id}.mp4",
            "thumbnail_url": "",
            "duration": 22.0,
            "width": 720,
            "text": f"Security footage from {handle}",
            "created_at": datetime.now(timezone.utc).timestamp(),
        }

    with patch.object(src, "_http_get_text", AsyncMock(return_value=None)), \
         patch.object(src, "_ddg_fallback_discover", AsyncMock(return_value=ddg_hits)), \
         patch.object(src, "_resolve_video_metadata", side_effect=_ok_meta):
        articles = await src.fetch_articles()

    assert len(articles) == 4


# ═══════════════════════════════════════════════════════════════════════
# Group 5: Context Enrichment, Eyewitness Attribution & Button Suppression
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_context_enricher_prefers_local_batch_then_ddg_and_caps_3():
    x_art = Article(
        canonical_url="https://x.com/Rawafiid/status/2222",
        original_url="https://x.com/Rawafiid/status/2222",
        source_name="Rawafid",
        original_title="Watch explosions near Kirkuk base",
        original_summary="Watch explosions near Kirkuk base",
    )
    local_reuters = Article(
        canonical_url="https://www.reuters.com/world/kirkuk-base-strike",
        original_url="https://www.reuters.com/world/kirkuk-base-strike",
        source_name="Reuters World",
        original_title="Explosions reported near Kirkuk military airbase",
        original_summary="Security sources confirmed blasts at a military base near Kirkuk.",
    )
    ddg_spy = AsyncMock(return_value=[])
    assert needs_context_enrichment(x_art) is True

    corrobs = await enrich_x_article_context(
        x_art,
        batch_articles=[x_art, local_reuters],
        ddg_search_fn=ddg_spy,
    )
    ddg_spy.assert_not_awaited()
    assert len(corrobs) == 1
    assert "[Corroborating Context]" in x_art.original_summary
    assert "Reuters World" in x_art.original_summary

    x_art2 = Article(
        canonical_url="https://x.com/ClashReport/status/3333",
        original_url="https://x.com/ClashReport/status/3333",
        source_name="Clash Report",
        original_title="Missile strike targets Aleppo depot",
        original_summary="Missile strike targets Aleppo depot",
    )
    ddg_hits = [
        {"title": "Duplicate on X", "url": "https://x.com/someone/status/1", "source": "x.com"},
        {"title": "Missile strike hits weapons depot in Aleppo", "url": "https://www.reuters.com/a1", "source": "Reuters"},
        {"title": "Aleppo depot targeted by missile strike", "url": "https://www.bbc.com/a2", "source": "BBC"},
        {"title": "Airstrikes and missile hit depot in Aleppo", "url": "https://apnews.com/a3", "source": "AP"},
        {"title": "Extra missile strike depot Aleppo report", "url": "https://www.aljazeera.com/a4", "source": "Al Jazeera"},
    ]
    corrobs2 = await enrich_x_article_context(
        x_art2,
        batch_articles=[],
        ddg_search_fn=AsyncMock(return_value=ddg_hits),
    )
    assert len(corrobs2) == 3
    assert all("x.com" not in c["source_name"].lower() for c in corrobs2)
    assert "[Corroborating Context]" in x_art2.original_summary


def test_x_attribution_and_inline_button_suppression():
    x_eyewitness = Article(
        canonical_url="https://x.com/unknown_user/status/4444",
        original_url="https://x.com/unknown_user/status/4444",
        source_name="unknown_user",
        original_title="Explosion heard",
        is_breaking=True,
        priority="CRITICAL",
        event_key="erbil_blast",
    )
    corrob = compute_corroboration(x_eyewitness, batch_articles=[x_eyewitness], recent_history=[])
    apply_verification_guard(x_eyewitness, corrob)
    assert x_eyewitness.is_breaking is False
    assert getattr(x_eyewitness, "_is_uncorroborated_x_video", False) is True
    tagged = prepend_x_video_attribution("دەنگی تەقینەوەیەک بیستراوە.")
    assert tagged.startswith(X_EYEWITNESS_ATTRIBUTION_PREFIX)

    assert build_source_button_markup(article=x_eyewitness) is None
    with patch.object(config, "ENABLE_X_SOURCE_LINK_BUTTON", True):
        markup = build_source_button_markup(article=x_eyewitness)
        assert markup is not None
        assert markup.inline_keyboard[0][0].url == "https://x.com/unknown_user/status/4444"


# ═══════════════════════════════════════════════════════════════════════
# Group 6: Zero Extra Gemini API Calls Verification
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_x_pipeline_uses_zero_extra_gemini_calls():
    fresh_iso = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    x_centcom = Article(
        id=101,
        canonical_url="https://x.com/CENTCOM/status/99999",
        original_url="https://x.com/CENTCOM/status/99999",
        source_name="CENTCOM",
        original_title="CENTCOM forces destroy ballistic launcher in region",
        original_summary="Official military video shows precision strike on ballistic missile launcher.",
        publication_date=fresh_iso,
        video_url="https://video.twimg.com/centcom_clip.mp4",
    )
    local_bbc = Article(
        id=102,
        canonical_url="https://www.bbc.com/news/centcom-launcher",
        original_url="https://www.bbc.com/news/centcom-launcher",
        source_name="BBC World",
        original_title="US military says CENTCOM forces destroy ballistic launcher in region",
        original_summary="Confirmed military strike targeted ballistic launcher overnight.",
        publication_date=fresh_iso,
    )

    ev_eval = RelevanceEvaluation(
        score=9,
        is_breaking=True,
        reason="Military strike",
        passed=True,
        category="جیهان",
        event_key="centcom_launcher_strike",
        event_entities=["CENTCOM"],
    )
    mock_db = _make_mock_db()
    eval_mock = AsyncMock(side_effect=lambda arts, **kw: [ev_eval for _ in arts])
    pipeline = _make_pipeline(mock_db, eval_mock)

    await pipeline.evaluate_batch_candidates([x_centcom, local_bbc])
    assert eval_mock.await_count == 1
