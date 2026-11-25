"""
Comprehensive unit and integration tests for:
1. Public-web Telegram news discovery (`sources/telegram_source.py`) scraping
   `https://t.me/s/{username}` for `SabrenNewss` (`Sabereen News`) and
   `Sabren_News1` (`Sabereen Plus`) without Telegram API credentials.
2. HTML parsing, canonical URL extraction, UTC timestamps, entity decoding,
   CSS `background-image` photo URL extraction, text cleaning, and junk rejection
   (empty, short <20 chars, malformed, stale, ads/promos, traffic/radar, municipal).
3. Per-channel failure isolation and restricted Telegram preview handling.
4. Reuters/BBC-style Neutral Journalistic Style & deterministic Kurdish/Arabic/English
   terminology normalization (`normalize_neutral_terminology`, prompts, translator,
   content formatter, digest generator, and pipeline).
5. End-to-end integration with `NewsEngine`, `ArticlePipeline`, `SemanticDeduplicator`,
   5-of-8 Bashur quota preservation (`is_bashur_article`, `QuotaBalancer`),
   strict `score >= 9.0` breaking gate, and image card pipeline.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from bs4 import BeautifulSoup
from PIL import Image

from config import (
    ArticleStatus,
    BASHUR_QUOTA_PER_8,
    BATCH_PUBLISH_LIMIT,
    ENABLE_TELEGRAM,
    SOURCE_CREDIBILITY,
    TELEGRAM_SOURCE_CHANNELS,
)
from core.balancer import QuotaBalancer
from core.engine import NewsEngine
from core.importance_ranker import (
    RelevanceEvaluation,
    check_blocked_topic,
    is_bashur_article,
    is_strict_breaking,
)
from core.pipeline import ArticlePipeline
from monitoring.health import HealthMonitor
from monitoring.metrics import Metrics
from processing.content_formatter import (
    ContentFormatter,
    normalize_neutral_terminology,
)
from processing.deduplicator import SemanticDeduplicator
from processing.digest_generator import (
    GEMINI_DIGEST_SYSTEM_PROMPT,
    DailyDigestGenerator,
)
from processing.extractor import ArticleExtractor
from processing.translator import (
    GEMINI_KURDISH_REWRITE_SYSTEM_PROMPT,
    GEMINI_TRANSLATION_SYSTEM_PROMPT,
    GeminiTranslationProvider,
    TranslationEngine,
)
from sources.base import BaseSource, RawArticle
from sources.telegram_source import (
    TelegramSource,
    clean_telegram_message_text,
    extract_telegram_photo_url,
    extract_telegram_video_thumb_url,
    extract_telegram_video_url,
    is_rejected_telegram_content,
    parse_telegram_timestamp,
)
from storage.database import Database
from storage.models import Article
from tg_bot.publisher import TelegramPublisher


class _MockResponseContext:
    """Async context manager wrapping a mocked aiohttp response or exception."""

    def __init__(self, response=None, error: BaseException | None = None) -> None:
        self._response = response
        self._error = error

    async def __aenter__(self):
        if self._error is not None:
            raise self._error
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


def _make_html_response(status: int, html: str) -> AsyncMock:
    resp = AsyncMock()
    resp.status = status
    resp.text = AsyncMock(return_value=html)
    return resp


def _make_gemini_json_session(payload: dict) -> MagicMock:
    resp = AsyncMock()
    resp.status = 200
    resp.json = AsyncMock(
        return_value={"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]}
    )
    resp.text = AsyncMock(return_value=json.dumps(payload))
    session = MagicMock()
    session.post = MagicMock(return_value=_MockResponseContext(resp))
    return session


def _make_jpeg_bytes(color: tuple[int, int, int] = (25, 65, 110)) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (640, 360), color=color)
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture
async def temp_db(tmp_path):
    db = Database(str(tmp_path / "test_telegram_source.db"))
    await db.connect()
    yield db
    await db.close()


def _iso_hours_ago(hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _build_channel_html(username: str, blocks: list[str]) -> str:
    joined = "\n".join(blocks)
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>{username}</title></head>
<body>
<main class="tgme_main">
  <section class="tgme_channel_history js-message_history">
    {joined}
  </section>
</main>
</body>
</html>"""


def _build_msg_block(
    post_ref: str,
    text_html: str,
    dt_str: str | None = None,
    photo_style: str | None = None,
) -> str:
    photo_div = (
        f'<a class="tgme_widget_message_photo_wrap" style="{photo_style}" href="https://t.me/{post_ref}">'
        '<i class="tgme_widget_message_photo"></i></a>'
        if photo_style is not None
        else ""
    )
    time_html = f'<time datetime="{dt_str}" class="time">10:00</time>' if dt_str is not None else ""
    text_div = f'<div class="tgme_widget_message_text js-message_text">{text_html}</div>' if text_html else ""
    return f"""
<div class="tgme_widget_message_wrap js-widget_message_wrap">
  <div class="tgme_widget_message js-widget_message" data-post="{post_ref}">
    <div class="tgme_widget_message_bubble">
      {photo_div}
      {text_div}
      <div class="tgme_widget_message_footer">
        <a class="tgme_widget_message_date" href="https://t.me/{post_ref}">{time_html}</a>
      </div>
    </div>
  </div>
</div>"""


class TestTelegramConfig:
    """Verify centralized Telegram channel config and dynamic channel extensibility."""

    def test_telegram_config_defaults(self):
        assert isinstance(ENABLE_TELEGRAM, bool)
        assert ENABLE_TELEGRAM is True

        by_user = {ch["username"]: ch for ch in TELEGRAM_SOURCE_CHANNELS}
        assert "SabrenNewss" in by_user
        assert "Sabren_News1" in by_user

        assert by_user["SabrenNewss"]["name"] == "Sabereen News"
        assert by_user["SabrenNewss"]["region_hint"] == "iraq_region"
        assert by_user["SabrenNewss"]["lang"] == "ar"

        assert by_user["Sabren_News1"]["name"] == "Sabereen Plus"
        assert by_user["Sabren_News1"]["region_hint"] == "iraq_region"
        assert by_user["Sabren_News1"]["lang"] == "ar"

        assert "sabereen news" in SOURCE_CREDIBILITY
        assert "sabereen plus" in SOURCE_CREDIBILITY

        # Validator allows configured canonical channel post URLs while still blocking spam/invite t.me links
        from processing.validator import ArticleValidator
        validator = ArticleValidator()
        ok_sab, _ = validator._validate_url("https://t.me/SabrenNewss/12345")
        ok_plus, _ = validator._validate_url("https://t.me/Sabren_News1/987")
        bad_spam, reason_spam = validator._validate_url("https://t.me/unknown_spam_channel/99")
        bad_invite, _ = validator._validate_url("https://t.me/+secretInviteCode")
        assert ok_sab is True
        assert ok_plus is True
        assert bad_spam is False and "Blocked domain" in reason_spam
        assert bad_invite is False

    def test_telegram_source_accepts_additional_config_channels(self):
        custom_channels = [
            {"username": "SabrenNewss", "name": "Sabereen News", "region_hint": "iraq_region", "lang": "ar"},
            {"username": "ExtraDispatch", "name": "Extra Dispatch", "region_hint": "world", "lang": "en"},
        ]
        src = TelegramSource(channels=custom_channels)
        assert src.name == "TelegramPublicWeb"
        assert src.source_type == "telegram_web"
        assert src.enabled is True
        assert [c["username"] for c in src._channels] == ["SabrenNewss", "ExtraDispatch"]

        fresh_iso = _iso_hours_ago(1.0)
        html = _build_channel_html(
            "ExtraDispatch",
            [
                _build_msg_block(
                    "ExtraDispatch/77",
                    (
                        "European defense ministers convene an emergency security session in Brussels "
                        "following overnight missile strikes on regional energy shipping corridors. "
                        "Delegations approved joint air defense reinforcements for allied bases."
                    ),
                    dt_str=fresh_iso,
                )
            ],
        )
        parsed = src.parse_channel_html(html, custom_channels[1])
        assert len(parsed) == 1
        assert parsed[0].url == "https://t.me/ExtraDispatch/77"
        assert parsed[0].source_name == "Extra Dispatch"
        assert parsed[0].region == "world"


class TestTelegramHtmlParsing:
    """Verify public t.me/s/{username} HTML parsing, URLs, timestamps, entities, and photos."""

    def test_parse_channel_html_extracts_canonical_urls_timestamps_entities_and_photos(self):
        fresh_utc = _iso_hours_ago(0.5)
        three_hours_ago_plus3 = (
            datetime.now(timezone(timedelta(hours=3))) - timedelta(hours=1.5)
        ).strftime("%Y-%m-%dT%H:%M:%S+03:00")

        html = _build_channel_html(
            "SabrenNewss",
            [
                _build_msg_block(
                    "SabrenNewss/12345",
                    (
                        "🚨🔴 وزارة الدفاع العراقية تعلن في بيان رسمي &quot;تنفيذ ضربات جوية&quot; ضد مواقع "
                        "مسلحة في منطقة الحدود &amp; تعزيز الانتشار العسكري.<br/>"
                        "وأكد المتحدث العسكري أن العملية استمرت ٤٥ دقيقة بمشاركة طائرات مسيرة واستطلاع جوي "
                        "دقيق لمتابعة التحركات الميدانية ومنع أي خروقات أمنية جديدة في المحافظة.<br/>"
                        "تقرير ميداني @SabrenNewss https://t.me/SabrenNewss/12345 #عاجل #صابرين_نيوز"
                    ),
                    dt_str=fresh_utc,
                    photo_style="background-image:url('https://cdn4.telegram-cdn.org/file/photo_single.jpg')",
                ),
                _build_msg_block(
                    "SabrenNewss/12346",
                    (
                        "رئيس الوزراء يعقد اجتماعاً أمنياً طارئاً في بغداد لبحث التطورات الإقليمية "
                        "وملف الحدود المشتركة مع الدول المجاورة.<br/>"
                        "وشدد المجتمعون على أهمية التنسيق الاستخباري وحماية البعثات الدبلوماسية والمطارات "
                        "المدنية في ظل التصعيد العسكري المتواصل في المنطقة."
                    ),
                    dt_str=three_hours_ago_plus3,
                    photo_style="background-image:url(https://cdn4.telegram-cdn.org/file/photo_unquoted.jpg)",
                ),
                _build_msg_block(
                    "SabrenNewss/12347",
                    (
                        "مجلس النواب يناقش مشروع قانون الموازنة الاتحادية والتعديلات المالية الجديدة "
                        "بحضور رؤساء الكتل البرلمانية ولجنة المالية.<br/>"
                        "ومن المقرر استكمال القراءة الثانية للمشروع خلال جلسة الأسبوع المقبل بعد تدقيق "
                        "التخصيصات الاستثمارية للمحافظات والمؤسسات الحكومية."
                    ),
                    dt_str=fresh_utc,
                    photo_style=None,
                ),
            ],
        )

        src = TelegramSource()
        cfg = {"username": "SabrenNewss", "name": "Sabereen News", "region_hint": "iraq_region", "lang": "ar"}
        articles = src.parse_channel_html(html, cfg)

        assert len(articles) == 3

        first = articles[0]
        assert first.url == "https://t.me/SabrenNewss/12345"
        assert first.source_name == "Sabereen News"
        assert first.region == "iraq_region"
        assert first.country == "iraq_region"
        # Static watermarked Telegram photos are intentionally ignored (card_generator uses clean theme background)
        assert first.image_url is None
        assert '"تنفيذ ضربات جوية"' in first.summary
        assert "الحدود & تعزيز" in first.summary
        assert "٤٥" in first.summary
        assert "@SabrenNewss" not in first.summary
        assert "https://t.me/" not in first.summary
        assert "#عاجل" not in first.summary
        assert "🚨" not in first.title
        assert first.publication_date.endswith("+00:00")
        assert len(first.summary) >= 160

        second = articles[1]
        assert second.url == "https://t.me/SabrenNewss/12346"
        assert second.image_url is None
        assert second.publication_date.endswith("+00:00")

        third = articles[2]
        assert third.url == "https://t.me/SabrenNewss/12347"
        assert third.image_url is None

    def test_extract_telegram_photo_url_all_quote_and_entity_variants(self):
        cases = [
            ("background-image:url(https://cdn1.telesco.pe/file/a.jpg)", "https://cdn1.telesco.pe/file/a.jpg"),
            ("background-image: url('https://cdn1.telesco.pe/file/b.jpg')", "https://cdn1.telesco.pe/file/b.jpg"),
            ('background-image: url("https://cdn1.telesco.pe/file/c.jpg")', "https://cdn1.telesco.pe/file/c.jpg"),
            ("background-image:url(&quot;https://cdn1.telesco.pe/file/d.jpg&quot;)", "https://cdn1.telesco.pe/file/d.jpg"),
            ("background-image:url(&#39;https://cdn1.telesco.pe/file/e.jpg&#39;)", "https://cdn1.telesco.pe/file/e.jpg"),
            ("background-image:url(data:image/png;base64,AAA)", ""),
        ]
        for style_attr, expected in cases:
            escaped_attr = style_attr.replace('"', "&quot;")
            soup = BeautifulSoup(
                f'<div class="tgme_widget_message"><a class="tgme_widget_message_photo_wrap" style="{escaped_attr}"></a></div>',
                "lxml",
            )
            tag = soup.select_one(".tgme_widget_message")
            assert extract_telegram_photo_url(tag) == expected

        soup_none = BeautifulSoup('<div class="tgme_widget_message"></div>', "lxml")
        assert extract_telegram_photo_url(soup_none.select_one(".tgme_widget_message")) == ""

    def test_parse_telegram_timestamp_variants(self):
        dt_z = parse_telegram_timestamp("2026-11-25T10:15:00Z")
        assert dt_z is not None and dt_z.tzinfo == timezone.utc and dt_z.hour == 10

        dt_plus3 = parse_telegram_timestamp("2026-11-25T13:15:00+03:00")
        assert dt_plus3 is not None and dt_plus3.tzinfo == timezone.utc and dt_plus3.hour == 10

        dt_naive = parse_telegram_timestamp("2026-11-25T10:15:00")
        assert dt_naive is not None and dt_naive.tzinfo == timezone.utc

        assert parse_telegram_timestamp("") is None
        assert parse_telegram_timestamp("not-a-valid-date") is None


class TestTelegramFilteringAndCleaning:
    """Verify junk rejection, text cleaning, and ranker topic blocking."""

    def test_rejects_empty_short_malformed_stale_ads_traffic_and_municipal(self):
        fresh_utc = _iso_hours_ago(1.0)
        stale_utc = _iso_hours_ago(48.0)

        blocks = [
            # 1. Empty captionless photo post
            _build_msg_block(
                "SabrenNewss/1001",
                "",
                dt_str=fresh_utc,
                photo_style="background-image:url('https://cdn4.telegram-cdn.org/file/capless.jpg')",
            ),
            # 2. Very short caption (<20 chars after stripping noise)
            _build_msg_block(
                "SabrenNewss/1002",
                "🚨 عاجل الآن 🔴 @SabrenNewss #عاجل https://t.me/SabrenNewss",
                dt_str=fresh_utc,
            ),
            # 3. Malformed non-numeric data-post
            _build_msg_block(
                "SabrenNewss/pinned_notice",
                "بيان رسمي طويل من الحكومة الاتحادية حول السياسة الخارجية والأمن القومي والاتفاقيات الدفاعية.",
                dt_str=fresh_utc,
            ),
            # 4. Missing timestamp
            _build_msg_block(
                "SabrenNewss/1004",
                "وزير الخارجية يجري مباحثات دبلوماسية موسعة مع نظرائه حول استقرار الملاحة الدولية والأمن الإقليمي.",
                dt_str=None,
            ),
            # 5. Malformed timestamp
            _build_msg_block(
                "SabrenNewss/1005",
                "مجلس الأمن الدولي يعقد جلسة طارئة لبحث الأوضاع العسكرية والإنسانية المتصاعدة في المنطقة.",
                dt_str="invalid-timestamp",
            ),
            # 6. Stale post older than max_age_hours
            _build_msg_block(
                "SabrenNewss/1006",
                "خبر قديم يعود إلى قبل يومين حول اجتماع وزاري روتيني في العاصمة لمناقشة الميزانية.",
                dt_str=stale_utc,
            ),
            # 7. Commercial ad / channel promo
            _build_msg_block(
                "SabrenNewss/1007",
                "تخفيضات كبرى بمناسبة الافتتاح الجديد! للاعلان والتواصل اشتركوا في قناتنا عبر الرابط التالي للحصول على الخصومات.",
                dt_str=fresh_utc,
            ),
            # 8. Traffic / radar post
            _build_msg_block(
                "SabrenNewss/1008",
                "تنويه مروري: مديرية المرور تعلن نشر رادارات وكاميرات المراقبة لمتابعة السرعة على الطريق السريع وسط ازدحام مروري.",
                dt_str=fresh_utc,
            ),
            # 9. Municipal utility outage / schedule post
            _build_msg_block(
                "SabrenNewss/1009",
                "تنويه هام للمواطنين: جدول قطع الكهرباء المبرمج وصيانة شبكة المياه وانقطاع التيار الكهربائي في الأحياء الجنوبية غداً.",
                dt_str=fresh_utc,
            ),
            # 10. Valid political/military news post -> MUST PASS
            _build_msg_block(
                "SabrenNewss/1010",
                (
                    "البرلمان العراقي يصوت خلال جلسة رسمية بحضور الكتل السياسية على قرار يخص الاتفاقية الأمنية "
                    "وحماية السيادة الوطنية والحدود المشتركة.<br/>"
                    "وأكدت رئاسة المجلس أن القرار يلزم الحكومة بتعزيز الدفاعات الجوية ومتابعة الانتهاكات العسكرية "
                    "عبر القنوات الدبلوماسية الدولية."
                ),
                dt_str=fresh_utc,
            ),
        ]

        html = _build_channel_html("SabrenNewss", blocks)
        src = TelegramSource(max_age_hours=24)
        cfg = {"username": "SabrenNewss", "name": "Sabereen News", "region_hint": "iraq_region", "lang": "ar"}
        articles = src.parse_channel_html(html, cfg)

        assert len(articles) == 1
        assert articles[0].url == "https://t.me/SabrenNewss/1010"

    def test_clean_telegram_message_text_preserves_facts_and_removes_noise(self):
        raw = (
            "⚡️🔴 عاجل | وزارة الطاقة تعلن توقيع عقد بقيمة 450 مليون دولار في 2026-11-25 "
            "لتطوير الحقول الغازية.\n"
            "للمزيد تابعوا @Sabren_News1 و https://t.me/Sabren_News1/999\n"
            "📌 اشترك في قناتنا الآن https://t.me/promo\n"
            "#بغداد #اقتصاد #اخبار_العراق"
        )
        cleaned = clean_telegram_message_text(raw)
        assert "450" in cleaned
        assert "2026-11-25" in cleaned
        assert "وزارة الطاقة" in cleaned
        assert "@Sabren_News1" not in cleaned
        assert "t.me/" not in cleaned
        assert "اشترك في قناتنا" not in cleaned
        assert "#بغداد" not in cleaned
        assert "⚡️" not in cleaned
        assert "🔴" not in cleaned
        rejected, _ = is_rejected_telegram_content(cleaned)
        assert rejected is False

    def test_importance_ranker_blocks_traffic_municipal_and_ads_but_keeps_war_politics(self):
        def _art(text: str) -> Article:
            return Article(
                original_url="https://t.me/SabrenNewss/1",
                canonical_url="https://t.me/SabrenNewss/1",
                original_title=text,
                original_summary=text,
                source_name="Sabereen News",
            )

        assert check_blocked_topic(_art("Traffic police install speed radar cameras on Erbil highway")) is not None
        assert check_blocked_topic(_art("مديرية المرور تعلن نشر رادارات وكاميرات المراقبة المرورية")) is not None
        assert check_blocked_topic(_art("بەڕێوەبەرایەتی هاتووچۆ هۆشداری لەبارەی کامێرای خێرایی دا")) is not None

        assert check_blocked_topic(_art("جدول قطع الكهرباء المبرمج وصيانة خطوط المياه غدا")) is not None
        assert check_blocked_topic(_art("Municipal water and electricity outage maintenance schedule")) is not None

        assert check_blocked_topic(_art("اشتركوا في قناتنا الجديدة عبر التليجرام للاعلان والترويج")) is not None
        assert check_blocked_topic(_art("Sponsored discount offer — follow us on Telegram")) is not None

        assert (
            check_blocked_topic(
                _art("Pentagon announces deployment of air defense batteries after ballistic missile attacks")
            )
            is None
        )
        assert (
            check_blocked_topic(
                _art("Kurdistan Regional Government delegation meets Iraqi Prime Minister over oil export resumption")
            )
            is None
        )


class TestTelegramFetchIsolation:
    """Verify per-channel HTTP errors, timeouts, and restricted Telegram pages never crash fetching."""

    @pytest.mark.asyncio
    async def test_channel_failure_isolation_and_restricted_preview(self):
        fresh_utc = _iso_hours_ago(0.5)
        valid_plus_html = _build_channel_html(
            "Sabren_News1",
            [
                _build_msg_block(
                    "Sabren_News1/501",
                    (
                        "انطلاق مباحثات دبلوماسية رفيعة المستوى في جنيف بمشاركة وفود دولية لبحث التهدئة "
                        "العسكرية في الشرق الأوسط وتأمين ممرات التجارة البحرية.<br/>"
                        "وأشار دبلوماسيون إلى طرح مسودة اتفاق أولي لوقف إطلاق النار وتبادل الأسرى."
                    ),
                    dt_str=fresh_utc,
                )
            ],
        )
        restricted_html = """<html><body>
            <div class="tgme_page_title">Sabereen News</div>
            <div class="tgme_page_description">Please open Telegram to view this post or channel.</div>
        </body></html>"""

        channels = [
            {"username": "SabrenNewss", "name": "Sabereen News", "region_hint": "iraq_region", "lang": "ar"},
            {"username": "Sabren_News1", "name": "Sabereen Plus", "region_hint": "iraq_region", "lang": "ar"},
        ]

        # Scenario 1: SabrenNewss returns 500, Sabren_News1 succeeds
        src = TelegramSource(channels=channels)

        def fake_get_500(url, **kwargs):
            if "SabrenNewss" in url:
                return _MockResponseContext(_make_html_response(500, "Server Error"))
            return _MockResponseContext(_make_html_response(200, valid_plus_html))

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(side_effect=fake_get_500)
        src._session = mock_session

        articles = await src.fetch_articles()
        assert len(articles) == 1
        assert articles[0].url == "https://t.me/Sabren_News1/501"
        assert articles[0].source_name == "Sabereen Plus"

        # Scenario 2: SabrenNewss raises TimeoutError, Sabren_News1 returns 429 -> returns [] cleanly
        def fake_get_both_fail(url, **kwargs):
            if "SabrenNewss" in url:
                return _MockResponseContext(error=asyncio.TimeoutError())
            return _MockResponseContext(_make_html_response(429, "Too Many Requests"))

        mock_session.get = MagicMock(side_effect=fake_get_both_fail)
        articles_empty = await src.fetch_articles()
        assert articles_empty == []

        # Scenario 3: SabrenNewss restricted preview page, Sabren_News1 ClientError -> no crash
        def fake_get_restricted(url, **kwargs):
            if "SabrenNewss" in url:
                return _MockResponseContext(_make_html_response(200, restricted_html))
            return _MockResponseContext(error=aiohttp.ClientConnectionError("network down"))

        mock_session.get = MagicMock(side_effect=fake_get_restricted)
        articles_restricted = await src.fetch_articles()
        assert articles_restricted == []


class TestNeutralTerminologyNormalization:
    """Verify Reuters/BBC-style neutral Sorani/Arabic/English terminology enforcement."""

    @pytest.mark.parametrize(
        ("raw", "expected_present", "forbidden"),
        [
            ("شاندێکی باکووری عێراق سەردانی بەغدای کرد", "هەرێمی کوردستان", ["باکووری عێراق"]),
            ("ڕەوشی باکوری عێراق و پەیوەندییەکان لە باکووری عێراقدا", "هەرێمی کوردستاندا", ["باکوری عێراق", "باکووری عێراقدا"]),
            ("وفد من شمال العراق يصل إلى بغداد", "هەرێمی کوردستان", ["شمال العراق"]),
            ("Talks held in Northern Iraq over oil exports", "هەرێمی کوردستان", ["Northern Iraq"]),
            ("ئەمریکای دوژمن و دوژمنی ئەمریکی هۆشدارییان دا", "ئەمریکا", ["ئەمریکای دوژمن", "دوژمنی ئەمریکی"]),
            ("هێزە داگیرکەرەکانی ئەمریکا و قوات الاحتلال الأمريكي والشيطان الأكبر", "هێزەکانی ئەمریکا", ["داگیرکەر", "الشيطان الأكبر"]),
            ("العدو الأمريكي يعلن تعزيز قواعده", "ئەمریکا", ["العدو الأمريكي"]),
            ("قەوارەی زایۆنی و کیانی زایۆنی و رژێمی زایۆنی بەیاننامەیان بڵاوکردەوە", "ئیسرائیل", ["قەوارەی زایۆنی", "کیانی زایۆنی", "رژێمی زایۆنی"]),
            ("سوپای داگیرکەری زایۆنی و جيش الاحتلال الإسرائيلي و الكيان الصهيوني", "سوپای ئیسرائیل", ["داگیرکەری زایۆنی", "جيش الاحتلال", "الكيان الصهيوني"]),
            ("قوات الاحتلال الإسرائيلي و هێزە داگیرکەرەکانی ئیسرائیل", "هێزەکانی ئیسرائیل", ["الاحتلال", "داگیرکەر"]),
            ("الكيان المؤقت و قەوارەی داگیرکەر و دوژمنی زایۆنی", "ئیسرائیل", ["الكيان المؤقت", "قەوارەی داگیرکەر", "دوژمنی زایۆنی"]),
            ("سووریای داگیرکراو و سوريا المحتلة", "سووریا", ["داگیرکراو", "المحتلة"]),
        ],
    )
    def test_normalize_neutral_terminology_mappings_and_idempotence(self, raw, expected_present, forbidden):
        normalized = normalize_neutral_terminology(raw)
        assert expected_present in normalized
        for bad in forbidden:
            assert bad not in normalized
        # Idempotence: second pass leaves result unchanged
        assert normalize_neutral_terminology(normalized) == normalized

    def test_normalize_neutral_terminology_preserves_factual_context_and_words(self):
        factual = (
            "نەتەوە یەکگرتووەکان لە ڕاپۆرتێکی یاساییدا باسی دۆخی ناوچە جێناکۆکەکان و "
            "بوونی هێزە سەربازییەکان دەکات لە هەرێمی کوردستاندا."
        )
        assert normalize_neutral_terminology(factual) == factual

    def test_content_formatter_normalizes_headline_and_body(self):
        art = Article(
            original_url="https://t.me/SabrenNewss/800",
            canonical_url="https://t.me/SabrenNewss/800",
            original_title="Biased draft",
            translated_title="قەوارەی زایۆنی و ئەمریکای دوژمن لە باکووری عێراق کۆبوونەوە",
            translated_summary=(
                "لە باکووری عێراقدا ڕاپۆرتەکان باس لە جووڵەی هێزە داگیرکەرەکانی ئەمریکا و "
                "کیانی زایۆنی دەکەن لە نزیک سووریای داگیرکراو."
            ),
            source_name="Sabereen News",
            category="military_conflict",
        )
        formatter = ContentFormatter()
        caption = formatter.format_caption(art)
        full_msg = formatter.format_telegram_message(art)

        for rendered in (caption, full_msg):
            assert "هەرێمی کوردستان" in rendered
            assert "ئیسرائیل" in rendered
            assert "ئەمریکا" in rendered
            assert "سووریا" in rendered
            for biased in ("باکووری عێراق", "قەوارەی زایۆنی", "کیانی زایۆنی", "ئەمریکای دوژمن", "سووریای داگیرکراو"):
                assert biased not in rendered

    @pytest.mark.asyncio
    async def test_gemini_translator_and_digest_prompts_and_outputs_enforce_neutrality(self):
        for prompt_text in (
            GEMINI_TRANSLATION_SYSTEM_PROMPT,
            GEMINI_KURDISH_REWRITE_SYSTEM_PROMPT,
            GEMINI_DIGEST_SYSTEM_PROMPT,
        ):
            assert "هەرێمی کوردستان" in prompt_text
            assert "ئەمریکا" in prompt_text
            assert "ئیسرائیل" in prompt_text
            assert "سووریا" in prompt_text
            assert "Reuters" in prompt_text or "BBC" in prompt_text or "neutral" in prompt_text.lower()

        # Verify GeminiTranslationProvider + TranslationEngine normalize non-neutral model outputs
        engine = TranslationEngine(gemini_api_key="gem_test_key", base_backoff=0.0)
        biased_json = {
            "title": "قەوارەی زایۆنی و ئەمریکای دوژمن لە باکووری عێراق",
            "summary": (
                "لە باکووری عێراقدا چاودێران باس لە جموجۆڵی هێزە داگیرکەرەکانی ئەمریکا و "
                "دوژمنی زایۆنی دەکەن. هەروەها دۆخی سووریای داگیرکراو تاوتوێ کرا."
            ),
        }
        engine._gemini_provider._client._get_session = AsyncMock(
            return_value=_make_gemini_json_session(biased_json)
        )

        tr_title, tr_summary = await engine.translate_article(
            title="Report on regional security developments",
            summary="Detailed report from Iraq and Syria border region regarding diplomatic talks.",
        )
        assert "هەرێمی کوردستان" in tr_title
        assert "ئیسرائیل" in tr_title
        assert "ئەمریکا" in tr_title
        assert "سووریا" in tr_summary
        for bad in ("قەوارەی زایۆنی", "ئەمریکای دوژمن", "باکووری عێراق", "سووریای داگیرکراو", "دوژمنی زایۆنی"):
            assert bad not in tr_title
            assert bad not in tr_summary

        # Also verify standalone translate_title / translate_summary paths normalize output
        with patch.object(
            engine._gemini_provider._client,
            "generate_json",
            new_callable=AsyncMock,
            return_value={"text": "قەوارەی زایۆنی و ئەمریکای دوژمن لە باکووری عێراق"},
        ):
            standalone_title = await engine.translate_title("Sample headline")
        assert standalone_title == "ئیسرائیل و ئەمریکا لە هەرێمی کوردستان"

        # Verify DailyDigestGenerator neutralizes synthesized output too
        digest_gen = DailyDigestGenerator(api_key="gem_test_key", base_backoff=0.0)
        pub_now = datetime.now(timezone.utc).isoformat()
        arts = [
            Article(
                original_url=f"https://example.com/d/{i}",
                canonical_url=f"https://example.com/d/{i}",
                original_title=f"Event {i}",
                translated_title=f"ڕووداوی ژمارە {i}",
                translated_summary="ڕاپۆرتی ڕۆژنامەوانی بێلایەن.",
                source_name="Reuters",
                category="military_conflict",
                published_at=pub_now,
                status=ArticleStatus.PUBLISHED,
            )
            for i in range(1, 5)
        ]
        biased_digest = {
            "stories": [
                {"emoji": "⚔️", "summary": "قەوارەی زایۆنی و ئەمریکای دوژمن لە باکووری عێراق هەڵوێستی نوێیان ڕاگەیاند."},
                {"emoji": "🏛️", "summary": "دۆخی سووریای داگیرکراو و کۆبوونەوەی نەتەوە یەکگرتووەکان لە نیویۆرک بەردەوامە."},
                {"emoji": "📉", "summary": "نرخی نەوت لە بازاڕە جیهانییەکاندا گۆڕانکاری بەخۆیەوە بینی."},
            ]
        }
        with patch.object(digest_gen._gemini, "generate_json", new_callable=AsyncMock, return_value=biased_digest):
            digest_res = await digest_gen.generate_digest(arts, digest_date="2026-11-25")

        assert "هەرێمی کوردستان" in digest_res.digest_text
        assert "ئیسرائیل" in digest_res.digest_text
        assert "ئەمریکا" in digest_res.digest_text
        assert "سووریا" in digest_res.digest_text
        assert "قەوارەی زایۆنی" not in digest_res.digest_text
        assert "باکووری عێراق" not in digest_res.digest_text


class TestTelegramPipelineAndEngineIntegration:
    """Verify NewsEngine registration, discovery isolation, pipeline gates, dedup, quota, and cards."""

    @pytest.mark.asyncio
    async def test_news_engine_registers_telegram_and_handles_discovery_isolation_and_canonical_dedup(
        self, tmp_path
    ):
        db = Database(str(tmp_path / "engine_tg.db"))
        await db.connect()
        engine = None
        engine_disabled = None
        try:
            engine = NewsEngine()
            engine._db = db
            await engine._init_sources()

            source_names = [s.name for s in engine._sources]
            assert "TelegramPublicWeb" in source_names
            assert "RSS" in source_names

            # Respects ENABLE_TELEGRAM=False cleanly
            with patch("core.engine.ENABLE_TELEGRAM", False):
                engine_disabled = NewsEngine()
                engine_disabled._db = db
                await engine_disabled._init_sources()
                assert "TelegramPublicWeb" not in [s.name for s in engine_disabled._sources]

            # Replace engine sources with 1 Telegram source + 1 mocked RSS source
            tg_src = next(s for s in engine._sources if isinstance(s, TelegramSource))
            rss_stub = MagicMock(spec=BaseSource)
            rss_stub.name = "RSS"
            rss_stub.source_type = "rss"
            rss_stub.enabled = True
            rss_stub.initialize = AsyncMock()
            rss_stub.cleanup = AsyncMock()

            fresh_utc = _iso_hours_ago(0.4)
            tg_article = RawArticle(
                url="https://t.me/SabrenNewss/555",
                title="Iraq security forces reinforce border perimeter near western Anbar",
                summary=(
                    "Iraq security forces reinforced the border perimeter near western Anbar following "
                    "nighttime drone activity and coordination talks between defense commanders in Baghdad. "
                    "Additional radar units and patrols were deployed along key highways to secure installations."
                ),
                source_name="Sabereen News",
                category="news",
                publication_date=fresh_utc,
                image_url="https://cdn4.telegram-cdn.org/file/tg555.jpg",
                country="iraq_region",
                region="iraq_region",
            )
            rss_article = RawArticle(
                url="https://reuters.com/world/middle-east/diplomacy-556",
                title="Foreign ministers hold urgent talks on Red Sea maritime security corridor",
                summary=(
                    "Foreign ministers convened an urgent ministerial session on Red Sea maritime security "
                    "and protection of commercial shipping lanes after renewed drone incidents overnight. "
                    "Delegates reviewed joint naval escort commitments."
                ),
                source_name="Reuters",
                category="World",
                publication_date=fresh_utc,
                image_url="",
                country="world",
                region="world",
            )

            tg_src.fetch_articles = AsyncMock(return_value=[tg_article])
            rss_stub.fetch_articles = AsyncMock(return_value=[rss_article])
            for src in engine._sources:
                if src is not tg_src:
                    await src.cleanup()
            engine._sources = [tg_src, rss_stub]

            # Patch batch publish execution so discovery cycle only measures discovery + DB persistence
            engine._pipeline = MagicMock()
            engine._pipeline.process_batch = AsyncMock(return_value={"published": 0})
            await engine._discovery_cycle()

            saved_tg = await db.get_article_by_canonical_url("https://t.me/SabrenNewss/555")
            saved_rss = await db.get_article_by_canonical_url("https://reuters.com/world/middle-east/diplomacy-556")
            assert saved_tg is not None
            assert saved_tg.source_name == "Sabereen News"
            assert saved_rss is not None

            # Second discovery cycle with exact same Telegram post -> no duplicate row inserted
            await engine._discovery_cycle()
            async with db.conn.execute("SELECT COUNT(*) FROM articles") as cur:
                row = await cur.fetchone()
            assert row[0] == 2

            # Telegram failure in discovery never blocks RSS source
            rss_second = RawArticle(
                url="https://reuters.com/world/middle-east/summit-557",
                title="G7 envoys release joint declaration on regional ceasefire talks",
                summary=(
                    "G7 diplomatic envoys issued a joint statement calling for sustained ceasefire negotiations "
                    "and unrestricted humanitarian aid access through border crossings across the region. "
                    "Further ministerial consultations are scheduled later this week."
                ),
                source_name="Reuters",
                category="World",
                publication_date=fresh_utc,
            )
            tg_src.fetch_articles = AsyncMock(side_effect=RuntimeError("Telegram network down"))
            rss_stub.fetch_articles = AsyncMock(return_value=[rss_second])
            await engine._discovery_cycle()
            assert await db.get_article_by_canonical_url("https://reuters.com/world/middle-east/summit-557") is not None
        finally:
            for eng in (engine, engine_disabled):
                if eng is not None:
                    for src in getattr(eng, "_sources", []):
                        try:
                            await src.cleanup()
                        except Exception:
                            pass
            await db.close()

    @pytest.mark.asyncio
    async def test_end_to_end_pipeline_filters_dedup_quota_breaking_normalization_and_card_image(
        self, temp_db
    ):
        # 1. Verify Baghdad Sabereen News item is NOT misclassified as Bashur, protecting 5-of-8 quota
        baghdad_tg_art = Article(
            original_url="https://t.me/SabrenNewss/9001",
            canonical_url="https://t.me/SabrenNewss/9001",
            original_title="Iraqi parliament debates federal security pact in Baghdad",
            original_summary=(
                "Lawmakers in Baghdad held a special parliamentary debate on the federal security pact, "
                "air defense radar procurement, and diplomatic coordination with neighboring capitals. "
                "Committee leaders urged faster implementation of joint border safeguards."
            ),
            source_name="Sabereen News",
            category="politics_world",
            importance_score=0.88,
            country="iraq_region",
            region="iraq_region",
            publication_date=_iso_hours_ago(0.2),
        )
        assert is_bashur_article(baghdad_tg_art) is False

        erbil_tg_art = Article(
            original_url="https://t.me/SabrenNewss/9002",
            canonical_url="https://t.me/SabrenNewss/9002",
            original_title="KRG and Peshmerga commanders inspect air defenses in Erbil",
            original_summary=(
                "Kurdistan Regional Government officials and Peshmerga commanders reviewed joint security "
                "arrangements in Erbil alongside talks with Baghdad on federal budget transfers."
            ),
            source_name="Sabereen News",
            category="kurdistan",
            importance_score=0.90,
            country="iraq_region",
            region="iraq_region",
            publication_date=_iso_hours_ago(0.2),
        )
        assert is_bashur_article(erbil_tg_art) is True

        # Confirm 5-of-8 QuotaBalancer selects up to 5 Bashur + 3 Global (treating baghdad_tg_art as Global)
        pool = [
            Article(
                original_url=f"https://rudaw.net/kurdistan/k{i}",
                canonical_url=f"https://rudaw.net/kurdistan/k{i}",
                original_title=f"KRG Parliament approves Erbil Sulaymaniyah security measure {i}",
                original_summary="Kurdistan Regional Government and Peshmerga forces coordinated policy in Erbil.",
                source_name="Rudaw",
                category="kurdistan",
                importance_score=0.85 - i * 0.01,
                publication_date=_iso_hours_ago(0.5),
            )
            for i in range(6)
        ] + [
            baghdad_tg_art,
            Article(
                original_url="https://reuters.com/world/g1",
                canonical_url="https://reuters.com/world/g1",
                original_title="NATO defense chiefs announce missile shield expansion",
                original_summary="NATO ministers announced new air defense batteries across eastern allies.",
                source_name="Reuters",
                category="military_conflict",
                importance_score=0.87,
                publication_date=_iso_hours_ago(0.5),
            ),
            Article(
                original_url="https://reuters.com/world/g2",
                canonical_url="https://reuters.com/world/g2",
                original_title="EU summit imposes new energy sanctions package",
                original_summary="European Union leaders endorsed a new sanctions framework targeting oil tankers.",
                source_name="Reuters",
                category="politics_world",
                importance_score=0.86,
                publication_date=_iso_hours_ago(0.5),
            ),
            Article(
                original_url="https://reuters.com/world/g3",
                canonical_url="https://reuters.com/world/g3",
                original_title="UN Security Council debates maritime shipping security",
                original_summary="Ambassadors at the UN Security Council discussed Red Sea commercial routes.",
                source_name="Reuters",
                category="politics_world",
                importance_score=0.84,
                publication_date=_iso_hours_ago(0.5),
            ),
        ]
        balancer = QuotaBalancer(batch_limit=BATCH_PUBLISH_LIMIT, bashur_per_8=BASHUR_QUOTA_PER_8)
        selection = balancer.select_batch(pool, limit=8)
        assert selection.bashur_selected == 5
        assert selection.global_selected == 3

        # 2. Full end-to-end TelegramPublisher + ArticlePipeline test with mocked network/Gemini
        mock_bot = AsyncMock()
        msg_obj = MagicMock()
        msg_obj.message_id = 7771
        mock_bot.send_photo = AsyncMock(return_value=msg_obj)
        mock_bot.send_message = AsyncMock(return_value=msg_obj)

        publisher = TelegramPublisher()
        publisher._channel_id = "@KurdishGlobalNews"
        publisher._bot = mock_bot
        publisher._image_handler.download_image = AsyncMock(return_value=_make_jpeg_bytes())

        pipeline = ArticlePipeline(temp_db, publisher, HealthMonitor(), Metrics())
        # Prevent real Gemini HTTP calls in SemanticDeduplicator
        pipeline._semantic_dedup = SemanticDeduplicator(temp_db, api_key="", base_backoff=0.0)

        tg_photo_url = "https://cdn4.telegram-cdn.org/file/real_strike_photo.jpg"
        baghdad_tg_art.image_url = tg_photo_url
        baghdad_tg_art.id = await temp_db.insert_article(baghdad_tg_art)

        # Case 2A: Score 9.0 statement with is_breaking=False -> strict breaking gate stays FALSE & silent!
        with patch.object(
            pipeline._ranker,
            "evaluate_global_relevance",
            new=AsyncMock(
                return_value=RelevanceEvaluation(
                    score=9,
                    is_breaking=False,
                    category="politics_world",
                    reason="High-level parliamentary pact without sudden kinetic emergency",
                    passed=True,
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
                    "پەرلەمان گفتوگۆ لەسەر ڕێککەوتنی ئەمنی باکووری عێراق و ئەمریکای دوژمن دەکات",
                    (
                        "لە بەغدا کۆبوونەوەی پەرلەمان لەسەر دۆخی ئەمنی و جووڵەی قەوارەی زایۆنی و "
                        "سووریای داگیرکراو بەڕێوەچوو. پەرلەمانتاران داوای هەماهەنگیی بەرگرییان کرد."
                    ),
                )
            ),
        ):
            ok = await pipeline.process_article(baghdad_tg_art)

        assert ok is True
        # Telegram channel static photos/og:image are suppressed so branded card avoids channel watermarks
        publisher._image_handler.download_image.assert_not_awaited()
        send_kwargs = mock_bot.send_photo.call_args.kwargs
        assert send_kwargs["disable_notification"] is True
        assert "🔴 بەپەلە | " not in send_kwargs["caption"]
        # Terminology normalized in published caption AND saved DB row
        assert "هەرێمی کوردستان" in send_kwargs["caption"]
        assert "ئەمریکا" in send_kwargs["caption"]
        assert "ئیسرائیل" in send_kwargs["caption"]
        assert "سووریا" in send_kwargs["caption"]
        for bad in ("باکووری عێراق", "ئەمریکای دوژمن", "قەوارەی زایۆنی", "سووریای داگیرکراو"):
            assert bad not in send_kwargs["caption"]

        saved_row = await temp_db.get_article_by_canonical_url(baghdad_tg_art.canonical_url)
        assert saved_row is not None and saved_row.status == ArticleStatus.PUBLISHED
        assert "باکووری عێراق" not in saved_row.translated_title
        assert "ئەمریکای دوژمن" not in saved_row.translated_title

        # Case 2B: Score 9.5 + is_breaking=True sudden military attack -> loud delivery + 🔴 بەپەلە |
        mock_bot.send_photo.reset_mock()
        breaking_tg = Article(
            original_url="https://t.me/Sabren_News1/9003",
            canonical_url="https://t.me/Sabren_News1/9003",
            original_title="Ballistic missile strikes hit major airbase in central region",
            original_summary=(
                "Multiple ballistic missiles struck a major airbase overnight triggering explosions and "
                "air defense sirens across surrounding districts. Military officials confirmed emergency "
                "deployments and casualty assessment teams at the site."
            ),
            source_name="Sabereen Plus",
            category="military_conflict",
            image_url="https://cdn4.telegram-cdn.org/file/breaking9003.jpg",
            publication_date=_iso_hours_ago(0.1),
        )
        breaking_tg.id = await temp_db.insert_article(breaking_tg)
        with patch.object(
            pipeline._ranker,
            "evaluate_global_relevance",
            new=AsyncMock(
                return_value=RelevanceEvaluation(
                    score=9.5,
                    is_breaking=True,
                    category="military_conflict",
                    reason="Sudden ongoing ballistic missile strike",
                    passed=True,
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
                    "مووشەکی بالیستیک بەر بنکەیەکی سەربازی کەوت و بەرگریی ئاسمانی کارا کرا",
                    (
                        "دەنگی تەقینەوەی بەهێز لە نزیک بنکە سەربازییەکە بیستراوە و تیمەکانی فریاگوزاری "
                        "گەیشتوونەتە شوێنی ڕووداوەکە. بەرپرسان دەڵێن لێکۆڵینەوە بەردەوامە."
                    ),
                )
            ),
        ):
            ok_breaking = await pipeline.process_article(breaking_tg)

        assert ok_breaking is True
        brk_kwargs = mock_bot.send_photo.call_args.kwargs
        assert brk_kwargs["disable_notification"] is False
        assert "🔴 بەپەلە | " in brk_kwargs["caption"]
        assert is_strict_breaking(breaking_tg) is True

        # Case 2C: Same military event later arriving from another Telegram channel -> zero-API local dedup skips!
        dup_tg = Article(
            original_url="https://t.me/SabrenNewss/9004",
            canonical_url="https://t.me/SabrenNewss/9004",
            original_title="Explosions reported after missile attack on central military airbase",
            original_summary=(
                "Emergency crews rushed to the central military airbase after reported missile strikes "
                "and air defense activations across nearby neighborhoods."
            ),
            source_name="Sabereen News",
            category="military_conflict",
            publication_date=_iso_hours_ago(0.05),
        )
        dup_tg.id = await temp_db.insert_article(dup_tg)
        with patch.object(
            pipeline._ranker,
            "evaluate_global_relevance",
            new=AsyncMock(
                return_value=RelevanceEvaluation(
                    score=8,
                    is_breaking=False,
                    category="military_conflict",
                    reason="Military strike report",
                    passed=True,
                )
            ),
        ):
            dup_published = await pipeline.process_article(dup_tg)

        assert dup_published is False
        assert await temp_db.is_url_skipped(dup_tg.canonical_url) is True

        # Case 2D: Blocked topic (traffic radar / municipal / sports) rejected & persisted to skipped_urls
        blocked_tg = Article(
            original_url="https://t.me/SabrenNewss/9005",
            canonical_url="https://t.me/SabrenNewss/9005",
            original_title="تنويه مروري: نشر رادارات وكاميرات المراقبة على الطريق السريع",
            original_summary="مديرية المرور تباشر بتفعيل كاميرات السرعة والرادارات لمتابعة المخالفات المرورية.",
            source_name="Sabereen News",
            category="news",
            publication_date=_iso_hours_ago(0.1),
        )
        blocked_tg.id = await temp_db.insert_article(blocked_tg)
        blocked_published = await pipeline.process_article(blocked_tg)
        assert blocked_published is False
        assert await temp_db.is_url_skipped(blocked_tg.canonical_url) is True


class TestTelegramVideoAndMediaExtraction:
    """Verify video/thumbnail extraction, Media-is-too-big handling, DB storage, and pipeline video flow."""

    def test_extract_video_and_thumbnail_from_video_wrap_and_too_big_previews(self):
        fresh_utc = _iso_hours_ago(0.4)
        html = f"""<!DOCTYPE html>
<html><body><main><section class="tgme_channel_history">
  <div class="tgme_widget_message_wrap">
    <div class="tgme_widget_message" data-post="SabrenNewss/7001">
      <a class="tgme_widget_message_video_player" href="https://t.me/SabrenNewss/7001">
        <div class="tgme_widget_message_video_wrap">
          <i class="tgme_widget_message_video_thumb" style="background-image:url('https://cdn4.telegram-cdn.org/file/vthumb_1.jpg')"></i>
          <video class="tgme_widget_message_video" src="https://cdn4.telegram-cdn.org/file/clip_7001.mp4"></video>
        </div>
      </a>
      <div class="tgme_widget_message_text">
        وزارة الدفاع تنشر مشاهد لعملية أمنية ومناورات ميدانية لتأمين المناطق الحدودية.<br/>
        وأكد البيان تعزيز وحدات الرصد الجوي ومواصلة المهام لحماية المنشآت السيادية والاستراتيجية في البلاد.
      </div>
      <div class="tgme_widget_message_footer">
        <a class="tgme_widget_message_date" href="https://t.me/SabrenNewss/7001"><time datetime="{fresh_utc}">12:00</time></a>
      </div>
    </div>
  </div>
  <div class="tgme_widget_message_wrap">
    <div class="tgme_widget_message" data-post="SabrenNewss/7002">
      <div class="tgme_widget_message_video_wrap">
        <i class="tgme_widget_message_video_thumb" style="background-image:url(&quot;https://cdn4.telegram-cdn.org/file/vthumb_2.jpg&quot;)"></i>
        <video class="tgme_widget_message_video"><source src="https://cdn4.telegram-cdn.org/file/clip_7002.mp4" type="video/mp4"></video>
      </div>
      <div class="tgme_widget_message_text">
        رئيس الوزراء يعقد جلسة طارئة مع القيادات العسكرية لمتابعة الملف الأمني وتنسيق الخطط الميدانية.<br/>
        وشدد الاجتماع على حماية الحدود ورفع جاهزية الدفاعات الجوية في مختلف القطاعات العسكرية.
      </div>
      <div class="tgme_widget_message_footer">
        <a class="tgme_widget_message_date" href="https://t.me/SabrenNewss/7002"><time datetime="{fresh_utc}">12:05</time></a>
      </div>
    </div>
  </div>
  <div class="tgme_widget_message_wrap">
    <div class="tgme_widget_message" data-post="SabrenNewss/7003">
      <div class="tgme_widget_message_video_wrap">
        <i class="tgme_widget_message_video_thumb" style="background-image:url('https://cdn4.telegram-cdn.org/file/too_big_thumb.jpg')"></i>
        <div class="message_media_not_supported_wrap"><div class="message_media_not_supported_label">MEDIA IS TOO BIG</div></div>
      </div>
      <div class="tgme_widget_message_text">
        اجتماع دبلوماسي موسع في بغداد لبحث الاتفاقيات الإقليمية وأمن الطاقة والحدود الدولية.<br/>
        وأكد المشاركون أهمية الحوار المباشر وتجنب التصعيد العسكري وحماية المصالح الوطنية المشتركة.
      </div>
      <div class="tgme_widget_message_footer">
        <a class="tgme_widget_message_date" href="https://t.me/SabrenNewss/7003"><time datetime="{fresh_utc}">12:10</time></a>
      </div>
    </div>
  </div>
  <div class="tgme_widget_message_wrap">
    <div class="tgme_widget_message" data-post="SabrenNewss/7004">
      <a class="tgme_widget_message_photo_wrap" style="background-image:url('https://cdn4.telegram-cdn.org/file/album_photo.jpg')"></a>
      <div class="tgme_widget_message_video_wrap">
        <video src="https://cdn4.telegram-cdn.org/file/album_vid.mp4"></video>
      </div>
      <div class="tgme_widget_message_text">
        القيادة العسكرية تتابع تطورات الميدان في القاطع الغربي وتصدر توجيهات جديدة بشأن الانتشار الأمني.<br/>
        كما جرى تقييم الجاهزية اللوجستية ودعم الوحدات القتالية في النقاط الحدودية المتقدمة.
      </div>
      <div class="tgme_widget_message_footer">
        <a class="tgme_widget_message_date" href="https://t.me/SabrenNewss/7004"><time datetime="{fresh_utc}">12:15</time></a>
      </div>
    </div>
  </div>
</section></main></body></html>"""

        src = TelegramSource()
        cfg = {"username": "SabrenNewss", "name": "Sabereen News", "region_hint": "iraq_region", "lang": "ar"}
        articles = src.parse_channel_html(html, cfg)

        assert len(articles) == 4

        # 1. Direct <video src> -> video_url extracted, thumb image_url ignored
        assert articles[0].url == "https://t.me/SabrenNewss/7001"
        assert articles[0].source_name == "Sabereen News"
        assert articles[0].video_url == "https://cdn4.telegram-cdn.org/file/clip_7001.mp4"
        assert articles[0].image_url is None

        # 2. Nested <video><source src> -> video_url extracted, thumb image_url ignored
        assert articles[1].url == "https://t.me/SabrenNewss/7002"
        assert articles[1].video_url == "https://cdn4.telegram-cdn.org/file/clip_7002.mp4"
        assert articles[1].image_url is None

        # 3. "Media is too big" -> video_url is None, static thumbnail ignored
        assert articles[2].url == "https://t.me/SabrenNewss/7003"
        assert articles[2].video_url is None
        assert articles[2].image_url is None

        # 4. Multi-media (photo + video) -> retains video_url, static photo image_url ignored
        assert articles[3].url == "https://t.me/SabrenNewss/7004"
        assert articles[3].video_url == "https://cdn4.telegram-cdn.org/file/album_vid.mp4"
        assert articles[3].image_url is None

        # Verify helper functions directly ignore malformed/non-http video attrs
        bad_soup = BeautifulSoup(
            '<div class="tgme_widget_message"><div class="tgme_widget_message_video_wrap"><video src="javascript:alert(1)"></video></div></div>',
            "lxml",
        )
        node = bad_soup.select_one(".tgme_widget_message")
        assert extract_telegram_video_url(node) is None
        assert extract_telegram_video_thumb_url(node) == ""

    @pytest.mark.asyncio
    async def test_video_url_persisted_through_extractor_database_and_pipeline(self, temp_db, tmp_path):
        fresh_utc = _iso_hours_ago(0.2)
        raw = RawArticle(
            url="https://t.me/Sabren_News1/8100",
            title="Sabereen Plus reports border security radar deployment",
            summary=(
                "Iraq defense units deployed mobile air surveillance systems along the western border "
                "following security consultations in Baghdad. Additional engineering teams fortified "
                "communications posts and monitoring towers across the sector."
            ),
            source_name="Sabereen Plus",
            category="military_conflict",
            publication_date=fresh_utc,
            image_url="https://cdn4.telegram-cdn.org/file/thumb_8100.jpg",
            video_url="https://cdn4.telegram-cdn.org/file/video_8100.mp4",
            country="iraq_region",
            region="iraq_region",
        )

        extracted = ArticleExtractor().extract(raw)
        assert extracted.video_url == "https://cdn4.telegram-cdn.org/file/video_8100.mp4"
        assert extracted.image_url == "https://cdn4.telegram-cdn.org/file/thumb_8100.jpg"
        assert extracted.source_name == "Sabereen Plus"

        art_id = await temp_db.insert_article(extracted)
        assert art_id is not None
        loaded = await temp_db.get_article_by_canonical_url(extracted.canonical_url)
        assert loaded is not None
        assert loaded.video_url == "https://cdn4.telegram-cdn.org/file/video_8100.mp4"
        assert loaded.image_url == "https://cdn4.telegram-cdn.org/file/thumb_8100.jpg"

        # Run through full ArticlePipeline + TelegramPublisher with mocked video processor & send_video
        branded_tmp = tmp_path / "kgn_pipeline_8100.mp4"
        branded_tmp.write_bytes(b"BRANDED_STREAM")

        mock_bot = AsyncMock()
        vid_msg = MagicMock()
        vid_msg.message_id = 9101
        mock_bot.send_video = AsyncMock(return_value=vid_msg)
        mock_bot.send_photo = AsyncMock()

        publisher = TelegramPublisher()
        publisher._channel_id = "@KurdishGlobalNews"
        publisher._bot = mock_bot

        pipeline = ArticlePipeline(temp_db, publisher, HealthMonitor(), Metrics())
        pipeline._semantic_dedup = SemanticDeduplicator(temp_db, api_key="", base_backoff=0.0)

        with patch.object(
            pipeline._ranker,
            "evaluate_global_relevance",
            new=AsyncMock(
                return_value=RelevanceEvaluation(
                    score=8.5,
                    is_breaking=False,
                    category="military_conflict",
                    reason="Border defense deployment",
                    passed=True,
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
                    "ئەمریکای دوژمن و باکووری عێراق لە پرسی سنوورەکاندا گفتوگۆیان کرد",
                    (
                        "هێزەکانی بەرگری لە باکووری عێراق و ناوچە سنوورییەکان ڕێوشوێنی نوێیان گرتەبەر. "
                        "بەرپرسان دەڵێن هەماهەنگییەکان بۆ سەقامگیری بەردەوام دەبن."
                    ),
                )
            ),
        ), patch(
            "tg_bot.publisher.download_and_brand_video",
            new=AsyncMock(return_value=str(branded_tmp)),
        ) as mock_brand:
            published = await pipeline.process_article(loaded)

        assert published is True
        mock_brand.assert_awaited_once_with("https://cdn4.telegram-cdn.org/file/video_8100.mp4")
        mock_bot.send_video.assert_awaited_once()
        mock_bot.send_photo.assert_not_awaited()
        sent_caption = mock_bot.send_video.call_args.kwargs["caption"]
        assert "Sabereen Plus" in sent_caption
        assert "https://t.me/Sabren_News1/8100" in sent_caption
        assert "هەرێمی کوردستان" in sent_caption
        assert "ئەمریکا" in sent_caption
        assert "باکووری عێراق" not in sent_caption
        assert "ئەمریکای دوژمن" not in sent_caption
        assert not branded_tmp.exists()

