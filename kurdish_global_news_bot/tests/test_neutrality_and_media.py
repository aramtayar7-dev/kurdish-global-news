"""
Comprehensive 100%-mocked automated tests covering:
  3.1 Deterministic Sorani editorial neutrality sanitizer & prompt hardening
  3.2 Telegram HTML image/album/preview/embed extraction & CDN download headers
  3.3 Resilient X / Twitter ingestion (rotation, sequential jitter, 429 breaker, fallbacks, budget)

Zero live network calls are permitted in this suite.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

BOT_ROOT = Path(__file__).resolve().parent.parent
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))

from PIL import Image

import config
from processing.content_formatter import (
    EMOJI_PATTERN,
    PUNCTUATION_PATTERN,
    ContentFormatter,
    _is_duplicate_summary,
    sanitize_neutral_sorani_text,
)
from processing.gemini_client import GeminiClient, resolve_gemini_system_instruction
from processing.openrouter_client import (
    NEUTRAL_EDITORIAL_SYSTEM_INSTRUCTION,
    OpenRouterClient,
    resolve_openrouter_system_prompt,
)
from processing.translator import (
    GEMINI_KURDISH_REWRITE_SYSTEM_PROMPT,
    GEMINI_TRANSLATION_SYSTEM_PROMPT,
    GeminiTranslationProvider,
    TranslationEngine,
)
from sources.telegram_source import (
    CHROME_DESKTOP_USER_AGENT as TG_CHROME_UA,
    TelegramSource,
    classify_telegram_media,
    extract_background_image,
    extract_telegram_image_candidates,
)
from sources.x_video_source import (
    SYNDICATION_HEADERS,
    XVideoSource,
    normalize_canonical_x_url,
)
from storage.models import Article
from tg_bot.publisher import TELEGRAM_CDN_DOWNLOAD_HEADERS, TelegramPublisher


@pytest.fixture(autouse=True)
def _block_live_duckduckgo_network():
    """Safety net ensuring no test accidentally invokes live DuckDuckGo/Bing searches."""
    with patch("duckduckgo_search.DDGS", side_effect=AssertionError("Live DDGS network call blocked in unit tests")):
        yield


class _MockAsyncResponseCtx:
    """Async context manager mock for aiohttp/httpx responses."""

    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _make_valid_png_bytes(width: int = 32, height: int = 32) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=(20, 60, 120))
    img.save(buf, format="PNG")
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════
# 3.1 SANITIZER & EDITORIAL NEUTRALITY TESTS
# ═══════════════════════════════════════════════════════════════════════════


def test_sanitize_removes_emojis():
    raw = "🤔🚀🔥 هەواڵێکی بەپەلە⚠️ لە بەغدا🇮🇶🏳️ تیمەکان👋🏽 و خێزان👨‍👩‍👧‍👦 گەیشتن𐌰"
    cleaned = sanitize_neutral_sorani_text(raw)
    assert not EMOJI_PATTERN.search(cleaned)
    for emoji in ("🤔", "🚀", "🔥", "⚠️", "🇮🇶", "🏳️", "👋🏽", "👨", "👩", "👧", "👦", "\u200D", "\uFE0F"):
        assert emoji not in cleaned
    # Preserves valid Sorani text AND non-emoji supplementary characters (e.g. U+10330)
    assert "هەواڵێکی بەپەلە لە بەغدا" in cleaned
    assert "تیمەکان و خێزان گەیشتن𐌰" in cleaned


def test_sanitize_collapses_excessive_punctuation():
    raw = "هێرشەکە کرا!!! ئایا زیان هەیە؟؟؟ زانیاری نوێ!!؟ ڕێژەکە 3.14 لە ڕێکەوتی 2026.11.27 بوو...."
    cleaned = sanitize_neutral_sorani_text(raw)
    assert not PUNCTUATION_PATTERN.search(cleaned)
    assert "!!!" not in cleaned
    assert "؟؟؟" not in cleaned
    assert "!!؟" not in cleaned
    assert "...." not in cleaned
    assert "3.14" in cleaned
    assert "2026.11.27" in cleaned
    # Single meaningful question mark stays untouched (not turned into an assertion)
    question = "ئایا دانوستانەکان بەردەوام دەبن؟"
    assert sanitize_neutral_sorani_text(question) == question


def test_sanitize_removes_sarcastic_country_quotes():
    assert sanitize_neutral_sorani_text('"ئیسرائیل" وەڵامی دا.') == "ئیسرائیل وەڵامی دا."
    assert sanitize_neutral_sorani_text("«ئەمریکا» و 'ئێران' کۆبوونەوە.") == "ئەمریکا و ئێران کۆبوونەوە."


def test_sanitize_preserves_direct_quotations():
    text = "وەزارەتەکە ڕایگەیاند: «ئەمە هێرشێکی سەربازییە»."
    assert sanitize_neutral_sorani_text(text) == text

    quoted_statement = 'سەرچاوە فەرمییەکە گوتی: «بەرخودان و ناوچەکە لە قۆناغی هەستیاردان».'
    assert sanitize_neutral_sorani_text(quoted_statement) == quoted_statement


def test_sanitize_preserves_attribution():
    attributed = (
        "بەپێی ڕاگەیەندراوی وەزارەتی بەرگری، هێزەکان ١٤ مووشەکیان بەرپەرچ داوەتەوە و "
        "ئەمەش بە دەستدرێژی ناو دەبرێت لەلایەن وتەبێژەکەوە."
    )
    cleaned = sanitize_neutral_sorani_text(attributed)
    assert "بەپێی ڕاگەیەندراوی وەزارەتی بەرگری" in cleaned
    assert "١٤" in cleaned
    assert "بە دەستدرێژی ناو" in cleaned


def test_sanitize_preserves_legal_context():
    legal_text = (
        "لە یاسای نێودەوڵەتیدا پرسی الاحتلال و ناوچە داگیرکراوەکان، "
        "هەروەها بەرخودان و ڕاپۆرتەکانی الشهداء و تیرۆریستان بە گوێرەی بەڵگە یاساییەکان هەڵدەسەنگێنرێت."
    )
    cleaned = sanitize_neutral_sorani_text(legal_text)
    assert "الاحتلال" in cleaned
    assert "بەرخودان" in cleaned
    assert "الشهداء" in cleaned
    assert "تیرۆریستان" in cleaned

    arabic_legal = "بەگوێرەی القوانين و جريمة العدوان لە دادگای نێودەوڵەتی لێکۆڵینەوە کرا."
    assert "العدوان" in sanitize_neutral_sorani_text(arabic_legal)


def test_sanitize_neutralizes_editorial_phrases():
    raw = "🤔🚀 قەوارەی زایۆنی!! دەستدرێژییەکی گەورە؟؟"
    cleaned = sanitize_neutral_sorani_text(raw)
    assert "🤔" not in cleaned
    assert "🚀" not in cleaned
    assert "!!" not in cleaned
    assert "؟؟" not in cleaned
    assert "قەوارەی زایۆنی" not in cleaned
    assert "ئیسرائیل" in cleaned
    assert "هێرشێکی" in cleaned

    # Also test editorial Arabic & Northern Iraq replacements
    raw2 = "سوپای الكيان الصهيوني و ئەمریکای دوژمن لە شمال العراق ڕێککەوتن."
    cleaned2 = sanitize_neutral_sorani_text(raw2)
    assert "ئیسرائیل" in cleaned2
    assert "ئەمریکا" in cleaned2
    assert "هەرێمی کوردستان" in cleaned2
    assert "الكيان الصهيوني" not in cleaned2
    assert "ئەمریکای دوژمن" not in cleaned2


def test_sanitize_does_not_overreplace_substrings():
    # Ensure boundary rules prevent corrupting partial substrings inside longer tokens
    sample = "بەپرسانی کوردستان و ئەمریکییەکان لە هەولێردا کۆبوونەوە و العدواني نەناسراو بوو."
    cleaned = sanitize_neutral_sorani_text(sample)
    assert "کوردستان" in cleaned
    assert "ئەمریکییەکان" in cleaned
    assert "العدواني" in cleaned  # attached suffix 'ي' is part of another word -> untouched

    attached = "قەوارەی زایۆنی!! لە بەرەی باشوور."
    assert sanitize_neutral_sorani_text(attached) == "ئیسرائیل. لە بەرەی باشوور."


def test_sanitize_handles_arabic_and_sorani():
    mixed = "جيش الاحتلال الإسرائيلي و هێزەکانی دوژمنی ئەمریکی لە قەوارەی زایۆنیدا و ئەمەریکای دوژمن کۆبوونەوە."
    cleaned = sanitize_neutral_sorani_text(mixed)
    assert "سوپای ئیسرائیل" in cleaned
    assert "هێزەکانی ئەمریکا" in cleaned
    assert "ئیسرائیلدا" in cleaned
    assert "ئەمریکا" in cleaned


def test_sanitize_is_idempotent():
    samples = [
        "🤔🚀 قەوارەی زایۆنی!! دەستدرێژییەکی گەورە؟؟",
        '"ئیسرائیل" و ئەمریکای دوژمن لە شمال العراق کۆبوونەوە!!',
        "وەزارەتەکە ڕایگەیاند: «ئەمە هێرشێکی سەربازییە».",
        "ڕاپۆرتە سەرەتاییەکان: 12.5 کیلۆمەتر لە 2026-11-27، و، بەرپرسان قسەیان کرد.",
    ]
    for sample in samples:
        once = sanitize_neutral_sorani_text(sample)
        twice = sanitize_neutral_sorani_text(once)
        assert twice == once


def test_sanitize_handles_none_and_invalid_input():
    assert sanitize_neutral_sorani_text(None) == ""  # type: ignore[arg-type]
    assert sanitize_neutral_sorani_text("") == ""
    assert sanitize_neutral_sorani_text(12345) == ""  # type: ignore[arg-type]
    assert sanitize_neutral_sorani_text(["invalid"]) == ""  # type: ignore[arg-type]
    assert sanitize_neutral_sorani_text({"bad": "dict"}) == ""  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_sanitizer_integrated_before_publish():
    # 1. Verify shared prompt policy in Gemini + OpenRouter + Translator prompts
    assert "neutral, fact-based Central Kurdish (Sorani)" in NEUTRAL_EDITORIAL_SYSTEM_INSTRUCTION
    assert NEUTRAL_EDITORIAL_SYSTEM_INSTRUCTION in GEMINI_TRANSLATION_SYSTEM_PROMPT
    assert NEUTRAL_EDITORIAL_SYSTEM_INSTRUCTION in GEMINI_KURDISH_REWRITE_SYSTEM_PROMPT
    assert resolve_gemini_system_instruction("") == NEUTRAL_EDITORIAL_SYSTEM_INSTRUCTION
    assert resolve_openrouter_system_prompt("") == NEUTRAL_EDITORIAL_SYSTEM_INSTRUCTION

    # 2. Verify Gemini -> OpenRouter failover sends editorial instruction in system role & sanitizes output
    import time

    gemini = GeminiClient(
        api_key="gemini-cooled-key",
        cooldown_store={"gemini-cooled-key": time.monotonic() + 120.0},
    )
    or_resp = AsyncMock()
    or_resp.status = 200
    or_resp.json = AsyncMock(
        return_value={
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "title": '🤔 "قەوارەی زایۆنی"!! دەستدرێژییەکی نوێ؟؟',
                                "summary": "🔥 سوپای قەوارەی زایۆنی و ئەمریکای دوژمن لە ناوچەکەدا ڕاگەیەندراوێکیان بڵاوکردەوە!!",
                                "event_key": "regional_alert",
                                "event_entities": ["Israel", "US"],
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }
    )
    mock_or_session = MagicMock()
    mock_or_session.closed = False
    mock_or_session.post = MagicMock(return_value=_MockAsyncResponseCtx(or_resp))
    or_client = OpenRouterClient(api_key="or-test-key", models=["m/free"], session=mock_or_session)
    gemini._openrouter_client = or_client

    provider = GeminiTranslationProvider(api_key="gemini-cooled-key")
    provider._client = gemini
    with patch.object(config, "ENABLE_OPENROUTER_FALLBACK", True), \
         patch.object(config, "OPENROUTER_API_KEY", "or-test-key"):
        title, summary = await provider.translate_article("Israel military statement", "Forces issued a statement in the region.")
    sent_payload = mock_or_session.post.call_args.kwargs["json"]
    assert sent_payload["messages"][0]["role"] == "system"
    assert NEUTRAL_EDITORIAL_SYSTEM_INSTRUCTION in sent_payload["messages"][0]["content"]
    assert "🤔" not in title and "!!" not in title and "؟؟" not in title
    assert "ئیسرائیل" in title
    assert "🔥" not in summary and "ئەمریکای دوژمن" not in summary and "ئەمریکا" in summary

    # 3. Verify ContentFormatter strips emojis/punctuation from headline & body while keeping UI badge
    formatter = ContentFormatter()
    dirty_article = Article(
        original_title="Raw title",
        translated_title='🚀 "ئیسرائیل" و شمال العراق!!',
        translated_summary="🤔 ئەمریکای دوژمن کۆبووەوە؟؟؟",
        source_name="Reuters",
        category="International Politics",
        original_url="https://reuters.com/world/item-1",
    )
    msg = formatter.format_telegram_message(dirty_article)
    assert "🚀" not in msg
    assert "🤔" not in msg
    assert "!!" not in msg
    assert "؟؟؟" not in msg
    assert '"ئیسرائیل"' not in msg
    assert "هەرێمی کوردستان" in msg

    # 4. Verify TelegramPublisher runs sanitizer immediately before Sorani gate & send_photo
    pub = TelegramPublisher()
    pub._bot = MagicMock()
    pub._bot.send_photo = AsyncMock(return_value=MagicMock(message_id=42))
    pub._card_generator = MagicMock()
    card_path = Path(BOT_ROOT / "data" / "test_dummy_card.jpg")
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_bytes(_make_valid_png_bytes())
    pub._card_generator.generate_card_file = AsyncMock(return_value=card_path)

    art = Article(
        original_title="Border news",
        translated_title='⚠️ قەوارەی زایۆنی!! لە شمال العراق',
        translated_summary='🚀 هێزەکانی ئەمریکای دوژمن و دەستدرێژییەکی نوێ لە سنوورەکاندا؟؟',
        source_name="Rudaw",
        category="Kurdistan",
        original_url="https://rudaw.net/test",
    )
    try:
        msg_id = await pub.publish_article(art)
    finally:
        card_path.unlink(missing_ok=True)

    assert msg_id == 42
    assert "⚠️" not in art.translated_title and "🚀" not in art.translated_summary
    sent_caption = pub._bot.send_photo.call_args.kwargs["caption"]
    assert "⚠️" not in sent_caption and "🚀" not in sent_caption
    assert "قەوارەی زایۆنی" not in sent_caption
    assert "هەرێمی کوردستان" in sent_caption
    assert "ئەمریکا" in sent_caption


def test_formatter_omits_duplicate_summary_on_short_breaking_dispatch():
    formatter = ContentFormatter()

    # Exact & near-identical short @AjaNews breaking post -> text appears ONLY ONCE (no repeated body paragraph)
    headline_txt = "ئایزنکۆت: ٨ ڕۆژی ماوە بۆ ئەرکی پێکهێنانی حکومەت، بەنیازم لەو ماوەیەدا حکومەت پێکبهێنم"
    near_identical_summary = "بەپەلە | ئایزنکۆت: ٨ ڕۆژی ماوە بۆ ئەرکی پێکهێنانی حکومەت، بەنیازم لەو ماوەیەدا حکومەت پێکبهێنم."
    assert _is_duplicate_summary(f"🔴 بەپەلە | {headline_txt}", near_identical_summary) is True
    assert _is_duplicate_summary(headline_txt, "٨ ڕۆژی ماوە بۆ ئەرکی پێکهێنانی حکومەت") is True

    short_breaking = Article(
        original_title="Eisencot: 8 days left to form government",
        translated_title=headline_txt,
        translated_summary=near_identical_summary,
        source_name="Al Jazeera",
        category="politics_world",
        original_url="https://t.me/AjaNews/7200",
        is_breaking=True,
        importance_score=9.2,
    )
    msg = formatter.format_telegram_message(short_breaking)
    caption = formatter.format_caption(short_breaking)
    assert msg.count(headline_txt) == 1
    assert caption.count(headline_txt) == 1
    assert "━━━━━━━━━━━━━━━" in msg
    assert "📂 سیاسەت و جیهان" in caption

    # Distinct headline + longer multi-sentence body -> BOTH headline and summary render normally
    normal_art = Article(
        original_title="Defense ministers hold security talks",
        translated_title="وەزیرانی بەرگری لەسەر ئاسایشی سنوورەکان کۆبوونەوە",
        translated_summary=(
            "لە کۆبوونەوەکەدا میکانیزمی نوێی هاوکاریی سەربازی و پاراستنی ئاسمانی ناوچەکە تاوتوێ کران. "
            "هەروەها هەردوولا ڕێککەوتن لەسەر ئاڵوگۆڕی زانیاریی هەواڵگری و بەهێزکردنی خاڵە سنوورییەکان."
        ),
        source_name="Reuters",
        category="military_conflict",
        original_url="https://reuters.com/world/defense-talks",
        is_breaking=False,
    )
    assert _is_duplicate_summary(normal_art.translated_title, normal_art.translated_summary) is False
    normal_msg = formatter.format_telegram_message(normal_art)
    normal_cap = formatter.format_caption(normal_art)
    assert normal_art.translated_title in normal_msg
    assert "میکانیزمی نوێی هاوکاریی سەربازی" in normal_msg
    assert normal_art.translated_title in normal_cap
    assert "میکانیزمی نوێی هاوکاریی سەربازی" in normal_cap


def test_normalize_sorani_orthography_and_preserve_verified_orgs():
    from processing.content_formatter import normalize_sorani_orthography

    # 1. Broken Arabic-keyboard 'ه ' spacing & lexical MT artifacts fixed cleanly
    assert normalize_sorani_orthography("وه زیری ده ره وه ی") == "وەزیری دەرەوەی"
    assert normalize_sorani_orthography("به رگری") == "بەرگری"
    assert normalize_sorani_orthography("به رده وامی") == "بەردەوامی"
    assert normalize_sorani_orthography("پشتراست کرده وه") == "پشتڕاست کردەوە"

    # Real word-initial 'ه' (\u0647) in Kurdistan words remains completely untouched
    authentic = "هێزەکانی پێشمەرگە لە هەولێر و بەغدا کۆبوونەوە."
    assert normalize_sorani_orthography(authentic) == authentic

    # 2. Stray hashtags stripped cleanly & verified organization names stay intact and natural
    hashtagged = "#حووسییەکان لە #سەعدە و #حیزبوڵڵا و #حەماس و #ئەنساروڵڵا ڕاگەیەندراویان بڵاوکردەوە."
    normalized_orgs = sanitize_neutral_sorani_text(hashtagged)
    assert "#حووسییەکان" not in normalized_orgs
    assert "#سەعدە" not in normalized_orgs
    for org in ("حووسییەکان", "سەعدە", "حیزبوڵڵا", "حەماس", "ئەنساروڵڵا"):
        assert org in normalized_orgs

    # Partisan/propaganda slurs stripped while verified organizations remain natural
    slurred = "ڕێکخراوی تیرۆریستی حیزبوڵڵا و ئەمریکای دوژمن لەسەر ناوچەکە قسەیان کرد."
    clean_slurred = sanitize_neutral_sorani_text(slurred)
    assert "حیزبوڵڵا" in clean_slurred
    assert "تیرۆریست" not in clean_slurred
    assert "ئەمریکای دوژمن" not in clean_slurred
    assert "ئەمریکا" in clean_slurred


@pytest.mark.asyncio
async def test_native_sorani_posts_bypass_all_translation_and_foreign_languages_translate(tmp_path):
    from core.importance_ranker import RelevanceEvaluation
    from core.pipeline import ArticlePipeline
    from monitoring.health import HealthMonitor
    from monitoring.metrics import Metrics
    from processing.deduplicator import SemanticDeduplicator
    from processing.translator import is_native_sorani_article, is_valid_kurdish_sorani
    from storage.database import Database

    # Verify Persian, Arabic, and English do NOT falsely pass native Sorani gate
    assert is_native_sorani_article("سەردانی وەزیرانی کوردستان بۆ هەولێر", "لە هەولێر کۆبوونەوەیەکی ئەمنی ئەنجامدرا.") is True
    assert is_native_sorani_article("فوری | سپاه پاسداران از حمله موشکی خبر داد", "نیروهای مسلح ایران در بیانیه ای اعلام کردند عملیات نظامی ادامه دارد.") is False
    assert is_valid_kurdish_sorani("گزارش نیروهای مسلح ایران درباره حمله موشکی در منطقه") is False
    assert is_native_sorani_article("عاجل | وزارة الدفاع العراقية تعلن تعزيز الانتشار الأمني") is False
    assert is_native_sorani_article("Defense ministers hold urgent talks in Baghdad") is False

    # 1. TranslationEngine bypasses Gemini, OpenRouter, and Google Translate completely on native Sorani (0 calls)
    engine = TranslationEngine(gemini_api_key="gem-live-key", delay=0.0)
    with patch.object(engine._gemini_provider, "translate_article", new_callable=AsyncMock) as mock_gem, \
         patch.object(engine._google_fallback, "translate", new_callable=AsyncMock) as mock_google, \
         patch.object(OpenRouterClient, "generate_json", new_callable=AsyncMock) as mock_or:
        for src_title, src_sum in [
            (
                "🔥 هێزەکانی پێشمەرگە لە هەولێر ئۆپەراسیۆنیان ئەنجامدا!!",
                "هێزەکانی پێشمەرگە و سوپای عێراق لە ناوچە سنوورییەکانی کەرکووک هەماهەنگیی سەربازییان زیاد کرد. #هەولێر",
            ),
            (
                "حکوومەتی هەرێمی کوردستان پڕۆژەیاسای بودجەی پەسەند کرد",
                "لە دانیشتنی ئەمڕۆدا پرسی ناردنی بەشەبودجە و مووچەی فەرمانبەران تاوتوێ کرا.",
            ),
        ]:
            tr_t, tr_s = await engine.translate_article(src_title, src_sum)
            assert "🔥" not in tr_t and "!!" not in tr_t
            assert "#هەولێر" not in tr_s
            assert "پێشمەرگە" in tr_t or "حکوومەتی هەرێمی کوردستان" in tr_t

        mock_gem.assert_not_awaited()
        mock_google.assert_not_awaited()
        mock_or.assert_not_awaited()

    # 2. Foreign language posts (Arabic, Persian, English) still translate & normalize broken fallback orthography
    engine_fallback = TranslationEngine(gemini_api_key="", delay=0.0)
    with patch.object(
        engine_fallback._google_fallback,
        "translate",
        new_callable=AsyncMock,
        side_effect=[
            "وه زیری ده ره وه ی ئێران هێرشه که ی پشتراست کرده وه",
            "وه زاره تی به رگری رایگه یاند که هێزه کان له به رده وامی دان.",
            "بەرپرسانی سەربازی لە #سەعدە وەڵامیان دایەوە",
            "حیزبوڵڵا و ئەمریکای دوژمن لە ناوچەکەدا هەڵوێستیان ڕاگەیاند.",
        ],
    ) as mock_google_fb:
        fb_t1, fb_s1 = await engine_fallback.translate_article(
            "عاجل | وزير الخارجية الإيراني يؤكد الهجوم",
            "أعلنت وزارة الدفاع استمرار العمليات العسكرية في المنطقة.",
        )
        assert fb_t1 == "وەزیری دەرەوەی ئێران هێرشەکەی پشتڕاست کردەوە"
        assert "وەزارەتی بەرگری" in fb_s1 and "بەردەوامی" in fb_s1

        fb_t2, fb_s2 = await engine_fallback.translate_article(
            "گزارش فوری از صعده درباره تحولات نظامی",
            "نیروهای مسلح بیانیه ای درباره حوادث منطقه صادر کردند.",
        )
        assert "#سەعدە" not in fb_t2 and "سەعدە" in fb_t2
        assert "حیزبوڵڵا" in fb_s2
        assert "ئەمریکای دوژمن" not in fb_s2 and "ئەمریکا" in fb_s2
        assert mock_google_fb.await_count == 4

    # 3. ArticlePipeline.prepare_article_for_publishing bypasses _translator completely on Rudaw/K24/NRT native Sorani
    db = Database(str(tmp_path / "native_sorani_pipe.db"))
    await db.connect()
    try:
        pub = TelegramPublisher()
        pub._channel_id = "@KurdishGlobalNews"
        pub._bot = AsyncMock()
        pipeline = ArticlePipeline(db, pub, HealthMonitor(), Metrics())
        pipeline._semantic_dedup = SemanticDeduplicator(db, api_key="", base_backoff=0.0)

        now_iso = datetime.now(timezone.utc).isoformat()
        k24_art = Article(
            original_url="https://t.me/Kurdistan24news/4100",
            canonical_url="https://t.me/Kurdistan24news/4100",
            original_title="🚀 وەزارەتی پێشمەرگە هێزەکانی سنووری هەولێری بەهێز کرد!!",
            normalized_title="وەزارەتی پێشمەرگە هێزەکانی سنووری هەولێری بەهێز کرد",
            original_summary="🔴 بەپەلە | وەزارەتی پێشمەرگە ڕایگەیاند هێزەکانی لە ناوچە سنوورییەکانی هەولێر جێگیر کران. #کوردستان",
            source_name="Kurdistan24",
            category="kurdistan",
            region="kurdistan_bashur",
            importance_score=8.8,
            publication_date=now_iso,
            discovered_at=now_iso,
        )
        k24_art.compute_content_hash()
        k24_art.compute_event_fingerprint()
        k24_art.id = await db.insert_article(k24_art)

        pipeline._extractor.fetch_article_content_and_image = AsyncMock(return_value=("", None))
        pipeline._translator.translate_article = AsyncMock(
            side_effect=AssertionError("Translator MUST NOT be called for native Sorani articles!")
        )
        ok = await pipeline.prepare_article_for_publishing(k24_art)
        assert ok is True
        pipeline._translator.translate_article.assert_not_called()
        assert "🚀" not in (k24_art.translated_title or "")
        assert "!!" not in (k24_art.translated_title or "")
        assert "پێشمەرگە" in (k24_art.translated_title or "")
    finally:
        await db.close()


def test_source_prioritization_and_noisy_western_feed_demotion():
    from config import RSS_FEEDS, SOURCE_CREDIBILITY, TELEGRAM_SOURCE_CHANNELS
    from core.balancer import select_paced_batch
    from core.importance_ranker import article_selection_sort_key, get_source_priority_rank

    tg_users = {ch["username"] for ch in TELEGRAM_SOURCE_CHANNELS}
    assert "rudawdigital" not in tg_users
    assert {"AjaNews", "Kurdistan24news"}.issubset(tg_users)

    noisy = {
        "The Guardian World", "DW News", "France24 EN", "CNBC World", "BleepingComputer",
        "BBC World", "BBC Technology", "AP News", "NPR News", "UN News", "Military Times",
    }
    enabled_names = {f["name"] for f in RSS_FEEDS if f.get("enabled", True)}
    assert noisy.isdisjoint(enabled_names)
    assert {"Al Jazeera", "Rudaw Sorani", "Kurdistan24 Sorani", "Shafaq News", "Reuters World"}.issubset(enabled_names)

    for hi in ("Al Jazeera", "Rudaw", "Kurdistan24", "NRT", "Shafaq News"):
        assert SOURCE_CREDIBILITY[hi.lower()] >= 0.90
        assert get_source_priority_rank(hi) == 0

    for lo in ("The Guardian", "France24", "DW", "CNBC", "BleepingComputer"):
        assert SOURCE_CREDIBILITY[lo.lower()] <= 0.45
        assert get_source_priority_rank(lo) == 2

    now_iso = datetime.now(timezone.utc).isoformat()
    aja = Article(
        id=1,
        original_url="https://t.me/AjaNews/8001",
        canonical_url="https://t.me/AjaNews/8001",
        original_title="Baghdad security command strengthens air defense in Iraq",
        source_name="Al Jazeera",
        category="military_conflict",
        region="middle_east",
        importance_score=8.1,
        publication_date=now_iso,
    )
    rudaw = Article(
        id=2,
        original_url="https://t.me/rudawdigital/8002",
        canonical_url="https://t.me/rudawdigital/8002",
        original_title="Erbil KRG parliament discusses border security agreement",
        source_name="Rudaw",
        category="kurdistan",
        region="kurdistan_bashur",
        importance_score=8.0,
        publication_date=now_iso,
    )
    guardian = Article(
        id=3,
        original_url="https://www.theguardian.com/world/2026/nov/27/eu-iraq-talks",
        canonical_url="https://www.theguardian.com/world/2026/nov/27/eu-iraq-talks",
        original_title="European ministers discuss Iraq diplomacy",
        source_name="The Guardian World",
        category="politics_world",
        region="middle_east",
        importance_score=8.7,
        publication_date=now_iso,
    )
    ranked = sorted([guardian, aja, rudaw], key=article_selection_sort_key)
    assert ranked[0] is rudaw
    assert ranked[1] is aja
    assert ranked[2] is guardian

    res = select_paced_batch([guardian, aja], target_posts=1, source_mix={"telegram": 1, "x": 0, "search": 0})
    assert res.selected == [aja]


@pytest.mark.asyncio
async def test_telegram_k24_image_url_preserved_through_pipeline_to_publisher(tmp_path):
    from core.importance_ranker import RelevanceEvaluation
    from core.pipeline import ArticlePipeline
    from monitoring.health import HealthMonitor
    from monitoring.metrics import Metrics
    from processing.deduplicator import SemanticDeduplicator
    from storage.database import Database

    db = Database(str(tmp_path / "k24_img_passthrough.db"))
    await db.connect()
    try:
        pub = TelegramPublisher()
        pub._channel_id = "@KurdishGlobalNews"
        pub._bot = AsyncMock()
        pub._bot.send_photo = AsyncMock(return_value=MagicMock(message_id=9191))

        valid_png = _make_valid_png_bytes(64, 64)
        fake_card = tmp_path / "k24_card.jpg"
        fake_card.write_bytes(b"JPG")
        pub._card_generator.generate_card_file = AsyncMock(return_value=fake_card)

        pipeline = ArticlePipeline(db, pub, HealthMonitor(), Metrics())
        pipeline._semantic_dedup = SemanticDeduplicator(db, api_key="", base_backoff=0.0)
        pipeline._ranker.evaluate_global_relevance = AsyncMock(
            return_value=RelevanceEvaluation(
                score=8,
                is_breaking=False,
                category="kurdistan",
                reason="Kurdistan security meeting",
                passed=True,
            )
        )
        pipeline._extractor.fetch_article_content_and_image = AsyncMock(
            return_value=("", "https://cdn4.telegram-cdn.org/file/ignored_og_preview.jpg")
        )

        # 1. Kurdistan24news post with extracted Telegram CDN photo -> preserved into _download_telegram_cdn_image & card_generator
        cdn_photo_url = "https://cdn4.telegram-cdn.org/file/k24_real_photo_777.jpg"
        now_iso = datetime.now(timezone.utc).isoformat()
        k24_with_photo = Article(
            original_url="https://t.me/Kurdistan24news/777",
            canonical_url="https://t.me/Kurdistan24news/777",
            original_title="هێزەکانی پێشمەرگە لە هەولێر کۆبوونەوەیەکی ئەمنییان ئەنجامدا",
            normalized_title="هێزەکانی پێشمەرگە لە هەولێر کۆبوونەوەیەکی ئەمنییان ئەنجامدا",
            original_summary="لە کۆبوونەوەکەدا پلانی پاراستنی سنوورەکانی هەرێمی کوردستان تاوتوێ کرا.",
            source_name="Kurdistan24",
            category="kurdistan",
            region="kurdistan_bashur",
            image_url=cdn_photo_url,
            publication_date=now_iso,
            discovered_at=now_iso,
        )
        k24_with_photo.compute_content_hash()
        k24_with_photo.compute_event_fingerprint()
        k24_with_photo.id = await db.insert_article(k24_with_photo)

        with patch.object(
            pub, "_download_telegram_cdn_image", new_callable=AsyncMock, return_value=valid_png
        ) as mock_cdn_dl:
            ok = await pipeline.process_article(k24_with_photo)

        assert ok is True
        assert k24_with_photo.image_url == cdn_photo_url
        mock_cdn_dl.assert_awaited_once()
        assert mock_cdn_dl.call_args.args[0] == cdn_photo_url
        assert pub._card_generator.generate_card_file.call_args.kwargs["image_bytes"] == valid_png

        # 2. Kurdistan24news post without extracted photo -> stays empty (no og:image leak) & falls back to luxury background
        pub._card_generator.generate_card_file.reset_mock()
        fake_card.write_bytes(b"JPG")
        k24_no_photo = Article(
            original_url="https://t.me/Kurdistan24news/778",
            canonical_url="https://t.me/Kurdistan24news/778",
            original_title="حکوومەتی هەرێمی کوردستان پڕۆژەیەکی نوێی سەربازی پەسەند کرد",
            normalized_title="حکوومەتی هەرێمی کوردستان پڕۆژەیەکی نوێی سەربازی پەسەند کرد",
            original_summary="وەزارەتی پێشمەرگە وردەکاریی بڕیارەکەی بۆ ڕێکخستنەوەی هێزەکان بڵاوکردەوە.",
            source_name="Kurdistan24",
            category="kurdistan",
            region="kurdistan_bashur",
            image_url="",
            publication_date=now_iso,
            discovered_at=now_iso,
        )
        k24_no_photo.compute_content_hash()
        k24_no_photo.compute_event_fingerprint()
        k24_no_photo.id = await db.insert_article(k24_no_photo)

        ok_no_photo = await pipeline.process_article(k24_no_photo)
        assert ok_no_photo is True
        assert k24_no_photo.image_url == ""
        assert pub._card_generator.generate_card_file.call_args.kwargs["image_bytes"] is None
    finally:
        await db.close()


# ═══════════════════════════════════════════════════════════════════════════
# 3.2 TELEGRAM MEDIA EXTRACTION & CDN DOWNLOAD TESTS
# ═══════════════════════════════════════════════════════════════════════════


def test_telegram_single_photo_background_image():
    html = """
    <div class="tgme_widget_message" data-post="chan/101">
      <div class="tgme_widget_message_photo_wrap"
           style="background-image:url('https://cdn.example/image.jpg')">
      </div>
    </div>
    """
    assert extract_background_image("background-image:url('https://cdn.example/image.jpg')") == "https://cdn.example/image.jpg"
    candidates = extract_telegram_image_candidates(html)
    assert candidates == ["https://cdn.example/image.jpg"]


def test_telegram_grouped_album_multiple_photos():
    html = """
    <div class="tgme_widget_message" data-post="chan/201">
      <div class="tgme_widget_message_grouped_layer">
        <div class="tgme_widget_message_photo_wrap"
             style="background-image:url('https://cdn.example/a.jpg')">
        </div>
        <div class="tgme_widget_message_photo_wrap"
             style="background-image:url('https://cdn.example/b.jpg')">
        </div>
      </div>
    </div>
    <div class="tgme_widget_message" data-post="chan/202">
      <div class="tgme_widget_message_photo_wrap"
           style="background-image:url('https://cdn.example/unrelated.jpg')">
      </div>
    </div>
    """
    candidates = extract_telegram_image_candidates(html)
    assert "https://cdn.example/a.jpg" in candidates
    assert "https://cdn.example/b.jpg" in candidates

    # Scoped per message in parse_channel_html so unrelated message images never mix
    now_iso = datetime.now(timezone.utc).isoformat()
    channel_html = f"""
    <div class="tgme_widget_message" data-post="chan/201">
      <a class="tgme_widget_message_date" href="https://t.me/chan/201"><time datetime="{now_iso}"></time></a>
      <div class="tgme_widget_message_grouped_layer">
        <div class="tgme_widget_message_photo_wrap" style="background-image:url('https://cdn.example/a.jpg')"></div>
        <div class="tgme_widget_message_photo_wrap" style="background-image:url('https://cdn.example/b.jpg')"></div>
      </div>
      <div class="tgme_widget_message_text">هەواڵی ئەلبوومی یەکەم لە بەغداوە بڵاوکرایەوە بۆ ڕای گشتی.</div>
    </div>
    <div class="tgme_widget_message" data-post="chan/202">
      <a class="tgme_widget_message_date" href="https://t.me/chan/202"><time datetime="{now_iso}"></time></a>
      <div class="tgme_widget_message_photo_wrap" style="background-image:url('https://cdn.example/unrelated.jpg')"></div>
      <div class="tgme_widget_message_text">هەواڵی دووەم لە هەولێرەوە سەبارەت بە ئاسایشی ناوچەکە.</div>
    </div>
    """
    src = TelegramSource()
    parsed = src.parse_channel_html(channel_html, {"username": "chan", "name": "Chan"})
    assert len(parsed) == 2
    assert parsed[0].image_url == "https://cdn.example/a.jpg"
    assert parsed[1].image_url == "https://cdn.example/unrelated.jpg"


def test_telegram_link_preview_image():
    html = """
    <div class="tgme_widget_message_link_preview">
      <img class="link_preview_image"
           src="https://cdn.example/preview.jpg">
    </div>
    """
    classified = classify_telegram_media(html)
    assert classified["link_previews"] == ["https://cdn.example/preview.jpg"]
    assert extract_telegram_image_candidates(html) == ["https://cdn.example/preview.jpg"]


def test_telegram_video_thumbnail():
    html = """
    <div class="tgme_widget_message" data-post="chan/301">
      <div class="tgme_widget_message_video_player">
        <i class="tgme_widget_message_video_thumb"
           style="background-image:url('https://cdn.example/vthumb.jpg')"></i>
      </div>
    </div>
    """
    classified = classify_telegram_media(html)
    assert classified["video_thumbnails"] == ["https://cdn.example/vthumb.jpg"]
    assert classified["photos"] == []
    assert extract_telegram_image_candidates(html) == []

    # Without a playable video URL, a standalone video thumbnail must NOT be published as an original photo
    now_iso = datetime.now(timezone.utc).isoformat()
    msg_html = f"""
    <div class="tgme_widget_message" data-post="chan/301">
      <a class="tgme_widget_message_date" href="https://t.me/chan/301"><time datetime="{now_iso}"></time></a>
      <div class="tgme_widget_message_video_player">
        <i class="tgme_widget_message_video_thumb" style="background-image:url('https://cdn.example/vthumb.jpg')"></i>
      </div>
      <div class="tgme_widget_message_text">Media is too big? No, Kurdish military deployment update from Kirkuk.</div>
    </div>
    """
    articles = TelegramSource().parse_channel_html(msg_html, {"username": "chan", "name": "Chan"})
    assert len(articles) == 1
    assert articles[0].image_url is None


def test_telegram_og_image_fallback():
    html = """
    <html><head>
      <meta property="og:image" content="https://cdn.example/og.jpg">
    </head><body></body></html>
    """
    classified = classify_telegram_media(html)
    assert classified["og_images"] == ["https://cdn.example/og.jpg"]
    assert extract_telegram_image_candidates(html) == ["https://cdn.example/og.jpg"]


@pytest.mark.asyncio
async def test_telegram_embed_single_fallback():
    now_iso = datetime.now(timezone.utc).isoformat()
    channel_html = f"""
    <div class="tgme_widget_message" data-post="rudawdigital/401">
      <a class="tgme_widget_message_date" href="https://t.me/rudawdigital/401"><time datetime="{now_iso}"></time></a>
      <div class="tgme_widget_message_text">ڕاپۆرتی سەربازی لە سنوورەکان بڵاوکرایەوە و هێزەکان چاودێری دەکەن.</div>
    </div>
    """
    embed_html = """
    <html><head><meta property="og:image" content="https://cdn.example/embed_og.jpg"></head>
    <body>
      <div class="tgme_widget_message_photo_wrap" style="background-image:url('https://cdn.example/embed_photo.jpg')"></div>
    </body></html>
    """
    calls: list[tuple[str, dict]] = []

    class _FakeResp:
        def __init__(self, status: int, body: str):
            self.status = status
            self._body = body
            self.content = None

        async def text(self, errors: str = "replace") -> str:
            return self._body

    def fake_get(url: str, headers=None, timeout=None):
        calls.append((url, dict(headers or {})))
        if "embed=1&single=1" in url:
            return _MockAsyncResponseCtx(_FakeResp(200, embed_html))
        return _MockAsyncResponseCtx(_FakeResp(200, channel_html))

    src = TelegramSource(channels=[{"username": "rudawdigital", "name": "Rudaw"}])
    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.get = MagicMock(side_effect=fake_get)
    src._session = mock_session

    articles = await src.fetch_channel({"username": "rudawdigital", "name": "Rudaw"})
    assert len(articles) == 1
    assert articles[0].image_url == "https://cdn.example/embed_photo.jpg"
    embed_calls = [c for c in calls if "embed=1&single=1" in c[0]]
    assert len(embed_calls) == 1
    assert embed_calls[0][0] == "https://t.me/rudawdigital/401?embed=1&single=1"
    assert embed_calls[0][1]["User-Agent"] == TG_CHROME_UA
    assert embed_calls[0][1]["Referer"] == "https://t.me/"
    assert embed_calls[0][1]["Accept-Language"] == "en-US,en;q=0.9"


def test_telegram_protocol_relative_url():
    html = """
    <div class="tgme_widget_message_photo_wrap"
         style="background-image:url('//cdn4.telegram-cdn.org/file/proto.jpg?token=abc')">
    </div>
    <div class="tgme_widget_message_link_preview">
      <img class="link_preview_image" src="/media/relative.jpg?v=2">
    </div>
    """
    candidates = extract_telegram_image_candidates(html, base_url="https://t.me/rudawdigital/10")
    assert "https://cdn4.telegram-cdn.org/file/proto.jpg?token=abc" in candidates
    assert "https://t.me/media/relative.jpg?v=2" in candidates


def test_telegram_srcset_selection():
    html = """
    <div class="tgme_widget_message_photo_wrap">
      <img src="https://cdn.example/small.jpg"
           srcset="https://cdn.example/low.jpg 320w, https://cdn.example/high.jpg 1280w, https://cdn.example/med.jpg 640w">
    </div>
    """
    candidates = extract_telegram_image_candidates(html)
    assert candidates[0] == "https://cdn.example/high.jpg"


def test_telegram_invalid_url_rejection():
    html = """
    <div class="tgme_widget_message_photo_wrap" style="background-image:url('javascript:alert(1)')"></div>
    <div class="tgme_widget_message_photo_wrap" style="background-image:url('data:image/png;base64,AAA')"></div>
    <div class="tgme_widget_message_link_preview">
      <img class="link_preview_image" src="vbscript:msgbox(1)">
      <img class="link_preview_image" src="https://bad url with spaces.com/a.jpg">
    </div>
    """
    assert extract_background_image("background-image:url('javascript:alert(1)')") is None
    assert extract_background_image("background-image:url('data:image/png;base64,BBB')") is None
    assert extract_telegram_image_candidates(html) == []


def test_telegram_duplicate_image_removal():
    html = """
    <meta property="og:image" content="https://cdn.example/dup.jpg">
    <div class="tgme_widget_message_grouped_layer">
      <div class="tgme_widget_message_photo_wrap" style="background-image:url('https://cdn.example/dup.jpg')"></div>
      <div class="tgme_widget_message_photo_wrap" style="background-image:url('https://cdn.example/second.jpg')"></div>
    </div>
    <div class="tgme_widget_message_link_preview">
      <img class="link_preview_image" src="https://cdn.example/dup.jpg">
    </div>
    """
    candidates = extract_telegram_image_candidates(html)
    assert candidates == ["https://cdn.example/dup.jpg", "https://cdn.example/second.jpg"]


@pytest.mark.asyncio
async def test_telegram_cdn_download_headers():
    png_bytes = _make_valid_png_bytes()
    captured: dict[str, dict[str, str]] = {}

    class _FakeHttpxResp:
        status_code = 200
        headers = {"content-type": "image/png", "content-length": str(len(png_bytes))}
        content = png_bytes

    async def fake_get(url: str, headers=None):
        captured["headers"] = dict(headers or {})
        return _FakeHttpxResp()

    mock_client = MagicMock()
    mock_client.get = AsyncMock(side_effect=fake_get)
    mock_client.aclose = AsyncMock()

    pub = TelegramPublisher()
    data = await pub._download_telegram_cdn_image(
        "https://cdn4.telegram-cdn.org/file/photo.png",
        job_id="job_11",
        client=mock_client,
    )
    assert data == png_bytes
    assert captured["headers"] == TELEGRAM_CDN_DOWNLOAD_HEADERS
    assert captured["headers"]["Referer"] == "https://t.me/"
    assert captured["headers"]["Accept-Language"] == "en-US,en;q=0.9"
    assert "Chrome/117.0.0.0" in captured["headers"]["User-Agent"]

    # Also verify _publish_with_card routes Telegram CDN URLs through _download_telegram_cdn_image (never bare ImageHandler)
    captured.clear()
    pub._bot = MagicMock()
    pub._bot.send_photo = AsyncMock(return_value=MagicMock(message_id=77))
    pub._card_generator = MagicMock()
    tmp_card = Path(BOT_ROOT / "data" / "test_tg_cdn_card.jpg")
    tmp_card.parent.mkdir(parents=True, exist_ok=True)
    tmp_card.write_bytes(png_bytes)
    pub._card_generator.generate_card_file = AsyncMock(return_value=tmp_card)

    art = Article(
        id=99,
        original_title="Defense news",
        translated_title="هەواڵی بەرگری لە هەولێر",
        translated_summary="هێزەکان جموجۆڵی نوێیان لە سنوورەکاندا ئەنجام دا.",
        source_name="Official Source",  # skip corner watermark check
        category="Kurdistan",
        original_url="https://reuters.com/test-99",
        image_url="https://cdn4.telegram-cdn.org/file/photo.png",
    )
    with patch("tg_bot.publisher.is_official_media_source", return_value=True), \
         patch("tg_bot.publisher.httpx.AsyncClient", return_value=mock_client):
        msg_id = await pub.publish_article(art)
    tmp_card.unlink(missing_ok=True)
    assert msg_id == 77
    assert captured["headers"] == TELEGRAM_CDN_DOWNLOAD_HEADERS
    assert pub._card_generator.generate_card_file.call_args.kwargs["image_bytes"] == png_bytes


@pytest.mark.asyncio
async def test_telegram_html_error_page_rejection():
    class _FakeHtmlErrorResp:
        status_code = 200
        headers = {"content-type": "image/jpeg"}  # Disguised HTML error page
        content = b"<!DOCTYPE html><html><body><h1>403 Forbidden</h1></body></html>"

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=_FakeHtmlErrorResp())
    mock_client.aclose = AsyncMock()

    pub = TelegramPublisher()
    res = await pub._download_telegram_cdn_image(
        "https://cdn4.telegram-cdn.org/file/disguised.jpg",
        client=mock_client,
    )
    assert res is None

    # Also reject explicit text/html content-type
    class _FakeTextHtmlResp:
        status_code = 200
        headers = {"content-type": "text/html; charset=utf-8"}
        content = _make_valid_png_bytes()

    mock_client.get = AsyncMock(return_value=_FakeTextHtmlResp())
    assert await pub._download_telegram_cdn_image("https://cdn4.telegram-cdn.org/file/x.jpg", client=mock_client) is None


# ═══════════════════════════════════════════════════════════════════════════
# 3.3 RESILIENT X / TWITTER INGESTION TESTS
# ═══════════════════════════════════════════════════════════════════════════


def _sample_x_accounts(count: int = 16) -> list[dict]:
    if count == 16 and len(config.X_VIDEO_SOURCE_ACCOUNTS) == 16:
        return list(config.X_VIDEO_SOURCE_ACCOUNTS)
    return [
        {
            "handle": f"acct_{i}",
            "display_name": f"Account {i}",
            "region": "middle_east" if i < 6 else "global",
            "lang": "en",
            "credibility_tier": 1 if i == 0 else 2,
            "source_group": f"grp_{i}",
            "is_official_media": i < 4,
        }
        for i in range(count)
    ]


def _build_syndication_html(handle: str, status_id: str, text: str, created_at: str) -> str:
    payload = {
        "props": {
            "pageProps": {
                "timeline": {
                    "entries": [
                        {
                            "content": {
                                "tweet": {
                                    "id_str": status_id,
                                    "created_at": created_at,
                                    "full_text": text,
                                    "user": {"screen_name": handle},
                                }
                            }
                        }
                    ]
                }
            }
        }
    }
    return f'<html><body><script id="__NEXT_DATA__" type="application/json">{json.dumps(payload)}</script></body></html>'


def test_x_rotating_subset():
    accounts = _sample_x_accounts(16)
    src = XVideoSource(accounts=accounts, batch_size=5)
    b1 = [a["handle"] for a in src.select_account_batch()]
    b2 = [a["handle"] for a in src.select_account_batch()]
    b3 = [a["handle"] for a in src.select_account_batch()]
    assert len(b1) == len(b2) == len(b3) == 5
    assert set(b1).isdisjoint(set(b2))
    assert set(b2).isdisjoint(set(b3))


def test_x_rotation_wraparound():
    accounts = [
        {"handle": "A", "region": "middle_east"},
        {"handle": "B", "region": "middle_east"},
        {"handle": "C", "region": "middle_east"},
        {"handle": "D", "region": "middle_east"},
        {"handle": "E", "region": "middle_east"},
        {"handle": "F", "region": "middle_east"},
    ]
    src = XVideoSource(accounts=accounts, batch_size=4)
    b1 = [a["handle"] for a in src.select_account_batch()]
    b2 = [a["handle"] for a in src.select_account_batch()]
    assert b1 == ["A", "B", "C", "D"]
    assert b2 == ["E", "F", "A", "B"]
    assert len(b2) == len(set(b2)) == 4


def test_x_priority_accounts_preserved():
    mixed = [
        {"handle": "Global1", "region": "global", "credibility_tier": 2, "is_official_media": False},
        {"handle": "KurdOfficial", "region": "kurdistan_bashur", "credibility_tier": 2, "is_official_media": True},
        {"handle": "ME_Official", "region": "middle_east", "credibility_tier": 1, "is_official_media": True},
        {"handle": "Global2", "region": "global", "credibility_tier": 2, "is_official_media": True},
    ]
    src = XVideoSource(accounts=mixed, batch_size=2)
    b1 = [a["handle"] for a in src.select_account_batch()]
    assert b1 == ["KurdOfficial", "ME_Official"]

    # Also verify priority leads inside real 16-account default config
    default_src = XVideoSource(batch_size=5)
    first_batch = default_src.select_account_batch()
    assert all(a["region"] in ("kurdistan_bashur", "middle_east") for a in first_batch)


@pytest.mark.asyncio
async def test_x_requests_are_sequential():
    accounts = _sample_x_accounts(3)
    in_flight = 0
    max_concurrent = 0
    seen_headers: list[dict] = []
    fresh_created = (datetime.now(timezone.utc) - timedelta(minutes=10)).strftime("%a %b %d %H:%M:%S +0000 %Y")

    async def fake_sleep(_s: float) -> None:
        return None

    src = XVideoSource(
        accounts=accounts,
        mirrors=[],
        batch_size=3,
        sleep_fn=fake_sleep,
        rand_uniform_fn=lambda a, b: 1.8,
    )

    class _SeqResp:
        status = 200
        headers = {"content-type": "text/html"}

        def __init__(self, handle: str, idx: int):
            self._html = _build_syndication_html(handle, f"440{idx}", f"Military update {idx}", fresh_created)

        async def text(self, errors: str = "replace") -> str:
            nonlocal in_flight, max_concurrent
            in_flight += 1
            max_concurrent = max(max_concurrent, in_flight)
            await asyncio.sleep(0)
            in_flight -= 1
            return self._html

    call_idx = [0]

    def fake_get(url: str, headers=None, allow_redirects=True, timeout=None):
        seen_headers.append(dict(headers or {}))
        handle = url.rstrip("/").rsplit("/", 1)[-1]
        idx = call_idx[0]
        call_idx[0] += 1
        return _MockAsyncResponseCtx(_SeqResp(handle, idx))

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.get = MagicMock(side_effect=fake_get)
    src._session = mock_session
    src._resolve_video_metadata = AsyncMock(  # type: ignore[method-assign]
        return_value={"video_url": "https://video.twimg.com/seq.mp4", "duration": 20, "width": 720, "text": "Military update"}
    )

    articles = await src.fetch_articles()
    assert len(articles) == 3
    assert max_concurrent == 1
    assert len(seen_headers) == 3
    for hdrs in seen_headers:
        assert hdrs == SYNDICATION_HEADERS


@pytest.mark.asyncio
async def test_x_jitter_is_called():
    accounts = _sample_x_accounts(4)
    rand_calls: list[tuple[float, float]] = []
    sleep_calls: list[float] = []

    def fake_rand(lo: float, hi: float) -> float:
        rand_calls.append((lo, hi))
        return 2.1

    async def fake_sleep(sec: float) -> None:
        sleep_calls.append(sec)

    src = XVideoSource(
        accounts=accounts,
        mirrors=[],
        batch_size=4,
        jitter_min=1.5,
        jitter_max=2.5,
        rand_uniform_fn=fake_rand,
        sleep_fn=fake_sleep,
    )
    src._http_get_with_status = AsyncMock(return_value=(200, "<html></html>", {}))  # type: ignore[method-assign]
    src._ddg_fallback_discover = AsyncMock(return_value=[])  # type: ignore[method-assign]

    await src.fetch_articles()
    # Jitter called cleanly between requests (3 sleeps for 4 sequential accounts)
    assert len(rand_calls) == 3
    assert all(lo == 1.5 and hi == 2.5 for lo, hi in rand_calls)
    assert sleep_calls == [2.1, 2.1, 2.1]


@pytest.mark.asyncio
async def test_x_429_trips_breaker():
    clock = [100.0]
    src = XVideoSource(
        accounts=_sample_x_accounts(2),
        mirrors=[],
        cooldown_seconds=600,
        monotonic_fn=lambda: clock[0],
    )
    src._http_get_with_status = AsyncMock(return_value=(429, None, {}))  # type: ignore[method-assign]

    assert src.is_syndication_cooldown_active() is False
    entries, ok = await src.fetch_channel_syndication(src._accounts[0])
    assert entries == [] and ok is False
    assert src.is_syndication_cooldown_active() is True
    assert src._syndication_degraded is True
    assert src._syndication_cooldown_until == pytest.approx(700.0)


@pytest.mark.asyncio
async def test_x_429_stops_remaining_syndication_requests():
    accounts = _sample_x_accounts(5)
    syn_urls: list[str] = []

    async def fake_sleep(_s: float) -> None:
        return None

    src = XVideoSource(
        accounts=accounts,
        mirrors=[],
        batch_size=5,
        sleep_fn=fake_sleep,
        rand_uniform_fn=lambda a, b: 1.5,
    )

    async def fake_http(url: str, headers=None):
        if "syndication.twitter.com" in url:
            syn_urls.append(url)
            if len(syn_urls) == 2:
                return 429, None, {}
            return 200, "<html></html>", {}
        return 404, None, {}

    src._http_get_with_status = fake_http  # type: ignore[method-assign]
    src._ddg_fallback_discover = AsyncMock(return_value=[])  # type: ignore[method-assign]
    await src.fetch_articles()
    # Stopped immediately after account #2 returned HTTP 429 (never queried #3, #4, #5)
    assert len(syn_urls) == 2


@pytest.mark.asyncio
async def test_x_cooldown_skips_syndication():
    clock = [500.0]
    syn_calls = 0

    src = XVideoSource(
        accounts=_sample_x_accounts(3),
        mirrors=[],
        batch_size=3,
        cooldown_seconds=600,
        sleep_fn=AsyncMock(),
        monotonic_fn=lambda: clock[0],
    )

    async def fake_http(url: str, headers=None):
        nonlocal syn_calls
        if "syndication.twitter.com" in url:
            syn_calls += 1
            return 429, None, {}
        return 404, None, {}

    src._http_get_with_status = fake_http  # type: ignore[method-assign]
    src._ddg_fallback_discover = AsyncMock(return_value=[])  # type: ignore[method-assign]
    await src.fetch_articles()
    assert syn_calls == 1

    # Cycle 2 during cooldown (t=800 < 1100): zero syndication requests
    clock[0] = 800.0
    await src.fetch_articles()
    assert syn_calls == 1

    # Cycle 3 after cooldown expired (t=1101 >= 1100): syndication resumes cleanly
    clock[0] = 1101.0
    await src.fetch_articles()
    assert syn_calls == 2


@pytest.mark.asyncio
async def test_x_retry_after_is_respected():
    clock = [1000.0]
    src = XVideoSource(
        accounts=_sample_x_accounts(2),
        mirrors=[],
        cooldown_seconds=60,
        monotonic_fn=lambda: clock[0],
    )
    src._http_get_with_status = AsyncMock(  # type: ignore[method-assign]
        return_value=(429, None, {"Retry-After": "900"})
    )
    await src.fetch_channel_syndication(src._accounts[0])

    clock[0] = 1500.0  # Past configured 60s, still before 900s Retry-After
    assert src.is_syndication_cooldown_active() is True
    clock[0] = 1901.0
    assert src.is_syndication_cooldown_active() is False


@pytest.mark.asyncio
async def test_x_rss_fallback_after_429():
    now = datetime.now(timezone.utc)
    fresh_pub = (now - timedelta(minutes=15)).strftime("%a, %d %b %Y %H:%M:%S +0000")
    rss_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel>
      <item>
        <title>CENTCOM air defense report</title>
        <link>https://nitter.net/CENTCOM/status/55001</link>
        <pubDate>{fresh_pub}</pubDate>
        <description>CENTCOM confirmed air defense interception.</description>
      </item>
    </channel></rss>"""

    accounts = [{"handle": "CENTCOM", "display_name": "CENTCOM", "region": "middle_east", "is_official_media": True}]
    src = XVideoSource(
        accounts=accounts,
        mirrors=["https://rsshub.app/twitter/user/{}", "https://nitter.net/{}/rss"],
        batch_size=1,
    )

    async def fake_http(url: str, headers=None):
        if "syndication.twitter.com" in url:
            return 429, None, {}
        if "rsshub.app" in url:
            return 503, None, {}
        if "nitter.net/CENTCOM/rss" in url:
            return 200, rss_xml, {}
        return 404, None, {}

    async def fake_json(url: str):
        if "api.fxtwitter.com/CENTCOM/status/55001" in url:
            return {
                "tweet": {
                    "text": "CENTCOM confirmed air defense interception.",
                    "media": {
                        "videos": [
                            {"type": "video", "url": "https://video.twimg.com/c55001.mp4", "duration": 18, "width": 1280}
                        ]
                    },
                }
            }
        return None

    src._http_get_with_status = fake_http  # type: ignore[method-assign]
    src._http_get_json = fake_json  # type: ignore[method-assign]
    articles = await src.fetch_articles()
    assert len(articles) == 1
    assert articles[0].url == "https://x.com/CENTCOM/status/55001"
    assert articles[0].video_url == "https://video.twimg.com/c55001.mp4"


@pytest.mark.asyncio
async def test_x_google_news_rss_fallback():
    now = datetime.now(timezone.utc)
    recent_epoch = (now - timedelta(minutes=25)).timestamp()
    gnews_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel>
      <item>
        <title>CENTCOM joint maritime defense update</title>
        <link>https://x.com/CENTCOM/status/66001</link>
        <pubDate>Tue, 27 Nov 2026 12:00:00 +0000</pubDate>
      </item>
      <item>
        <title>Unverified timestamp post without created_at in fxtwitter</title>
        <link>https://x.com/CENTCOM/status/66002</link>
        <pubDate>Tue, 27 Nov 2026 12:00:00 +0000</pubDate>
      </item>
    </channel></rss>"""

    accounts = [{"handle": "CENTCOM", "display_name": "CENTCOM", "region": "middle_east", "is_official_media": True}]
    src = XVideoSource(accounts=accounts, mirrors=["https://nitter.net/{}/rss"], batch_size=1)
    src.trip_syndication_breaker()
    requested_urls: list[str] = []

    async def fake_http(url: str, headers=None):
        requested_urls.append(url)
        if "news.google.com/rss/search" in url:
            return 200, gnews_xml, {}
        return 503, None, {}

    async def fake_json(url: str):
        if "/66001" in url:
            return {
                "tweet": {
                    "text": "CENTCOM joint maritime defense update",
                    "created_timestamp": recent_epoch,
                    "media": {"videos": [{"type": "video", "url": "https://video.twimg.com/c66001.mp4", "duration": 15, "width": 720}]},
                }
            }
        if "/66002" in url:
            # Missing created_timestamp: must be rejected rather than trusting Google News index pubDate
            return {
                "tweet": {
                    "text": "Unverified timestamp post",
                    "media": {"videos": [{"type": "video", "url": "https://video.twimg.com/c66002.mp4", "duration": 15, "width": 720}]},
                }
            }
        return None

    src._http_get_with_status = fake_http  # type: ignore[method-assign]
    src._http_get_json = fake_json  # type: ignore[method-assign]

    articles = await src.fetch_articles()
    assert any("site%3Ax.com+from%3ACENTCOM" in u for u in requested_urls)
    assert [a.url for a in articles] == ["https://x.com/CENTCOM/status/66001"]


@pytest.mark.asyncio
async def test_x_fxtwitter_media_resolution():
    src = XVideoSource(accounts=[{"handle": "CENTCOM", "is_official_media": True}])
    src._http_get_json = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "tweet": {
                "text": "Precision strike footage",
                "media": {
                    "videos": [
                        {
                            "type": "video",
                            "url": "https://video.twimg.com/fx_hd.mp4",
                            "thumbnail_url": "https://pbs.twimg.com/fx_thumb.jpg",
                            "duration": 20,
                            "width": 1280,
                        }
                    ]
                },
            }
        }
    )
    meta = await src._resolve_media_metadata("CENTCOM", "77100")
    assert meta is not None
    assert meta["video_url"] == "https://video.twimg.com/fx_hd.mp4"
    assert meta["image_url"] == "https://pbs.twimg.com/fx_thumb.jpg"


@pytest.mark.asyncio
async def test_x_vxtwitter_fallback():
    src = XVideoSource(accounts=[{"handle": "CENTCOM", "is_official_media": True}])

    async def fake_json(url: str):
        if "fxtwitter" in url:
            return None
        if "vxtwitter" in url:
            return {
                "text": "VxTwitter backup footage",
                "media_extended": [
                    {
                        "type": "video",
                        "url": "https://video.twimg.com/vx_backup.mp4",
                        "thumbnail_url": "https://pbs.twimg.com/vx_thumb.jpg",
                        "duration_millis": 14000,
                        "size": {"width": 960, "height": 540},
                    }
                ],
            }
        return None

    src._http_get_json = fake_json  # type: ignore[method-assign]
    meta = await src._resolve_media_metadata("CENTCOM", "77200")
    assert meta is not None
    assert meta["video_url"] == "https://video.twimg.com/vx_backup.mp4"
    assert meta["duration"] == 14.0
    assert meta["width"] == 960


@pytest.mark.asyncio
async def test_x_deduplicates_cross_tier_results():
    assert normalize_canonical_x_url("https://twitter.com/CENTCOM/status/88001") == "https://x.com/CENTCOM/status/88001"
    assert normalize_canonical_x_url("https://fxtwitter.com/CENTCOM/status/88001") == "https://x.com/CENTCOM/status/88001"
    assert normalize_canonical_x_url("https://vxtwitter.com/CENTCOM/status/88001") == "https://x.com/CENTCOM/status/88001"

    now = datetime.now(timezone.utc)
    fresh_created = (now - timedelta(minutes=10)).strftime("%a %b %d %H:%M:%S +0000 %Y")
    fresh_pub = (now - timedelta(minutes=10)).strftime("%a, %d %b %Y %H:%M:%S +0000")
    accounts = [
        {"handle": "CENTCOM", "display_name": "CENTCOM", "region": "middle_east", "is_official_media": True},
        {"handle": "RudawEnglish", "display_name": "Rudaw", "region": "kurdistan_bashur", "is_official_media": True},
    ]
    syn_html = _build_syndication_html("CENTCOM", "88001", "Operation statement", fresh_created)
    rss_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel>
      <item>
        <title>Operation statement</title>
        <link>https://vxtwitter.com/RudawEnglish/status/88001</link>
        <pubDate>{fresh_pub}</pubDate>
      </item>
    </channel></rss>"""

    src = XVideoSource(accounts=accounts, mirrors=["https://nitter.net/{}/rss"], batch_size=2, sleep_fn=AsyncMock())

    async def fake_http(url: str, headers=None):
        if "syndication.twitter.com/srv/timeline-profile/screen-name/CENTCOM" in url:
            return 200, syn_html, {}
        if "syndication.twitter.com" in url:
            return 200, "<html></html>", {}
        if "nitter.net/RudawEnglish/rss" in url:
            return 200, rss_xml, {}
        return 404, None, {}

    src._http_get_with_status = fake_http  # type: ignore[method-assign]
    src._resolve_video_metadata = AsyncMock(  # type: ignore[method-assign]
        return_value={"video_url": "https://video.twimg.com/v88001.mp4", "thumbnail_url": "", "duration": 20, "width": 720, "text": "Operation statement"}
    )

    articles = await src.fetch_articles()
    assert len(articles) == 1
    assert articles[0].url == "https://x.com/CENTCOM/status/88001"


@pytest.mark.asyncio
async def test_x_rejects_stale_results():
    stale_created = (datetime.now(timezone.utc) - timedelta(hours=12)).strftime("%a %b %d %H:%M:%S +0000 %Y")
    accounts = [{"handle": "CENTCOM", "display_name": "CENTCOM", "region": "middle_east", "is_official_media": True}]
    syn_html = _build_syndication_html("CENTCOM", "99001", "Old archived alert", stale_created)

    src = XVideoSource(accounts=accounts, mirrors=[], max_age_hours=6.0, batch_size=1)
    src._http_get_with_status = AsyncMock(return_value=(200, syn_html, {}))  # type: ignore[method-assign]
    src._ddg_fallback_discover = AsyncMock(return_value=[])  # type: ignore[method-assign]
    src._resolve_video_metadata = AsyncMock(  # type: ignore[method-assign]
        return_value={"video_url": "https://video.twimg.com/old.mp4", "duration": 20, "width": 720, "text": "Old archived alert"}
    )

    articles = await src.fetch_articles()
    assert articles == []


@pytest.mark.asyncio
async def test_x_rejects_invalid_media():
    src = XVideoSource(accounts=[{"handle": "CENTCOM", "is_official_media": True}])

    # 1. GIF instead of video
    src._http_get_json = AsyncMock(  # type: ignore[method-assign]
        return_value={"tweet": {"text": "gif", "media": {"videos": [{"type": "gif", "url": "https://v/g.mp4", "duration": 10, "width": 720}]}}}
    )
    assert await src._resolve_video_metadata("CENTCOM", "1") is None

    # 2. Duration too short (< 4s) & too long (> 120s)
    src._http_get_json = AsyncMock(  # type: ignore[method-assign]
        return_value={"tweet": {"text": "short", "media": {"videos": [{"type": "video", "url": "https://v/s.mp4", "duration": 2, "width": 720}]}}}
    )
    assert await src._resolve_video_metadata("CENTCOM", "2") is None

    src._http_get_json = AsyncMock(  # type: ignore[method-assign]
        return_value={"tweet": {"text": "long", "media": {"videos": [{"type": "video", "url": "https://v/l.mp4", "duration": 240, "width": 720}]}}}
    )
    assert await src._resolve_video_metadata("CENTCOM", "3") is None

    # 3. Width below X_MIN_VIDEO_WIDTH (< 480)
    src._http_get_json = AsyncMock(  # type: ignore[method-assign]
        return_value={"tweet": {"text": "narrow", "media": {"videos": [{"type": "video", "url": "https://v/n.mp4", "duration": 15, "width": 320}]}}}
    )
    assert await src._resolve_video_metadata("CENTCOM", "4") is None


@pytest.mark.asyncio
async def test_x_cycle_request_budget():
    accounts = _sample_x_accounts(5)
    calls = 0

    src = XVideoSource(
        accounts=accounts,
        mirrors=["https://rsshub.app/twitter/user/{}", "https://nitter.net/{}/rss"],
        batch_size=5,
        max_requests_per_cycle=4,
        sleep_fn=AsyncMock(),
    )

    async def fake_http(url: str, headers=None):
        nonlocal calls
        calls += 1
        return 503, None, {}

    src._http_get_with_status = fake_http  # type: ignore[method-assign]
    src._ddg_fallback_discover = AsyncMock(return_value=[])  # type: ignore[method-assign]

    await src.fetch_articles()
    assert calls <= 4
