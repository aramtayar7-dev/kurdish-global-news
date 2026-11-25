"""
Comprehensive automated tests for Inline Source URL Buttons (`InlineKeyboardMarkup`) (Idea 6).
100% offline/mocked — zero live Telegram, network, or Gemini API calls.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import InlineKeyboardMarkup
from telegram.error import BadRequest

import config
from config import (
    ENABLE_INLINE_SOURCE_BUTTON,
    ENABLE_TELEGRAM_SOURCE_LINK_BUTTON,
    INLINE_SOURCE_BUTTON_MAX_LABEL_LEN,
)
from monitoring.watchdog import AdminWatchdog
from processing.content_formatter import ContentFormatter
from processing.gemini_client import GeminiClient
from storage.models import Article
from tg_bot.inline_buttons import (
    DEFAULT_BUTTON_LABEL,
    build_source_button_markup,
    format_source_button_label,
    is_valid_button_url,
)
from tg_bot.publisher import TelegramPublisher


def _make_article(
    url: str = "https://www.reuters.com/world/middle-east/erbil-talks-2026",
    source_name: str = "Reuters",
    category: str = "politics_world",
    translated_title: str = "هەولێر و بەغدا تاوتوێی پڕۆژەیاسای ئاسایشیان کرد",
    translated_summary: str = "وەفدەکانی هەولێر و بەغدا لە کۆبوونەوەیەکدا ڕێککەوتنی نوێیان واژۆ کرد.",
    video_url: str | None = None,
    thread_root_message_id: int | None = None,
    is_breaking: bool = False,
    score: float = 8.4,
) -> Article:
    return Article(
        original_url=url,
        canonical_url=Article.canonicalize_url(url),
        original_title="Erbil and Baghdad hold security consultations",
        translated_title=translated_title,
        translated_summary=translated_summary,
        source_name=source_name,
        category=category,
        video_url=video_url,
        thread_root_message_id=thread_root_message_id,
        is_breaking=is_breaking,
        importance_score=score,
    )


# ═══════════════════════════════════════════════════
# 1. Config Defaults & Valid Web/RSS URL Markup
# ═══════════════════════════════════════════════════
class TestInlineButtonBuilderAndValidation:
    def test_config_defaults(self):
        assert ENABLE_INLINE_SOURCE_BUTTON is True
        assert ENABLE_TELEGRAM_SOURCE_LINK_BUTTON is False
        assert INLINE_SOURCE_BUTTON_MAX_LABEL_LEN == 36

    @pytest.mark.parametrize(
        "url,source_name,expected_label",
        [
            ("https://www.reuters.com/world/story-1", "Reuters", "🔗 سەرچاوە — Reuters"),
            ("https://www.kurdistan24.net/ckb/story/123", "Kurdistan24 Sorani", "🔗 سەرچاوە — Kurdistan24"),
            ("https://www.bbc.com/news/world-middle-east-99", "BBC World", "🔗 سەرچاوە — BBC World"),
            ("https://www.aljazeera.com/news/1", "Al Jazeera", "🔗 سەرچاوە — Al Jazeera"),
            ("https://www.theguardian.com/world/test", "", DEFAULT_BUTTON_LABEL),
            ("https://example.com/a", "Very Long International News Organization Network Dispatch Name", DEFAULT_BUTTON_LABEL),
        ],
    )
    def test_valid_web_urls_produce_single_row_inline_markup(
        self, url: str, source_name: str, expected_label: str
    ):
        markup = build_source_button_markup(url=url, source_name=source_name)
        assert isinstance(markup, InlineKeyboardMarkup)
        assert len(markup.inline_keyboard) == 1
        assert len(markup.inline_keyboard[0]) == 1
        btn = markup.inline_keyboard[0][0]
        assert btn.text == expected_label
        assert btn.url == url

    # ═══════════════════════════════════════════════
    # 2. External Telegram Retention Protection
    # ═══════════════════════════════════════════════
    @pytest.mark.parametrize(
        "tg_url",
        [
            "https://t.me/SabrenNewss/232612",
            "https://telegram.me/rudawdigital/123",
            "https://telegram.dog/Kurdistan24news/45",
            "https://sub.t.me/channel/1",
        ],
    )
    def test_external_telegram_links_blocked_by_default_and_allowed_when_enabled(
        self, tg_url: str, monkeypatch
    ):
        # Default False -> subscriber retention protection blocks t.me/ link buttons
        assert build_source_button_markup(url=tg_url, source_name="Sabereen News") is None

        # Explicitly enabled -> produces clean button
        monkeypatch.setattr(config, "ENABLE_TELEGRAM_SOURCE_LINK_BUTTON", True)
        markup = build_source_button_markup(url=tg_url, source_name="Sabereen News")
        assert isinstance(markup, InlineKeyboardMarkup)
        assert markup.inline_keyboard[0][0].url == tg_url

    # ═══════════════════════════════════════════════
    # 3. Invalid / Unsafe URLs
    # ═══════════════════════════════════════════════
    @pytest.mark.parametrize(
        "bad_url",
        [
            "",
            "   ",
            None,
            "not-a-url",
            "javascript:alert(1)",
            "mailto:admin@example.com",
            "tg://resolve?domain=SabrenNewss",
            "ftp://example.com/file",
            "http://localhost/test",
            "http://127.0.0.1/admin",
            "http://192.168.1.10/feed",
            "https://example.com/has space/news",
            "https://example.com/ctrl\nchar",
            "https://intranet/path",
            "https://example.com/" + ("a" * 2050),
        ],
    )
    def test_invalid_or_unsafe_urls_return_none_safely(self, bad_url):
        assert is_valid_button_url(bad_url) is False
        assert build_source_button_markup(url=bad_url, source_name="Reuters") is None

    # ═══════════════════════════════════════════════
    # 4. Feature Flag Disable
    # ═══════════════════════════════════════════════
    def test_feature_flag_false_disables_all_buttons(self, monkeypatch):
        monkeypatch.setattr(config, "ENABLE_INLINE_SOURCE_BUTTON", False)
        assert (
            build_source_button_markup(
                url="https://www.reuters.com/world/test-1",
                source_name="Reuters",
            )
            is None
        )


# ═══════════════════════════════════════════════════
# 5. Caption Deduplication
# ═══════════════════════════════════════════════════
class TestCaptionDeduplication:
    def test_caption_and_message_omit_bottom_link_only_when_inline_button_present(self):
        formatter = ContentFormatter()
        art = _make_article(
            url="https://www.reuters.com/world/summit-101",
            source_name="Reuters",
        )

        # With inline button -> clean caption WITHOUT redundant bottom link anchor
        cap_with_btn = formatter.format_caption(art, has_inline_source_button=True)
        msg_with_btn = formatter.format_telegram_message(art, has_inline_source_button=True)
        assert "بابەتی ئەسڵی" not in cap_with_btn
        assert "reuters.com" not in cap_with_btn
        assert "Reuters" in cap_with_btn
        assert len(cap_with_btn) <= 1024
        assert "بابەتی ئەسڵی" not in msg_with_btn

        # Without inline button -> keeps fallback bottom link anchor cleanly
        cap_without_btn = formatter.format_caption(art, has_inline_source_button=False)
        msg_without_btn = formatter.format_telegram_message(art, has_inline_source_button=False)
        assert '<a href="https://www.reuters.com/world/summit-101">بابەتی ئەسڵی</a>' in cap_without_btn
        assert '<a href="https://www.reuters.com/world/summit-101">بابەتی ئەسڵی</a>' in msg_without_btn

        # Clean formatted label check
        assert format_source_button_label("Reuters") == "🔗 سەرچاوە — Reuters"


# ═══════════════════════════════════════════════════
# 6. Publisher Integration across Card, Video, Text, Threading & Retry
# ═══════════════════════════════════════════════════
class TestPublisherInlineButtonIntegration:
    @pytest.mark.asyncio
    async def test_send_photo_and_send_message_attach_reply_markup_with_thread_reply_id(self, tmp_path):
        fake_card = tmp_path / "card.jpg"
        fake_card.write_bytes(b"FAKE_JPEG")

        mock_bot = AsyncMock()
        mock_bot.send_photo = AsyncMock(return_value=MagicMock(message_id=501))
        mock_bot.send_message = AsyncMock(return_value=MagicMock(message_id=502))

        pub = TelegramPublisher()
        pub._channel_id = "@KurdishGlobalNews"
        pub._bot = mock_bot
        pub._image_handler.download_image = AsyncMock(return_value=None)
        pub._card_generator.generate_card_file = AsyncMock(return_value=fake_card)

        threaded_art = _make_article(
            url="https://www.kurdistan24.net/ckb/story/555",
            source_name="Kurdistan24 Sorani",
            thread_root_message_id=400,
        )

        with patch.object(GeminiClient, "generate_json", new=AsyncMock()) as mock_gemini:
            # 1. Branded card (send_photo) with BOTH reply_markup & reply_to_message_id
            with patch("tg_bot.publisher.ENABLE_IMAGES", True):
                msg_id = await pub.publish_article(threaded_art)
            assert msg_id == 501
            p_kw = mock_bot.send_photo.call_args.kwargs
            assert p_kw["reply_to_message_id"] == 400
            assert "بابەتی ئەسڵی" not in p_kw["caption"]
            markup = p_kw["reply_markup"]
            assert isinstance(markup, InlineKeyboardMarkup)
            assert markup.inline_keyboard[0][0].text == "🔗 سەرچاوە — Kurdistan24"
            assert markup.inline_keyboard[0][0].url == "https://www.kurdistan24.net/ckb/story/555"

            # 2. Text fallback (send_message) with BOTH reply_markup & reply_to_message_id
            with patch("tg_bot.publisher.ENABLE_IMAGES", False):
                txt_id = await pub.publish_article(threaded_art)
            assert txt_id == 502
            m_kw = mock_bot.send_message.call_args.kwargs
            assert m_kw["reply_to_message_id"] == 400
            assert "بابەتی ئەسڵی" not in m_kw["text"]
            assert isinstance(m_kw["reply_markup"], InlineKeyboardMarkup)

        mock_gemini.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_button_url_invalid_triggers_clean_retry_without_reply_markup(self, tmp_path):
        fake_card = tmp_path / "card_retry.jpg"
        fake_card.write_bytes(b"FAKE_JPEG_DATA")

        mock_bot = AsyncMock()
        mock_bot.send_photo = AsyncMock(
            side_effect=[
                BadRequest("BUTTON_URL_INVALID"),
                MagicMock(message_id=601),
            ]
        )

        pub = TelegramPublisher()
        pub._channel_id = "@KurdishGlobalNews"
        pub._bot = mock_bot
        pub._image_handler.download_image = AsyncMock(return_value=None)
        pub._card_generator.generate_card_file = AsyncMock(return_value=fake_card)

        art = _make_article(
            url="https://www.reuters.com/world/unusual-button-url",
            source_name="Reuters",
            thread_root_message_id=321,
        )

        with patch("tg_bot.publisher.ENABLE_IMAGES", True):
            msg_id = await pub.publish_article(art)

        assert msg_id == 601
        assert mock_bot.send_photo.await_count == 2
        first_kw = mock_bot.send_photo.call_args_list[0].kwargs
        second_kw = mock_bot.send_photo.call_args_list[1].kwargs
        assert isinstance(first_kw["reply_markup"], InlineKeyboardMarkup)
        assert "reply_markup" not in second_kw
        # Threading anchor preserved across button fallback retry!
        assert second_kw["reply_to_message_id"] == 321

    @pytest.mark.asyncio
    async def test_daily_digest_and_admin_watchdog_have_no_inline_source_buttons(self, tmp_path):
        mock_bot = AsyncMock()
        mock_bot.send_photo = AsyncMock(return_value=MagicMock(message_id=701))
        mock_bot.send_message = AsyncMock(return_value=MagicMock(message_id=702))

        pub = TelegramPublisher()
        pub._channel_id = "@KurdishGlobalNews"
        pub._bot = mock_bot

        digest_card = tmp_path / "digest.jpg"
        digest_card.write_bytes(b"JPG")
        await pub.publish_daily_digest("🌙 <b>پوختەی شەوانە</b>", digest_card)
        assert "reply_markup" not in mock_bot.send_photo.call_args.kwargs

        wd = AdminWatchdog(
            db=None,
            publisher=pub,
            admin_chat_id="123456789",
            public_channel_id="@KurdishGlobalNews",
        )
        ok = await wd.dispatch_alert("SOURCE_OUTAGE", title="Test outage", details="Source timeout")
        assert ok is True
        assert "reply_markup" not in mock_bot.send_message.call_args.kwargs
