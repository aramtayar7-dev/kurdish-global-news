from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from admin_webhook import create_webhook_app
from tg_bot.admin_poster import (
    build_admin_article,
    extract_and_strip_admin_url,
    is_authorized_admin_message,
    parse_admin_post_text,
    publish_admin_manual_post,
)


def _make_mock_bot() -> MagicMock:
    bot = MagicMock()
    bot.defaults = None
    bot.send_photo = AsyncMock(return_value=SimpleNamespace(message_id=4321))
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=99))

    tg_file = MagicMock()
    tg_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"\xff\xd8\xff\xe0FAKEJPEG"))
    bot.get_file = AsyncMock(return_value=tg_file)
    return bot


def _make_message(
    *,
    user_id: int = 111222333,
    chat_type: str = "private",
    text: str | None = None,
    caption: str | None = None,
    photos: list | None = None,
    message_id: int = 10,
) -> SimpleNamespace:
    return SimpleNamespace(
        message_id=message_id,
        from_user=SimpleNamespace(id=user_id),
        chat=SimpleNamespace(id=user_id, type=chat_type),
        text=text,
        caption=caption,
        photo=photos,
    )


def test_parse_admin_post_text_multiline_and_singleline():
    headline, summary, is_breaking = parse_admin_post_text(
        "بەپەلە | کۆبوونەوەی ئەنجومەنی وەزیران لە هەولێر دەستیپێکرد\n"
        "وەزیرەکان گفتوگۆ لەسەر مووچە و دۆخی ئەمنی دەکەن."
    )
    assert is_breaking is True
    assert headline == "کۆبوونەوەی ئەنجومەنی وەزیران لە هەولێر دەستیپێکرد"
    assert summary == "وەزیرەکان گفتوگۆ لەسەر مووچە و دۆخی ئەمنی دەکەن."

    h2, s2, b2 = parse_admin_post_text("سەرۆک بارزانی پێشوازی لە شاندێکی دیپلۆماسی کرد.")
    assert b2 is False
    assert h2 == "سەرۆک بارزانی پێشوازی لە شاندێکی دیپلۆماسی کرد."
    assert s2 == ""


def test_is_authorized_admin_message_strict():
    admin_msg = _make_message(user_id=777888, chat_type="private", text="hello")
    stranger_msg = _make_message(user_id=999000, chat_type="private", text="hello")
    group_msg = _make_message(user_id=777888, chat_type="group", text="hello")

    assert is_authorized_admin_message(admin_msg, admin_chat_id="777888") is True
    assert is_authorized_admin_message(stranger_msg, admin_chat_id="777888") is False
    assert is_authorized_admin_message(group_msg, admin_chat_id="777888") is False
    assert is_authorized_admin_message(admin_msg, admin_chat_id="") is False


def test_build_admin_article_preserves_kurdish_verbatim():
    text = "بەپەلە: هێرشی مووشەکی بۆ سەر بنکەیەکی سەربازی لە هەولێر\nوردەکاریی زیاتر لە ڕاگەیەندراوی فەرمیدا بڵاودەکرێتەوە."
    article = build_admin_article(text)
    assert article.translated_title == "هێرشی مووشەکی بۆ سەر بنکەیەکی سەربازی لە هەولێر"
    assert article.translated_summary == "وردەکاریی زیاتر لە ڕاگەیەندراوی فەرمیدا بڵاودەکرێتەوە."
    assert article.is_breaking is True
    assert article.importance_score >= 9.0
    assert article.category == "kurdistan"


@pytest.mark.parametrize(
    "raw_text,expected_clean,expected_url",
    [
        (
            "کۆبوونەوەی ئەمنی لە هەولێر http://t.me/Kurdistan24news",
            "کۆبوونەوەی ئەمنی لە هەولێر",
            "http://t.me/Kurdistan24news",
        ),
        (
            "سەردێڕی هەواڵ\nدرێژەی بابەتەکە لێرەیە.\nسەرچاوە: t.me/Kurdistan24news/9988",
            "سەردێڕی هەواڵ\nدرێژەی بابەتەکە لێرەیە.",
            "https://t.me/Kurdistan24news/9988",
        ),
        (
            "بەپەلە | ڕاگەیەندراوی فەرمی\nوردەکاری لە کۆنگرەکە https://www.reuters.com/world/item-1",
            "بەپەلە | ڕاگەیەندراوی فەرمی\nوردەکاری لە کۆنگرەکە",
            "https://www.reuters.com/world/item-1",
        ),
    ],
)
def test_extract_and_strip_admin_url(raw_text: str, expected_clean: str, expected_url: str):
    cleaned, url = extract_and_strip_admin_url(raw_text)
    assert cleaned == expected_clean
    assert url == expected_url


@pytest.mark.asyncio
async def test_admin_post_with_url_strips_raw_url_and_renders_clean_source_link(tmp_path: Path):
    bot = _make_mock_bot()
    fake_card = tmp_path / "card_url.jpg"
    fake_card.write_bytes(b"JPEGDATA")

    card_gen = MagicMock()
    card_gen.generate_card_file = AsyncMock(return_value=fake_card)

    msg = _make_message(
        user_id=111222333,
        text=(
            "هەرێمی کوردستان پلانی نوێی ڕاگەیاند\n"
            "وەزارەت وردەکاری پرۆژەکەی بڵاوکردەوە http://t.me/Kurdistan24news"
        ),
    )

    result = await publish_admin_manual_post(
        bot,
        msg,
        card_generator=card_gen,
        channel_id="@KurdishGlobalNews",
        admin_chat_id=111222333,
    )
    assert result["status"] == "published"

    sent_article = card_gen.generate_card_file.await_args.args[0]
    assert "http://" not in sent_article.translated_title
    assert "http://" not in sent_article.translated_summary
    assert sent_article.original_url == "http://t.me/Kurdistan24news"

    caption = bot.send_photo.await_args.kwargs["caption"]
    assert '<a href="http://t.me/Kurdistan24news">سەرچاوە</a>' in caption
    assert "🔗 <a href=" in caption
    assert "بابەتی ئەسڵی" not in caption
    # Raw URL must NOT appear in body lines outside the bottom <a href="..."> anchor
    caption_without_anchor = caption.replace('<a href="http://t.me/Kurdistan24news">سەرچاوە</a>', "")
    assert "http://t.me/Kurdistan24news" not in caption_without_anchor


@pytest.mark.asyncio
async def test_non_admin_is_blocked_completely():
    bot = _make_mock_bot()
    card_gen = MagicMock()
    card_gen.generate_card_file = AsyncMock()

    stranger = _make_message(user_id=55555, text="هەواڵی ساختە")
    result = await publish_admin_manual_post(
        bot,
        stranger,
        card_generator=card_gen,
        channel_id="@KurdishGlobalNews",
        admin_chat_id=111222333,
    )

    assert result["status"] == "ignored"
    card_gen.generate_card_file.assert_not_called()
    bot.send_photo.assert_not_called()
    bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_admin_photo_plus_caption_publishes_card(tmp_path: Path):
    bot = _make_mock_bot()
    fake_card = tmp_path / "card.jpg"
    fake_card.write_bytes(b"JPEGDATA")

    card_gen = MagicMock()
    card_gen.generate_card_file = AsyncMock(return_value=fake_card)

    photos = [
        SimpleNamespace(file_id="low_res_id"),
        SimpleNamespace(file_id="high_res_id"),
    ]
    msg = _make_message(
        user_id=111222333,
        caption="پەرلەمانی کوردستان کۆدەبێتەوە\nبڕیارە پرۆژەیاسای بودجە تاوتوێ بکرێت.",
        photos=photos,
    )

    result = await publish_admin_manual_post(
        bot,
        msg,
        card_generator=card_gen,
        channel_id="@KurdishGlobalNews",
        admin_chat_id=111222333,
        channel_handle="@KurdishGlobalNews",
    )

    assert result["status"] == "published"
    assert result["used_photo"] is True
    bot.get_file.assert_awaited_once_with("high_res_id")

    call_args = card_gen.generate_card_file.await_args
    assert call_args.kwargs["image_bytes"] == b"\xff\xd8\xff\xe0FAKEJPEG"
    assert call_args.args[0].translated_title == "پەرلەمانی کوردستان کۆدەبێتەوە"
    assert call_args.args[0].translated_summary == "بڕیارە پرۆژەیاسای بودجە تاوتوێ بکرێت."

    send_photo_kwargs = bot.send_photo.await_args.kwargs
    assert send_photo_kwargs["chat_id"] == "@KurdishGlobalNews"
    assert "پەرلەمانی کوردستان کۆدەبێتەوە" in send_photo_kwargs["caption"]
    assert "بڕیارە پرۆژەیاسای بودجە تاوتوێ بکرێت." in send_photo_kwargs["caption"]
    assert not fake_card.exists()

    reply_text = bot.send_message.await_args.kwargs["text"]
    assert "✅ بڵاوکرایەوە" in reply_text
    assert "https://t.me/KurdishGlobalNews/4321" in reply_text


@pytest.mark.asyncio
async def test_admin_text_only_uses_background_fallback_and_no_duplicate_body(tmp_path: Path):
    bot = _make_mock_bot()
    fake_card = tmp_path / "card_text.jpg"
    fake_card.write_bytes(b"JPEGDATA")

    card_gen = MagicMock()
    card_gen.generate_card_file = AsyncMock(return_value=fake_card)

    msg = _make_message(
        user_id=111222333,
        text="بەپەلە | تەقینەوەیەک لە بەغدا ڕوویدا",
        photos=None,
    )

    result = await publish_admin_manual_post(
        bot,
        msg,
        card_generator=card_gen,
        channel_id="-1001234567890",
        admin_chat_id=111222333,
        channel_handle="",
    )

    assert result["status"] == "published"
    assert result["used_photo"] is False
    assert result["is_breaking"] is True
    call_args = card_gen.generate_card_file.await_args
    assert call_args.kwargs["image_bytes"] is None
    assert call_args.kwargs["is_breaking"] is True

    caption = bot.send_photo.await_args.kwargs["caption"]
    assert caption.count("تەقینەوەیەک لە بەغدا ڕوویدا") == 1
    assert "🔴 بەپەلە |" in caption


@pytest.mark.asyncio
async def test_admin_publish_error_reports_exact_error_to_admin():
    bot = _make_mock_bot()
    card_gen = MagicMock()
    card_gen.generate_card_file = AsyncMock(side_effect=RuntimeError("font missing"))

    msg = _make_message(user_id=111222333, text="هەواڵی تاقیکردنەوە")
    result = await publish_admin_manual_post(
        bot,
        msg,
        card_generator=card_gen,
        channel_id="@KurdishGlobalNews",
        admin_chat_id=111222333,
    )

    assert result["status"] == "error"
    reply_text = bot.send_message.await_args.kwargs["text"]
    assert "❌ هەڵە لە بڵاوکردنەوە" in reply_text
    assert "font missing" in reply_text


@pytest.mark.asyncio
async def test_webhook_server_health_and_update_handling(tmp_path: Path):
    bot = _make_mock_bot()
    fake_card = tmp_path / "card_wh.jpg"
    fake_card.write_bytes(b"JPEGDATA")

    card_gen = MagicMock()
    card_gen.generate_card_file = AsyncMock(return_value=fake_card)

    app = create_webhook_app(
        bot,
        webhook_secret="my-secret",
        card_generator=card_gen,
        channel_id="@KurdishGlobalNews",
        admin_chat_id=111222333,
    )

    async with TestClient(TestServer(app)) as client:
        health = await client.get("/health")
        assert health.status == 200
        health_json = await health.json()
        assert health_json["status"] == "ok"

        forbidden = await client.post("/webhook", json={"update_id": 1})
        assert forbidden.status == 403

        ok = await client.post(
            "/webhook",
            headers={"X-Telegram-Bot-Api-Secret-Token": "my-secret"},
            json={
                "update_id": 2,
                "message": {
                    "message_id": 55,
                    "date": 1700000000,
                    "chat": {"id": 111222333, "type": "private"},
                    "from": {"id": 111222333, "is_bot": False, "first_name": "Admin"},
                    "text": "سەردێڕی تاقیکردنەوەی وێبهوک",
                },
            },
        )
        assert ok.status == 200
        ok_json = await ok.json()
        assert ok_json["status"] == "published"
        bot.send_photo.assert_awaited_once()
