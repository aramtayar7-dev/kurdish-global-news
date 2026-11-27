from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from api.webhook import handle_webhook_request


def _make_mock_bot() -> MagicMock:
    bot = MagicMock()
    bot.defaults = None
    bot.send_photo = AsyncMock(return_value=SimpleNamespace(message_id=777))
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=888))
    return bot


def test_vercel_json_routes_api_webhook():
    vercel_cfg = json.loads(
        (Path(__file__).resolve().parent.parent / "vercel.json").read_text(encoding="utf-8")
    )
    assert vercel_cfg["version"] == 2
    assert any(
        r.get("source") == "/api/webhook" and r.get("destination") == "/api/webhook.py"
        for r in vercel_cfg.get("rewrites", [])
    )


def test_vercel_webhook_rejects_bad_secret_and_invalid_json():
    bot = _make_mock_bot()

    status_403, body_403 = handle_webhook_request(
        b"{}",
        headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"},
        webhook_secret="expected",
        bot=bot,
    )
    assert status_403 == 403
    assert body_403["error"] == "forbidden"

    status_400, body_400 = handle_webhook_request(
        b"not-json",
        headers={"X-Telegram-Bot-Api-Secret-Token": "expected"},
        webhook_secret="expected",
        bot=bot,
    )
    assert status_400 == 400
    assert body_400["error"] == "invalid_json"


def test_vercel_webhook_publishes_admin_post_and_blocks_non_admin(tmp_path: Path):
    bot = _make_mock_bot()
    fake_card = tmp_path / "vercel_card.jpg"
    fake_card.write_bytes(b"JPEGDATA")

    card_gen = MagicMock()
    card_gen.generate_card_file = AsyncMock(return_value=fake_card)

    non_admin_payload = json.dumps(
        {
            "update_id": 100,
            "message": {
                "message_id": 1,
                "date": 1700000000,
                "chat": {"id": 99999, "type": "private"},
                "from": {"id": 99999, "is_bot": False, "first_name": "Other"},
                "text": "پۆستی نایاسایی",
            },
        }
    ).encode("utf-8")

    st_ignored, res_ignored = handle_webhook_request(
        non_admin_payload,
        headers={"X-Telegram-Bot-Api-Secret-Token": "sec"},
        webhook_secret="sec",
        bot=bot,
        card_generator=card_gen,
        channel_id="@KurdishGlobalNews",
        admin_chat_id=111222333,
    )
    assert st_ignored == 200
    assert res_ignored["status"] == "ignored"
    bot.send_photo.assert_not_called()

    admin_payload = json.dumps(
        {
            "update_id": 101,
            "message": {
                "message_id": 2,
                "date": 1700000001,
                "chat": {"id": 111222333, "type": "private"},
                "from": {"id": 111222333, "is_bot": False, "first_name": "Admin"},
                "text": "بەپەلە | کۆبوونەوەی گرنگ لە هەولێر",
            },
        }
    ).encode("utf-8")

    st_ok, res_ok = handle_webhook_request(
        admin_payload,
        headers={"X-Telegram-Bot-Api-Secret-Token": "sec"},
        webhook_secret="sec",
        bot=bot,
        card_generator=card_gen,
        channel_id="@KurdishGlobalNews",
        admin_chat_id=111222333,
    )
    assert st_ok == 200
    assert res_ok["status"] == "published"
    assert res_ok["is_breaking"] is True
    bot.send_photo.assert_awaited_once()
    assert "✅ بڵاوکرایەوە" in bot.send_message.await_args.kwargs["text"]
