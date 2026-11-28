"""
Vercel Python Serverless Function for Instant Admin Telegram Webhook.

Routes POST /api/webhook Telegram updates to tg_bot.admin_poster.publish_admin_manual_post,
which enforces ADMIN_CHAT_ID security, renders the branded 1280x720 card using
card_generator.py / background_selector.py / content_formatter.py, publishes to
TELEGRAM_CHANNEL_ID, and replies '✅ بڵاوکرایەوە' to the admin.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Optional

# Ensure kurdish_global_news_bot root is importable when Vercel invokes api/webhook.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from telegram import Bot, Update

import config
from tg_bot.admin_poster import extract_admin_video_file_id, publish_admin_manual_post


async def process_webhook_payload(
    payload: dict[str, Any],
    *,
    bot: Optional[Bot] = None,
    card_generator: Any = None,
    formatter: Any = None,
    channel_id: Any = None,
    admin_chat_id: Any = None,
) -> dict[str, Any]:
    """Process a parsed Telegram Update JSON payload asynchronously."""
    owns_bot = bot is None
    tg_bot = bot or Bot(token=config.TELEGRAM_BOT_TOKEN)
    if owns_bot:
        await tg_bot.initialize()

    try:
        update = Update.de_json(payload, tg_bot)
        message = update.effective_message if update is not None else None
        if message is None:
            return {"ok": True, "status": "ignored"}

        video_file_id = extract_admin_video_file_id(message)
        video_caption = getattr(message, "caption", None) if video_file_id else None

        result = await publish_admin_manual_post(
            tg_bot,
            message,
            card_generator=card_generator,
            formatter=formatter,
            channel_id=channel_id,
            admin_chat_id=admin_chat_id,
            video_file_id=video_file_id,
            caption_override=video_caption,
        )
        return {"ok": True, **result}
    finally:
        if owns_bot:
            await tg_bot.shutdown()


def handle_webhook_request(
    body_bytes: bytes,
    headers: Optional[dict[str, str]] = None,
    *,
    webhook_secret: Optional[str] = None,
    bot: Optional[Bot] = None,
    card_generator: Any = None,
    formatter: Any = None,
    channel_id: Any = None,
    admin_chat_id: Any = None,
) -> tuple[int, dict[str, Any]]:
    """Validate secret/JSON and execute the webhook update synchronously for BaseHTTPRequestHandler."""
    secret = (
        webhook_secret
        if webhook_secret is not None
        else os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()
    )
    if secret:
        hdrs = {k.lower(): v for k, v in (headers or {}).items()}
        if hdrs.get("x-telegram-bot-api-secret-token", "") != secret:
            return 403, {"ok": False, "error": "forbidden"}

    try:
        payload = json.loads(body_bytes.decode("utf-8"))
        if not isinstance(payload, dict):
            return 400, {"ok": False, "error": "invalid_json"}
    except Exception:
        return 400, {"ok": False, "error": "invalid_json"}

    result = asyncio.run(
        process_webhook_payload(
            payload,
            bot=bot,
            card_generator=card_generator,
            formatter=formatter,
            channel_id=channel_id,
            admin_chat_id=admin_chat_id,
        )
    )
    return 200, result


class handler(BaseHTTPRequestHandler):
    """Vercel Python Serverless Function entrypoint."""

    def _send_json(self, status_code: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        self._send_json(
            200,
            {"status": "ok", "service": "kurdish_global_news_admin_webhook"},
        )

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        raw_body = self.rfile.read(content_length) if content_length > 0 else b""
        hdrs = {k: v for k, v in self.headers.items()}
        status_code, response_body = handle_webhook_request(raw_body, hdrs)
        self._send_json(status_code, response_body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return
