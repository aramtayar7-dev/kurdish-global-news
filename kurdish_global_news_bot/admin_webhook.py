"""
Kurdish Global News Bot — Lightweight Admin Webhook & Polling Server

Render-ready aiohttp web server listening on PORT (default 10000) with:
  - GET  /health           -> Health check endpoint (matching render.yaml)
  - POST /webhook          -> Telegram Bot Webhook endpoint for instant admin posts
  - POST /telegram-webhook -> Alias for Telegram Bot Webhook endpoint

Local testing mode:
  python admin_webhook.py --polling
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from typing import Any, Optional

from aiohttp import web
from telegram import Bot, Update

import config
from tg_bot.admin_poster import publish_admin_manual_post

logger = logging.getLogger(__name__)


def create_webhook_app(
    bot: Optional[Bot] = None,
    *,
    webhook_secret: Optional[str] = None,
    card_generator: Any = None,
    formatter: Any = None,
    channel_id: Any = None,
    admin_chat_id: Any = None,
) -> web.Application:
    """Create the lightweight aiohttp application for health checks and Telegram webhooks."""
    tg_bot = bot or Bot(token=config.TELEGRAM_BOT_TOKEN)
    secret = (
        webhook_secret
        if webhook_secret is not None
        else os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()
    )

    async def health_handler(_request: web.Request) -> web.Response:
        return web.json_response(
            {"status": "ok", "service": "kurdish_global_news_admin_bot"}
        )

    async def webhook_handler(request: web.Request) -> web.Response:
        if secret:
            header_secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
            if header_secret != secret:
                return web.json_response({"ok": False, "error": "forbidden"}, status=403)

        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid_json"}, status=400)

        update = Update.de_json(payload, tg_bot)
        message = update.effective_message if update is not None else None
        if message is None:
            return web.json_response({"ok": True, "status": "ignored"})

        result = await publish_admin_manual_post(
            tg_bot,
            message,
            card_generator=card_generator,
            formatter=formatter,
            channel_id=channel_id,
            admin_chat_id=admin_chat_id,
        )
        return web.json_response({"ok": True, **result})

    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_post("/webhook", webhook_handler)
    app.router.add_post("/telegram-webhook", webhook_handler)
    return app


async def run_polling_loop(
    bot: Optional[Bot] = None,
    *,
    poll_timeout: int = 30,
) -> None:
    """Run lightweight long-polling for local admin manual posting tests."""
    tg_bot = bot or Bot(token=config.TELEGRAM_BOT_TOKEN)
    await tg_bot.initialize()
    try:
        try:
            await tg_bot.delete_webhook(drop_pending_updates=False)
        except Exception as exc:
            logger.debug("delete_webhook skipped: %s", exc)

        offset: Optional[int] = None
        logger.info("Admin manual poster polling started (ADMIN_CHAT_ID=%s)", config.ADMIN_CHAT_ID)
        while True:
            try:
                updates = await tg_bot.get_updates(
                    offset=offset,
                    timeout=poll_timeout,
                    allowed_updates=["message"],
                )
                for upd in updates:
                    offset = upd.update_id + 1
                    if upd.message is not None:
                        await publish_admin_manual_post(tg_bot, upd.message)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Polling error: %s", exc)
                await asyncio.sleep(3)
    finally:
        await tg_bot.shutdown()


async def _serve_webhook(host: str, port: int) -> None:
    tg_bot = Bot(token=config.TELEGRAM_BOT_TOKEN)
    await tg_bot.initialize()

    webhook_url = os.getenv("TELEGRAM_WEBHOOK_URL", "").strip()
    webhook_secret = os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()
    if webhook_url:
        try:
            await tg_bot.set_webhook(
                url=webhook_url,
                secret_token=webhook_secret or None,
                allowed_updates=["message"],
            )
            logger.info("Registered Telegram webhook: %s", webhook_url)
        except Exception as exc:
            logger.warning("Failed to auto-register webhook: %s", exc)

    app = create_webhook_app(tg_bot, webhook_secret=webhook_secret)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()
    logger.info("Admin webhook server listening on %s:%s", host, port)

    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await runner.cleanup()
        await tg_bot.shutdown()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Kurdish Global News Bot — Instant Admin Manual Poster"
    )
    parser.add_argument(
        "--polling",
        action="store_true",
        help="Run Telegram long-polling instead of the HTTP webhook server.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Webhook bind host.")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", str(config.PORT or 10000))),
        help="Webhook listen port (default: PORT env or 10000).",
    )
    args = parser.parse_args()

    if args.polling:
        asyncio.run(run_polling_loop())
    else:
        asyncio.run(_serve_webhook(args.host, args.port))


if __name__ == "__main__":
    main()
