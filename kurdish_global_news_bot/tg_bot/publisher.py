"""
Kurdish Global News Bot — Telegram Publisher

Publishes translated articles to a Telegram channel with images
and handles rate limiting and delivery confirmation.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Optional

import httpx
from telegram import Bot
from telegram.request import HTTPXRequest
from telegram.constants import ParseMode
from telegram.error import RetryAfter, TelegramError, TimedOut, NetworkError, BadRequest

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHANNEL_ID, ENABLE_IMAGES
from storage.models import Article
from processing.content_formatter import ContentFormatter
from tg_bot.image_handler import ImageHandler

logger = logging.getLogger("tg_bot.publisher")


class TelegramPublisher:
    """Publishes news articles to a Telegram channel."""

    def __init__(self) -> None:
        self._bot: Optional[Bot] = None
        self._formatter = ContentFormatter()
        self._image_handler = ImageHandler()
        self._channel_id = TELEGRAM_CHANNEL_ID

    async def initialize(self) -> None:
        """Initialize the Telegram bot and image handler."""
        if not TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN is not configured.")
        if not self._channel_id:
            raise ValueError("TELEGRAM_CHANNEL_ID is not configured.")

        # Configure HTTPXRequest with generous cloud timeouts and IPv4 transport
        # Prevents GitHub Actions / Azure CI from hanging on broken IPv6 routes to api.telegram.org
        transport = httpx.AsyncHTTPTransport(local_address="0.0.0.0", retries=3)
        request = HTTPXRequest(
            connect_timeout=30.0,
            read_timeout=30.0,
            write_timeout=30.0,
            pool_timeout=30.0,
            connection_pool_size=8,
            httpx_kwargs={"transport": transport},
        )
        self._bot = Bot(token=TELEGRAM_BOT_TOKEN, request=request)
        await self._image_handler.initialize()

        # Verify bot connectivity with retries for cloud/CI network jitter
        max_init_retries = 3
        for attempt in range(max_init_retries):
            try:
                me = await self._bot.get_me()
                logger.info("Telegram bot connected: @%s (%s)", me.username, me.first_name)
                break
            except (TimedOut, NetworkError) as e:
                if attempt < max_init_retries - 1:
                    wait_time = 2 * (attempt + 1)
                    logger.warning(
                        "Telegram connection attempt %d/%d timed out or network error: %s. Retrying in %ds...",
                        attempt + 1, max_init_retries, e, wait_time,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("Failed to connect to Telegram after %d attempts: %s", max_init_retries, e)
                    raise
            except TelegramError as e:
                logger.error("Failed to connect to Telegram: %s", e)
                raise

    async def cleanup(self) -> None:
        """Clean up resources."""
        await self._image_handler.cleanup()
        if self._bot:
            await self._bot.shutdown()

    async def publish_article(self, article: Article) -> Optional[int]:
        """
        Publish an article to the Telegram channel.

        Returns the Telegram message_id on success, or None on failure.
        Does NOT wait for a batch or hourly schedule — publishes immediately.
        """
        if not self._bot:
            raise RuntimeError("Publisher not initialized. Call initialize() first.")

        # Try with image first (if enabled and available)
        if ENABLE_IMAGES and article.image_url:
            message_id = await self._publish_with_image(article)
            if message_id is not None:
                return message_id
            # Fall back to text-only if image fails
            logger.info("Image publish failed, falling back to text-only.")

        # Text-only publish
        return await self._publish_text_only(article)

    async def _publish_with_image(self, article: Article) -> Optional[int]:
        """Publish an article with an image."""
        assert self._bot is not None

        image_data = await self._image_handler.download_image(article.image_url)
        if not image_data:
            return None

        caption = self._formatter.format_caption(article)

        try:
            message = await self._send_with_retry(
                self._bot.send_photo,
                chat_id=self._channel_id,
                photo=io.BytesIO(image_data),
                caption=caption,
                parse_mode=ParseMode.HTML,
            )
            if message:
                logger.info("Published with image: msg_id=%d, title='%s'",
                            message.message_id, article.original_title[:60])
                return message.message_id
        except BadRequest as e:
            # Image might be invalid for Telegram
            logger.warning("Telegram rejected image: %s", str(e)[:200])
        except Exception as e:
            logger.error("Image publish error: %s", str(e)[:200])

        return None

    async def _publish_text_only(self, article: Article) -> Optional[int]:
        """Publish a text-only article."""
        assert self._bot is not None

        text = self._formatter.format_telegram_message(article)

        try:
            message = await self._send_with_retry(
                self._bot.send_message,
                chat_id=self._channel_id,
                text=text,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=False,
            )
            if message:
                logger.info("Published text-only: msg_id=%d, title='%s'",
                            message.message_id, article.original_title[:60])
                return message.message_id
        except Exception as e:
            logger.error("Text publish error: %s", str(e)[:200])

        return None

    async def _send_with_retry(self, method, max_retries: int = 3, **kwargs) -> Optional[object]:
        """Send a Telegram message with retry and rate-limit handling."""
        for attempt in range(max_retries):
            try:
                result = await method(**kwargs)
                return result
            except RetryAfter as e:
                wait_time = e.retry_after + 1
                logger.warning("Telegram rate limit: waiting %d seconds.", wait_time)
                await asyncio.sleep(wait_time)
            except TimedOut:
                wait_time = 5 * (attempt + 1)
                logger.warning("Telegram timeout. Retrying in %d seconds.", wait_time)
                await asyncio.sleep(wait_time)
            except NetworkError as e:
                wait_time = 10 * (attempt + 1)
                logger.warning("Telegram network error: %s. Retrying in %ds.",
                               str(e)[:100], wait_time)
                await asyncio.sleep(wait_time)
            except BadRequest as e:
                # Don't retry bad requests (message format errors)
                logger.error("Telegram BadRequest: %s", str(e)[:200])
                raise
            except TelegramError as e:
                logger.error("Telegram error (attempt %d/%d): %s",
                             attempt + 1, max_retries, str(e)[:200])
                if attempt < max_retries - 1:
                    await asyncio.sleep(5 * (attempt + 1))

        logger.error("All %d publish attempts failed.", max_retries)
        return None

    async def test_connection(self) -> bool:
        """Test that the bot can reach the channel."""
        if not self._bot:
            return False
        try:
            chat = await self._bot.get_chat(self._channel_id)
            logger.info("Channel verified: %s (type=%s)", chat.title, chat.type)
            return True
        except TelegramError as e:
            logger.error("Channel verification failed: %s", str(e))
            return False
