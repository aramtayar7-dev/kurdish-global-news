"""
Kurdish Global News Bot — Telegram Publisher

Publishes Gemini-translated Kurdish news articles with branded 1280x720 news
graphic cards via send_photo, clean sentence-boundary caption trimming,
IPv4-hardened HTTPXRequest, and temporary card file cleanup.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import httpx
from telegram import Bot, LinkPreviewOptions
from telegram.constants import ParseMode
from telegram.error import BadRequest, NetworkError, RetryAfter, TelegramError, TimedOut
from telegram.request import HTTPXRequest

from config import (
    ENABLE_IMAGES,
    NOTIFY_DAILY_DIGEST,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHANNEL_ID,
)
from core.importance_ranker import is_strict_breaking
from core.source_verifier import is_official_media_source, is_x_source_url
from processing.card_generator import NewsCardGenerator
from processing.content_formatter import ContentFormatter
from processing.translator import is_valid_kurdish_sorani, trim_at_sentence_boundary
from processing.video_processor import (
    WatermarkRejectedError,
    _WATERMARK_POLICY_CTX,
    cleanup_video_file,
    download_and_brand_video,
)
from processing.video_watermark_detector import has_image_corner_watermark
from storage.models import Article
from tg_bot.image_handler import ImageHandler
from tg_bot.inline_buttons import build_source_button_markup

logger = logging.getLogger("tg_bot.publisher")


class TelegramPublisher:
    """Publishes news articles with branded graphic cards to a Telegram channel."""

    def __init__(self) -> None:
        self._bot: Optional[Bot] = None
        self._formatter = ContentFormatter()
        self._image_handler = ImageHandler()
        self._card_generator = NewsCardGenerator()
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

    @staticmethod
    def _trim_caption_at_sentence(caption: str, max_len: int = 1024) -> str:
        """Ensure caption never exceeds Telegram's limit or breaks mid-sentence."""
        if len(caption) <= max_len:
            return caption
        return trim_at_sentence_boundary(caption, max_len=max_len)

    async def publish_article(self, article: Article) -> Optional[int]:
        """
        Publish an article immediately:
        - When `video_url` is present: download & watermark with KGN logo and send via sendVideo.
        - Otherwise (or on video fallback): generate branded 1280x720 card and send via sendPhoto.
        - Falls back cleanly to text-only if photo card upload fails.
        """
        if not self._bot:
            raise RuntimeError("Publisher not initialized. Call initialize() first.")

        # Strict Sorani Kurdish publish gate: never generate cards or publish untranslated/Arabic text
        if not is_valid_kurdish_sorani(article.translated_title or ""):
            logger.warning(
                "Publish ABORTED (Sorani gate): non-Sorani or empty translated headline for '%s'",
                (article.original_title or "")[:80],
            )
            return None
        if (article.translated_summary or "").strip() and not is_valid_kurdish_sorani(article.translated_summary):
            logger.warning(
                "Publish ABORTED (Sorani gate): non-Sorani translated summary for '%s'",
                (article.original_title or "")[:80],
            )
            return None

        if getattr(article, "video_url", None):
            message_id = await self._publish_with_video(article)
            if message_id is not None:
                return message_id
            if getattr(article, "_watermark_rejected_reason", None):
                logger.info(
                    "Publish ABORTED (Watermark gate): %s — skipping card/text fallback.",
                    getattr(article, "_watermark_rejected_reason", ""),
                )
                return None
            logger.info("Video publish/branding failed or unavailable, falling back to branded image card.")

        if ENABLE_IMAGES:
            message_id = await self._publish_with_card(article)
            if message_id is not None:
                return message_id
            logger.info("Card photo publish failed, falling back to text-only.")

        return await self._publish_text_only(article)

    async def _send_with_optional_reply(
        self,
        method,
        *,
        reply_to_message_id: Optional[int] = None,
        rewind_file: Optional[object] = None,
        **kwargs,
    ) -> Optional[object]:
        """
        Send via Telegram Bot API with reply metadata when `reply_to_message_id` is set.
        Falls back cleanly to non-reply publishing if Telegram rejects a deleted root message.
        """
        send_kwargs = dict(kwargs)
        if reply_to_message_id is not None:
            send_kwargs["reply_to_message_id"] = int(reply_to_message_id)
            send_kwargs["allow_sending_without_reply"] = True

        try:
            return await self._send_with_retry(method, **send_kwargs)
        except BadRequest as exc:
            err_lower = str(exc).lower()
            is_button_err = send_kwargs.get("reply_markup") is not None and any(
                tok in err_lower
                for tok in ("button_url_invalid", "reply_markup", "inline keyboard", "button_user_invalid")
            )
            if is_button_err:
                logger.warning(
                    "Telegram rejected inline reply_markup (%s); retrying cleanly without button.",
                    str(exc)[:140],
                )
                if rewind_file is not None and hasattr(rewind_file, "seek"):
                    try:
                        rewind_file.seek(0)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                send_kwargs.pop("reply_markup", None)
                try:
                    return await self._send_with_retry(method, **send_kwargs)
                except BadRequest as exc2:
                    exc = exc2
                    err_lower = str(exc2).lower()

            is_reply_err = reply_to_message_id is not None and any(
                tok in err_lower
                for tok in ("reply", "message to be replied", "replied", "not found")
            )
            if not is_reply_err:
                raise
            logger.warning(
                "Root message %s unavailable for reply (%s); publishing cleanly without reply anchor.",
                reply_to_message_id,
                str(exc)[:120],
            )
            if rewind_file is not None and hasattr(rewind_file, "seek"):
                try:
                    rewind_file.seek(0)  # type: ignore[attr-defined]
                except Exception:
                    pass
            send_kwargs.pop("reply_to_message_id", None)
            send_kwargs.pop("allow_sending_without_reply", None)
            return await self._send_with_retry(method, **send_kwargs)

    async def _publish_with_video(self, article: Article) -> Optional[int]:
        """Download & watermark video with KGN logo, send via send_video, and clean up temp file."""
        assert self._bot is not None
        if not article.video_url:
            return None

        breaking = is_strict_breaking(article)
        article.is_breaking = breaking
        disable_notification = not breaking
        reply_root_id = getattr(article, "thread_root_message_id", None)

        pre_branded = getattr(article, "_prepared_branded_video_path", None)
        if pre_branded and Path(pre_branded).is_file():
            branded_video_path = str(pre_branded)
            article._prepared_branded_video_path = None  # type: ignore[attr-defined]
        else:
            is_official = is_official_media_source(
                getattr(article, "source_name", "") or "",
                getattr(article, "original_url", "") or "",
            )
            is_x = is_x_source_url(getattr(article, "original_url", "") or "")
            text_hint = (
                f"{getattr(article, 'original_title', '') or ''} "
                f"{getattr(article, 'original_summary', '') or ''} "
                f"{getattr(article, 'original_url', '') or ''}"
            )
            ctx_token = _WATERMARK_POLICY_CTX.set(
                {
                    "is_official_media": is_official,
                    "check_watermarks": bool(is_x or not is_official),
                    "text_hint": text_hint,
                }
            )
            try:
                branded_video_path = await download_and_brand_video(article.video_url)
            except WatermarkRejectedError as exc:
                article._watermark_rejected_reason = str(exc)  # type: ignore[attr-defined]
                logger.info("Video rejected by watermark policy: %s", exc)
                return None
            finally:
                _WATERMARK_POLICY_CTX.reset(ctx_token)
            if not branded_video_path:
                return None

        reply_markup = build_source_button_markup(article=article)
        caption = self._trim_caption_at_sentence(
            self._formatter.format_caption(article, has_inline_source_button=(reply_markup is not None))
        )
        try:
            with open(branded_video_path, "rb") as video_file:
                send_extra = {"reply_markup": reply_markup} if reply_markup is not None else {}
                message = await self._send_with_optional_reply(
                    self._bot.send_video,
                    reply_to_message_id=reply_root_id,
                    rewind_file=video_file,
                    chat_id=self._channel_id,
                    video=video_file,
                    caption=caption,
                    parse_mode=ParseMode.HTML,
                    supports_streaming=True,
                    disable_notification=disable_notification,
                    **send_extra,
                )
            if message:
                logger.info(
                    "Published branded video: msg_id=%d, reply_to=%s, breaking=%s, silent=%s, source=%s, category=%s, title='%s'",
                    message.message_id,
                    reply_root_id,
                    breaking,
                    disable_notification,
                    article.source_name,
                    article.category,
                    article.original_title[:60],
                )
                return message.message_id
        except BadRequest as e:
            logger.warning("Telegram rejected branded video: %s", str(e)[:200])
        except Exception as e:
            logger.error("Video publish error: %s", str(e)[:200])
        finally:
            cleanup_video_file(branded_video_path)

        return None

    async def _publish_with_card(self, article: Article) -> Optional[int]:
        """Generate branded 1280x720 news card, send via send_photo, and clean up temp file."""
        assert self._bot is not None

        breaking = is_strict_breaking(article)
        article.is_breaking = breaking
        disable_notification = not breaking
        reply_root_id = getattr(article, "thread_root_message_id", None)

        image_bytes: Optional[bytes] = None
        if article.image_url:
            image_bytes = await self._image_handler.download_image(article.image_url)
            if image_bytes is not None:
                is_official_img = is_official_media_source(
                    getattr(article, "source_name", "") or "",
                    getattr(article, "original_url", "") or "",
                )
                if not is_official_img:
                    img_hint = (
                        f"{getattr(article, 'original_title', '') or ''} "
                        f"{getattr(article, 'original_url', '') or ''}"
                    )
                    has_wm, wm_reason = has_image_corner_watermark(image_bytes, text_hint=img_hint)
                    if has_wm:
                        logger.info(
                            "Stripping watermarked image (%s) from unofficial source '%s'; using clean background.",
                            wm_reason,
                            getattr(article, "source_name", "Unknown"),
                        )
                        image_bytes = None
                        article.image_url = ""

        reply_markup = build_source_button_markup(article=article)
        caption = self._trim_caption_at_sentence(
            self._formatter.format_caption(article, has_inline_source_button=(reply_markup is not None))
        )
        card_path: Optional[Path] = None

        try:
            card_path = await self._card_generator.generate_card_file(
                article, image_bytes=image_bytes, is_breaking=breaking
            )
            with open(card_path, "rb") as photo_file:
                send_extra = {"reply_markup": reply_markup} if reply_markup is not None else {}
                message = await self._send_with_optional_reply(
                    self._bot.send_photo,
                    reply_to_message_id=reply_root_id,
                    rewind_file=photo_file,
                    chat_id=self._channel_id,
                    photo=photo_file,
                    caption=caption,
                    parse_mode=ParseMode.HTML,
                    disable_notification=disable_notification,
                    **send_extra,
                )
            if message:
                logger.info(
                    "Published branded card: msg_id=%d, reply_to=%s, breaking=%s, silent=%s, category=%s, title='%s'",
                    message.message_id,
                    reply_root_id,
                    breaking,
                    disable_notification,
                    article.category,
                    article.original_title[:60],
                )
                return message.message_id
        except BadRequest as e:
            logger.warning("Telegram rejected branded card: %s", str(e)[:200])
        except Exception as e:
            logger.error("Card publish error: %s", str(e)[:200])
        finally:
            if card_path is not None:
                try:
                    card_path.unlink(missing_ok=True)
                except Exception as cleanup_err:
                    logger.debug("Temp card cleanup failed (%s): %s", card_path, cleanup_err)

        return None

    async def _publish_text_only(self, article: Article) -> Optional[int]:
        """Publish a text-only article with web page preview disabled."""
        assert self._bot is not None

        breaking = is_strict_breaking(article)
        article.is_breaking = breaking
        disable_notification = not breaking
        reply_root_id = getattr(article, "thread_root_message_id", None)
        reply_markup = build_source_button_markup(article=article)
        text = self._formatter.format_telegram_message(
            article, has_inline_source_button=(reply_markup is not None)
        )

        try:
            send_extra = {"reply_markup": reply_markup} if reply_markup is not None else {}
            message = await self._send_with_optional_reply(
                self._bot.send_message,
                reply_to_message_id=reply_root_id,
                chat_id=self._channel_id,
                text=text,
                parse_mode=ParseMode.HTML,
                link_preview_options=LinkPreviewOptions(is_disabled=True),
                disable_notification=disable_notification,
                **send_extra,
            )
            if message:
                logger.info(
                    "Published text-only: msg_id=%d, reply_to=%s, breaking=%s, silent=%s, title='%s'",
                    message.message_id,
                    reply_root_id,
                    breaking,
                    disable_notification,
                    article.original_title[:60],
                )
                return message.message_id
        except Exception as e:
            logger.error("Text publish error: %s", str(e)[:200])

        return None

    async def publish_daily_digest(
        self,
        digest_text: str,
        card_path: Path,
        header_caption: str = "",
        disable_notification: Optional[bool] = None,
    ) -> Optional[int]:
        """
        Publish the nightly Kurdish news digest cover card and roundup text:
        - If digest_text <= 1024 chars: send photo with digest_text as caption.
        - If digest_text > 1024 chars: send photo with short header caption,
          followed immediately by the full digest text (web preview disabled).
        Defaults to silent delivery unless NOTIFY_DAILY_DIGEST is True.
        Always cleans up temporary card_path afterward.
        """
        if not self._bot:
            raise RuntimeError("Publisher not initialized. Call initialize() first.")

        silent = (not NOTIFY_DAILY_DIGEST) if disable_notification is None else bool(disable_notification)

        try:
            if len(digest_text) <= 1024:
                with open(card_path, "rb") as photo_file:
                    message = await self._send_with_retry(
                        self._bot.send_photo,
                        chat_id=self._channel_id,
                        photo=photo_file,
                        caption=digest_text,
                        parse_mode=ParseMode.HTML,
                        disable_notification=silent,
                    )
                if message:
                    logger.info(
                        "Published nightly digest card + caption: msg_id=%d, silent=%s",
                        message.message_id, silent,
                    )
                    return message.message_id
                return None

            # > 1024 chars: photo with concise header + full text follow-up
            short_caption = (
                header_caption.strip()
                or "<b>🌙 پوختەی گرنگترین هەواڵەکانی ئەمڕۆ</b>\n📡 @KurdishGlobalNews"
            )[:1024]
            with open(card_path, "rb") as photo_file:
                photo_msg = await self._send_with_retry(
                    self._bot.send_photo,
                    chat_id=self._channel_id,
                    photo=photo_file,
                    caption=short_caption,
                    parse_mode=ParseMode.HTML,
                    disable_notification=silent,
                )

            text_msg = await self._send_with_retry(
                self._bot.send_message,
                chat_id=self._channel_id,
                text=digest_text,
                parse_mode=ParseMode.HTML,
                link_preview_options=LinkPreviewOptions(is_disabled=True),
                disable_notification=silent,
            )
            if text_msg:
                logger.info(
                    "Published nightly digest card + full text follow-up: msg_id=%d, silent=%s",
                    text_msg.message_id, silent,
                )
                return text_msg.message_id
            return photo_msg.message_id if photo_msg else None
        except Exception as e:
            logger.error("Daily digest publish error: %s", str(e)[:200])
            return None
        finally:
            try:
                card_path.unlink(missing_ok=True)
            except Exception:
                pass

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
            except BadRequest as e:
                logger.error("Telegram BadRequest: %s", str(e)[:200])
                raise
            except NetworkError as e:
                wait_time = 10 * (attempt + 1)
                logger.warning(
                    "Telegram network error: %s. Retrying in %ds.",
                    str(e)[:100],
                    wait_time,
                )
                await asyncio.sleep(wait_time)
            except TelegramError as e:
                logger.error(
                    "Telegram error (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    str(e)[:200],
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(5 * (attempt + 1))

        logger.error("All %d publish attempts failed.", max_retries)
        return None

    async def send_admin_message(
        self,
        chat_id: str,
        text: str,
        disable_notification: bool = False,
    ) -> bool:
        """
        Send a private operational watchdog message strictly to `chat_id`.
        Refuses if target is missing, invalid, or matches public channel `self._channel_id`.
        Never raises exceptions (non-blocking).
        """
        from monitoring.watchdog import is_valid_admin_target, redact_secrets

        if not is_valid_admin_target(chat_id, self._channel_id):
            logger.warning(
                "Refusing admin message: invalid target or matches public channel (%s).",
                self._channel_id,
            )
            return False

        if not self._bot:
            logger.warning("Cannot send admin message: Telegram bot client not initialized.")
            return False

        safe_text = redact_secrets(text)[:4000]
        try:
            await self._bot.send_message(
                chat_id=chat_id.strip(),
                text=safe_text,
                parse_mode=ParseMode.HTML,
                link_preview_options=LinkPreviewOptions(is_disabled=True),
                disable_notification=disable_notification,
            )
            return True
        except Exception as exc:
            logger.warning(
                "Failed to deliver private admin watchdog message (non-blocking): %s",
                redact_secrets(str(exc))[:200],
            )
            return False

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
