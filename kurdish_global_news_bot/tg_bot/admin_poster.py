"""
Kurdish Global News Bot — Instant Admin Manual Poster

Allows the authorized channel admin (ADMIN_CHAT_ID) to send a Photo+Caption
or Text-only message directly to the Telegram bot in private chat and
immediately publish a branded 1280x720 news card to TELEGRAM_CHANNEL_ID.

- Strict security: only accepts messages from int(ADMIN_CHAT_ID).
- Zero LLM / translation calls: preserves admin Kurdish text verbatim.
- Photo+Caption: downloads highest-resolution photo (photo[-1]) for card_generator.py.
- Text-only: passes image_bytes=None so background_selector.py picks the matching theme.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

from telegram import Bot, Message
from telegram.constants import ParseMode

import config
from core.importance_ranker import (
    is_bashur_article,
    is_cybersecurity_article,
    is_economy_article,
    is_politics_or_war,
)
from processing.card_generator import NewsCardGenerator
from processing.content_formatter import ContentFormatter
from storage.models import Article

logger = logging.getLogger(__name__)

_BREAKING_PREFIX_RE = re.compile(
    r"^(?:🔴\s*)?بەپەلە(?:\s*[:|\-–—]\s*|\s+|$)"
)
_ADMIN_URL_RE = re.compile(
    r"(?:https?://[^\s<>\"'()]+|\b(?:t\.me|telegram\.me|telegram\.dog)/[^\s<>\"'()]+)",
    re.IGNORECASE,
)
_URL_TRAILING_PUNCT = ".,;!?،؛»)\"'|"


def extract_and_strip_admin_url(raw_text: str) -> tuple[str, str]:
    """
    Extract the first URL (`http(s)://...` or `t.me/...`) from admin text
    and strip all raw URLs from the body so captions only show a clean `🔗 سەرچاوە` link.
    """
    text = (raw_text or "").replace("\r\n", "\n").replace("\r", "\n")
    extracted_url = ""

    def _replace(match: re.Match[str]) -> str:
        nonlocal extracted_url
        candidate = match.group(0).rstrip(_URL_TRAILING_PUNCT)
        if not extracted_url and candidate:
            if not candidate.lower().startswith(("http://", "https://")):
                candidate = f"https://{candidate}"
            extracted_url = candidate
        return " "

    stripped = _ADMIN_URL_RE.sub(_replace, text)
    cleaned_lines: list[str] = []
    for line in stripped.split("\n"):
        # Clean stray prefix markers left after removing a standalone source URL
        line_clean = re.sub(
            r"(?:🔗|سەرچاوە|source)\s*[:\-–—]?\s*$",
            "",
            line,
            flags=re.IGNORECASE,
        )
        line_clean = re.sub(r"[ \t]{2,}", " ", line_clean).strip()
        if line_clean:
            cleaned_lines.append(line_clean)

    return "\n".join(cleaned_lines), extracted_url


def _parse_int_id(value: Any) -> Optional[int]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def is_authorized_admin_message(
    message: Optional[Message],
    admin_chat_id: Any = None,
) -> bool:
    """
    Strictly verify that the incoming Telegram message comes from ADMIN_CHAT_ID
    in a private chat. Any other user or missing ID is rejected/ignored.
    """
    if message is None:
        return False

    target_id = _parse_int_id(
        admin_chat_id if admin_chat_id is not None else config.ADMIN_CHAT_ID
    )
    if target_id is None:
        return False

    from_user = getattr(message, "from_user", None)
    if from_user is None or getattr(from_user, "id", None) is None:
        return False

    try:
        sender_id = int(from_user.id)
    except (TypeError, ValueError):
        return False

    if sender_id != target_id:
        return False

    chat = getattr(message, "chat", None)
    chat_type = getattr(chat, "type", None) if chat is not None else None
    if chat_type is not None and str(chat_type).lower() != "private":
        return False

    return True


def parse_admin_post_text(raw_text: str) -> tuple[str, str, bool]:
    """
    Parse admin message text into (headline, summary, is_breaking).

    - Detects leading 'بەپەلە' to set is_breaking=True without double-prefixing.
    - Multi-line: line 1 becomes headline, remaining lines become summary.
    - Single-line: becomes headline with empty summary (no duplicate body).
    """
    text = (raw_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return "", "", False

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return "", "", False

    is_breaking = False
    first_line = lines[0]
    match = _BREAKING_PREFIX_RE.match(first_line)
    if match:
        is_breaking = True
        remainder = first_line[match.end():].strip()
        if remainder:
            lines[0] = remainder
        elif len(lines) > 1:
            lines = lines[1:]
        else:
            lines[0] = first_line

    headline = lines[0]
    summary = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
    return headline, summary, is_breaking


def infer_admin_article_category(headline: str, summary: str) -> str:
    """
    Infer the best-fitting visual category badge and background theme hint
    from the admin's Kurdish text without altering the text itself.
    """
    probe = Article(
        original_title=headline,
        translated_title=headline,
        translated_summary=summary,
        source_name="",
    )
    if is_bashur_article(probe):
        return "kurdistan"
    if is_cybersecurity_article(probe):
        return "cybersecurity"
    if is_economy_article(probe):
        return "economy_energy"
    if is_politics_or_war(probe):
        return "military_conflict"
    return "politics_world"


def build_admin_article(raw_text: str, source_name: str = "") -> Article:
    """
    Construct an Article directly from the admin's Kurdish text with zero
    translation or LLM rewriting, extracting any embedded source URL cleanly.
    """
    clean_text, extracted_url = extract_and_strip_admin_url(raw_text)
    headline, summary, is_breaking = parse_admin_post_text(clean_text)
    category = infer_admin_article_category(headline, summary)
    return Article(
        original_title=headline,
        translated_title=headline,
        translated_summary=summary,
        original_url=extracted_url,
        category=category,
        source_name=source_name,
        is_breaking=is_breaking,
        importance_score=9.5 if is_breaking else 7.5,
    )


async def download_highest_res_photo(
    bot: Bot,
    message: Message,
) -> Optional[bytes]:
    """Download the highest-resolution photo attached to a Telegram message."""
    photos = getattr(message, "photo", None)
    if not photos:
        return None
    largest = photos[-1]
    tg_file = await bot.get_file(largest.file_id)
    data = await tg_file.download_as_bytearray()
    return bytes(data)


def extract_admin_video_file_id(message: Optional[Message]) -> Optional[str]:
    """
    Extract Telegram `file_id` for video, animation, or video/* document
    without downloading or re-encoding the media.
    """
    if message is None:
        return None

    for attr in ("video", "animation"):
        media = getattr(message, attr, None)
        file_id = getattr(media, "file_id", None) if media is not None else None
        if isinstance(file_id, str) and file_id.strip():
            return file_id.strip()

    doc = getattr(message, "document", None)
    if doc is not None:
        mime = str(getattr(doc, "mime_type", "") or "").strip().lower()
        file_id = getattr(doc, "file_id", None)
        if mime.startswith("video/") and isinstance(file_id, str) and file_id.strip():
            return file_id.strip()

    return None


def _build_channel_message_link(
    channel_id: Any,
    message_id: Optional[int],
    channel_handle: str = "",
) -> str:
    if not message_id:
        return ""
    handle = (channel_handle or "").strip().lstrip("@")
    if handle:
        return f"https://t.me/{handle}/{message_id}"
    cid = str(channel_id or "").strip()
    if cid.startswith("@"):
        return f"https://t.me/{cid.lstrip('@')}/{message_id}"
    if cid.startswith("-100") and cid[4:].isdigit():
        return f"https://t.me/c/{cid[4:]}/{message_id}"
    return ""


async def _reply_admin(bot: Bot, message: Message, text: str) -> None:
    chat_id = getattr(getattr(message, "chat", None), "id", None) or getattr(
        getattr(message, "from_user", None), "id", None
    )
    if chat_id is None:
        return
    try:
        await bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_to_message_id=getattr(message, "message_id", None),
            disable_web_page_preview=True,
        )
    except Exception as exc:
        logger.warning("Failed to reply to admin chat_id=%s: %s", chat_id, exc)


async def publish_admin_manual_post(
    bot: Bot,
    message: Optional[Message],
    *,
    card_generator: Optional[NewsCardGenerator] = None,
    formatter: Optional[ContentFormatter] = None,
    channel_id: Any = None,
    admin_chat_id: Any = None,
    channel_handle: Optional[str] = None,
    video_file_id: Optional[str] = None,
    caption_override: Optional[str] = None,
) -> dict[str, Any]:
    """
    Validate admin authority, render the branded news card (or pass through video file_id),
    publish immediately to TELEGRAM_CHANNEL_ID, and confirm back to the admin in private chat.
    """
    if not is_authorized_admin_message(message, admin_chat_id=admin_chat_id):
        sender = getattr(getattr(message, "from_user", None), "id", None)
        logger.warning("Blocked unauthorized manual post attempt from user_id=%s", sender)
        return {"status": "ignored", "reason": "unauthorized"}

    assert message is not None
    raw_text = (
        caption_override
        if caption_override is not None
        else (
            getattr(message, "caption", None)
            or getattr(message, "text", None)
            or ""
        )
    ).strip()

    if not raw_text:
        await _reply_admin(
            bot,
            message,
            "⚠️ تکایە دەق یان سەردێڕ بنێرە لەگەڵ پۆستەکەدا.",
        )
        return {"status": "rejected", "reason": "empty_text"}

    target_channel = channel_id if channel_id is not None else config.TELEGRAM_CHANNEL_ID
    handle = channel_handle if channel_handle is not None else config.CHANNEL_HANDLE
    generator = card_generator or NewsCardGenerator()
    fmt = formatter or ContentFormatter()
    resolved_video_id = (
        video_file_id.strip()
        if isinstance(video_file_id, str) and video_file_id.strip()
        else extract_admin_video_file_id(message)
    )

    card_path: Optional[Path] = None
    try:
        article = build_admin_article(raw_text)
        if not article.translated_title:
            await _reply_admin(
                bot,
                message,
                "⚠️ تکایە دەق یان سەردێڕ بنێرە لەگەڵ پۆستەکەدا.",
            )
            return {"status": "rejected", "reason": "empty_text"}

        caption = fmt.format_caption(article, has_inline_source_button=False)

        if resolved_video_id:
            sent_msg = await bot.send_video(
                chat_id=target_channel,
                video=resolved_video_id,
                caption=caption[:1024],
                parse_mode=ParseMode.HTML,
                disable_notification=not article.is_breaking,
                supports_streaming=True,
            )
            used_photo = False
            used_video = True
        else:
            image_bytes = await download_highest_res_photo(bot, message)
            card_path = await generator.generate_card_file(
                article,
                image_bytes=image_bytes,
                is_breaking=article.is_breaking,
            )
            with open(card_path, "rb") as photo_fh:
                sent_msg = await bot.send_photo(
                    chat_id=target_channel,
                    photo=photo_fh,
                    caption=caption[:1024],
                    parse_mode=ParseMode.HTML,
                    disable_notification=not article.is_breaking,
                )
            used_photo = image_bytes is not None
            used_video = False

        msg_id = getattr(sent_msg, "message_id", None)
        link = _build_channel_message_link(target_channel, msg_id, handle)
        confirm = f"✅ بڵاوکرایەوە\n🆔 {msg_id}" if msg_id else "✅ بڵاوکرایەوە"
        if link:
            confirm = f"{confirm}\n🔗 {link}"
        await _reply_admin(bot, message, confirm)

        return {
            "status": "published",
            "message_id": msg_id,
            "is_breaking": article.is_breaking,
            "category": article.category,
            "used_photo": used_photo,
            "used_video": used_video,
        }
    except Exception as exc:
        logger.exception("Admin manual publish failed: %s", exc)
        await _reply_admin(bot, message, f"❌ هەڵە لە بڵاوکردنەوە:\n{exc}")
        return {"status": "error", "error": str(exc)}
    finally:
        if card_path is not None:
            try:
                Path(card_path).unlink(missing_ok=True)
            except Exception as cleanup_exc:
                logger.debug("Temp card cleanup failed (%s): %s", card_path, cleanup_exc)
