"""
Kurdish Global News Bot — Public Telegram Web News Discovery Source

Fetches public web preview pages (`https://t.me/s/{username}`) via aiohttp and
BeautifulSoup without Telegram authentication, login credentials, or private access.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import html
import logging
import re
from typing import Any, Optional

import aiohttp
from bs4 import BeautifulSoup, Tag
from dateutil import parser as dt_parser

from config import MAX_ARTICLE_AGE_HOURS, REQUEST_TIMEOUT, TELEGRAM_SOURCE_CHANNELS
from processing.validator import SPAM_PATTERNS, ArticleValidator
from sources.base import BaseSource, RawArticle

logger = logging.getLogger("sources.telegram")

TELEGRAM_WEB_BASE_URL = "https://t.me/s/{username}"

TELEGRAM_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_RESTRICTED_PREVIEW_MARKERS = (
    "please open telegram to view this post",
    "this channel can't be displayed",
    "channel cannot be displayed",
    "you can view and join",
)

# Decorative propaganda & channel-formatting artifacts stripped cleanly from text
_DECORATIVE_ARTIFACTS_RE = re.compile(
    r"[\U0001F1E6-\U0001F1FF]|"
    r"[🔺🔻⭕▪️▫️🟡🟢🟠🔵⚫⚪🚨💥🔥⚡📌📍📸🎥📢📲⚠️✅❌🔴🟢⬇️⬆️➡️⬅️⏩⏪🔔📢💬✍️🕊️🇮🇶🇺🇸🇮🇱🇮🇷🇵🇸🇱🇧🇾🇪🇸🇾🇹🇷]+|"
    r"[\u200d\ufe0f]",
)

_HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{3,}\b")
_TME_URL_RE = re.compile(r"(?:https?://)?(?:t|telegram)\.me/[^\s<>()]+", re.IGNORECASE)
_GENERIC_URL_RE = re.compile(r"https?://[^\s<>()]+", re.IGNORECASE)
_HASHTAG_ONLY_LINE_RE = re.compile(r"^\s*(?:#[\w\u0621-\u064A\u0671-\u06D5]+\s*)+$")
_INLINE_HASHTAG_RE = re.compile(r"#([\w\u0621-\u064A\u0671-\u06D5]+)")

_SIGNATURE_LINE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*(?:قناة\s+)?صابرين\s+نيوز(?:\s*بلس)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*sabereen\s+(?:news|plus)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:للمزيد|للاشتراك|تابعونا|انضموا\s+إلى|اشترك\s+في)\b.*", re.IGNORECASE),
    re.compile(r"^\s*(?:منقول|عبر\s*:|تێلێگرام\s*:|تلغرام\s*:).*$", re.IGNORECASE),
    re.compile(r"^\s*(?:media\s+is\s+too\s+big|please\s+open\s+telegram\b.*)$", re.IGNORECASE),
]

_TRAFFIC_PROMO_MUNICIPAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(للاعلان(?:ات)?|للتواصل\s+والإعلان|تخفيضات|خصم\s+\d+%|عرض\s+خاص|"
        r"اشترك(?:وا)?\s+(?:الآن|في\s+قناتنا)|احجز\s+الآن|فرصة\s+عمل|تعيينات\s+شركة|ڕیکلام)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(تنويه\s+مروري|مديرية\s+المرور|شرطة\s+المرور|مفارز\s+المرور|غرامة\s+مرورية|"
        r"رادار(?:ات)?(?:\s+المرور|\s+السرعة)?|كاميرات\s+(?:المراقبة\s+المرورية|السرعة|المرور)|"
        r"ازدحام\s+مروري|کامێرای\s+(?:خێرایی|هاتووچۆ)|بەڕێوەبەرایەتی\s+هاتووچۆ|"
        r"traffic\s+radar|speed\s+camera)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(جدول\s+(?:قطع\s+)?الكهرباء|انقطاع\s+التيار\s+الكهربائي|جدول\s+المولدات|"
        r"صيانة\s+شبكة\s+المياه|أمانة\s+بغداد.{0,40}صيانة|انقطاع\s+المياه.{0,40}صيانة|"
        r"ساعات\s+التشغيل\s+للمولدات|خشتەی\s+کارەبا|پچڕانی\s+کارەبا)",
        re.IGNORECASE,
    ),
]

_BACKGROUND_IMAGE_URL_RE = re.compile(
    r"background-image\s*:\s*url\(\s*(?:\\?['\"])?(https?://[^'\"\)\s\\]+)(?:\\?['\"])?\s*\)",
    re.IGNORECASE,
)

_MEANINGFUL_CHAR_RE = re.compile(r"[A-Za-z0-9\u0621-\u064A\u0671-\u06D5]")


def clean_telegram_message_text(raw_text: str) -> str:
    """
    Clean raw Telegram message text by removing irrelevant @handles, t.me URLs,
    promotional hashtags, forwarded signatures, watermarks, and decorative emojis
    while preserving factual sentences, names, locations, dates, and numbers.
    """
    if not raw_text:
        return ""

    text = html.unescape(raw_text)
    cleaned_lines: list[str] = []

    for line in text.splitlines():
        work = line.strip()
        if not work:
            continue
        if _HASHTAG_ONLY_LINE_RE.match(work):
            continue

        work = _TME_URL_RE.sub("", work)
        work = _GENERIC_URL_RE.sub("", work)
        work = _HANDLE_RE.sub("", work)
        work = _DECORATIVE_ARTIFACTS_RE.sub(" ", work)
        # Turn inline topical hashtags like #بغداد into plain words while dropping # symbol
        work = _INLINE_HASHTAG_RE.sub(r"\1", work)
        work = re.sub(r"\s+", " ", work).strip(" \t\r-–—|")

        if not work:
            continue
        if any(sig.match(work) for sig in _SIGNATURE_LINE_PATTERNS):
            continue
        cleaned_lines.append(work)

    result = "\n".join(cleaned_lines).strip()
    return re.sub(r"\n{3,}", "\n\n", result)


def is_rejected_telegram_content(cleaned_text: str) -> tuple[bool, str]:
    """
    Validate cleaned Telegram text against minimum meaningful length, advertisements,
    channel promotions, traffic/radar notices, and municipal utility alerts.
    """
    if not cleaned_text or not cleaned_text.strip():
        return True, "Empty message text"

    meaningful_count = len(_MEANINGFUL_CHAR_RE.findall(cleaned_text))
    if meaningful_count < 20:
        return True, f"Caption too short ({meaningful_count} < 20 meaningful chars)"

    for spam_re in SPAM_PATTERNS:
        if spam_re.search(cleaned_text):
            return True, "Matched spam/advertisement pattern"

    for promo_re in _TRAFFIC_PROMO_MUNICIPAL_PATTERNS:
        if promo_re.search(cleaned_text):
            return True, "Rejected Telegram ad, traffic, or municipal-service notice"

    return False, ""


def _extract_background_image_url_from_tags(tags: list[Tag]) -> str:
    """Extract first valid http(s) URL from inline CSS `background-image:url(...)` attributes."""
    for wrap in tags:
        raw_style = wrap.get("style")
        if not raw_style or not isinstance(raw_style, str):
            continue
        unescaped_style = html.unescape(raw_style)
        match = _BACKGROUND_IMAGE_URL_RE.search(unescaped_style)
        if match:
            candidate = match.group(1).strip()
            if candidate.startswith(("http://", "https://")):
                return candidate
    return ""


def extract_telegram_photo_url(msg_tag: Tag) -> str:
    """
    Extract attached photo URL from `.tgme_widget_message_photo_wrap` CSS
    `background-image:url(...)`, handling quoted, unquoted, and escaped variants.
    Returns empty string if missing or invalid.
    """
    return _extract_background_image_url_from_tags(
        msg_tag.select(".tgme_widget_message_photo_wrap")
    )


def extract_telegram_video_url(msg_tag: Tag) -> Optional[str]:
    """
    Extract direct MP4/video URL from Telegram video containers inside `.tgme_widget_message`.
    Checks `<video src="...">`, `<video><source src="...">`, and `data-src`.
    Returns None cleanly when missing or when Telegram renders a 'Media is too big' placeholder.
    """
    video_tags = msg_tag.select(
        ".tgme_widget_message_video_wrap video, "
        ".tgme_widget_message_video_player video, "
        ".tgme_widget_message_roundvideo_wrap video, "
        "video"
    )
    for video in video_tags:
        candidates: list[object] = [video.get("src"), video.get("data-src")]
        for source_tag in video.select("source"):
            candidates.extend([source_tag.get("src"), source_tag.get("data-src")])
        for raw_cand in candidates:
            if not raw_cand or not isinstance(raw_cand, str):
                continue
            cand = html.unescape(raw_cand).strip()
            if cand.startswith(("http://", "https://")):
                return cand
    return None


def extract_telegram_video_thumb_url(msg_tag: Tag) -> str:
    """
    Extract video preview thumbnail URL from `.tgme_widget_message_video_thumb`
    (or `.tgme_widget_message_roundvideo_thumb` / `<video poster="...">`) as a reliable fallback.
    """
    thumb_url = _extract_background_image_url_from_tags(
        msg_tag.select(
            ".tgme_widget_message_video_thumb, "
            ".tgme_widget_message_roundvideo_thumb, "
            ".tgme_widget_message_video_wrap [style]"
        )
    )
    if thumb_url:
        return thumb_url

    for video in msg_tag.select(
        ".tgme_widget_message_video_wrap video, "
        ".tgme_widget_message_video_player video, "
        "video"
    ):
        poster = video.get("poster")
        if poster and isinstance(poster, str):
            cand = html.unescape(poster).strip()
            if cand.startswith(("http://", "https://")):
                return cand
    return ""


def parse_telegram_timestamp(raw_datetime: str) -> Optional[datetime]:
    """Parse `time[datetime]` value into a timezone-aware UTC datetime, or None."""
    if not raw_datetime or not raw_datetime.strip():
        return None
    try:
        dt = dt_parser.parse(raw_datetime.strip())
    except Exception:
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _derive_headline_and_summary(cleaned_text: str) -> tuple[str, str]:
    """Build concise headline + complete cleaned summary for RawArticle."""
    flat = re.sub(r"\s+", " ", cleaned_text).strip()
    lines = [ln.strip() for ln in cleaned_text.splitlines() if ln.strip()]
    first_block = lines[0] if lines else flat

    # Split at first sentence boundary inside first line if long
    sentence_match = re.search(r"[.!؟]", first_block[:130])
    if sentence_match and sentence_match.start() >= 25:
        headline = first_block[: sentence_match.start() + 1].strip()
    elif len(first_block) <= 120:
        headline = first_block
    else:
        cut = first_block.rfind(" ", 0, 120)
        headline = first_block[: (cut if cut > 40 else 120)].strip()

    # Keep summary <= 1500 chars cleanly (matching RSSSource truncation cap)
    return headline[:200], flat[:1500]


class TelegramSource(BaseSource):
    """Fetches public Telegram web preview feeds (`https://t.me/s/{username}`)."""

    def __init__(
        self,
        channels: Optional[list[dict[str, str]]] = None,
        request_timeout: Optional[int] = None,
        max_age_hours: Optional[float] = None,
        user_agent: str = TELEGRAM_BROWSER_USER_AGENT,
    ) -> None:
        super().__init__(name="TelegramPublicWeb", source_type="telegram_web", enabled=True)
        self._channels: list[dict[str, str]] = (
            list(TELEGRAM_SOURCE_CHANNELS) if channels is None else list(channels)
        )
        self._timeout = REQUEST_TIMEOUT if request_timeout is None else int(request_timeout)
        self._max_age_hours = MAX_ARTICLE_AGE_HOURS if max_age_hours is None else float(max_age_hours)
        self._headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ar,en-US;q=0.9,en;q=0.8",
        }
        self._session: Optional[aiohttp.ClientSession] = None

    async def initialize(self) -> None:
        """Create aiohttp client session with realistic browser headers."""
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        self._session = aiohttp.ClientSession(
            timeout=timeout,
            headers=self._headers,
        )

    async def cleanup(self) -> None:
        """Close owned aiohttp client session safely."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def parse_channel_html(self, html_content: str, channel_cfg: dict[str, str]) -> list[RawArticle]:
        """
        Parse Telegram public-web HTML (`t.me/s/{username}`) into RawArticle entries.
        Handles missing fields, malformed DOM fragments, and restricted previews safely.
        """
        if not html_content or not html_content.strip():
            return []

        username = (channel_cfg.get("username") or "").strip().lstrip("@")
        display_name = (channel_cfg.get("name") or username).strip()
        region_hint = (channel_cfg.get("region_hint") or "").strip()
        if not username:
            return []

        lower_html = html_content.lower()
        soup = BeautifulSoup(html_content, "html.parser")
        message_nodes = soup.select(".tgme_widget_message")

        if not message_nodes:
            if any(marker in lower_html for marker in _RESTRICTED_PREVIEW_MARKERS) or soup.select_one(
                ".tgme_channel_info_error"
            ):
                logger.warning("Telegram channel '%s' preview restricted or unavailable.", username)
            return []

        articles: list[RawArticle] = []
        id_href_re = re.compile(rf"^https?://t\.me/{re.escape(username)}/(\d+)/?$", re.IGNORECASE)

        for msg_node in message_nodes:
            try:
                post_id = self._extract_post_id(msg_node, username, id_href_re)
                if not post_id:
                    continue

                pub_dt = self._extract_pub_datetime(msg_node)
                if pub_dt is None:
                    continue

                pub_iso = pub_dt.isoformat()
                is_recent, _ = ArticleValidator.is_recent(
                    pub_iso, max_age_hours=self._max_age_hours
                )
                if not is_recent:
                    continue

                text_node = msg_node.select_one(".tgme_widget_message_text")
                if text_node is None:
                    continue

                raw_text = text_node.get_text("\n")
                cleaned_text = clean_telegram_message_text(raw_text)
                rejected, _ = is_rejected_telegram_content(cleaned_text)
                if rejected:
                    continue

                headline, summary = _derive_headline_and_summary(cleaned_text)
                if not headline or not summary:
                    continue

                photo_url = extract_telegram_photo_url(msg_node)
                video_thumb_url = extract_telegram_video_thumb_url(msg_node)
                image_url = photo_url or video_thumb_url
                video_url = extract_telegram_video_url(msg_node)
                canonical_url = f"https://t.me/{username}/{post_id}"

                articles.append(
                    RawArticle(
                        url=canonical_url,
                        title=headline,
                        summary=summary,
                        source_name=display_name,
                        category="",
                        publication_date=pub_iso,
                        image_url=image_url,
                        video_url=video_url,
                        country=region_hint,
                        region=region_hint,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Malformed Telegram message in channel '%s': %s",
                    username,
                    str(exc)[:150],
                )
                continue

        return articles

    @staticmethod
    def _extract_post_id(
        msg_node: Tag,
        username: str,
        id_href_re: re.Pattern[str],
    ) -> str:
        """Extract numeric post_id from data-post attribute or canonical date link."""
        raw_post = str(msg_node.get("data-post") or "").strip()
        if "/" in raw_post:
            _, candidate_id = raw_post.split("/", 1)
            candidate_id = candidate_id.strip()
            if candidate_id.isdigit():
                return candidate_id

        for link in msg_node.select("a.tgme_widget_message_date[href], a[href]"):
            href = str(link.get("href") or "").strip()
            m = id_href_re.match(href)
            if m:
                return m.group(1)

        return ""

    @staticmethod
    def _extract_pub_datetime(msg_node: Tag) -> Optional[datetime]:
        """Extract and normalize time[datetime] into UTC timezone-aware datetime."""
        time_tag = msg_node.select_one("time[datetime]")
        if time_tag is None:
            return None
        return parse_telegram_timestamp(str(time_tag.get("datetime") or ""))

    async def fetch_channel(self, channel_cfg: dict[str, str]) -> list[RawArticle]:
        """Fetch and parse one public Telegram channel, isolating any HTTP/parse failure."""
        if not self._session or self._session.closed:
            await self.initialize()

        username = (channel_cfg.get("username") or "").strip().lstrip("@")
        if not username:
            return []

        url = TELEGRAM_WEB_BASE_URL.format(username=username)
        assert self._session is not None

        try:
            async with self._session.get(url) as response:
                if response.status != 200:
                    logger.warning(
                        "Telegram channel '%s' returned HTTP %d.",
                        username,
                        response.status,
                    )
                    return []
                html_text = await response.text(errors="replace")
        except asyncio.TimeoutError:
            logger.warning("Telegram channel '%s' timed out.", username)
            return []
        except aiohttp.ClientError as exc:
            logger.warning("Telegram channel '%s' HTTP client error: %s", username, str(exc)[:160])
            return []
        except Exception as exc:
            logger.warning("Telegram channel '%s' error: %s", username, str(exc)[:160])
            return []

        return self.parse_channel_html(html_text, channel_cfg)

    async def fetch_articles(self) -> list[RawArticle]:
        """Fetch public posts from all configured Telegram channels concurrently."""
        if not self._channels:
            return []
        if not self._session or self._session.closed:
            await self.initialize()

        tasks = [self.fetch_channel(ch) for ch in self._channels]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_articles: list[RawArticle] = []
        for idx, res in enumerate(results):
            ch_name = self._channels[idx].get("username", "unknown")
            if isinstance(res, Exception):
                logger.warning("Telegram channel '%s' failed: %s", ch_name, str(res)[:160])
                continue
            if isinstance(res, list):
                all_articles.extend(res)

        logger.info(
            "Telegram public-web source discovered %d posts across %d channels.",
            len(all_articles),
            len(self._channels),
        )
        return all_articles
