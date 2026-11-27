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
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup, Tag
from dateutil import parser as dt_parser

from config import MAX_ARTICLE_AGE_HOURS, REQUEST_TIMEOUT, TELEGRAM_SOURCE_CHANNELS
from processing.validator import SPAM_PATTERNS, ArticleValidator
from sources.base import BaseSource, RawArticle

logger = logging.getLogger("sources.telegram")

TELEGRAM_WEB_BASE_URL = "https://t.me/s/{username}"
TELEGRAM_EMBED_BASE_URL = "https://t.me/{username}/{post_id}?embed=1&single=1"
MAX_TELEGRAM_HTML_BYTES = 2 * 1024 * 1024  # 2 MiB safety bound for channel/embed HTML
MAX_EMBED_FALLBACKS_PER_CHANNEL = 3

CHROME_DESKTOP_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/117.0.0.0 Safari/537.36"
)
TELEGRAM_BROWSER_USER_AGENT = CHROME_DESKTOP_USER_AGENT
TELEGRAM_DOWNLOAD_HEADERS: dict[str, str] = {
    "User-Agent": CHROME_DESKTOP_USER_AGENT,
    "Referer": "https://t.me/",
    "Accept-Language": "en-US,en;q=0.9",
}

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
    re.compile(r"^\s*(?:قناة\s+)?الجزيرة\s*(?:[-–|]?\s*(?:عاجل|مباشر|العربية))?\s*$", re.IGNORECASE),
    re.compile(r"^\s*#(?:الجزيرة|AjaNews)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:للمزيد|للاشتراك|تابعونا|انضموا\s+إلى|اشترك\s+في)\b.*", re.IGNORECASE),
    re.compile(r"^\s*(?:منقول|عبر\s*:|تێلێگرام\s*:|تلغرام\s*:).*$", re.IGNORECASE),
    re.compile(r"^\s*(?:media\s+is\s+too\s+big|please\s+open\s+telegram\b.*)$", re.IGNORECASE),
    # Kurdish-language channel signatures (Rudaw, Kurdistan24, NRT, general)
    re.compile(r"^\s*(?:#Rudaw|#رووداو|تۆڕی\s+میدیایی\s+رووداو|رووداو\s*[-–]?\s*(?:ئەنقەرە|هەولێر|بەغدا|سلێمانی|دهۆک))\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:#Kurdistan24|#کوردستان24|#K24|بۆ\s+بینینی\s+زیاتر|بۆ\s+خوێندنەوەی\s+زیاتر)\b.*$", re.IGNORECASE),
    re.compile(r"^\s*(?:#NRT|#ئێن\.?\s*ئا[رڕ]\.?\s*تی|#nrttvofficial)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:سەرچاوە|بەدوادا\s+بچن|بۆ\s+زانیاری\s+زیاتر)\b.*$", re.IGNORECASE),
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
    r"background(?:-image)?\s*:[^;]*?\burl\(\s*(?:\\?['\"])?([^'\"\)\s\\]+)(?:\\?['\"])?\s*\)",
    re.IGNORECASE,
)

_FORBIDDEN_IMAGE_SCHEMES = ("javascript:", "data:", "vbscript:", "file:", "blob:")
_ALLOWED_IMAGE_SCHEMES = frozenset({"http", "https"})

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


def normalize_telegram_media_url(
    raw_url: Optional[str],
    base_url: str = "https://t.me/",
) -> Optional[str]:
    """
    Validate and normalize a candidate Telegram media URL.
    Supports protocol-relative (`//`) and relative URLs via `base_url`,
    preserves CDN query parameters, and rejects unsafe or malformed URLs.
    """
    if not raw_url or not isinstance(raw_url, str):
        return None
    cand = html.unescape(raw_url).strip().strip("\"'")
    if not cand or any(ch.isspace() for ch in cand) or "<" in cand or ">" in cand:
        return None

    lowered = cand.lower()
    if lowered.startswith(_FORBIDDEN_IMAGE_SCHEMES):
        return None

    effective_base = base_url if (base_url or "").startswith(("http://", "https://")) else "https://t.me/"
    if cand.startswith("//"):
        base_scheme = urlparse(effective_base).scheme or "https"
        cand = f"{base_scheme}:{cand}"
    elif ":" in cand.split("/", 1)[0] and not lowered.startswith(("http://", "https://")):
        return None
    elif not lowered.startswith(("http://", "https://")):
        cand = urljoin(effective_base, cand)

    try:
        parsed = urlparse(cand)
    except Exception:
        return None

    if parsed.scheme.lower() not in _ALLOWED_IMAGE_SCHEMES:
        return None
    host = (parsed.hostname or "").strip().lower()
    if not host or "." not in host and host != "localhost":
        return None
    return cand


def extract_background_image(
    style: Optional[str],
    base_url: str = "https://t.me/",
) -> Optional[str]:
    """
    Safely parse inline CSS `background-image:url(...)` and return a validated,
    absolute HTTP(S) URL (resolving protocol-relative and relative paths via `base_url`),
    or None if missing, unsafe (`javascript:`, `data:`), or malformed.
    """
    if not style or not isinstance(style, str):
        return None
    unescaped = html.unescape(style)
    for match in _BACKGROUND_IMAGE_URL_RE.finditer(unescaped):
        normalized = normalize_telegram_media_url(match.group(1), base_url=base_url)
        if normalized:
            return normalized
    return None


def _select_best_srcset_url(
    srcset_value: Optional[str],
    base_url: str = "https://t.me/",
) -> Optional[str]:
    """Select the highest-resolution valid image candidate from an HTML `srcset` attribute."""
    if not srcset_value or not isinstance(srcset_value, str):
        return None

    best_score: float = -1.0
    best_url: Optional[str] = None
    for part in srcset_value.split(","):
        tokens = part.strip().split()
        if not tokens:
            continue
        url_cand = normalize_telegram_media_url(tokens[0], base_url=base_url)
        if not url_cand:
            continue
        score = 1.0
        if len(tokens) > 1:
            desc = tokens[1].lower()
            try:
                if desc.endswith("w"):
                    score = float(desc[:-1])
                elif desc.endswith("x"):
                    score = float(desc[:-1]) * 1000.0
            except ValueError:
                score = 1.0
        if score > best_score:
            best_score = score
            best_url = url_cand
    return best_url


def _extract_img_tag_candidates(img: Tag, base_url: str = "https://t.me/") -> list[str]:
    """Extract ordered image URLs from an `<img>` element (`srcset`, `data-src`, `src`, `style`)."""
    candidates: list[str] = []
    # Prefer highest-resolution srcset candidate first when reliable descriptor exists
    srcset_raw = img.get("srcset")
    if isinstance(srcset_raw, str):
        best_srcset = _select_best_srcset_url(srcset_raw, base_url=base_url)
        if best_srcset:
            candidates.append(best_srcset)
    for attr in ("data-src", "src", "data-original"):
        val = img.get(attr)
        if isinstance(val, str):
            norm = normalize_telegram_media_url(val, base_url=base_url)
            if norm:
                candidates.append(norm)
    style_val = img.get("style")
    if isinstance(style_val, str):
        bg = extract_background_image(style_val, base_url=base_url)
        if bg:
            candidates.append(bg)
    return candidates


def _is_decorative_telegram_img(img: Tag) -> bool:
    """Return True if an `<img>` is a Telegram avatar, emoji, or decorative UI icon."""
    classes = " ".join(img.get("class") or []).lower()
    if "emoji" in classes or "tgme_widget_message_user_photo" in classes:
        return True
    for parent in img.parents:
        if not isinstance(parent, Tag):
            continue
        p_classes = " ".join(parent.get("class") or []).lower()
        if any(
            skip in p_classes
            for skip in (
                "tgme_widget_message_user_photo",
                "tgme_channel_info_header_photo",
                "tgme_widget_message_author",
                "tgme_widget_message_sticker",
            )
        ):
            return True
    return False


def _deduplicate_urls(urls: list[str]) -> list[str]:
    """Deduplicate validated URLs preserving exact priority order."""
    seen: set[str] = set()
    unique: list[str] = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def _extract_background_image_url_from_tags(
    tags: list[Tag],
    base_url: str = "https://t.me/",
) -> str:
    """Extract first valid http(s) URL from inline CSS `background-image:url(...)` attributes."""
    for wrap in tags:
        raw_style = wrap.get("style")
        if isinstance(raw_style, str):
            found = extract_background_image(raw_style, base_url=base_url)
            if found:
                return found
    return ""


def classify_telegram_media(
    html_or_tag: str | Tag,
    base_url: str = "https://t.me/",
) -> dict[str, Any]:
    """
    Classify Telegram HTML media candidates into distinct semantic buckets:
      - `photos`: direct `.tgme_widget_message_photo_wrap` photos
      - `album_photos`: nested grouped-album photos inside `.tgme_widget_message_grouped_layer`
      - `link_previews`: `.tgme_widget_message_link_preview` images
      - `og_images`: `<meta property="og:image">` candidates
      - `video_thumbnails`: `.tgme_widget_message_video_thumb` / poster images (kept distinct!)
      - `candidates`: deduplicated priority list of photograph/preview/og candidates (excluding video thumbnails)
    """
    if isinstance(html_or_tag, Tag):
        root = html_or_tag
    else:
        if not html_or_tag or not isinstance(html_or_tag, str):
            return {
                "photos": [],
                "album_photos": [],
                "link_previews": [],
                "og_images": [],
                "video_thumbnails": [],
                "candidates": [],
            }
        root = BeautifulSoup(html_or_tag, "html.parser")

    photos: list[str] = []
    album_photos: list[str] = []
    link_previews: list[str] = []
    og_images: list[str] = []
    video_thumbnails: list[str] = []

    # 1. Grouped album wrappers first/distinct + standalone message photo wrappers
    for wrap in root.select(".tgme_widget_message_photo_wrap"):
        in_album = any(
            isinstance(p, Tag) and "tgme_widget_message_grouped_layer" in (p.get("class") or [])
            for p in wrap.parents
        )
        found_in_wrap: list[str] = []
        if isinstance(wrap.get("style"), str):
            bg = extract_background_image(wrap.get("style"), base_url=base_url)
            if bg:
                found_in_wrap.append(bg)
        for img in wrap.select("img"):
            if not _is_decorative_telegram_img(img):
                found_in_wrap.extend(_extract_img_tag_candidates(img, base_url=base_url))
        if in_album:
            album_photos.extend(found_in_wrap)
        else:
            photos.extend(found_in_wrap)

    # Also catch any <img> directly inside grouped_layer without photo_wrap
    for layer in root.select(".tgme_widget_message_grouped_layer"):
        if isinstance(layer.get("style"), str):
            bg = extract_background_image(layer.get("style"), base_url=base_url)
            if bg:
                album_photos.append(bg)
        for img in layer.select("img"):
            if not _is_decorative_telegram_img(img):
                album_photos.extend(_extract_img_tag_candidates(img, base_url=base_url))

    # 2. Link preview images
    for preview in root.select(".tgme_widget_message_link_preview"):
        if isinstance(preview.get("style"), str):
            bg = extract_background_image(preview.get("style"), base_url=base_url)
            if bg:
                link_previews.append(bg)
        for sub in preview.select(".link_preview_image, [style]"):
            if isinstance(sub.get("style"), str):
                bg = extract_background_image(sub.get("style"), base_url=base_url)
                if bg:
                    link_previews.append(bg)
        for img in preview.select("img"):
            if not _is_decorative_telegram_img(img):
                link_previews.extend(_extract_img_tag_candidates(img, base_url=base_url))

    # 3. Video thumbnails (kept strictly separate from original photos!)
    for vthumb in root.select(
        ".tgme_widget_message_video_thumb, "
        ".tgme_widget_message_roundvideo_thumb, "
        ".tgme_widget_message_video_wrap [style]"
    ):
        if isinstance(vthumb.get("style"), str):
            bg = extract_background_image(vthumb.get("style"), base_url=base_url)
            if bg:
                video_thumbnails.append(bg)
        for img in vthumb.select("img"):
            video_thumbnails.extend(_extract_img_tag_candidates(img, base_url=base_url))
    for video in root.select(".tgme_widget_message_video_wrap video, .tgme_widget_message_video_player video, video"):
        poster = video.get("poster")
        if isinstance(poster, str):
            norm = normalize_telegram_media_url(poster, base_url=base_url)
            if norm:
                video_thumbnails.append(norm)

    # 4. og:image meta tags
    for meta in root.select('meta[property="og:image"], meta[name="og:image"]'):
        content = meta.get("content")
        if isinstance(content, str):
            norm = normalize_telegram_media_url(content, base_url=base_url)
            if norm:
                og_images.append(norm)

    # 5. Any remaining non-decorative <img> inside the message not already categorized
    thumb_set = set(video_thumbnails)
    extra_imgs: list[str] = []
    for img in root.select("img"):
        if _is_decorative_telegram_img(img):
            continue
        in_video = any(
            isinstance(p, Tag)
            and any(
                vc in (p.get("class") or [])
                for vc in ("tgme_widget_message_video_wrap", "tgme_widget_message_video_player")
            )
            for p in img.parents
        )
        if in_video:
            continue
        for c in _extract_img_tag_candidates(img, base_url=base_url):
            if c not in thumb_set:
                extra_imgs.append(c)

    photos = _deduplicate_urls(photos)
    album_photos = _deduplicate_urls(album_photos)
    link_previews = _deduplicate_urls(link_previews)
    og_images = _deduplicate_urls(og_images)
    video_thumbnails = _deduplicate_urls(video_thumbnails)

    # Priority: original photos -> grouped album photos -> og:image -> link previews -> extra non-thumb imgs
    candidates = _deduplicate_urls(photos + album_photos + og_images + link_previews + extra_imgs)
    return {
        "photos": photos,
        "album_photos": album_photos,
        "link_previews": link_previews,
        "og_images": og_images,
        "video_thumbnails": video_thumbnails,
        "candidates": candidates,
    }


def extract_telegram_image_candidates(
    html_or_tag: str | Tag,
    base_url: str = "https://t.me/",
    *,
    include_video_thumbs: bool = False,
) -> list[str]:
    """
    Extract prioritized, deduplicated, scheme-validated image URLs from Telegram HTML.
    By default excludes video thumbnails so video thumbs are never mistaken for original photos.
    """
    classified = classify_telegram_media(html_or_tag, base_url=base_url)
    if include_video_thumbs:
        return _deduplicate_urls(classified["candidates"] + classified["video_thumbnails"])
    return list(classified["candidates"])


def extract_telegram_photo_url(msg_tag: Tag, base_url: str = "https://t.me/") -> str:
    """
    Extract best original photo / album / preview URL from a Telegram message node,
    excluding video thumbnails. Returns empty string if missing or invalid.
    """
    classified = classify_telegram_media(msg_tag, base_url=base_url)
    for bucket in (
        classified["photos"],
        classified["album_photos"],
        classified["link_previews"],
        classified["og_images"],
    ):
        if bucket:
            return bucket[0]
    return ""


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
            "Accept-Language": "ar,ckb,en-US;q=0.9,en;q=0.8",
            "Cache-Control": "no-cache",
            "Upgrade-Insecure-Requests": "1",
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

                canonical_url = f"https://t.me/{username}/{post_id}"
                media_info = classify_telegram_media(msg_node, base_url=canonical_url)
                photo_url = extract_telegram_photo_url(msg_node, base_url=canonical_url)
                video_url = extract_telegram_video_url(msg_node)
                # Video thumbnails stay strictly distinct and are only carried when a real video exists
                thumb_url = (
                    (media_info["video_thumbnails"][0] if media_info["video_thumbnails"] else extract_telegram_video_thumb_url(msg_node))
                    if (not photo_url and video_url)
                    else ""
                )
                resolved_image_url = (photo_url or thumb_url) or None
                if photo_url:
                    logger.debug("Telegram extraction tier=channel_dom for %s", canonical_url)
                elif video_url:
                    logger.debug("Telegram extraction tier=video_media for %s", canonical_url)

                articles.append(
                    RawArticle(
                        url=canonical_url,
                        title=headline,
                        summary=summary,
                        source_name=display_name,
                        category="",
                        publication_date=pub_iso,
                        image_url=resolved_image_url,
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

    async def fetch_embed_image_fallback(
        self,
        username: str,
        post_id: str,
    ) -> Optional[str]:
        """
        Controlled single-message embed fallback:
        https://t.me/{username}/{post_id}?embed=1&single=1
        Uses Chrome desktop User-Agent + t.me Referer, bounded response size,
        status validation, and at most 1 retry on transient error.
        Never promotes a video thumbnail without video into a photograph.
        """
        clean_user = (username or "").strip().lstrip("@")
        clean_id = (post_id or "").strip()
        if not clean_user or not clean_id.isdigit():
            return None

        if not self._session or self._session.closed:
            await self.initialize()
        assert self._session is not None

        embed_url = TELEGRAM_EMBED_BASE_URL.format(username=clean_user, post_id=clean_id)
        timeout = aiohttp.ClientTimeout(total=self._timeout)

        for attempt in range(2):  # Initial attempt + at most 1 bounded retry
            try:
                async with self._session.get(
                    embed_url,
                    headers=TELEGRAM_DOWNLOAD_HEADERS,
                    timeout=timeout,
                ) as response:
                    if response.status >= 500 and attempt == 0:
                        continue
                    if response.status != 200:
                        logger.debug(
                            "Telegram embed fallback %s returned HTTP %d.",
                            embed_url,
                            response.status,
                        )
                        return None

                    raw_bytes: Optional[bytes] = None
                    content_stream = getattr(response, "content", None)
                    if content_stream is not None and hasattr(content_stream, "read") and callable(content_stream.read):
                        try:
                            maybe_bytes = await content_stream.read(MAX_TELEGRAM_HTML_BYTES + 1)
                            if isinstance(maybe_bytes, (bytes, bytearray)):
                                raw_bytes = bytes(maybe_bytes)
                        except Exception:
                            raw_bytes = None

                    if raw_bytes is not None:
                        if len(raw_bytes) > MAX_TELEGRAM_HTML_BYTES:
                            logger.warning(
                                "Telegram embed fallback %s exceeded max HTML size (%d bytes).",
                                embed_url,
                                MAX_TELEGRAM_HTML_BYTES,
                            )
                            return None
                        embed_html = raw_bytes.decode("utf-8", errors="replace")
                    else:
                        embed_html = await response.text(errors="replace")
                        if len(embed_html.encode("utf-8", errors="ignore")) > MAX_TELEGRAM_HTML_BYTES:
                            logger.warning("Telegram embed fallback %s exceeded max HTML size.", embed_url)
                            return None

                    classified = classify_telegram_media(embed_html, base_url=embed_url)
                    # Priority: message photo -> grouped/nested embed photo -> og:image -> link preview (never bare video thumbnail)
                    for bucket_name in ("photos", "album_photos", "og_images", "link_previews"):
                        bucket = classified.get(bucket_name) or []
                        if bucket:
                            logger.info(
                                "Telegram extraction tier=embed_single bucket=%s recovered image for @%s/%s",
                                bucket_name,
                                clean_user,
                                clean_id,
                            )
                            return bucket[0]
                    return None
            except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
                if attempt == 0:
                    continue
                logger.debug(
                    "Telegram embed fallback failed for %s: %s",
                    embed_url,
                    str(exc)[:140],
                )
                return None
            except Exception as exc:
                logger.debug("Telegram embed fallback error for %s: %s", embed_url, str(exc)[:140])
                return None
        return None

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

        articles = self.parse_channel_html(html_text, channel_cfg)
        embed_used = 0
        for art in articles:
            if embed_used >= MAX_EMBED_FALLBACKS_PER_CHANNEL:
                break
            if not art.image_url and not art.video_url and art.url:
                post_id = art.url.rstrip("/").rsplit("/", 1)[-1]
                if post_id.isdigit():
                    embed_used += 1
                    recovered = await self.fetch_embed_image_fallback(username, post_id)
                    if recovered:
                        art.image_url = recovered
        return articles

    async def fetch_articles(self) -> list[RawArticle]:
        """Fetch public posts from all configured Telegram channels concurrently with bounded concurrency."""
        if not self._channels:
            return []
        if not self._session or self._session.closed:
            await self.initialize()

        semaphore = asyncio.Semaphore(5)  # Bounded concurrency: at most 5 channels at once

        async def _fetch_bounded(ch: dict[str, str]) -> list[RawArticle]:
            async with semaphore:
                return await self.fetch_channel(ch)

        tasks = [_fetch_bounded(ch) for ch in self._channels]
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
