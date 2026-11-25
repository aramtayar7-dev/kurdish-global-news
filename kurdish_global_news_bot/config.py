"""
Kurdish Global News Bot — Configuration Module

All configurable parameters loaded from environment variables with sane defaults.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).parent.resolve()
load_dotenv(_PROJECT_ROOT / ".env")


def _bool(val: str | None, default: bool = True) -> bool:
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes")


def _int(val: str | None, default: int) -> int:
    try:
        return int(val) if val else default
    except (ValueError, TypeError):
        return default


def _float(val: str | None, default: float) -> float:
    try:
        return float(val) if val else default
    except (ValueError, TypeError):
        return default


# ═══════════════════════════════════════════════════
# Telegram
# ═══════════════════════════════════════════════════
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHANNEL_ID: str = os.getenv("TELEGRAM_CHANNEL_ID", "")
NOTIFY_DAILY_DIGEST: bool = _bool(os.getenv("NOTIFY_DAILY_DIGEST"), False)

# ═══════════════════════════════════════════════════
# Discovery & Scheduling
# ═══════════════════════════════════════════════════
NEWS_CHECK_INTERVAL_SECONDS: int = _int(os.getenv("NEWS_CHECK_INTERVAL_SECONDS"), 60)
AUTO_PUBLISH: bool = _bool(os.getenv("AUTO_PUBLISH"), True)

# ═══════════════════════════════════════════════════
# Concurrency
# ═══════════════════════════════════════════════════
MAX_CONCURRENT_SOURCES: int = _int(os.getenv("MAX_CONCURRENT_SOURCES"), 5)
MAX_CONCURRENT_ARTICLES: int = _int(os.getenv("MAX_CONCURRENT_ARTICLES"), 3)

# ═══════════════════════════════════════════════════
# Importance & Quality
# ═══════════════════════════════════════════════════
MIN_IMPORTANCE_SCORE: float = _float(os.getenv("MIN_IMPORTANCE_SCORE"), 0.4)
MIN_SOURCE_CREDIBILITY: float = _float(os.getenv("MIN_SOURCE_CREDIBILITY"), 0.3)
MIN_GLOBAL_RELEVANCE_SCORE: int = _int(os.getenv("MIN_GLOBAL_RELEVANCE_SCORE"), 7)

# ═══════════════════════════════════════════════════
# Feature Toggles
# ═══════════════════════════════════════════════════
ENABLE_IMAGES: bool = _bool(os.getenv("ENABLE_IMAGES"), True)
ENABLE_RSS: bool = _bool(os.getenv("ENABLE_RSS"), True)
ENABLE_DDG: bool = _bool(os.getenv("ENABLE_DDG"), True)
ENABLE_GDELT: bool = _bool(os.getenv("ENABLE_GDELT"), True)
ENABLE_TELEGRAM: bool = _bool(os.getenv("ENABLE_TELEGRAM"), True)
ENABLE_TRANSLATION_FALLBACK: bool = _bool(os.getenv("ENABLE_TRANSLATION_FALLBACK"), True)
ENABLE_LOCAL_LLM: bool = _bool(os.getenv("ENABLE_LOCAL_LLM"), False)

# ═══════════════════════════════════════════════════
# Cloud Deployment & Health Web Server
# ═══════════════════════════════════════════════════
PORT: int = _int(os.getenv("PORT"), 10000)

# ═══════════════════════════════════════════════════
# Gemini AI Engine (Primary Classification, Translation & Summarization)
# ═══════════════════════════════════════════════════
def _load_gemini_keys() -> list[str]:
    """Load & deduplicate comma-separated keys from GEMINI_API_KEYS and GEMINI_API_KEY."""
    raw_combined = ",".join(
        part
        for part in (os.getenv("GEMINI_API_KEYS", ""), os.getenv("GEMINI_API_KEY", ""))
        if part
    )
    keys: list[str] = []
    seen: set[str] = set()
    for item in raw_combined.split(","):
        key = item.strip()
        if key and key not in seen:
            keys.append(key)
            seen.add(key)
    return keys


GEMINI_API_KEYS: list[str] = _load_gemini_keys()
GEMINI_API_KEY: str = ",".join(GEMINI_API_KEYS)
ENABLE_GEMINI_PROOFREAD: bool = _bool(os.getenv("ENABLE_GEMINI_PROOFREAD"), True)
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
MAX_GEMINI_CANDIDATES_PER_CYCLE: int = _int(os.getenv("MAX_GEMINI_CANDIDATES_PER_CYCLE"), 25)
GEMINI_RANK_BATCH_SIZE: int = max(1, min(15, _int(os.getenv("GEMINI_RANK_BATCH_SIZE"), 12)))
GEMINI_KEY_COOLDOWN_SECONDS: float = _float(os.getenv("GEMINI_KEY_COOLDOWN_SECONDS"), 60.0)
GEMINI_QUOTA_COOLDOWN_SECONDS: float = _float(os.getenv("GEMINI_QUOTA_COOLDOWN_SECONDS"), 300.0)
GEMINI_REQUEST_PACING_SECONDS: float = _float(os.getenv("GEMINI_REQUEST_PACING_SECONDS"), 0.5)

# Branding & Card Generation
CHANNEL_HANDLE: str = os.getenv("CHANNEL_HANDLE", "@KurdishGlobalNews").strip() or "@KurdishGlobalNews"
ASSETS_DIR: Path = _PROJECT_ROOT / "assets"
CARD_WIDTH: int = 1280
CARD_HEIGHT: int = 720

# ═══════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

# ═══════════════════════════════════════════════════
# Database (SQLite default; Supabase PostgREST when SUPABASE_URL + SUPABASE_KEY are set)
# ═══════════════════════════════════════════════════
DATABASE_PATH: str = os.getenv("DATABASE_PATH", str(_PROJECT_ROOT / "data" / "news_bot.db"))
SUPABASE_URL: str = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "").strip()

# ═══════════════════════════════════════════════════
# Article Filtering & Recency
# ═══════════════════════════════════════════════════
MAX_ARTICLE_AGE_HOURS: float = float(os.getenv("MAX_ARTICLE_AGE_HOURS", "3.0"))
MAX_ARTICLE_AGE_MINUTES: int = _int(os.getenv("MAX_ARTICLE_AGE_MINUTES"), 30)
DEDUP_WINDOW_HOURS: int = _int(os.getenv("DEDUP_WINDOW_HOURS"), 12)
DEDUP_HISTORY_HOURS: int = _int(os.getenv("DEDUP_HISTORY_HOURS"), 72)

# ═══════════════════════════════════════════════════
# Live Event Threading (Reply-based Material Updates)
# ═══════════════════════════════════════════════════
ENABLE_LIVE_EVENT_THREADING: bool = _bool(os.getenv("ENABLE_LIVE_EVENT_THREADING"), True)
EVENT_THREAD_WINDOW_HOURS: int = _int(os.getenv("EVENT_THREAD_WINDOW_HOURS"), 24)
THREAD_UPDATE_COOLDOWN_MINUTES: int = _int(os.getenv("THREAD_UPDATE_COOLDOWN_MINUTES"), 20)
MAX_UPDATES_PER_THREAD: int = _int(os.getenv("MAX_UPDATES_PER_THREAD"), 3)

# ═══════════════════════════════════════════════════
# News Prioritization & Balancer (5-of-8 Bashur Quota + 60/40 Category Ratio)
# ═══════════════════════════════════════════════════
PRIMARY_CATEGORY_RATIO: float = _float(os.getenv("PRIMARY_CATEGORY_RATIO"), 0.60)
SECONDARY_CATEGORY_RATIO: float = _float(os.getenv("SECONDARY_CATEGORY_RATIO"), 0.40)
FALLBACK_WINDOW_MINUTES: int = _int(os.getenv("FALLBACK_WINDOW_MINUTES"), 20)

BATCH_PUBLISH_LIMIT: int = _int(os.getenv("BATCH_PUBLISH_LIMIT"), 8)
BASHUR_QUOTA_PER_8: int = _int(os.getenv("BASHUR_QUOTA_PER_8"), 5)
GLOBAL_QUOTA_PER_8: int = max(0, 8 - BASHUR_QUOTA_PER_8)
BASHUR_TARGET_RATIO: float = BASHUR_QUOTA_PER_8 / 8.0
GLOBAL_TARGET_RATIO: float = 1.0 - BASHUR_TARGET_RATIO

# Strict Topic Mix Quota per 8 posts: min 7 Politics/War + max 1 Economy OR Cybersecurity
POLITICS_WAR_MIN_PER_8: int = _int(os.getenv("POLITICS_WAR_MIN_PER_8"), 7)
ECON_CYBER_MAX_PER_8: int = max(0, 8 - POLITICS_WAR_MIN_PER_8)
ECON_CYBER_MIN_SCORE: float = _float(os.getenv("ECON_CYBER_MIN_SCORE"), 7.5)

# ═══════════════════════════════════════════════════
# Network
# ═══════════════════════════════════════════════════
REQUEST_TIMEOUT: int = _int(os.getenv("REQUEST_TIMEOUT"), 30)
MAX_RETRIES: int = _int(os.getenv("MAX_RETRIES"), 3)
MAX_IMAGE_SIZE_MB: int = _int(os.getenv("MAX_IMAGE_SIZE_MB"), 5)
MAX_IMAGE_SIZE_BYTES: int = MAX_IMAGE_SIZE_MB * 1024 * 1024

# ═══════════════════════════════════════════════════
# User-Agent for HTTP requests
# ═══════════════════════════════════════════════════
USER_AGENT: str = "KurdishGlobalNewsBot/1.0 (news aggregation research)"

# ═══════════════════════════════════════════════════
# Backup
# ═══════════════════════════════════════════════════
BACKUP_DIR: str = str(_PROJECT_ROOT / "backups")
BACKUP_INTERVAL_HOURS: int = _int(os.getenv("BACKUP_INTERVAL_HOURS"), 6)

# ═══════════════════════════════════════════════════
# Canonical News Categories & Metadata
# ═══════════════════════════════════════════════════
CATEGORY_META: dict[str, dict[str, str]] = {
    "kurdistan": {"label": "کوردستان", "emoji": "☀️"},
    "military_conflict": {"label": "شەڕ و سەربازی", "emoji": "⚔️"},
    "politics_world": {"label": "سیاسەت و جیهان", "emoji": "🌍"},
    "economy_markets": {"label": "ئابووری و بازاڕ", "emoji": "📈"},
    "ai_tech": {"label": "تەکنەلۆژیا و زیرەکیی دەستکرد", "emoji": "🤖"},
    "cybersecurity": {"label": "ئاسایشی سایبەری", "emoji": "🛡️"},
    "science_space": {"label": "زانست و بۆشایی ئاسمان", "emoji": "🚀"},
}

CANONICAL_CATEGORIES: tuple[str, ...] = tuple(CATEGORY_META.keys())

NEWS_CATEGORIES: list[str] = list(CANONICAL_CATEGORIES) + [
    "International Politics",
    "World Leaders & Governments",
    "Wars & Armed Conflicts",
    "Military & Defense",
    "Diplomacy & International Relations",
    "Global Economy & Financial Markets",
    "AI & Technology",
    "Science & Research",
    "Health & Medical",
    "Climate & Environment",
    "Natural Disasters & Emergencies",
    "Energy, Oil & Gas",
    "Cybersecurity",
    "International Organizations",
    "Human Rights & Humanitarian",
    "Criminal Investigations & Security",
    "Space & Astronomy",
    "International Sports",
    "Business & Corporate",
    "Other Significant Events",
]

# Primary category focus (~60% target)
PRIMARY_CATEGORIES: set[str] = {
    "kurdistan",
    "military_conflict",
    "politics_world",
    "Wars & Armed Conflicts",
    "Military & Defense",
    "World Leaders & Governments",
    "International Politics",
    "Diplomacy & International Relations",
}

# Secondary category focus (~40% target)
SECONDARY_CATEGORIES: set[str] = {
    cat for cat in NEWS_CATEGORIES if cat not in PRIMARY_CATEGORIES
}


def is_primary_category(category: str | None) -> bool:
    """Return True if category belongs to Primary Focus (Kurdistan, War, Military, World Politics)."""
    if not category:
        return False
    return category in PRIMARY_CATEGORIES


# ═══════════════════════════════════════════════════
# Search Keywords (for DuckDuckGo and other search sources)
# Weighted 60% Primary Focus / 40% Secondary Focus
# ═══════════════════════════════════════════════════
SEARCH_KEYWORDS: list[str] = [
    # ── Primary Focus (~60%): Kurdistan (4 Parts), Wars, Military, Geopolitics ──
    "Kurdistan Erbil Sulaymaniyah Kirkuk",
    "Rojava Qamishli Kobani SDF Syria",
    "Diyarbakir Amed Van Kurdish Turkey",
    "Rojhelat Sanandaj Mahabad Kermanshah Iran",
    "war armed conflict combat",
    "military defense airstrike missile",
    "world leaders president prime minister",
    "UN Security Council resolution",
    "NATO military alliance summit",
    "geopolitical crisis escalation conflict",
    "international sanctions crisis",
    "ceasefire peace negotiations treaty",
    # ── Secondary Focus (~40%): AI, Cyber, Economy, Science ──
    "artificial intelligence AI model breakthrough",
    "cybersecurity major cyberattack ransomware breach",
    "zero-day exploit cyber espionage hacking",
    "global economy inflation financial crisis",
    "stock market semiconductors Nvidia tech earnings",
    "science space mission exploration NASA",
    "global energy oil gas supply crisis",
    "major earthquake natural disaster emergency",
]

# ═══════════════════════════════════════════════════
# RSS Feeds (Global Sources — general feeds avoid hardcoded war labels)
# Weighted ~60% Primary Focus / ~40% Secondary Focus
# ═══════════════════════════════════════════════════
RSS_FEEDS: list[dict[str, str]] = [
    # ── Primary Focus Feeds (~60%): Politics, Diplomacy, Defense ──
    {"name": "Reuters World", "url": "https://feeds.reuters.com/Reuters/worldNews", "category": "politics_world"},
    {"name": "BBC World", "url": "https://feeds.bbci.co.uk/news/world/rss.xml", "category": "politics_world"},
    {"name": "Al Jazeera", "url": "https://www.aljazeera.com/xml/rss/all.xml", "category": "politics_world"},
    {"name": "AP News", "url": "https://rsshub.app/apnews/topics/apf-topnews", "category": "politics_world"},
    {"name": "France24 EN", "url": "https://www.france24.com/en/rss", "category": "politics_world"},
    {"name": "DW News", "url": "https://rss.dw.com/rdf/rss-en-all", "category": "politics_world"},
    {"name": "The Guardian World", "url": "https://www.theguardian.com/world/rss", "category": "politics_world"},
    {"name": "NPR News", "url": "https://feeds.npr.org/1001/rss.xml", "category": "politics_world"},
    {"name": "UN News", "url": "https://news.un.org/feed/subscribe/en/news/all/rss.xml", "category": "politics_world"},
    {"name": "Defense News Global", "url": "https://www.defensenews.com/arc/outboundfeeds/rss/category/global/", "category": "military_conflict"},
    {"name": "Military Times", "url": "https://www.militarytimes.com/arc/outboundfeeds/rss/category/news/", "category": "military_conflict"},
    {"name": "GDELT Events", "url": "http://data.gdeltproject.org/gdeltv2/lastupdate.txt", "category": "military_conflict"},

    # ── Secondary Focus Feeds (~40%): AI, Cyber, Economy, Science ──
    {"name": "Reuters Technology", "url": "https://feeds.reuters.com/reuters/technologyNews", "category": "ai_tech"},
    {"name": "BBC Technology", "url": "https://feeds.bbci.co.uk/news/technology/rss.xml", "category": "ai_tech"},
    {"name": "The Hacker News", "url": "https://feeds.feedburner.com/TheHackersNews", "category": "cybersecurity"},
    {"name": "BleepingComputer", "url": "https://www.bleepingcomputer.com/feed/", "category": "cybersecurity"},
    {"name": "Reuters Business", "url": "https://feeds.reuters.com/reuters/businessNews", "category": "economy_markets"},
    {"name": "CNBC World", "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100727362", "category": "economy_markets"},
    {"name": "Reuters Science", "url": "https://feeds.reuters.com/reuters/scienceNews", "category": "science_space"},
    {"name": "Space.com", "url": "https://www.space.com/feeds/all", "category": "science_space"},
]

# ═══════════════════════════════════════════════════
# Public Telegram News Channels (https://t.me/s/{username})
# ═══════════════════════════════════════════════════
TELEGRAM_SOURCE_CHANNELS: list[dict[str, str]] = [
    {"username": "SabrenNewss", "name": "Sabereen News", "region_hint": "iraq_region", "lang": "ar"},
    {"username": "Sabren_News1", "name": "Sabereen Plus", "region_hint": "iraq_region", "lang": "ar"},
]

# ═══════════════════════════════════════════════════
# Source Credibility Scores
# ═══════════════════════════════════════════════════
SOURCE_CREDIBILITY: dict[str, float] = {
    "reuters": 0.95,
    "bbc": 0.93,
    "associated press": 0.95,
    "ap news": 0.95,
    "al jazeera": 0.85,
    "france24": 0.85,
    "dw": 0.85,
    "the guardian": 0.88,
    "cnbc": 0.82,
    "npr": 0.88,
    "cnn": 0.80,
    "new york times": 0.90,
    "washington post": 0.88,
    "bbc news": 0.93,
    "sky news": 0.78,
    "space.com": 0.80,
    "un news": 0.90,
    "who": 0.92,
    "kurdistan24": 0.86,
    "rudaw": 0.86,
    "pukmedia": 0.78,
    "north press agency": 0.82,
    "rojava information center": 0.82,
    "bianet": 0.78,
    "khrn": 0.82,
    "defense news": 0.84,
    "the hacker news": 0.84,
    "bleepingcomputer": 0.84,
    "sabereen news": 0.72,
    "sabereen plus": 0.72,
    "google news": 0.78,
    "default": 0.50,
}

# ═══════════════════════════════════════════════════
# Category-to-emoji mapping for Telegram posts
# ═══════════════════════════════════════════════════
CATEGORY_EMOJIS: dict[str, str] = {
    # Canonical 7 categories
    "kurdistan": "☀️",
    "military_conflict": "⚔️",
    "politics_world": "🌍",
    "economy_markets": "📈",
    "ai_tech": "🤖",
    "cybersecurity": "🛡️",
    "science_space": "🚀",
    # Legacy fallbacks
    "International Politics": "🌍",
    "World Leaders & Governments": "🌍",
    "Wars & Armed Conflicts": "⚔️",
    "Military & Defense": "⚔️",
    "Diplomacy & International Relations": "🌍",
    "Global Economy & Financial Markets": "📈",
    "AI & Technology": "🤖",
    "Science & Research": "🚀",
    "Health & Medical": "🌍",
    "Climate & Environment": "🌍",
    "Natural Disasters & Emergencies": "🌍",
    "Energy, Oil & Gas": "📈",
    "Cybersecurity": "🛡️",
    "International Organizations": "🌍",
    "Human Rights & Humanitarian": "🌍",
    "Criminal Investigations & Security": "⚔️",
    "Space & Astronomy": "🚀",
    "International Sports": "🌍",
    "Business & Corporate": "📈",
    "Other Significant Events": "🌍",
}

# ═══════════════════════════════════════════════════
# Article statuses
# ═══════════════════════════════════════════════════
class ArticleStatus:
    DISCOVERED = "DISCOVERED"
    VALIDATING = "VALIDATING"
    PROCESSING = "PROCESSING"
    TRANSLATING = "TRANSLATING"
    READY_TO_PUBLISH = "READY_TO_PUBLISH"
    PUBLISHING = "PUBLISHING"
    PUBLISHED = "PUBLISHED"
    FAILED = "FAILED"
    RETRY_PENDING = "RETRY_PENDING"
    REJECTED_DUPLICATE = "REJECTED_DUPLICATE"
    SKIPPED = "SKIPPED"


# ═══════════════════════════════════════════════════
# Priority levels
# ═══════════════════════════════════════════════════
class Priority:
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"


# Validate essential config
def validate_config() -> list[str]:
    """Return list of configuration errors (empty = valid)."""
    errors: list[str] = []
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "your_bot_token_here":
        errors.append("TELEGRAM_BOT_TOKEN is not set.")
    if not TELEGRAM_CHANNEL_ID or TELEGRAM_CHANNEL_ID == "@your_channel_id_here":
        errors.append("TELEGRAM_CHANNEL_ID is not set.")
    if NEWS_CHECK_INTERVAL_SECONDS < 10:
        errors.append("NEWS_CHECK_INTERVAL_SECONDS must be >= 10.")
    return errors
