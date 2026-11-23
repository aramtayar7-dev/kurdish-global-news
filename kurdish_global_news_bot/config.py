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

# ═══════════════════════════════════════════════════
# Feature Toggles
# ═══════════════════════════════════════════════════
ENABLE_IMAGES: bool = _bool(os.getenv("ENABLE_IMAGES"), True)
ENABLE_RSS: bool = _bool(os.getenv("ENABLE_RSS"), True)
ENABLE_DDG: bool = _bool(os.getenv("ENABLE_DDG"), True)
ENABLE_GDELT: bool = _bool(os.getenv("ENABLE_GDELT"), True)
ENABLE_TRANSLATION_FALLBACK: bool = _bool(os.getenv("ENABLE_TRANSLATION_FALLBACK"), True)
ENABLE_LOCAL_LLM: bool = _bool(os.getenv("ENABLE_LOCAL_LLM"), False)

# ═══════════════════════════════════════════════════
# Cloud Deployment & Health Web Server
# ═══════════════════════════════════════════════════
PORT: int = _int(os.getenv("PORT"), 10000)

# ═══════════════════════════════════════════════════
# Translation / Groq LLM
# ═══════════════════════════════════════════════════
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

# ═══════════════════════════════════════════════════
# Gemini AI Proofreading & Quality Check (Stage 4)
# ═══════════════════════════════════════════════════
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
ENABLE_GEMINI_PROOFREAD: bool = _bool(os.getenv("ENABLE_GEMINI_PROOFREAD"), True)
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

# ═══════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

# ═══════════════════════════════════════════════════
# Database
# ═══════════════════════════════════════════════════
DATABASE_PATH: str = os.getenv("DATABASE_PATH", str(_PROJECT_ROOT / "data" / "news_bot.db"))

# ═══════════════════════════════════════════════════
# Article Filtering & Recency
# ═══════════════════════════════════════════════════
MAX_ARTICLE_AGE_HOURS: float = float(os.getenv("MAX_ARTICLE_AGE_HOURS", "3.0"))
MAX_ARTICLE_AGE_MINUTES: int = _int(os.getenv("MAX_ARTICLE_AGE_MINUTES"), 30)

# ═══════════════════════════════════════════════════
# News Prioritization & Category Balancer (60/40 Split)
# ═══════════════════════════════════════════════════
PRIMARY_CATEGORY_RATIO: float = _float(os.getenv("PRIMARY_CATEGORY_RATIO"), 0.60)
SECONDARY_CATEGORY_RATIO: float = _float(os.getenv("SECONDARY_CATEGORY_RATIO"), 0.40)
FALLBACK_WINDOW_MINUTES: int = _int(os.getenv("FALLBACK_WINDOW_MINUTES"), 20)

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
# News Categories
# ═══════════════════════════════════════════════════
NEWS_CATEGORIES: list[str] = [
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
    """Return True if category belongs to Primary Focus (War, Military, World Leaders, Geopolitics)."""
    if not category:
        return False
    return category in PRIMARY_CATEGORIES


# ═══════════════════════════════════════════════════
# Search Keywords (for DuckDuckGo and other search sources)
# Weighted 60% Primary Focus / 40% Secondary Focus
# ═══════════════════════════════════════════════════
SEARCH_KEYWORDS: list[str] = [
    # ── Primary Focus (~60%): Wars, Military, World Leaders, Geopolitics ──
    "war armed conflict combat",
    "military defense airstrike missile",
    "world leaders president prime minister",
    "UN Security Council resolution",
    "NATO military alliance summit",
    "geopolitical crisis escalation conflict",
    "frontline casualties military operation",
    "defense missile nuclear deterrent",
    "government leader summit diplomacy",
    "international sanctions crisis",
    "ceasefire peace negotiations treaty",
    "state leaders bilateral meeting crisis",
    # ── Secondary Focus (~40%): AI, Cyber, Economy, Disasters, Science ──
    "artificial intelligence AI model breakthrough",
    "cybersecurity major cyberattack ransomware breach",
    "zero-day exploit cyber espionage hacking",
    "global economy inflation financial crisis",
    "major earthquake natural disaster tsunami",
    "science space mission exploration NASA",
    "global energy oil gas supply crisis",
    "global health disease outbreak WHO",
]

# ═══════════════════════════════════════════════════
# RSS Feeds
# Weighted ~60% Primary Focus / ~40% Secondary Focus
# ═══════════════════════════════════════════════════
RSS_FEEDS: list[dict[str, str]] = [
    # ── Primary Focus Feeds (~60%): Wars, Military, World Leaders, Geopolitics ──
    {"name": "Reuters World", "url": "https://feeds.reuters.com/Reuters/worldNews", "category": "International Politics"},
    {"name": "BBC World", "url": "https://feeds.bbci.co.uk/news/world/rss.xml", "category": "International Politics"},
    {"name": "Al Jazeera", "url": "https://www.aljazeera.com/xml/rss/all.xml", "category": "Wars & Armed Conflicts"},
    {"name": "AP News", "url": "https://rsshub.app/apnews/topics/apf-topnews", "category": "International Politics"},
    {"name": "France24 EN", "url": "https://www.france24.com/en/rss", "category": "International Politics"},
    {"name": "DW News", "url": "https://rss.dw.com/rdf/rss-en-all", "category": "International Politics"},
    {"name": "The Guardian World", "url": "https://www.theguardian.com/world/rss", "category": "International Politics"},
    {"name": "NPR News", "url": "https://feeds.npr.org/1001/rss.xml", "category": "International Politics"},
    {"name": "UN News", "url": "https://news.un.org/feed/subscribe/en/news/all/rss.xml", "category": "Diplomacy & International Relations"},
    {"name": "Defense News Global", "url": "https://www.defensenews.com/arc/outboundfeeds/rss/category/global/", "category": "Military & Defense"},
    {"name": "Military Times", "url": "https://www.militarytimes.com/arc/outboundfeeds/rss/category/news/", "category": "Military & Defense"},
    {"name": "GDELT Events", "url": "http://data.gdeltproject.org/gdeltv2/lastupdate.txt", "category": "Wars & Armed Conflicts"},

    # ── Secondary Focus Feeds (~40%): AI, Cyber, Economy, Disasters, Science ──
    {"name": "Reuters Technology", "url": "https://feeds.reuters.com/reuters/technologyNews", "category": "AI & Technology"},
    {"name": "BBC Technology", "url": "https://feeds.bbci.co.uk/news/technology/rss.xml", "category": "AI & Technology"},
    {"name": "The Hacker News", "url": "https://feeds.feedburner.com/TheHackersNews", "category": "Cybersecurity"},
    {"name": "BleepingComputer", "url": "https://www.bleepingcomputer.com/feed/", "category": "Cybersecurity"},
    {"name": "Reuters Business", "url": "https://feeds.reuters.com/reuters/businessNews", "category": "Global Economy & Financial Markets"},
    {"name": "CNBC World", "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100727362", "category": "Global Economy & Financial Markets"},
    {"name": "Reuters Science", "url": "https://feeds.reuters.com/reuters/scienceNews", "category": "Science & Research"},
    {"name": "Space.com", "url": "https://www.space.com/feeds/all", "category": "Space & Astronomy"},
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
    "default": 0.50,
}

# ═══════════════════════════════════════════════════
# Category-to-emoji mapping for Telegram posts
# ═══════════════════════════════════════════════════
CATEGORY_EMOJIS: dict[str, str] = {
    "International Politics": "🌍",
    "World Leaders & Governments": "🏛️",
    "Wars & Armed Conflicts": "⚔️",
    "Military & Defense": "🛡️",
    "Diplomacy & International Relations": "🤝",
    "Global Economy & Financial Markets": "📊",
    "AI & Technology": "🤖",
    "Science & Research": "🔬",
    "Health & Medical": "🏥",
    "Climate & Environment": "🌿",
    "Natural Disasters & Emergencies": "🚨",
    "Energy, Oil & Gas": "⛽",
    "Cybersecurity": "🔒",
    "International Organizations": "🏢",
    "Human Rights & Humanitarian": "✊",
    "Criminal Investigations & Security": "🔍",
    "Space & Astronomy": "🚀",
    "International Sports": "🏆",
    "Business & Corporate": "💼",
    "Other Significant Events": "📰",
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
