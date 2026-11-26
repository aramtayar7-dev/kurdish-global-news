"""
Kurdish Global News Bot — Importance Ranker & Gemini Global Relevance Classifier

Evaluates global relevance, assigns one of the 7 strict Kurdish categories,
detects breaking developments, prioritizes all four parts of Kurdistan,
and provides rule-based fallback if the Gemini API is unavailable.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional

from storage.models import Article
from storage.utils import normalize_event_entities, normalize_event_key
from processing.extractor import ArticleExtractor
from processing.gemini_client import GeminiClient, GeminiAPIError, GeminiQuotaExhaustedError
from processing.translator import strip_news_dateline
import config
from config import (
    ALLOWED_EDITORIAL_CATEGORIES,
    CANONICAL_CATEGORIES,
    DISABLED_EDITORIAL_CATEGORIES,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_RANK_BATCH_SIZE,
    MAX_GEMINI_CANDIDATES_PER_CYCLE,
    MAX_GLOBAL_POSTS_PER_BATCH,
    MAX_GLOBAL_POSTS_PER_DAY,
    MIDDLE_EAST_TARGET_RATIO,
    MIN_GLOBAL_RELEVANCE_SCORE,
    MIN_IMPORTANCE_SCORE,
    MIN_SOURCE_CREDIBILITY,
    Priority,
    REGION_GLOBAL,
    REGION_KURDISTAN_BASHUR,
    REGION_MIDDLE_EAST,
    VALID_GEOGRAPHIC_REGIONS,
    is_primary_category,
)

logger = logging.getLogger("core.importance")

# Keywords that indicate critical/breaking news
CRITICAL_KEYWORDS: list[str] = [
    "breaking", "urgent", "just in", "developing",
    "explosion", "killed", "casualties", "massacre",
    "declared war", "nuclear", "invasion", "coup",
    "assassination", "emergency declared", "tsunami warning",
    "earthquake magnitude", "pandemic declared",
    "ceasefire agreement", "peace deal signed",
    "president dead", "prime minister resign",
]

HIGH_KEYWORDS: list[str] = [
    "sanctions", "summit", "agreement", "legislation",
    "election results", "military operation", "trade war",
    "major discovery", "breakthrough", "record",
    "crisis", "protest", "revolution", "referendum",
    "supreme court", "federal reserve", "interest rate",
    "vaccine approved", "ai regulation",
    "kurdistan", "rojava", "erbil", "diyarbakir", "rojhelat",
]

KURDISTAN_PATTERN = re.compile(
    r"(kurdistan|kurdish|kurds|erbil|hewler|sulaymaniyah|duhok|kirkuk|halabja|peshmerga|krg|"
    r"rojava|qamishli|kobani|hasakah|\bsdf\b|diyarbakir|amed|bakur|rojhelat|sanandaj|mahabad|urmia|kermanshah|"
    r"کوردستان|کورد|هەولێر|سلێمانی|دهۆک|کەرکوو?ک|هەڵەبجە|پێشمەرگە|ڕۆژئاوا|قامیشلۆ|کۆبانێ|حەسەکە|ئامەد|سنە|مەهاباد|ورمێ|کرماشان)",
    re.IGNORECASE,
)

# Southern Kurdistan (Bashur) geographic, governmental, and security identifiers
BASHUR_PATTERN = re.compile(
    r"(erbil|hewler|sulaymaniyah|slemani|\bsuli\b|duhok|halabja|kirkuk|peshmerga|\bkrg\b|"
    r"kurdistan region|iraqi kurdistan|south kurdistan|\bbashur\b(?!\s+(?:beirut|lebanon|gaza|israel|syria|iraq|iran|damascus|tehran|baghdad))|"
    r"erbil[\s\-–]+baghdad|baghdad[\s\-–]+erbil|nechirvan|masrour|barzani|talabani|disputed areas?|article 140|"
    r"هەولێر|سلێمانی|دهۆک|هەڵەبجە|کەرکوو?ک|پێشمەرگە|هەرێمی کوردستان|حکومەتی هەرێم|"
    r"کوردستانی باشوور|باشووری کوردستان|باشوور(?!ی\s+(?:بەیروت|لوبنان|غەززە|ئیسرائیل|سوریا|عێراق|تاران|بەغدا|دیمەشق|ئەمریکا|یەمان|سودان|کەرتی|ئێران|تەورێز|کەناری))|"
    r"بارزانی|تاڵەبانی|هەولێر و بەغدا|مووچە)",
    re.IGNORECASE,
)

NON_BASHUR_KURDISTAN_PATTERN = re.compile(
    r"(rojava|qamishli|kobani|hasakah|\bsdf\b|north and east syria|diyarbakir|amed|\bbakur\b|"
    r"van turkey|dem party|rojhelat|sanandaj|mahabad|kermanshah|urmia|"
    r"ڕۆژئاوا|قامیشلۆ|کۆبانێ|حەسەکە|باکوور|ئامەد|ڕۆژهەڵات|سنە|مەهاباد|ورمێ|کرماشان)",
    re.IGNORECASE,
)

# Locations in Syria that are non-Kurdish (e.g. Tartous naval port, Latakia, Damascus, etc.)
# News taking place in these locations must NEVER be classified as "kurdistan" unless they
# explicitly and genuinely involve Kurdish territories/forces (Rojava, SDF, Qamishli, Kobani, Hasakah, etc.)
NON_KURDISTAN_SYRIA_PATTERN = re.compile(
    r"\b(tartous|tartus|latakia|lattakia|damascus|homs|hama|daraa|deraa|suwayda|sweida|palmyra|tadmur|baniyas|quneitra)\b|"
    r"(طرطوس|تەرتووس|تەرتوس|اللاذقية|لازقیە|لادقیە|دمشق|دیمەشق|حمص|حمس|حماة|حەما|درعا|دەرعا|السويداء|سوەیدا|تدمر|تەدمور|بانياس|بانیاس|القنيطرة)",
    re.IGNORECASE,
)

# Politics, government, diplomacy, war, security, drone/missile incidents
POLITICS_WAR_PATTERN = re.compile(
    r"(\bwar\b|\bwars\b|armed conflict|\bmilitary\b|\bdefense\b|peshmerga|security force|"
    r"airstrike|airstrikes|\bstrike\b|\bstrikes\b|\bdrone\b|\bdrones\b|\bmissile\b|\bmissiles\b|"
    r"\brocket\b|\brockts\b|\bshelling\b|\battack\b|\battacks\b|\bcombat\b|ceasefire|"
    r"\btroops\b|\bsecurity\b|\bintelligence\b|\belection\b|\bparliament\b|\bgovernment\b|"
    r"\bcabinet\b|prime minister|\bpresident\b|(?:foreign|defense|interior)\s+ministry|\bdiplomac|\btreaty\b|\bsummit\b|"
    r"\bsanctions\b|\bcrisis\b|political|strategic|erbil[\s\-–]+baghdad|baghdad[\s\-–]+erbil|"
    r"disputed area|article 140|budget dispute|شەڕ|سەربازی|هێرش|درۆن|موشەک|تۆپباران|"
    r"پێشمەرگە|ئەمنی|ئاسایش|سیاسی|حکومەت|پەرلەمان|هەڵبژاردن|دیپلۆماسی|سەرۆک|وەزیران|قەیران|ململانێ)",
    re.IGNORECASE,
)

# Regional geopolitics prioritized ahead of generic non-political international stories
REGIONAL_GEOPOLITICAL_PATTERN = re.compile(
    r"(\biraq\b|baghdad|\biran\b|tehran|turkey|ankara|syria|damascus|united states|\bu\.s\.?\b|"
    r"washington|middle east|israel|\bgaza\b|lebanon|hezbollah|pentagon|\bnato\b|white house|kremlin|"
    r"عێراق|بەغدا|ئێران|تورکیا|سوریا|ئەمریکا|واشنتن|ڕۆژهەڵاتی ناوەڕاست|ئیسرائیل|غەززە|لوبنان)",
    re.IGNORECASE,
)

PRIORITY_TIER_1A = "1A"
PRIORITY_TIER_1B = "1B"
PRIORITY_TIER_2 = "2"
PRIORITY_TIER_3 = "3"

# ── Deterministic Second-Layer Blocked Topic Guards ──
BLOCKED_CATEGORY_NAMES: set[str] = {
    "international sports",
    "sports",
    "lifestyle",
    "entertainment",
    "culture",
}

INTERNATIONAL_COURT_OR_STATE_CRISIS_PATTERN = re.compile(
    r"(international\s+court\s+of\s+justice|international\s+criminal\s+court|\bicj\b|\bicc\b|"
    r"war\s+crimes\s+tribunal|un\s+security\s+council|constitutional\s+crisis|impeachment)",
    re.IGNORECASE,
)

DOMESTIC_CRIME_COURT_PATTERN = re.compile(
    r"(sentenced\s+to\s+\d+|years?\s+in\s+(?:federal\s+)?prison|months?\s+in\s+prison|"
    r"pleaded\s+guilty|u\.?s\.?\s+district\s+court|federal\s+court\s+in\s+[a-z]+|"
    r"federal\s+prosecutors?|district\s+judge|jury\s+convicted|indicted\s+(?:on|for)|"
    r"lying\s+to\s+(?:the\s+)?fbi|fbi\s+(?:arrest|agent|agents|affidavit)|"
    r"(?:u\.?s\.?\s+)?department\s+of\s+justice.{0,90}(?:sentenc|plead|indict|guilty|arrest)|"
    r"justice\s+department.{0,90}(?:sentenc|plead|indict|guilty|arrest)|"
    r"(?:houston|texas|florida|california|new\s+york|chicago|ohio|michigan)\s+(?:man|woman|resident)|"
    r"municipal\s+court|local\s+police|police\s+(?:arrested|charged)\s+(?:a\s+|two\s+|three\s+)?(?:man|woman|suspect|teen)|"
    r"\bdeportation\s+(?:case|hearing|order)\b|\bimmigration\s+(?:court|judge|hearing)\b|\babrego\s+garcia\b|"
    r"prison\s+sentence|bail\s+hearing|extremism\s+trial|propaganda\s+trial|material\s+support\s+to\s+(?:isis|terror)|"
    r"سزای\s+\d+\s+ساڵ\s+زیندان|دادگای\s+فیدراڵی\s+ئەمریکا|تۆمەتی\s+درۆکردن\s+بۆ\s+ئێف\s*بی\s*ئای|"
    r"پڕوپاگەندەی\s+داعش.{0,40}دادگای)",
    re.IGNORECASE,
)

SPORTS_PATTERN = re.compile(
    r"(\bfootball\b|\bsoccer\b|\bafcon\b|africa\s+cup(?:\s+of\s+nations)?|"
    r"(?:world|africa|nations|champions|european|asian)\s+cup|\bolympics?\b|\bfifa\b|\buefa\b|"
    r"champions\s+league|premier\s+league|la\s+liga|serie\s+a|bundesliga|"
    r"\bnba\b|\bnfl\b|\bmlb\b|\bnhl\b|\bufc\b|\bwimbledon\b|grand\s+slam|"
    r"\bathlete\b|\bfootballer\b|\bgoalkeeper\b|\bstriker\b|\bmidfielder\b|\bpenalty\s+shootout\b|\bhat-trick\b|\bred\s+card\b|"
    r"\b(?:football|soccer|basketball|tennis|cricket|golf|rugby|hockey|baseball|swimming)\s+(?:match|player|star|club|team|coach|tournament|league|championship)\b|"
    r"\bformula\s+1\b|\bf1\s+(?:grand\s+prix|race)\b|\btransfer\s+window\b|"
    r"\bnational\s+(?:football|soccer|basketball)?\s*team\b|"
    r"\b(?:semifinal|quarterfinal|cup\s+final|sports?\s+tournament|sports?\s+championship)\b|"
    r"\b(?:win|wins|won|beat|beats|defeat|defeats|scores?|scored)\b.{0,40}\b(?:match|cup|goal|coach|fc|stadium|morocco|senegal)\b|"
    r"تۆپی\s+پێ|تۆپی\s+سەبەتە|جامی\s+نەتەوەکانی\s+ئەفریقا|مۆندیال|ئۆڵۆمپیاد|"
    r"خولی\s+(?:پاڵەوانەکان|نایاب|ئەستێرەکان|ئینگلیزی|ئیسپانی)|پاڵەوانێتی\s+وەرزشی|"
    r"هەڵبژاردەی\s+تۆپی\s+پێ|یانەی\s+(?:تۆپی\s+پێ|وەرزشی)|یاریزان(?:ان|ی)|گۆڵپارێز|یاریگا|گۆڵکار|وەرزش|"
    r"كرة\s+القدم|دوري\s+الأبطال|كأس\s+العالم|المنتخب\s+الوطني|رياضة|مباراة\s+كرة)",
    re.IGNORECASE,
)

CULTURE_LIFESTYLE_PATTERN = re.compile(
    r"(\bcelebrity\b|\bcelebrities\b|\bred\s+carpet\b|\bbox\s+office\b|\boscar(?:s)?\b|\bacademy\s+awards?\b|"
    r"\bgrammy\s+awards?\b|\bemmy\s+awards?\b|\bgolden\s+globe\b|\bfashion\s+week\b|"
    r"\breality\s+tv\b|\breality\s+show\b|\bhollywood\b|\bbollywood\b|\bbroadway\b|"
    r"\bpop\s+star\b|\bmovie\s+star\b|\bfilm\s+star\b|\btv\s+star\b|\btelevision\s+star\b|"
    r"(?<!state\s)(?<!threat\s)(?<!bad\s)(?<!malicious\s)(?<!non-state\s)(?<!cyber\s)\bactors?\b|\bactress(?:es)?\b|"
    r"\bsinger\b|\bsongwriter\b|\bmusician\b|\brapper\b|\bcomedian\b|\b(?:film|movie)\s+director\b|"
    r"\bmovie\s+review\b|\bfilm\s+review\b|\balbum\s+review\b|\bculinary\s+festival\b|\bcinema\b|\bentertainment\b|"
    r"\bmusic\s+concert\b|\bart\s+exhibition\b|\bhoroscope\b|\bzodiac\s+sign\b|"
    r"\btravel\s+(?:guide|tips)\b|\btourist\s+destination\b|\bvacation\s+(?:spot|guide)\b|"
    r"\bbest\s+beaches\b|\bluxury\s+hotel\b|\bcruise\s+ship\b|\bweekend\s+getaway\b|"
    r"\b(?:weekend|sunny)\s+weather\b|\bweather\s+trivia\b|\bpollen\s+forecast\b|\bcherry\s+blossom\b|"
    r"\bsmartphone\b|\biphone\b|\bsamsung\s+galaxy\b|\bandroid\s+(?:phone|update|app)\b|"
    r"\bios\s+update\b|\bapp\s+update\b|\bwhatsapp\s+update\b|\binstagram\s+feature\b|\btiktok\s+trend\b|"
    r"\bgadget\s+review\b|\bsmartwatch\b|\bearbuds\b|\bgaming\s+console\b|\bplaystation\b|\bxbox\b|\bnintendo\b|"
    r"\bviral\s+(?:video|clip)\b|\bpet\s+rescue\b|\blottery\s+winner\b|\bcooking\s+recipe\b|\bdiet\s+trend\b|"
    r"شوێنی\s+گەشتیاری|فاشیۆن|گۆرانیبێژ(?:ی|ان)?|توقعات\s+الطقس\s+اليومية|مشاهير|عروض\s+سياحية|"
    r"هۆلیوود|ئەکتەر(?:ی|ێک|ێکی|ان)?|سینەماکار|ستێرەی\s+(?:سینەما|هۆلیوود)|سینەما|مۆسیقا|فیلم(?:ی|ەکان)?|"
    r"هونەرمەند(?:ی|ێک|ێکی|ان)?|کۆچی\s+دوایی.{0,60}(?:ئەکتەر|هونەرمەند|گۆرانیبێژ|سینەما|هۆلیوود)|"
    r"[پب]ێشانگای?(?:ەک|ەکی)?(?:دا)?\s*(?:هونەری|وێنە|وێنەکێشان|شێوەکاری|کتێب|فۆتۆگرافی)|"
    r"[پب]ێشانگایەکدا.{0,40}(?:نمایش|تابلۆ|هونەر)|تابلۆی\s+(?:هونەری|شێوەکاری)|شێوەکار(?:ی|ان)|"
    r"ف[یێ]ستیڤاڵی?\s*(?:هونەری|کلتووری|فیلم|شانۆ|گۆرانی|شیعر)|کۆنسێرت|شانۆگەری|"
    r"ئاهەنگی\s+(?:گۆرانی|مۆسیقا)|وەرزشی\s+(?:تۆپی\s+پێ|وشتنی\s+باری|مەلەوانگە)|خولی\s+(?:نایاب|ئەستێرەکان)|"
    r"ڕووداو(?:ی|ێکی)\s+هاتوو?چۆ|پێکدادانی\s+ئۆتۆمبێل|پێشبینی\s+کەشوهەوا|"
    r"هونەری\s+(?:نمایش|پێشانگا)|فەستیڤاڵ|نمایشگەی\s+هونەری|معرض\s+فني|مهرجان\s+(?:فني|ثقافي)|"
    r"هوليوود|ممثل(?:ة|ين)?|فنان(?:ة|ين)?|مغني(?:ة)?|نجم\s+(?:سينما|هوليوود)|وفاة\s+(?:الممثل|الفنان|المغني|النجم)|"
    r"حفل\s+(?:موسيقي|غنائي)|عرض\s+فيلم|مباراة\s+ودية|مسابقة\s+رياضية)",
    re.IGNORECASE,
)

# Multi-story podcast / newsletter briefing & composite roundup rejection
_ROUNDUP_URL_PATTERN = re.compile(
    r"/(?:podcasts?|newsletters?|up-first|morning-briefing|evening-briefing|news-briefing|live-updates)(?:/|$|\?)",
    re.IGNORECASE,
)
_ROUNDUP_TEXT_PATTERN = re.compile(
    r"(\bup\s+first\b|\b(?:morning|evening|daily|weekend|midday|tonight'?s)\s+(?:briefing|roundup|digest|newsletter)\b|"
    r"\bnews\s+(?:summary|roundup|digest|briefing)\b|\b(?:five|5|top)\s+things\s+(?:to\s+know|happening)\b|"
    r"\bwhat\s+(?:to\s+know|happened)\s+today\b|^\s*live\s+updates\s*[:\-–]|"
    r"\btoday\s+on\s+up\s+first\b|\bin\s+this\s+episode\b|\bfirst\s+up\b.{0,80}\balso\s+(?:today|ahead)\b|"
    r"پوختەی\s+هەواڵەکانی\s+ڕۆژ)",
    re.IGNORECASE,
)
_ROUNDUP_ALLCAPS_AND_TITLE_RE = re.compile(r"\S+(?:\s+\S+){2,}\s+AND\s+\S+(?:\s+\S+){2,}")
_ROUNDUP_SEMICOLON_TITLE_RE = re.compile(r"\S+(?:\s+\S+){3,}\s*;\s+\S+(?:\s+\S+){3,}")

# Publisher brand/signature stripping so outlet names (e.g. Kurdistan24) never trigger geographic classification
_OUTLET_BRAND_STRIP_RE = re.compile(
    r"(kurdistan\s*24(?:\s+(?:en|sorani|arabic|english))?|کوردستان\s*(?:24|٢٤)|#k24\b|"
    r"rudaw\s*(?:digital|english|arabic|net)?|تۆڕی\s+میدیایی\s+رووداو|north\s+press(?:\s+agency)?|نۆرس\s+پرێس|"
    r"pukmedia|nrttvofficial|sabereen\s+(?:news|plus))",
    re.IGNORECASE,
)

_ECONOMY_TOPIC_RE = re.compile(
    r"(\beconom|\btrade\b|\btariff|\bmarket|\bstock\b|\boil\b|\bgas\b|\bopec\b|\binflation\b|\bbank\b|"
    r"بازرگانی|ئابووری|نەوت|گاز|بانک|بۆرسە|گومرگ|هەڵاوسان|هەناردە|دۆلار)",
    re.IGNORECASE,
)
_MILITARY_TOPIC_RE = re.compile(
    r"(\bwar\b|\bmilitary\b|\bairstrike|\bmissile|\bdrone|\bdefense|\btroops|\bshelling|\battack|"
    r"شەڕ|سەربازی|هێرش|درۆن|موشەک|بۆردومان|ئەمنی)",
    re.IGNORECASE,
)
_CYBER_TOPIC_RE = re.compile(
    r"(\bcyber|\bransomware|\bhack|\bmalware|\bzero-day|سایبەری|ئەلیکترۆنی|هاک)",
    re.IGNORECASE,
)

TRAFFIC_MUNICIPAL_AD_PATTERN = re.compile(
    r"(تنويه\s+مروري|مديرية\s+المرور|شرطة\s+المرور|مفارز\s+المرور|غرامة\s+مرورية|"
    r"رادار(?:ات)?\s+(?:المرور|السرعة)|كاميرات\s+(?:المراقبة\s+المرورية|السرعة|المرور)|"
    r"ازدحام\s+مروري|[کك]امێرای\s+(?:خێرایی|هاتوو?چۆ)|بەڕێوەبەرایەتی\s+هاتوو?چۆ|"
    r"\bspeed\s+(?:radar\s+)?cameras?\b|\btraffic\s+(?:radar|police)\b|"
    r"جدول\s+(?:قطع\s+)?الكهرباء|انقطاع\s+التيار\s+الكهربائي|صيانة\s+شبكة\s+المياه|"
    r"أمانة\s+بغداد.{0,40}صيانة|ساعات\s+التشغيل\s+للمولدات|"
    r"\bwater\s+and\s+electricity\s+outage\b|\boutage\s+maintenance\s+schedule\b|"
    r"للاعلان(?:ات)?|للتواصل\s+والإعلان|عرض\s+خاص|تخفيضات|خصم\s+\d+%|"
    r"اشترك(?:وا)?\s+(?:في|عبر)\s+قناتنا|انضموا\s+إلى\s+قناتنا|ڕیکلام|"
    r"\bsponsored\s+discount\b|\bfollow\s+us\s+on\s+telegram\b)",
    re.IGNORECASE,
)

# Pre-filter patterns for routine non-political market/consumer-tech/science noise
# skipped BEFORE sending candidates to Gemini (unless tied to war, sanctions, cyberwar, or Kurdistan).
_OFFTOPIC_ECONOMY_PATTERN = re.compile(
    r"(\bstock\s+market\b|\bwall\s+street\b|\bs&p\s*500\b|\bdow\s+jones\b|\bnasdaq\b|"
    r"\bquarterly\s+(?:earnings|revenue|profit)\b|\bshare\s+price\b|\bstock\s+surge[sd]?\b|"
    r"\bbitcoin\b|\bcryptocurrency\b|\bethereum\b|\bmortgage\s+rates?\b|\bretail\s+sales\b|"
    r"\bhousing\s+market\b|\bipo\s+debut\b|\bdividend\b)",
    re.IGNORECASE,
)

_OFFTOPIC_TECH_SCIENCE_PATTERN = re.compile(
    r"(\bipad\b|\bmacbook\b|\bgraphics\s+card\b|\bchip\s+benchmark\b|\bapp\s+store\b|"
    r"\bsocial\s+media\s+app\b|\bstreaming\s+service\b|\bexoplanet\b|\bspace\s+telescope\b|\bcomet\b)",
    re.IGNORECASE,
)

_GEOPOLITICAL_SAFEGUARD_PATTERN = re.compile(
    r"(\bwar\b|\bmilitary\b|\bsanctions?\b|\bembargo\b|\bairstrike\b|\bmissile\b|\bdrone\b|"
    r"\bcyberattack\b|\bstate-sponsored\b|\bransomware\b|\bnuclear\b|\bnato\b|\bopec\b|"
    r"\boil\s+exports?\b|\bbudget\s+dispute\b|\bparliament\b|\btreaty\b|\bceasefire\b|"
    r"kurdistan|erbil|baghdad|iran|israel|ukraine|russia|syria|turkey)",
    re.IGNORECASE,
)

_OFFTOPIC_FEED_CATEGORIES: set[str] = {
    "ai_tech",
    "ai & technology",
    "business & corporate",
    "science_space",
    "science & research",
    "space & astronomy",
}

LEGACY_TO_CANONICAL_CATEGORY: dict[str, str] = {
    "kurdistan": "kurdistan",
    "wars & armed conflicts": "military_conflict",
    "military & defense": "military_conflict",
    "criminal investigations & security": "military_conflict",
    "international politics": "politics_world",
    "world leaders & governments": "politics_world",
    "diplomacy & international relations": "politics_world",
    "international organizations": "politics_world",
    "human rights & humanitarian": "politics_world",
    "health & medical": "politics_world",
    "climate & environment": "politics_world",
    "natural disasters & emergencies": "politics_world",
    "other significant events": "politics_world",
    "world news": "politics_world",
    "global economy & financial markets": "economy_markets",
    "energy, oil & gas": "economy_markets",
    "business & corporate": "economy_markets",
    "ai & technology": "ai_tech",
    "ai & tech": "ai_tech",
    "cybersecurity": "cybersecurity",
    "science & research": "science_space",
    "science & space": "science_space",
    "space & astronomy": "science_space",
}

GLOBAL_RELEVANCE_SYSTEM_PROMPT = (
    "You are the chief international news intelligence editor for Kurdish Global News, "
    "an independent, neutral, professional Central Kurdish (Sorani) international news organization.\n"
    "Your channel specializes EXCLUSIVELY in Military, Security, War, and High-Level Geopolitical Intelligence, "
    "plus verified Southern Kurdistan (Bashur) political, military, and security developments.\n\n"
    "Evaluate the importance, relevance, breaking urgency, category, and geographic region of the given news article.\n\n"
    "Allowed values for \"category\" (MUST be strictly one of these 3 active keys):\n"
    "- \"military_conflict\": Active armed conflicts, airstrikes, missile/drone attacks, defense treaties, military operations, troop mobilization/deployments, weapons deployment, air-defense, or high-level assassinations.\n"
    "- \"politics_world\": High-level geopolitics, world leaders, government collapse/decisions, parliamentary crises, national election crises, UN/NATO diplomacy, strategic state-level treaties, major international sanctions, major diplomatic crises.\n"
    "- \"kurdistan\": ONLY verified political, military, or security developments directly taking place in Southern Kurdistan (Bashur — Erbil, Sulaymaniyah, Duhok, Halabja, Kirkuk, KRG, Peshmerga, Erbil-Baghdad relations). Non-Bashur news must NEVER receive \"kurdistan\".\n\n"
    "Allowed values for \"region\" (MUST be strictly one of these 3 keys):\n"
    "- \"kurdistan_bashur\": Verified Southern Kurdistan political, military, or security events directly involving Erbil, Sulaymaniyah, Duhok, Halabja, Kirkuk, KRG, or Peshmerga.\n"
    "- \"middle_east\": Events occurring in or directly concerning Iraq, Iran, Syria, Turkey, Israel/Palestine, Lebanon, Yemen, Jordan, Saudi Arabia, Gulf states, OR international military/diplomatic intervention directly in the Middle East (e.g. US strikes in Iraq/Syria, EU sanctions on Iran, CENTCOM operations, Pentagon statements on Middle East security). A story is NEVER global merely because the US, EU, or NATO is involved.\n"
    "- \"global\": Genuinely international events taking place outside the Middle East with no direct Middle East operational connection (e.g. Ukraine-Russia war in Europe, China-Taiwan confrontation, US domestic politics, European internal government crises).\n\n"
    "STRICT Editorial Exclusions (MUST set score 1 to 3, passed=false):\n"
    "- ALL Economy / Finance / Business: stock markets, GDP reports, inflation, interest rates, cryptocurrency/Bitcoin, oil prices/tickers, corporate earnings, IPOs, mergers, trade deals, business investments.\n"
    "- ALL Technology / AI / Consumer Tech: AI models/OpenAI, software releases, smartphones, gadgets, space missions/NASA, ordinary hacking/ransomware.\n"
    "- ALL Ceremonial / Protocol / PR Fluff: festivals, exhibitions, ceremonies, book fairs, awards, marathons, charity foundation meetings with delegations, routine delegation visits, ribbon-cutting, inaugurating parks/schools/hospitals.\n"
    "- ALL Diplomatic Pleasantries: congratulatory messages, holiday greetings, routine condolences, ambassadors presenting credentials, courtesy visits, routine phone calls.\n"
    "- ALL sports, celebrity/entertainment news, domestic crime/court trials, traffic accidents, municipal service notices.\n\n"
    "STRICT Breaking News Rule (`is_breaking`):\n"
    "- Set \"is_breaking\": true ONLY for sudden, unfolding, emergency, high-impact events happening right now (and ONLY when score >= 9): "
    "active military strikes, explosions, missile/drone attacks, assassinations, sudden declarations of war or ceasefires, major earthquakes/natural disasters, "
    "emergency government or court decrees taking immediate effect, or major critical-infrastructure cyberattacks.\n"
    "- Explicitly set \"is_breaking\": false for ALL routine political statements, interviews, speeches, party-leader commentary, diplomatic meetings, "
    "opinions, analyses, forecasts, routine economic updates, or reactions — EVEN IF the article is high-priority Kurdistan news with score 8 or 9.\n\n"
    "Scoring Criteria (1 to 10):\n"
    "- 9 to 10 (HIGHEST PRIORITY): Major Southern Kurdistan (Bashur) security/military/drone/missile incidents or major armed conflicts, military escalations, world leader summits, and severe state crises.\n"
    "- 8 (HIGH PASS): Genuine political, military, or security developments taking place in Southern Kurdistan; major regional geopolitics (Iraq, Iran, Turkey, Syria, Israel, Middle East); major strategic state-level treaties or critical sanctions.\n"
    "- 7 (PASS): Notable international diplomacy, cross-border military crises, or ICJ/ICC state rulings.\n"
    "- 1 to 6 (STRICTLY REJECT): All off-topic, economy, tech, ceremonial fluff, routine meetings, local crime, sports, entertainment.\n\n"
    "Cross-Language Event Identification & Live Threading:\n"
    "- Include concise standardized ENGLISH `event_key` (lowercase snake_case identifying the SPECIFIC incident), `event_entities` (3 to 5 lowercase English entity/location/topic tokens), and optional `update_type`.\n\n"
    "Respond ONLY with valid JSON (no markdown fences, no commentary):\n"
    '{"score": <integer 1-10>, "is_breaking": <true or false>, "category": "<military_conflict|politics_world|kurdistan>", '
    '"region": "<kurdistan_bashur|middle_east|global>", '
    '"event_key": "<english_snake_case_event_id>", "event_entities": ["entity1", "entity2", "topic"], '
    '"update_type": "<new_story|casualties_update|official_response|retaliation|ceasefire|repeat_coverage>", '
    '"reason": "<short one-sentence justification>"}'
)

BATCH_RELEVANCE_SYSTEM_PROMPT = (
    GLOBAL_RELEVANCE_SYSTEM_PROMPT.rsplit("Respond ONLY with valid JSON", 1)[0]
    + "Batch Evaluation Mode:\n"
    + "- You will receive a JSON array of candidate articles, each identified by its integer `idx`.\n"
    + "- Evaluate EVERY article independently using the exact rules above.\n"
    + "- Respond ONLY with valid JSON containing an `items` array with one object per input `idx`:\n"
    + '{"items": [{"idx": <integer>, "score": <integer 1-10>, "is_breaking": <true or false>, '
    + '"category": "<military_conflict|politics_world|kurdistan>", "region": "<kurdistan_bashur|middle_east|global>", '
    + '"event_key": "<english_snake_case_event_id>", "event_entities": ["entity1", "entity2", "topic"], '
    + '"update_type": "<optional_update_type>", "reason": "<short one-sentence justification>"}]}'
)


@dataclass
class RelevanceEvaluation:
    """Result of the Gemini Global Importance & Category Gatekeeper check."""
    score: int
    is_breaking: bool
    reason: str
    passed: bool
    category: str = "politics_world"
    used_fallback: bool = False
    event_key: str = ""
    event_entities: list[str] = field(default_factory=list)
    update_type: str = ""
    region: str = ""


def clean_article_event_text(article: Article) -> str:
    """
    Extract normalized event text strictly from actual article content
    (title, summary, translated content, event_entities) after stripping leading
    agency datelines and publisher brand names (`Kurdistan24`, `Rudaw`, etc.)
    so source names and feed metadata never spoof geographic classification.
    """
    clean_title = _OUTLET_BRAND_STRIP_RE.sub(" ", strip_news_dateline(article.original_title or ""))
    clean_summary = _OUTLET_BRAND_STRIP_RE.sub(" ", strip_news_dateline(article.original_summary or ""))
    tr_title = _OUTLET_BRAND_STRIP_RE.sub(" ", strip_news_dateline(getattr(article, "translated_title", "") or ""))
    tr_summary = _OUTLET_BRAND_STRIP_RE.sub(" ", strip_news_dateline(getattr(article, "translated_summary", "") or ""))
    ents = " ".join(str(e) for e in (getattr(article, "event_entities", None) or []))
    return re.sub(r"\s+", " ", f"{clean_title} {clean_summary} {tr_title} {tr_summary} {ents}").strip()


def is_multi_story_roundup(article: Article) -> bool:
    """
    Return True for podcast briefings (e.g., NPR Up First), newsletters, or composite
    multi-story roundups joining unrelated topics into one feed item.
    """
    url = (getattr(article, "original_url", "") or getattr(article, "canonical_url", "") or "").strip()
    if url and _ROUNDUP_URL_PATTERN.search(url):
        return True

    title = (getattr(article, "original_title", "") or "").strip()
    summary = (getattr(article, "original_summary", "") or "").strip()
    if title and (_ROUNDUP_ALLCAPS_AND_TITLE_RE.search(title) or _ROUNDUP_SEMICOLON_TITLE_RE.search(title)):
        return True

    combined = f"{title} {summary}".strip()
    if combined and _ROUNDUP_TEXT_PATTERN.search(combined):
        return True
    return False


def is_genuine_kurdistan_event(article: Article) -> bool:
    """
    Return True ONLY when stripped event text directly involves Kurdish
    territories, governments, or forces (never merely datelines, outlet name, or foreign ISIS trials).
    Non-Kurdish Syrian locations (Tartous, Latakia, Damascus, etc.) are strictly excluded unless
    the event genuinely involves Kurdish territories, institutions, or forces.
    """
    clean_text = clean_article_event_text(article)
    if not clean_text:
        return False

    # Check for non-Kurdistan Syrian locations
    if NON_KURDISTAN_SYRIA_PATTERN.search(clean_text):
        # Only consider Kurdistan if there is an explicit and genuine Kurdish territorial / defense presence
        has_bashur = bool(BASHUR_PATTERN.search(clean_text))
        has_rojava = bool(NON_BASHUR_KURDISTAN_PATTERN.search(clean_text))
        if not (has_bashur or has_rojava):
            return False

    return bool(KURDISTAN_PATTERN.search(clean_text))


def check_blocked_topic(article: Article) -> Optional[str]:
    """
    Deterministic second-layer guard rejecting domestic crime/individual court trials,
    celebrity/Hollywood obituaries, sports/tournaments, multi-story roundups,
    and culture/lifestyle fluff before classification or publishing.
    Returns rejection reason string if blocked, else None.
    """
    cat = (getattr(article, "category", "") or "").strip().lower()
    if cat in BLOCKED_CATEGORY_NAMES:
        return f"Blocked category: {article.category}"

    if is_multi_story_roundup(article):
        return "Blocked topic: multi-story news roundup or podcast briefing excluded"

    clean_text = clean_article_event_text(article)
    if not clean_text:
        return None

    if SPORTS_PATTERN.search(clean_text):
        return "Blocked topic: sports/tournament news excluded from channel"

    if DOMESTIC_CRIME_COURT_PATTERN.search(clean_text) and not INTERNATIONAL_COURT_OR_STATE_CRISIS_PATTERN.search(clean_text):
        return "Blocked topic: foreign/domestic crime or individual court sentencing"

    if CULTURE_LIFESTYLE_PATTERN.search(clean_text):
        return "Blocked topic: culture/lifestyle/celebrity/entertainment fluff"

    if TRAFFIC_MUNICIPAL_AD_PATTERN.search(clean_text):
        return "Blocked topic: traffic/municipal-service notice or advertisement"

    return None


# ═════════════════════════════════════════════════════════════════════
# Deterministic Pre-LLM Editorial Gate: Military & High-Level Politics
# ═════════════════════════════════════════════════════════════════════

HARD_BANNED_ECONOMY_PATTERN = re.compile(
    r"(\b(?:stock|stocks|financial|crude|mortgage)\s+market[s]?\b|"
    r"\bwall\s+street\b|\bs&p\s*500\b|\bdow\s+jones\b|\bnasdaq\b|\bbourse\b|"
    r"\b(?:share|stock)\s+prices?\b|\bshares\s+(?:fall|rise|surge|drop|slide)\b|"
    r"\bbillionaire\s+signs\b|\blukoil\b|\boil\s+deal\b|\bcorporate\s+earnings\b|"
    r"\bfinancial\s+earnings\b|\bannual\s+financial\b|\bfinancial\s+results\b|"
    r"\bquarterly\s+(?:profit|earnings|revenue)\b|\bearnings\s+reports?\b|\bipo\s+debut\b|"
    r"\bmergers?\s+(?:and|&)\s+acquisitions?\b|\bm&a\b|\bcompany\s+deal\b|\bcommercial\s+deal\b|"
    r"\bcompany\s+announces\b.{0,60}\b(?:deal|earnings|results|profit|revenue|dividend)\b|"
    r"\bcommercial\s+contracts?\b|\bcorporate\s+delegations?\b|"
    r"\binvestment\s+announcements?\b|\binvestment\s+agreements?\b|\breal\s+estate\b|"
    r"\boil\s+prices?\s+(?:increase|rise|fall|drop|surge|slide|jump)\b|\bcrude\s+oil\s+prices?\b|"
    r"\bprice\s+of\s+oil\b|\bbarrel\s+of\s+oil\b|\bgdp\s+growth\b|\bgdp\s+reports?\b|"
    r"\binflation\s+(?:rate|report|data|figures?)\b|\bcentral\s+bank\s+(?:raises|cuts|changes|hikes)\s+interest\b|"
    r"\binterest\s+rates?\b|\bbitcoins?\b|\bcryptocurrenc(?:y|ies)\b|\bethereum\b|\bmortgage\s+rates?\b|"
    r"ئابووری|بازاڕی\s+بۆرسە|بۆرسە|نرخی\s+نەوت|نرخی\s+دۆلار|وەبەرهێنان|کۆمپانیا|"
    r"داهاتی\s+کۆمپانیا|قازانجی\s+کۆمپانیا|گرێبەستی\s+بازرگانی|هەڵاوسان|سوودی\s+بانکی|"
    r"بورصة|أسهم|أسعار\s+النفط|سعر\s+النفط|برميل\s+النفط|سعر\s+الدولار|استثمار|أرباح\s+الشركات|"
    r"التضخم|الفائدة\s+المصرفية|سعر\s+الفائدة|صفقة\s+تجارية|عقد\s+تجاري|بيتكوين|عملات\s+رقمية)",
    re.IGNORECASE,
)

HARD_BANNED_TECH_AI_CYBER_PATTERN = re.compile(
    r"(\bnew\s+ai\s+model\b|\bai\s+model\s+(?:launch|releases?|announced)\b|\bopenai\s+releases\b|"
    r"\bchatgpt\s+updates?\b|\bclaude\s+model\b|\bgemini\s+model\b|\bsoftware\s+releases?\b|"
    r"\bnew\s+smartphones?\b|\bsmartphones?\s+announced\b|\biphones?\s+\d+\b|\bsamsung\s+galaxy\b|"
    r"\bconsumer\s+electronics\b|\bgadgets?\s+review\b|\bspace\s+telescopes?\b|\bexoplanets?\b|"
    r"\bnasa\s+announcements?\b|\bspace\s+missions?\b|\bspace\s+launch\b|\bspacex\s+starship\b|"
    r"\bransomware\s+cyberattacks?\b|\bransomware\s+attacks?\b|\bhits\s+banks?\b|\bdata\s+breaches?\b|"
    r"\bmalware\b|\broutine\s+cyberattacks?\b|\bphishing\b|\bhacking\s+incidents?\b|"
    r"ژیریی\s+دەستکرد|تەکنەلۆژیا|مۆبایلی\s+نوێ|ئایفۆن|سامسۆنگ|هاککردنی\s+بانک|"
    r"باجی\s+سایبەری|ئاسایشی\s+سایبەری|هاکەر|باجی\s+ئەلیکترۆنی|"
    r"ذكاء\s+اصطناعي|نموذج\s+ذكاء|تطبيق\s+جديد|هاتف\s+ذكي|إطلاق\s+هاتف|هجوم\s+فدية|"
    r"برامج\s+الفدية|اختراق\s+بنك|تسريب\s+بيانات|أمن\s+سيبراني)",
    re.IGNORECASE,
)

HARD_BANNED_PROTOCOL_FLUFF_PATTERN = re.compile(
    r"(\bannual\s+(?:cultural\s+)?festival\b|\bcultural\s+festival\b|\bfilm\s+festival\b|\bmusic\s+festival\b|"
    r"\bbook\s+fair\b|\bannual\s+exhibition\b|\bart\s+exhibition\b|\baward\s+ceremony\b|\bmarathons?\b|"
    r"\bconcerts?\b|\bcelebration\b|\bceremony\s+held\b|\banniversary\s+celebration\b|"
    r"\b(?:charity\s+)?foundation\s+met\b|\bmet\s+with\s+(?:a\s+)?(?:foreign\s+|european\s+)?delegation\b|"
    r"\bwelcomed?\s+(?:a\s+)?(?:foreign\s+|european\s+)?delegation\b|\breceived?\s+(?:a\s+)?(?:foreign\s+|european\s+)?delegation\b|"
    r"\bcharity\s+activities\b|\bngo\s+meetings?\b|\broutine\s+(?:workshop|seminar|conference)\b|\bcorporate\s+meetings?\b|"
    r"\bribbon[\s\-]cuttings?\b|\binaugurated?\s+(?:a\s+)?(?:park|school|hospital|bridge|building)\b|"
    r"\bcompany\s+welcomes\s+(?:foreign\s+)?delegation\b|\bcongratulatory\s+messages?\b|"
    r"\bofficial\s+sends\s+congratulatory\b|\bcongratulates\b|\bholiday\s+greetings\b|"
    r"\broutine\s+(?:diplomatic\s+)?condolences?\b|\bsent\s+condolences\b|\bambassadors?\s+(?:presents?|presented)\s+credentials\b|"
    r"\broutine\s+ambassador\s+courtesy\s+visits?\b|\bcourtesy\s+visits?\b|\broutine\s+(?:diplomatic\s+)?telephone\s+calls?\b|"
    r"\bdiplomatic\s+pleasantries\b|"
    r"فێستیڤاڵ|پێشانگا(?:ی|دا)?|مەراسیم(?:ی)?|خەڵات(?:ی)?|کۆنسێرت|ئاهەنگ|ماراسۆن|نمایشگەی\s+هونەری|"
    r"دامەزراوە(?:ی)?\s*(?:خێرخوازی)?.{0,40}شاند|دەزگای\s+خێرخوازی.{0,40}شاند|کۆبوونەوەی\s+دامەزراوە|"
    r"پێشوازی\s+لە\s+شاند|کۆمپانیا.{0,30}شاند|فاوندەیشن|وۆرکشۆپ|سیمینار|"
    r"کردنەوەی\s+(?:باخچە|قوتابخانە|نەخۆشخانە|پرد|باڵەخانە)|"
    r"پەیامی\s+پیرۆزبایی|بروسکەی\s+سەرەخۆشی|پیرۆزبایی\s+(?:ئاراستە|دەکات|کرد)|سەردانی\s+پرۆتۆکۆڵی|پێشکەشکردنی\s+باوەڕنامە|"
    r"مهرجان|معرض|احتفال|حفل\s+تكريم|ماراثون|حفل\s+غنائي|"
    r"استقبل\s+وفداً|التقى\s+وفداً|مؤسسة\s+خيرية|جمعية\s+خيرية|ورشة\s+عمل|افتتاح\s+(?:مدرسة|حديقة|مستشفى|جسر)|"
    r"برقية\s+تهنئة|برقية\s+تعزية|زيارة\s+مجاملة|تقديم\s+أوراق\s+الاعتماد|اتصال\s+بروتوكولي)",
    re.IGNORECASE,
)

SUBSTANTIVE_MILITARY_PATTERN = re.compile(
    r"(\bairstrikes?\b|\bdrone\s+strikes?\b|\bdrone\s+attacks?\b|\bmissile\s+attacks?\b|"
    r"\brockets?\s+fired\b|\bartillery\s+shelling\b|\bexplosions?\b|\bbombing\b|"
    r"\bmilitary\s+clashes\b|\barmed\s+clashes\b|\bcross-border\s+(?:military\s+)?operations?\b|"
    r"\bmilitary\s+operations?\b|\btroop\s+(?:deployments?|mobilization)\b|\bmilitary\s+mobilization\b|"
    r"\bcombat\s+operations?\b|\bair\s+defense\s+(?:systems?|intercepted)\b|\bintercepted\s+(?:missiles?|drones?)\b|"
    r"\bweapons\s+deployments?\b|\bpatriot\s+missiles?\b|\bassassination\s+of\b|\bcommanders?\s+killed\b|"
    r"\bsecurity\s+forces\s+(?:kill|neutralize|target)\b|\bmajor\s+military\s+withdrawals?\b|"
    r"\bceasefire\s+(?:agreements?|violations?|collapse)\b|\barmed\s+conflict\b|\bstate\s+security\s+operations?\b|"
    r"\bstate[\s\-]sponsored\s+cyberattacks?\b|\bcyber\s+warfare\b|\benergy\s+grid\b|"
    r"هێرشی\s+ئاسمانی|بۆردومانی\s+ئاسمانی|هێرشی\s+درۆنی|هێرشی\s+موشەکی|پێکدادانی\s+سەربازی|"
    r"ئۆپەراسیۆنی\s+سەربازی|تۆپباران|ڕووبەڕووبوونەوەی\s+چەکداری|کشانەوەی\s+سەربازی|ئاگربەست|"
    r"بەرگری\s+ئاسمانی|کوژرانی\s+فەرماندە|هێزە\s+ئەمنییەکان|"
    r"ضربة\s+جوية|غارة\s+جوية|قصف\s+جوي|هجوم\s+بطائرة\s+مسيرة|هجوم\s+صاروخي|اشتباكات\s+مسلحة|"
    r"عملية\s+عسكرية|قصف\s+مدفعي|وقف\s+إطلاق\s+النار|دفاع\s+جوي|اغتيال\s+قيادي|مقتل\s+قائد\s+عسكري)",
    re.IGNORECASE,
)

HIGH_LEVEL_POLITICS_PATTERN = re.compile(
    r"(\bgovernment\s+collapse\b|\bgovernment\s+resigns?\b|\bparliamentary\s+crisis\b|"
    r"\bconstitutional\s+crisis\b|\belection\s+crisis\b|\bvot(?:e|ing)\s+of\s+no\s+confidence\b|"
    r"\bstate\s+of\s+emergency\b|\bstrategic\s+(?:defense\s+)?treat(?:y|ies)\b|\bmajor\s+international\s+sanctions\b|"
    r"\bnew\s+sanctions\s+targeting\b|\bstrategic\s+international\s+agreements?\b|\bstate-level\s+treat(?:y|ies)\b|"
    r"\bmajor\s+diplomatic\s+crisis\b|\bsevered\s+diplomatic\s+ties\b|\bexpelled\s+ambassadors?\b|"
    r"\bmilitary\s+alliances?\b|\bgeopolitical\s+negotiations\b|\bforeign\s+policy\s+(?:shift|change)\b|"
    r"\bkrg\s+and\s+baghdad\b|\berbil\s+and\s+baghdad\b|\bpeshmerga\s+security\b|"
    r"\bregional\s+civil\s+service\s+salaries\b|\bbudget\s+mechanisms?\b|"
    r"\b(?:major|mass|widespread|large|anti-government)?\s*(?:protests?|demonstrations?|civil\s+unrest|popular\s+uprising)\b|"
    r"خۆپیشاندان(?:ی|ەکان)?|تەقینەوەی\s+جەماوەری|تظاهر(?:ات)?\s*(?:حاشدة|شعبية|عارمة)?|احتجاج(?:ات)?\s*(?:شعبية|واسعة)?|"
    r"هەڵوەشاندنەوەی\s+حکومەت|دەستلەکارکێشانەوەی\s+حکومەت|قەیرانی\s+پەرلەمانی|قەیرانی\s+دەستووری|"
    r"سزای\s+نێودەوڵەتی|سزاکانی\s+ئەمریکا|سزاکانی\s+ئەوروپا|پەیماننامەی\s+ستراتیژی|ڕێککەوتنی\s+ستراتیژی|"
    r"قەیرانی\s+دیپلۆماسی|دەرکردنی\s+باڵیۆز|هەولێر\s+و\s+بەغدا|مووچەی\s+فەرمانبەرانی\s+هەرێم|"
    r"سقوط\s+الحكومة|استقالة\s+الحكومة|أزمة\s+برلمانية|أزمة\s+دستورية|عقوبات\s+دولية|"
    r"معاهدة\s+استراتيجية|اتفاقية\s+استراتيجية|أزمة\s+دبلوماسية|طرد\s+السفير|تحالف\s+عسكري)",
    re.IGNORECASE,
)

MIDDLE_EAST_GEOGRAPHY_PATTERN = re.compile(
    r"(\biraq\b|\bbaghdad\b|\bbasra\b|\bmosul\b|\bkerkuk\b|\bkirkuk\b|\bpmf\b|\bhashd\b|"
    r"\biran\b|\btehran\b|\birgc\b|\bquds\s+force\b|"
    r"\bsyria\b|\bdamascus\b|\btartous\b|\btartus\b|\blatakia\b|\baleppo\b|\bhoms\b|\bhama\b|\brojava\b|\bsdf\b|\bqamishli\b|\bkobani\b|\bhasakah\b|"
    r"\bturkey\b|\bturkiye\b|\bankara\b|\bisis\b|\bdaesh\b|"
    r"\bisrael\b|\btel\s+aviv\b|\bjerusalem\b|\bidf\b|\bnetanyahu\b|\bknesset\b|\bmossad\b|"
    r"\bpalestine\b|\bgaza\b|\bwest\s+bank\b|\bhamas\b|\bpij\b|\bpalestinian\b|"
    r"\blebanon\b|\bbeirut\b|\bhezbollah\b|"
    r"\byemen\b|\bsanaa\b|\bhouthi[s]?\b|\bansarallah\b|\bred\s+sea\b|\bbab\s+el[\s\-]mandeb\b|"
    r"\bjordan\b|\bamman\b|\bsaudi\s+arabia\b|\briyadh\b|\buae\b|\babu\s+dhabi\b|\bdubai\b|\bqatar\b|\bdoha\b|\bkuwait\b|\bbahrain\b|\boman\b|\begypt\b|\bcairo\b|\bsuez\b|"
    r"\bcentcom\b|\bmiddle\s+east\b|\bpersian\s+gulf\b|"
    r"عێراق|بەغدا|بەسڕە|موسڵ|کەرکووک|حەشدی\s+شەعبی|"
    r"ئێران|تاران|سوپای\s+پاسداران|قودس|"
    r"سووریا|سوریا|دیمەشق|تەرتووس|لازقیە|حەلەب|حومس|حەما|ڕۆژئاوا|قامیشلۆ|کۆبانێ|حەسەکە|"
    r"تورکیا|ئەنقەرە|"
    r"ئیسرائیل|تەلئەبیب|قودس|غەززە|کەرتی\s+غەززە|لوبنان|بەیروت|حزبوڵڵا|حەماس|فەلەستین|"
    r"یەمان|سەنعا|حوسی|حوسییەکان|دەریای\s+سوور|"
    r"ئوردن|سعوودیە|ڕیاز|ئیمارات|قەتەر|کووەیت|بەحرێن|عومان|میسر|قاهیرە|"
    r"ڕۆژهەڵاتی\s+ناوەڕاست|کەنداوی\s+فارس|"
    r"العراق|بغداد|البصرة|الموصل|كركوك|الحشد\s+الشعبي|"
    r"إيران|طهران|الحرس\s+الثوري|"
    r"سوريا|دمشق|طرطوس|اللاذقية|حلب|حمص|حماة|"
    r"تركيا|أنقرة|"
    r"إسرائيل|تل\s+أبيب|القدس|غزة|قطاع\s+غزة|لبنان|بيروت|حزب\s+الله|حماس|فلسطين|"
    r"اليمن|صنعاء|الحوثي|الحوثيون|البحر\s+الأحمر|"
    r"الأردن|عمان|السعودية|الرياض|الإمارات|أبوظبي|دبي|قطر|الدوحة|الكويت|البحرين|عمان|مصر|القاهرة|"
    r"الشرق\s+الأوسط|الخليج)",
    re.IGNORECASE,
)

FOREIGN_ACTOR_IN_REGIONAL_SECURITY_PATTERN = re.compile(
    r"\b(pentagon|centcom|white\s+house|state\s+department|nato|european\s+union|\beu\b)\b",
    re.IGNORECASE,
)


def extract_article_text(article: Article) -> str:
    """Extract raw combined textual content from article."""
    parts = [
        getattr(article, "original_title", "") or "",
        getattr(article, "original_summary", "") or "",
        getattr(article, "translated_title", "") or "",
        getattr(article, "translated_summary", "") or "",
        " ".join(str(e) for e in (getattr(article, "event_entities", None) or [])),
    ]
    return " ".join(p.strip() for p in parts if p.strip()).strip()


def normalize_article_text(article: Article) -> str:
    """Extract and normalize clean event text after stripping datelines and outlet brands."""
    return clean_article_event_text(article)


def detect_hard_banned_domain(clean_text: str, article: Optional[Article] = None) -> Optional[str]:
    """Detect if content belongs to hard-banned economy, technology, cyber, or disabled categories."""
    if article is not None:
        raw_cat = (getattr(article, "category", "") or "").strip().lower()
        if raw_cat in ("economy_markets", "business", "corporate"):
            return "economy"
        if raw_cat in ("ai_tech", "science_space"):
            return "technology"
        if raw_cat in ("cybersecurity",):
            return "cybersecurity"

    if HARD_BANNED_ECONOMY_PATTERN.search(clean_text):
        return "economy"

    if HARD_BANNED_TECH_AI_CYBER_PATTERN.search(clean_text):
        m = clean_text.lower()
        if any(k in m for k in ("cyber", "hack", "ransomware", "سایبەری", "هاک", "أمن سيبراني", "فدية")):
            return "cybersecurity"
        return "technology"

    return None


def detect_protocol_fluff(clean_text: str, article: Optional[Article] = None) -> Optional[str]:
    """Detect ceremonial fluff, festivals, delegation meetings, courtesy visits, or condolences."""
    m = HARD_BANNED_PROTOCOL_FLUFF_PATTERN.search(clean_text)
    if not m:
        return None
    matched = m.group(0).lower()
    if any(k in matched for k in ("festival", "فێستیڤاڵ", "مهرجان", "exhibition", "پێشانگا", "معرض")):
        return f"festival_or_exhibition ({matched})"
    if any(k in matched for k in ("delegation", "شاند", "وفد", "foundation", "دامەزراوە", "مؤسسة", "charity")):
        return f"routine_delegation_or_charity ({matched})"
    if any(k in matched for k in ("congratulat", "پیرۆزبایی", "تهنئة", "condolence", "سەرەخۆشی", "تعزية", "courtesy", "باوەڕنامە")):
        return f"diplomatic_pleasantry ({matched})"
    return f"ceremonial_fluff ({matched})"


def detect_substantive_military_event(clean_text: str, article: Optional[Article] = None) -> tuple[bool, str]:
    """Detect active military strikes, clashes, mobilization, troop movements, or defense operations."""
    m = SUBSTANTIVE_MILITARY_PATTERN.search(clean_text)
    if m:
        return True, m.group(0).strip()
    return False, ""


def detect_high_level_political_event(clean_text: str, article: Optional[Article] = None) -> tuple[bool, str]:
    """Detect high-level political crises, government collapse, state-level treaties, or major sanctions."""
    m = HIGH_LEVEL_POLITICS_PATTERN.search(clean_text)
    if m:
        return True, m.group(0).strip()
    return False, ""


def _is_state_treaty_or_crisis(clean_text: str) -> bool:
    """Check if text contains state-level treaty, ceasefire, or major diplomatic crisis."""
    return bool(re.search(
        r"(\bstate-level\s+treaty\b|\bstrategic\s+(?:defense\s+)?treaty\b|\bceasefire\s+agreement\b|"
        r"\bmajor\s+international\s+sanctions\b|\bexpelled\s+ambassadors?\b|\bsevered\s+diplomatic\b|"
        r"پەیماننامەی\s+ستراتیژی|ڕێککەوتنی\s+ستراتیژی|ئاگربەست|دەرکردنی\s+باڵیۆز|"
        r"معاهدة\s+استراتيجية|اتفاقية\s+استراتيجية|طرد\s+السفير|وقف\s+إطلاق\s+النار)",
        clean_text,
        re.IGNORECASE,
    ))


def classify_geographic_region(article: Article) -> str:
    """
    Classify an article into exactly one of three geographic regions:
      1. 'kurdistan_bashur': Verified Southern Kurdistan (Erbil, Suli, Duhok, Kirkuk, KRG, Peshmerga)
      2. 'middle_east': Iraq, Iran, Syria, Turkey, Israel/Palestine, Lebanon, Yemen, Gulf,
         OR international intervention directly in the Middle East (US strikes in Iraq/Syria, EU sanctions on Iran, etc.)
      3. 'global': Genuinely outside the Middle East with no direct Middle East operational connection
         (e.g., Ukraine-Russia, China-Taiwan, US domestic politics, European internal crises)
    """
    clean_text = clean_article_event_text(article)
    has_me = bool(MIDDLE_EAST_GEOGRAPHY_PATTERN.search(clean_text))
    has_foreign_actor = bool(FOREIGN_ACTOR_IN_REGIONAL_SECURITY_PATTERN.search(clean_text))

    # Critical Middle East Relevance Rule (1.4):
    # If a foreign actor is discussing or operating in the Middle East, it belongs to Middle East
    if has_foreign_actor and (has_me or bool(BASHUR_PATTERN.search(clean_text))):
        return REGION_MIDDLE_EAST

    if is_bashur_article(article):
        return REGION_KURDISTAN_BASHUR

    if has_me:
        return REGION_MIDDLE_EAST

    country = (getattr(article, "country", "") or "").strip().lower()
    if country in (
        "iraq", "iran", "syria", "turkey", "israel", "palestine",
        "lebanon", "yemen", "jordan", "saudi arabia", "uae",
        "qatar", "kuwait", "bahrain", "oman", "egypt",
    ):
        return REGION_MIDDLE_EAST

    return REGION_GLOBAL


def is_strictly_military_or_political(article: Article) -> tuple[bool, str]:
    """
    Deterministic pre-LLM editorial gate.
    Returns (True, explanation) or (False, rejection_reason).
    Fast, deterministic, testable, runs with 0 Gemini/OpenRouter calls.
    """
    # 1. Existing hard blocked topics (sports, domestic crime/court, celebrity obituaries, etc.)
    blocked = check_blocked_topic(article)
    if blocked:
        return False, blocked

    clean_text = normalize_article_text(article)
    if not clean_text:
        return False, "insufficient_content"

    # 2. Check for substantive military / war / security event
    is_mil, mil_reason = detect_substantive_military_event(clean_text, article)

    # 3. Check for high-level politics / geopolitics event
    is_pol, pol_reason = detect_high_level_political_event(clean_text, article)

    # 4. Check for hard-banned ceremonial, protocol, PR, festival, or delegation fluff
    protocol_fluff = detect_protocol_fluff(clean_text, article)
    if protocol_fluff:
        if is_mil:
            return True, f"substantive_military: {mil_reason} (overrode protocol fluff)"
        if is_pol and _is_state_treaty_or_crisis(clean_text):
            return True, f"high_level_politics: {pol_reason} (state-level treaty/crisis overrode ceremony)"
        return False, f"banned_protocol: {protocol_fluff}"

    # 5. Check for hard-banned domain (economy/finance/crypto/tech/gadgets/routine cyber)
    banned_domain = detect_hard_banned_domain(clean_text, article)
    if banned_domain:
        if is_mil:
            return True, f"substantive_military: {mil_reason} (military conflict involving {banned_domain})"
        return False, f"banned_{banned_domain}"

    # 6. Verify genuine Southern Kurdistan political/military/security event
    art_url = str(getattr(article, "canonical_url", "") or getattr(article, "original_url", "") or "").lower()
    is_short_tg_or_x = any(dom in art_url for dom in ("t.me/", "telegram.me/", "x.com/", "twitter.com/")) and len(clean_text) <= 260

    if is_bashur_article(article):
        if is_mil:
            return True, f"verified_southern_kurdistan_military: {mil_reason}"
        if is_pol or is_politics_or_war(article):
            return True, "verified_southern_kurdistan_politics: KRG/Bashur political development"
        if is_short_tg_or_x:
            return True, "short_telegram_x_bashur_breaking_pass_to_llm"
        return False, "insufficient_political_or_military_importance: local Kurdistan news lacks state-level impact"

    # 7. For non-Bashur stories: MUST be substantive military or high-level politics
    if is_mil:
        return True, f"substantive_military: {mil_reason}"
    if is_pol:
        return True, f"high_level_politics: {pol_reason}"
    if POLITICS_WAR_PATTERN.search(clean_text):
        return True, "politics_or_war: qualified geopolitical event"

    # Allow concise Telegram/X breaking dispatches that cleared all blocked/protocol/banned checks
    # to reach LLM/context enrichment instead of dropping short Arabic/Kurdish/OSINT captions early.
    if is_short_tg_or_x:
        return True, "short_telegram_x_breaking_pass_to_llm"

    return False, "insufficient_political_or_military_importance"


def check_pre_filter_off_topic(article: Article) -> Optional[str]:
    """
    Deterministic pre-Gemini topic filter eliminating obvious off-focus items
    (blocked sports/crime/traffic/ads plus routine non-political market/consumer-tech fluff)
    so Gemini quota is reserved cleanly for Kurdistan, politics, war, and security news.
    """
    is_ok, reason = is_strictly_military_or_political(article)
    if not is_ok:
        return reason
    return None


def _candidate_recency_ts(article: Article) -> float:
    raw_dt = getattr(article, "publication_date", None) or getattr(article, "discovered_at", None) or ""
    if not raw_dt:
        return 0.0
    try:
        from dateutil import parser as dateparser
        dt = dateparser.parse(raw_dt)
        if dt is None:
            return 0.0
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return 0.0


def select_top_candidates_for_gemini(
    articles: list[Article],
    max_candidates: int = MAX_GEMINI_CANDIDATES_PER_CYCLE,
) -> tuple[list[Article], list[tuple[Article, str]]]:
    """
    Filter out locally off-topic articles BEFORE Gemini and cap the remaining pool to the
    `max_candidates` freshest and most relevant candidates (Kurdistan & Politics/War first).
    When the 4:4:2 Batch Pacing Engine is active across multiple source buckets, preserves
    balanced headroom across telegram / x / search buckets before filling remaining capacity.
    Returns `(selected_candidates, rejected_with_reasons)`.
    """
    if not articles:
        return [], []

    kept: list[Article] = []
    rejected: list[tuple[Article, str]] = []

    for art in articles:
        reason = check_pre_filter_off_topic(art)
        if reason:
            rejected.append((art, reason))
        else:
            kept.append(art)

    if max_candidates <= 0:
        for art in kept:
            rejected.append((art, "Pre-filtered: zero Gemini candidate cap"))
        return [], rejected

    def _fresh_relevance_key(art: Article) -> tuple:
        kurd_rank = 0 if is_genuine_kurdistan_event(art) else 1
        pol_rank = 0 if is_politics_or_war(art) else 1
        reg_rank = 0 if is_regional_geopolitics(art) else 1
        cred = -ArticleExtractor.get_source_credibility(art.source_name)
        ts = -_candidate_recency_ts(art)
        return (kurd_rank, pol_rank, reg_rank, cred, ts)

    ordered = sorted(kept, key=_fresh_relevance_key)
    if (
        len(ordered) <= max_candidates
        or not getattr(config, "ENABLE_BATCH_PACING_ENGINE", True)
        or max_candidates < 5
    ):
        selected = ordered[:max_candidates]
        for overflow in ordered[max_candidates:]:
            rejected.append(
                (overflow, f"Deferred: exceeded max {max_candidates} Gemini candidates per cycle")
            )
        return selected, rejected

    # Reserve balanced multi-bucket representation (telegram / x / search) when overflowing
    from core.balancer import get_source_bucket

    by_bucket: dict[str, list[Article]] = {"telegram": [], "x": [], "search": []}
    for art in ordered:
        b = get_source_bucket(art)
        by_bucket.setdefault(b, []).append(art)

    non_empty_buckets = sum(1 for v in by_bucket.values() if v)
    if non_empty_buckets < 2:
        selected = ordered[:max_candidates]
        for overflow in ordered[max_candidates:]:
            rejected.append(
                (overflow, f"Deferred: exceeded max {max_candidates} Gemini candidates per cycle")
            )
        return selected, rejected

    mix = getattr(config, "BATCH_SOURCE_MIX", {"telegram": 2, "x": 2, "search": 1})
    selected: list[Article] = []
    selected_ids: set[int] = set()
    for bucket in ("telegram", "x", "search"):
        reserve = max(int(mix.get(bucket, 0)), 3)
        for art in by_bucket.get(bucket, [])[:reserve]:
            if len(selected) >= max_candidates:
                break
            selected.append(art)
            selected_ids.add(id(art))

    for art in ordered:
        if len(selected) >= max_candidates:
            break
        if id(art) not in selected_ids:
            selected.append(art)
            selected_ids.add(id(art))

    selected.sort(key=_fresh_relevance_key)
    for overflow in ordered:
        if id(overflow) not in selected_ids:
            rejected.append(
                (overflow, f"Deferred: exceeded max {max_candidates} Gemini candidates per cycle")
            )
    return selected, rejected


# Leading or embedded agency dateline pattern (e.g. "QAMISHLI, Syria (North Press) - ", "ERBIL (Reuters) - ")
_EMBEDDED_DATELINE_RE = re.compile(
    r"(?:^|[\s\.\!\?])"
    r"[A-Za-z\u0600-\u06FF\s,،]{1,40}?"
    r"\([A-Za-z\u0600-\u06FF\s]{2,30}\)"
    r"\s*(?:[-–—−]+|:)\s*",
)


def _strip_all_datelines(text: str) -> str:
    """Strip both leading and mid-text agency datelines and publisher signatures."""
    if not text:
        return ""
    # Strip leading datelines
    res = strip_news_dateline(text)
    # Strip embedded datelines
    res = _EMBEDDED_DATELINE_RE.sub(" ", res)
    # Strip outlet brands
    res = _OUTLET_BRAND_STRIP_RE.sub(" ", res)
    return re.sub(r"\s+", " ", res).strip()


def normalize_category(raw_category: Optional[str], text_hint: str = "") -> str:
    """Map any category string strictly onto one of the 7 canonical category keys."""
    cleaned_hint = _strip_all_datelines(text_hint or "")
    def _is_hint_kurdish(hint: str) -> bool:
        if not hint or not KURDISTAN_PATTERN.search(hint):
            return False
        if NON_KURDISTAN_SYRIA_PATTERN.search(hint):
            has_b = bool(BASHUR_PATTERN.search(hint))
            has_r = bool(NON_BASHUR_KURDISTAN_PATTERN.search(hint))
            if not (has_b or has_r):
                return False
        return True

    if raw_category:
        cleaned = raw_category.strip().lower()
        if cleaned in {"kurdistan", "کوردستان", "كردستان"} or "kurd" in cleaned:
            # If hint is provided, keep 'kurdistan' ONLY when actual content verified to involve Kurdistan/Kurdish entities
            if cleaned_hint:
                if _is_hint_kurdish(cleaned_hint):
                    return "kurdistan"
                if _CYBER_TOPIC_RE.search(cleaned_hint):
                    return "cybersecurity"
                if _ECONOMY_TOPIC_RE.search(cleaned_hint):
                    return "economy_markets"
                if _MILITARY_TOPIC_RE.search(cleaned_hint):
                    return "military_conflict"
                return "politics_world"
            return "kurdistan"
        if cleaned in CANONICAL_CATEGORIES:
            return cleaned
        if cleaned in LEGACY_TO_CANONICAL_CATEGORY:
            return LEGACY_TO_CANONICAL_CATEGORY[cleaned]
        # Partial keyword heuristics if model slightly varied format
        if "military" in cleaned or "war" in cleaned or "conflict" in cleaned or "defense" in cleaned:
            return "military_conflict"
        if "cyber" in cleaned or "hack" in cleaned:
            return "cybersecurity"
        if "ai" in cleaned or "tech" in cleaned:
            return "ai_tech"
        if "econom" in cleaned or "market" in cleaned or "business" in cleaned or "oil" in cleaned or "ئابووری" in cleaned:
            return "economy_markets"
        if "science" in cleaned or "space" in cleaned:
            return "science_space"

    if _is_hint_kurdish(cleaned_hint):
        return "kurdistan"
    if cleaned_hint and _ECONOMY_TOPIC_RE.search(cleaned_hint) and not _MILITARY_TOPIC_RE.search(cleaned_hint):
        return "economy_markets"
    return "politics_world"


def is_strict_breaking(article: Article) -> bool:
    """
    Enforce the strict breaking-news gate: return True ONLY when BOTH
    `article.is_breaking` is True AND `article.importance_score >= 9.0`.
    """
    return bool(
        getattr(article, "is_breaking", False)
        and float(getattr(article, "importance_score", 0.0) or 0.0) >= 9.0
    )


def is_bashur_article(article: Article) -> bool:
    """
    Strictly identify Southern Kurdistan (Bashur) events directly involving Erbil,
    Sulaymaniyah, Duhok, Halabja, Kirkuk, KRG, Peshmerga, or Erbil-Baghdad relations.
    Strips leading agency datelines and never relies on publisher name, raw region tag,
    or foreign ISIS mentions.
    """
    if check_blocked_topic(article) is not None:
        return False

    clean_text = clean_article_event_text(article)
    if BASHUR_PATTERN.search(clean_text):
        return True
    return False


def is_politics_or_war(article: Article) -> bool:
    """Return True if an article focuses on politics, governance, diplomacy, or war/security."""
    category = (getattr(article, "category", "") or "").strip().lower()
    if category in {
        "military_conflict",
        "politics_world",
        "wars & armed conflicts",
        "military & defense",
        "world leaders & governments",
        "international politics",
        "diplomacy & international relations",
    }:
        return True
    text = f"{article.original_title or ''} {article.original_summary or ''}"
    return bool(POLITICS_WAR_PATTERN.search(text))


def is_economy_article(article: Article) -> bool:
    """Return True when the article's primary category is Economy/Markets (non-Kurdistan slot)."""
    raw_cat = (getattr(article, "category", "") or "").strip().lower()
    clean_text = clean_article_event_text(article)
    norm_cat = normalize_category(raw_cat, text_hint=clean_text)
    return norm_cat == "economy_markets"


def is_cybersecurity_article(article: Article) -> bool:
    """Return True when the article's primary category is Cybersecurity."""
    raw_cat = (getattr(article, "category", "") or "").strip().lower()
    clean_text = clean_article_event_text(article)
    norm_cat = normalize_category(raw_cat, text_hint=clean_text)
    return norm_cat == "cybersecurity"


def classify_topic_bucket(article: Article) -> str:
    """
    Classify an article into one of the 7+1 editorial topic buckets:
      - 'economy': Global/major Economy & Markets (competes for max 1-of-8 Econ/Cyber slot)
      - 'cybersecurity': Major Cybersecurity / breaches (competes for max 1-of-8 Econ/Cyber slot)
      - 'politics_war': Kurdistan politics/war, Military/Conflict, or Politics/Diplomacy (7-of-8 primary)
      - 'other': General tech/science outside editorial focus
    """
    if is_economy_article(article) and not is_genuine_kurdistan_event(article):
        return "economy"
    if is_cybersecurity_article(article) and not is_genuine_kurdistan_event(article):
        return "cybersecurity"

    raw_cat = (getattr(article, "category", "") or "").strip().lower()
    clean_text = clean_article_event_text(article)
    norm_cat = normalize_category(raw_cat, text_hint=clean_text)
    if norm_cat in {"kurdistan", "politics_world", "military_conflict"} or is_politics_or_war(article):
        return "politics_war"
    if is_economy_article(article):
        # Bashur KRG/Baghdad economic/budget news in kurdistan stays politics_war when recognized above;
        # explicitly categorized economy_markets stays in 'economy' bucket
        return "economy"
    if is_cybersecurity_article(article):
        return "cybersecurity"
    return "other"


def is_regional_geopolitics(article: Article) -> bool:
    """Return True if an article covers Iraq, Iran, Turkey, Syria, US, or Middle East geopolitics."""
    text = f"{article.original_title or ''} {article.original_summary or ''} {article.country or ''}"
    return bool(REGIONAL_GEOPOLITICAL_PATTERN.search(text))


def classify_priority_tier(article: Article) -> str:
    """
    Assign explicit priority tiers for 5-of-8 Bashur selection:
      - '1A': Southern Kurdistan (Bashur) politics, war, security/drone/missile, Peshmerga, Erbil-Baghdad
      - '1B': Other major Southern Kurdistan news (salaries/مووچە, oil/economy, decrees)
      - '2': Global & regional politics/war/conflict (Iraq, Iran, Turkey, Syria, US, Middle East first)
      - '3': Non-political/lifestyle/other global stories
    """
    if is_bashur_article(article):
        if is_politics_or_war(article):
            return PRIORITY_TIER_1A
        return PRIORITY_TIER_1B

    if is_politics_or_war(article):
        return PRIORITY_TIER_2

    return PRIORITY_TIER_3


def article_selection_sort_key(article: Article) -> tuple:
    """
    Sort key ordering candidates within selection pools:
      - Bashur Priority 1A (politics & war) strictly ahead of 1B (local/economic Bashur)
      - Global Priority 2 (politics & war; regional Middle East/Iraq/Iran/Turkey/Syria/US first)
        strictly ahead of Priority 3 (non-political/other global)
      - Higher importance_score wins within the same sub-tier
    """
    tier = classify_priority_tier(article)
    tier_order = {
        PRIORITY_TIER_1A: 0,
        PRIORITY_TIER_1B: 1,
        PRIORITY_TIER_2: 2,
        PRIORITY_TIER_3: 3,
    }[tier]
    breaking_rank = 0 if is_strict_breaking(article) else 1
    regional_rank = 0 if is_regional_geopolitics(article) else 1
    score = -float(getattr(article, "importance_score", 0.0) or 0.0)
    pub = getattr(article, "published_at", "") or getattr(article, "publication_date", "") or ""
    return (tier_order, breaking_rank, regional_rank, score, pub)


class ImportanceRanker:
    """Scores and classifies article importance using Gemini AI + rule-based scoring."""

    def __init__(
        self,
        api_key: Optional[str | list[str]] = None,
        model: Optional[str] = None,
        min_relevance_score: Optional[int] = None,
        session: Optional[object] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
        base_backoff: float = 1.5,
        cooldown_store: Optional[dict[str, float]] = None,
    ) -> None:
        self._min_relevance_score = (
            MIN_GLOBAL_RELEVANCE_SCORE if min_relevance_score is None else min_relevance_score
        )
        self._gemini = GeminiClient(
            api_key=GEMINI_API_KEY if api_key is None else api_key,
            model=GEMINI_MODEL if model is None else model,
            session=session,  # type: ignore[arg-type]
            semaphore=semaphore if semaphore is not None else asyncio.Semaphore(1),
            base_backoff=base_backoff,
            cooldown_store=cooldown_store,
        )

    async def cleanup(self) -> None:
        """Close the underlying Gemini HTTP session."""
        await self._gemini.cleanup()

    def consume_gemini_cycle_stats(self) -> dict[str, int]:
        """Return and reset Gemini HTTP 429 & exhaustion counters from the ranker client."""
        return self._gemini.consume_cycle_stats()

    def _rule_based_fallback(
        self, article: Article, err_reason: str
    ) -> RelevanceEvaluation:
        """Fail-open safely using rule-based importance scoring when Gemini classifier errors out."""
        blocked_reason = check_blocked_topic(article)
        if blocked_reason:
            return RelevanceEvaluation(
                score=2,
                is_breaking=False,
                category="politics_world",
                reason=blocked_reason,
                passed=False,
                used_fallback=True,
            )

        rule_score, _ = self.score_article(article)
        credibility = ArticleExtractor.get_source_credibility(article.source_name)
        passed = self.is_eligible(rule_score, credibility, article=article)
        scaled_score = max(1, min(10, round(rule_score * 10)))
        clean_text = clean_article_event_text(article)
        cat = normalize_category(article.category, text_hint=clean_text)
        if is_genuine_kurdistan_event(article):
            cat = "kurdistan"
            scaled_score = max(8, scaled_score)
            passed = True
        elif cat == "kurdistan":
            cat = normalize_category("kurdistan", text_hint=clean_text or "politics")

        # Never let Kurdistan focus boost or Priority.CRITICAL flip is_breaking to True.
        # Only mark breaking on genuine sudden-emergency keywords without statement/interview markers.
        lower_text = clean_text.lower()
        non_breaking_markers = (
            "interview", "statement", "speech", "remarks", "commentary",
            "analysis", "opinion", "forecast", "says ", "said ", "urges ",
            "stresses ", "welcomes ", "meeting", "talks", "conference",
        )
        emergency_hits = sum(
            1 for kw in (
                "airstrike", "missile strike", "explosion", "assassination",
                "declared war", "ceasefire agreement", "earthquake magnitude",
                "invasion", "coup", "emergency declared",
            )
            if kw in lower_text
        )
        is_breaking = bool(
            emergency_hits >= 1
            and scaled_score >= 9
            and not any(m in lower_text for m in non_breaking_markers)
        )
        reg = classify_geographic_region(article)
        return RelevanceEvaluation(
            score=scaled_score,
            is_breaking=is_breaking,
            category=cat,
            reason=f"Rule-based fallback ({err_reason}; rule_score={rule_score:.2f})",
            passed=passed,
            used_fallback=True,
            region=reg,
        )

    def _build_evaluation_from_payload(
        self,
        data: dict,
        article: Article,
        threshold: int,
    ) -> RelevanceEvaluation:
        """Convert parsed Gemini classifier JSON into a gated RelevanceEvaluation."""
        if "score" not in data:
            raise GeminiAPIError("Missing 'score' field in Gemini classifier response.")

        score = max(1, min(10, int(round(float(data["score"])))))
        raw_is_breaking = bool(data.get("is_breaking", False))
        raw_category = str(data.get("category", "")).strip()
        clean_hint = clean_article_event_text(article)
        category = normalize_category(raw_category, text_hint=clean_hint)

        # Enforce content check for "kurdistan" category: if LLM returned "kurdistan"
        # but content is not genuine Kurdistan (e.g. Tartous Syria, non-Kurdish international story),
        # coerce cleanly to military_conflict or politics_world.
        if category == "kurdistan" and not is_genuine_kurdistan_event(article):
            category = "military_conflict" if is_politics_or_war(article) else "politics_world"

        # Enforce high priority ONLY for genuine Kurdistan events scoring >= threshold
        if category == "kurdistan" and is_genuine_kurdistan_event(article) and threshold <= score < 8:
            score = 8

        # Strict breaking gate: only True if Gemini returned True AND score >= 9
        is_breaking = bool(raw_is_breaking and score >= 9)
        reason = str(data.get("reason", "")).strip() or "Classified by Gemini relevance gatekeeper"
        ev_key = normalize_event_key(data.get("event_key", ""))
        ev_entities = normalize_event_entities(data.get("event_entities"))
        update_type = str(data.get("update_type", "")).strip().lower()

        raw_region = str(data.get("region", "")).strip().lower()
        if raw_region in ("kurdistan_bashur", "middle_east", "global"):
            region = raw_region
        else:
            region = classify_geographic_region(article)

        return RelevanceEvaluation(
            score=score,
            is_breaking=is_breaking,
            category=category,
            reason=reason,
            passed=(score >= threshold),
            used_fallback=False,
            event_key=ev_key,
            event_entities=ev_entities,
            update_type=update_type,
            region=region,
        )

    async def evaluate_global_relevance(
        self,
        article: Article,
        min_score: Optional[int] = None,
    ) -> RelevanceEvaluation:
        """
        Run lightweight Gemini classification check (`temperature=0.0`)
        on article title, source, region, and snippet.
        Returns relevance score (1-10), breaking flag, and one of the 7 categories.
        Fails open to rule-based scoring if GEMINI_API_KEY is missing or API errors out.
        """
        threshold = self._min_relevance_score if min_score is None else min_score

        pre_rejected = check_pre_filter_off_topic(article)
        if pre_rejected:
            reg = classify_geographic_region(article)
            return RelevanceEvaluation(
                score=2,
                is_breaking=False,
                category="politics_world",
                reason=pre_rejected,
                passed=False,
                used_fallback=False,
                region=reg,
            )

        blocked_reason = check_blocked_topic(article)
        if blocked_reason:
            reg = classify_geographic_region(article)
            return RelevanceEvaluation(
                score=2,
                is_breaking=False,
                category="politics_world",
                reason=blocked_reason,
                passed=False,
                used_fallback=False,
                region=reg,
            )

        has_active_engine = (
            self._gemini.has_keys and not self._gemini.all_keys_in_cooldown()
        ) or getattr(self._gemini, "has_fallback", False)

        if not has_active_engine:
            return self._rule_based_fallback(
                article,
                "All Gemini keys in cooldown" if self._gemini.has_keys else "GEMINI_API_KEY not configured",
            )

        clean_title = strip_news_dateline(article.original_title or "")
        snippet = strip_news_dateline(article.original_summary or "").strip()[:500]
        region_line = (
            f"Region Tag: {article.region}\n"
            if article.region and is_genuine_kurdistan_event(article)
            else ""
        )
        user_content = (
            f"Title: {clean_title}\n"
            f"Source: {article.source_name or 'Unknown'}\n"
            f"{region_line}"
            f"Snippet: {snippet}"
        )

        try:
            data = await self._gemini.generate_json(
                system_prompt=GLOBAL_RELEVANCE_SYSTEM_PROMPT,
                user_content=user_content,
                temperature=0.0,
                max_tokens=220,
            )
            return self._build_evaluation_from_payload(data, article, threshold)
        except Exception as e:
            err_msg = str(e)[:150]
            logger.warning(
                "Gemini relevance classifier error (%s); failing open to rule-based score for '%s'.",
                err_msg,
                article.original_title[:60],
            )
            return self._rule_based_fallback(article, err_msg)

    async def evaluate_batch_relevance(
        self,
        articles: list[Article],
        min_score: Optional[int] = None,
        batch_size: int = GEMINI_RANK_BATCH_SIZE,
    ) -> list[RelevanceEvaluation]:
        """
        Batch-evaluate candidate articles (10-15 per single Gemini JSON request)
        after applying deterministic blocked/off-topic pre-filters locally.
        Preserves single-article `evaluate_global_relevance` if monkeypatched in tests.
        """
        if not articles:
            return []

        threshold = self._min_relevance_score if min_score is None else min_score
        results: list[Optional[RelevanceEvaluation]] = [None] * len(articles)
        to_classify: list[tuple[int, Article]] = []

        # Pass 1: zero-API deterministic topic filter
        for idx, art in enumerate(articles):
            pre_rejected = check_pre_filter_off_topic(art)
            if pre_rejected:
                reg = classify_geographic_region(art)
                results[idx] = RelevanceEvaluation(
                    score=2,
                    is_breaking=False,
                    category="politics_world",
                    reason=pre_rejected,
                    passed=False,
                    used_fallback=False,
                    region=reg,
                )
            else:
                to_classify.append((idx, art))

        if not to_classify:
            return [r for r in results if r is not None]

        # Compatibility hook: if a test monkeypatched `evaluate_global_relevance` directly
        is_single_overridden = (
            "evaluate_global_relevance" in self.__dict__
            or type(self).evaluate_global_relevance is not ImportanceRanker.evaluate_global_relevance
        )
        if is_single_overridden:
            for idx, art in to_classify:
                results[idx] = await self.evaluate_global_relevance(art)
            return [r for r in results if r is not None]

        has_active_engine = (
            self._gemini.has_keys and not self._gemini.all_keys_in_cooldown()
        ) or getattr(self._gemini, "has_fallback", False)

        if not has_active_engine:
            reason = (
                "All Gemini keys in cooldown" if self._gemini.has_keys else "GEMINI_API_KEY not configured"
            )
            for idx, art in to_classify:
                results[idx] = self._rule_based_fallback(art, reason)
            return [r for r in results if r is not None]

        effective_batch = max(1, min(15, int(batch_size or GEMINI_RANK_BATCH_SIZE)))
        quota_exhausted = False

        for start in range(0, len(to_classify), effective_batch):
            chunk = to_classify[start: start + effective_batch]
            chunk_engine_ready = (
                not self._gemini.all_keys_in_cooldown()
                or getattr(self._gemini, "has_fallback", False)
            )
            if quota_exhausted or not chunk_engine_ready:
                for idx, art in chunk:
                    results[idx] = self._rule_based_fallback(art, "All Gemini keys exhausted (cooldown)")
                continue

            payload_items = []
            for idx, art in chunk:
                clean_title = strip_news_dateline(art.original_title or "")
                snippet = strip_news_dateline(art.original_summary or "").strip()[:280]
                item_dict: dict[str, object] = {
                    "idx": idx,
                    "title": clean_title,
                    "source": art.source_name or "Unknown",
                    "snippet": snippet,
                }
                if art.region and is_genuine_kurdistan_event(art):
                    item_dict["region"] = art.region
                payload_items.append(item_dict)

            user_content = (
                f"Evaluate these {len(payload_items)} candidate news articles:\n"
                f"{json.dumps(payload_items, ensure_ascii=False)}"
            )

            try:
                resp_data = await self._gemini.generate_json(
                    system_prompt=BATCH_RELEVANCE_SYSTEM_PROMPT,
                    user_content=user_content,
                    temperature=0.0,
                    max_tokens=min(2400, 180 * len(chunk) + 200),
                )
                raw_items = resp_data.get("items") or resp_data.get("results")
                if not isinstance(raw_items, list):
                    raise GeminiAPIError("Batch Gemini response missing 'items' list.")

                parsed_by_idx: dict[int, dict] = {}
                for entry in raw_items:
                    if isinstance(entry, dict) and "idx" in entry:
                        try:
                            parsed_by_idx[int(entry["idx"])] = entry
                        except (TypeError, ValueError):
                            continue

                for idx, art in chunk:
                    entry = parsed_by_idx.get(idx)
                    if entry is None:
                        results[idx] = self._rule_based_fallback(art, "Missing item in Gemini batch output")
                        continue
                    try:
                        results[idx] = self._build_evaluation_from_payload(entry, art, threshold)
                    except Exception as item_exc:
                        results[idx] = self._rule_based_fallback(art, str(item_exc)[:120])

            except GeminiQuotaExhaustedError as q_exc:
                quota_exhausted = True
                logger.warning(
                    "Gemini batch classifier hit 429 quota exhaustion (%s); "
                    "aborting remaining batch LLM calls and failing open cleanly.",
                    str(q_exc)[:140],
                )
                for idx, art in chunk:
                    results[idx] = self._rule_based_fallback(art, "Gemini quota exhausted")

            except Exception as exc:
                err_msg = str(exc)[:150]
                logger.warning(
                    "Gemini batch relevance classifier error (%s); failing open for %d article(s).",
                    err_msg,
                    len(chunk),
                )
                for idx, art in chunk:
                    results[idx] = self._rule_based_fallback(art, err_msg)

        return [
            r if r is not None else self._rule_based_fallback(articles[i], "Unclassified candidate")
            for i, r in enumerate(results)
        ]

    def score_article(self, article: Article) -> tuple[float, str]:
        """
        Calculate importance score (0.0 to 1.0) and priority level.
        Returns (score, priority).
        """
        scores: list[tuple[float, float]] = []

        credibility = ArticleExtractor.get_source_credibility(article.source_name)
        scores.append((credibility, 0.20))

        keyword_score = self._score_keywords(article)
        scores.append((keyword_score, 0.30))

        completeness = self._score_completeness(article)
        scores.append((completeness, 0.15))

        recency = self._score_recency(article)
        scores.append((recency, 0.15))

        title_quality = self._score_title_quality(article)
        scores.append((title_quality, 0.10))

        category_score = self._score_category(article)
        scores.append((category_score, 0.10))

        total_score = sum(s * w for s, w in scores)
        total_weight = sum(w for _, w in scores)
        base_score = total_score / total_weight if total_weight > 0 else 0.0

        focus_boost = self._calculate_focus_boost(article)
        final_score = max(0.0, min(1.0, base_score + focus_boost))
        priority = self._determine_priority(final_score, article)
        return final_score, priority

    def _calculate_focus_boost(self, article: Article) -> float:
        """
        Calculate priority score boost:
        - +0.20 for Primary focus (Kurdistan 4 parts, war, military strikes, conflicts, world leaders, geopolitics).
        - +0.10 for key Secondary focus (AI breakthroughs, major cyberattacks, major economic shifts).
        """
        text = clean_article_event_text(article).lower()

        if KURDISTAN_PATTERN.search(text) or is_primary_category(article.category):
            return 0.20

        primary_patterns = [
            r"\bwar\b", r"\bwars\b", r"\bmilitary\b", r"\bstrike\b", r"\bstrikes\b",
            r"\bairstrike\b", r"\bairstrikes\b", r"\bmissile\b", r"\bmissiles\b",
            r"\bconflict\b", r"\bconflicts\b", r"\btroops\b", r"\bworld leaders?\b",
            r"\bpresidents?\b", r"\bprime ministers?\b", r"\bnato\b",
            r"\bun security council\b", r"\bgeopolitics?\b", r"\bgeopolitical\b",
            r"\bescalat\w*", r"\binvasion\b", r"\bceasefire\b", r"\bfrontline\b",
            r"\bcombat\b", r"\bpentagon\b", r"\bkremlin\b",
        ]
        if any(re.search(pattern, text) for pattern in primary_patterns):
            return 0.20

        secondary_categories = {
            "ai_tech", "cybersecurity", "economy_markets",
            "AI & Technology", "Cybersecurity", "Global Economy & Financial Markets",
        }
        if article.category in secondary_categories:
            return 0.10

        secondary_patterns = [
            r"\bartificial intelligence\b", r"\bai model\b", r"\bai breakthrough\b",
            r"\bllm\b", r"\bcyberattacks?\b", r"\bransomware\b", r"\bzero-day\b",
            r"\bhackers?\b", r"\bhacking\b", r"\bcyber breach\b", r"\bdata breach\b",
            r"\bstate-sponsored\b", r"\beconomic crisis\b", r"\brecession\b",
            r"\binterest rates?\b", r"\binflation\b", r"\bfinancial collapse\b",
            r"\bmarket crash\b",
        ]
        if any(re.search(pattern, text) for pattern in secondary_patterns):
            return 0.10

        return 0.0

    def is_eligible(
        self,
        score: float,
        credibility_score: float | None = None,
        article: Optional[Article] = None,
    ) -> bool:
        """Check if an article meets minimum thresholds and passes blocked-topic guards."""
        if article is not None and check_blocked_topic(article) is not None:
            return False
        if score < MIN_IMPORTANCE_SCORE:
            return False
        if credibility_score is not None and credibility_score < MIN_SOURCE_CREDIBILITY:
            return False
        return True

    def _score_keywords(self, article: Article) -> float:
        text = clean_article_event_text(article).lower()
        critical_hits = sum(1 for kw in CRITICAL_KEYWORDS if kw in text)
        if critical_hits >= 2:
            return 1.0
        if critical_hits == 1:
            return 0.85

        high_hits = sum(1 for kw in HIGH_KEYWORDS if kw in text)
        if high_hits >= 2:
            return 0.75
        if high_hits == 1:
            return 0.60
        return 0.40

    def _score_completeness(self, article: Article) -> float:
        score = 0.0
        if article.original_title:
            score += 0.3
        if article.original_summary and len(article.original_summary) > 50:
            score += 0.3
        if article.source_name:
            score += 0.15
        if article.publication_date:
            score += 0.10
        if article.category:
            score += 0.10
        if article.image_url:
            score += 0.05
        return min(score, 1.0)

    def _score_recency(self, article: Article) -> float:
        if not article.publication_date:
            return 0.5
        try:
            from dateutil import parser as dateparser
            pub_dt = dateparser.parse(article.publication_date)
            if not pub_dt:
                return 0.5
            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=timezone.utc)
            hours = (datetime.now(timezone.utc) - pub_dt).total_seconds() / 3600
            if hours < 1:
                return 1.0
            if hours < 3:
                return 0.9
            if hours < 6:
                return 0.75
            if hours < 12:
                return 0.6
            if hours < 24:
                return 0.4
            if hours < 48:
                return 0.25
            return 0.1
        except Exception:
            return 0.5

    def _score_title_quality(self, article: Article) -> float:
        title = article.original_title
        if not title:
            return 0.0
        score = 0.5
        if 20 <= len(title) <= 150:
            score += 0.2
        if re.search(r"\d", title):
            score += 0.1
        if re.search(r"[A-Z][a-z]+", title):
            score += 0.1
        if title.isupper():
            score -= 0.3
        return max(0.0, min(1.0, score))

    def _score_category(self, article: Article) -> float:
        high_priority_categories = {
            "kurdistan", "military_conflict", "politics_world",
            "Wars & Armed Conflicts", "International Politics",
            "Natural Disasters & Emergencies", "World Leaders & Governments",
            "Military & Defense", "Diplomacy & International Relations",
        }
        medium_priority_categories = {
            "economy_markets", "ai_tech", "cybersecurity", "science_space",
            "Global Economy & Financial Markets", "AI & Technology",
            "Health & Medical", "Cybersecurity",
            "Energy, Oil & Gas", "International Organizations",
        }
        if article.category in high_priority_categories:
            return 0.9
        if article.category in medium_priority_categories:
            return 0.7
        return 0.5

    def _determine_priority(self, score: float, article: Article) -> str:
        text = clean_article_event_text(article).lower()
        critical_hits = sum(1 for kw in CRITICAL_KEYWORDS if kw in text)
        if critical_hits >= 2 or score >= 0.85:
            return Priority.CRITICAL
        if critical_hits == 1 or score >= 0.70:
            return Priority.HIGH
        if score >= MIN_IMPORTANCE_SCORE:
            return Priority.NORMAL
        return Priority.LOW
