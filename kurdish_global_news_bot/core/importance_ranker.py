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
    CANONICAL_CATEGORIES,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_RANK_BATCH_SIZE,
    MAX_GEMINI_CANDIDATES_PER_CYCLE,
    MIN_GLOBAL_RELEVANCE_SCORE,
    MIN_IMPORTANCE_SCORE,
    MIN_SOURCE_CREDIBILITY,
    Priority,
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
    "You are the chief international news editor for a Central Kurdish (Sorani) global breaking news channel.\n"
    "Evaluate the global importance, relevance, breaking urgency, and exact category of the given news article.\n\n"
    "Allowed values for \"category\" (MUST be strictly one of these 7 keys):\n"
    "- \"kurdistan\": Significant news from ANY of the four parts of Kurdistan — South (Erbil, Sulaymaniyah, Duhok, Kirkuk, KRG), "
    "West (Rojava, North & East Syria, Qamishli, Kobani, Hasakah, SDF), North (Diyarbakir/Amed, Van, Kurdish issues in Turkey), "
    "or East (Rojhelat, Sanandaj/Sine, Mahabad, Urmia, Kermanshah, Kurdish human rights/developments in Iran).\n"
    "- \"military_conflict\": Active armed conflicts, airstrikes, wars, missile/drone attacks, defense treaties, or military operations.\n"
    "- \"politics_world\": Geopolitics, world leaders, elections, UN/NATO diplomacy, government policy, treaties, royal/state affairs, major crises.\n"
    "- \"economy_markets\": Stock markets, Nvidia/company earnings, oil/energy, central banks, trade tariffs, sanctions, inflation.\n"
    "- \"ai_tech\": Artificial intelligence models, semiconductors, major tech releases, consumer/enterprise technology breakthroughs.\n"
    "- \"cybersecurity\": Cyberattacks, ransomware, state-sponsored hacking, major data breaches, zero-day vulnerabilities.\n"
    "- \"science_space\": Space missions, astronomy, NASA/SpaceX, scientific breakthroughs, major medical discoveries.\n\n"
    "IMPORTANT Geographic & Categorization Accuracy:\n"
    "- Determine geography strictly from WHERE the actual event took place and whose government, territory, or forces are directly involved.\n"
    "- NEVER classify an article as \"kurdistan\" merely because it was published by a Kurdish/regional outlet (e.g., North Press Agency, Rudaw, Kurdistan24, NRT), "
    "because of an agency dateline (e.g., 'Qamishli, Syria (North Press) –', 'Erbil -'), or because it mentions ISIS/داعش in a foreign/non-Kurdistan context. "
    "If a Kurdish outlet reports international news (e.g., China-US trade, Russia-Ukraine, Gaza, Europe), classify it strictly by its true topic (\"economy_markets\", \"politics_world\", \"military_conflict\", or \"cybersecurity\") and NEVER \"kurdistan\".\n"
    "- Classify strictly by actual subject, NOT the publisher name (e.g., Nvidia stock news from Al Jazeera is \"economy_markets\" or \"ai_tech\"; US court/FBI news published by North Press is NOT \"kurdistan\").\n\n"
    "STRICT Breaking News Rule (`is_breaking`):\n"
    "- Set \"is_breaking\": true ONLY for sudden, unfolding, emergency, high-impact events happening right now (and ONLY when score >= 9): "
    "active military strikes, explosions, missile/drone attacks, assassinations, sudden declarations of war or ceasefires, major earthquakes/natural disasters, "
    "emergency government or court decrees taking immediate effect, or major critical-infrastructure cyberattacks.\n"
    "- Explicitly set \"is_breaking\": false for ALL routine political statements, interviews, speeches, party-leader commentary, diplomatic meetings, "
    "opinions, analyses, forecasts, routine economic updates, or reactions — EVEN IF the article is high-priority Kurdistan news with score 8 or 9.\n"
    "- High importance alone does NOT make an article breaking news.\n\n"
    "Scoring Criteria (1 to 10):\n"
    "- Editorial Focus (7/8 Politics & War + Max 1/8 Economy OR Cybersecurity): This channel prioritizes Politics and War/Military news first. Major Economy or Cybersecurity stories compete for at most 1 slot per 8 posts.\n"
    "- 9 to 10 (HIGHEST PRIORITY): Major Southern Kurdistan (Bashur — Erbil, Sulaymaniyah, Duhok, Halabja, Kirkuk, KRG, Peshmerga, Erbil-Baghdad) politics, "
    "war, security/drone/missile incidents; or major wars, military escalations, world leader summits, and severe crises.\n"
    "- 8 (HIGH PASS): Genuine political, security, economic (salaries/مووچە, oil/energy, official decrees), or human-rights development taking place in ANY of the 4 parts of Kurdistan; "
    "regional geopolitics (Iraq, Iran, Turkey, Syria, US, Middle East); major macroeconomic crises/sanctions/central banks or critical state-level cyberattacks.\n"
    "- 7 (PASS): Notable international diplomacy, cross-border crises, or ICJ/ICC international court rulings.\n"
    "- 1 to 6 (STRICTLY REJECT): Domestic crime, FBI arrests, single-person immigration/deportation court cases with no state-level geopolitical impact, individual extremism/propaganda trials, local police reports, personal prison sentences (EVEN if ISIS/terrorism is mentioned in a foreign domestic court); "
    "ALL sports/tournaments/athletes; "
    "ALL celebrity/actor/actress/Hollywood/cinema/music/singer/athlete obituaries or entertainment news (whether English, Arabic, or Kurdish); "
    "ALL multi-story news roundups or podcast/newsletter briefings (e.g., Up First, Morning Briefing, composite headlines joining unrelated stories); "
    "travel/tourism fluff, weather trivia/routine forecasts, consumer phones/gadgets/apps, routine human-interest fluff, traffic/municipal outage notices, and advertisements.\n\n"
    "Cross-Language Event Identification & Live Threading:\n"
    "- Include concise standardized ENGLISH `event_key` (lowercase snake_case identifying the SPECIFIC incident, e.g. \"erbil_airport_drone_attack\", \"baghdad_krg_salary_agreement\", \"trump_iran_war_funding\" — never just a country/city), `event_entities` (3 to 5 lowercase English entity/location/topic tokens), and optional `update_type` (`\"new_story\"`, `\"casualties_update\"`, `\"official_response\"`, `\"retaliation\"`, `\"ceasefire\"`, `\"investigation_result\"`, or `\"repeat_coverage\"`).\n\n"
    "Respond ONLY with valid JSON (no markdown fences, no commentary):\n"
    '{"score": <integer 1-10>, "is_breaking": <true or false>, "category": "<one of the 7 keys>", '
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
    + '"category": "<one of the 7 keys>", "event_key": "<english_snake_case_event_id>", '
    + '"event_entities": ["entity1", "entity2", "topic"], "update_type": "<optional_update_type>", '
    + '"reason": "<short one-sentence justification>"}]}'
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
    """
    clean_text = clean_article_event_text(article)
    if not clean_text:
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


def check_pre_filter_off_topic(article: Article) -> Optional[str]:
    """
    Deterministic pre-Gemini topic filter eliminating obvious off-focus items
    (blocked sports/crime/traffic/ads plus routine non-political market/consumer-tech fluff)
    so Gemini quota is reserved cleanly for Kurdistan, politics, war, and security news.
    """
    blocked = check_blocked_topic(article)
    if blocked:
        return blocked

    clean_text = clean_article_event_text(article)
    if not clean_text:
        return None

    # Never pre-reject genuine Kurdistan news, politics/war, or geopolitical/security stories
    if is_genuine_kurdistan_event(article) or is_politics_or_war(article):
        return None
    if _GEOPOLITICAL_SAFEGUARD_PATTERN.search(clean_text):
        return None

    if _OFFTOPIC_ECONOMY_PATTERN.search(clean_text):
        return "Pre-filtered topic: routine market/earnings noise without geopolitical focus"
    if _OFFTOPIC_TECH_SCIENCE_PATTERN.search(clean_text):
        return "Pre-filtered topic: consumer tech/gadget/space fluff without security focus"

    raw_cat = (getattr(article, "category", "") or "").strip().lower()
    if raw_cat in _OFFTOPIC_FEED_CATEGORIES:
        return f"Pre-filtered topic: secondary feed '{article.category}' outside politics/war focus"

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


def normalize_category(raw_category: Optional[str], text_hint: str = "") -> str:
    """Map any category string strictly onto one of the 7 canonical category keys."""
    cleaned_hint = _OUTLET_BRAND_STRIP_RE.sub(" ", strip_news_dateline(text_hint or "")).strip()
    if raw_category:
        cleaned = raw_category.strip().lower()
        if cleaned in {"kurdistan", "کوردستان", "كردستان"} or "kurd" in cleaned:
            # Keep 'kurdistan' ONLY when actual content involves Kurdistan/Kurdish entities
            if not cleaned_hint or KURDISTAN_PATTERN.search(cleaned_hint):
                return "kurdistan"
            if _CYBER_TOPIC_RE.search(cleaned_hint):
                return "cybersecurity"
            if _ECONOMY_TOPIC_RE.search(cleaned_hint):
                return "economy_markets"
            if _MILITARY_TOPIC_RE.search(cleaned_hint):
                return "military_conflict"
            return "politics_world"
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

    if cleaned_hint and KURDISTAN_PATTERN.search(cleaned_hint):
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
        return RelevanceEvaluation(
            score=scaled_score,
            is_breaking=is_breaking,
            category=cat,
            reason=f"Rule-based fallback ({err_reason}; rule_score={rule_score:.2f})",
            passed=passed,
            used_fallback=True,
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

        # Enforce high priority ONLY for genuine Kurdistan events scoring >= threshold
        if category == "kurdistan" and is_genuine_kurdistan_event(article) and threshold <= score < 8:
            score = 8

        # Strict breaking gate: only True if Gemini returned True AND score >= 9
        is_breaking = bool(raw_is_breaking and score >= 9)
        reason = str(data.get("reason", "")).strip() or "Classified by Gemini relevance gatekeeper"
        ev_key = normalize_event_key(data.get("event_key", ""))
        ev_entities = normalize_event_entities(data.get("event_entities"))
        update_type = str(data.get("update_type", "")).strip().lower()

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

        blocked_reason = check_blocked_topic(article)
        if blocked_reason:
            return RelevanceEvaluation(
                score=2,
                is_breaking=False,
                category="politics_world",
                reason=blocked_reason,
                passed=False,
                used_fallback=False,
            )

        if not self._gemini.has_keys or self._gemini.all_keys_in_cooldown():
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
                results[idx] = RelevanceEvaluation(
                    score=2,
                    is_breaking=False,
                    category="politics_world",
                    reason=pre_rejected,
                    passed=False,
                    used_fallback=False,
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

        if not self._gemini.has_keys or self._gemini.all_keys_in_cooldown():
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
            if quota_exhausted or self._gemini.all_keys_in_cooldown():
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
