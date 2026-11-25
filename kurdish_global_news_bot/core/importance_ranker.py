"""
Kurdish Global News Bot — Importance Ranker & Gemini Global Relevance Classifier

Evaluates global relevance, assigns one of the 7 strict Kurdish categories,
detects breaking developments, prioritizes all four parts of Kurdistan,
and provides rule-based fallback if the Gemini API is unavailable.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import re
from datetime import datetime, timezone
from typing import Optional

from storage.models import Article
from processing.extractor import ArticleExtractor
from processing.gemini_client import GeminiClient, GeminiAPIError
from processing.translator import strip_news_dateline
from config import (
    CANONICAL_CATEGORIES,
    GEMINI_API_KEY,
    GEMINI_MODEL,
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
    r"کوردستان|کورد|هەولێر|سلێمانی|دهۆک|کەرکوک|هەڵەبجە|پێشمەرگە|ڕۆژئاوا|قامیشلۆ|کۆبانێ|حەسەکە|ئامەد|سنە|مەهاباد|ورمێ|کرماشان)",
    re.IGNORECASE,
)

# Southern Kurdistan (Bashur) geographic, governmental, and security identifiers
BASHUR_PATTERN = re.compile(
    r"(erbil|hewler|sulaymaniyah|slemani|\bsuli\b|duhok|halabja|kirkuk|peshmerga|\bkrg\b|"
    r"kurdistan region|iraqi kurdistan|south kurdistan|bashur|erbil[\s\-–]+baghdad|"
    r"baghdad[\s\-–]+erbil|nechirvan|masrour|barzani|talabani|disputed areas?|article 140|"
    r"هەولێر|سلێمانی|دهۆک|هەڵەبجە|کەرکوک|پێشمەرگە|هەرێمی کوردستان|حکومەتی هەرێم|"
    r"کوردستانی باشوور|باشوور|بارزانی|تاڵەبانی|هەولێر و بەغدا|مووچە)",
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
    r"\bgoalkeeper\b|\bstriker\b|\bmidfielder\b|\bpenalty\s+shootout\b|\bhat-trick\b|\bred\s+card\b|"
    r"\b(?:football|soccer|basketball|tennis|cricket)\s+(?:match|player|star|club|team|coach|tournament)\b|"
    r"\bnational\s+(?:football|soccer|basketball)?\s*team\b|"
    r"\b(?:semifinal|quarterfinal|cup\s+final|sports?\s+tournament)\b|"
    r"\b(?:win|wins|won|beat|beats|defeat|defeats|scores?|scored)\b.{0,40}\b(?:match|cup|goal|coach|fc|stadium|morocco|senegal)\b|"
    r"تۆپی\s+پێ|جامی\s+نەتەوەکانی\s+ئەفریقا|مۆندیال|ئۆڵۆمپیاد|خولی\s+پاڵەوانەکان|یاریگا|گۆڵکار|وەرزش)",
    re.IGNORECASE,
)

CULTURE_LIFESTYLE_PATTERN = re.compile(
    r"(\bcelebrity\s+gossip\b|\bred\s+carpet\b|\bbox\s+office\b|\boscar\s+nominees?\b|"
    r"\bgrammy\s+awards?\b|\bfashion\s+week\b|\breality\s+tv\b|\breality\s+show\b|"
    r"\bmovie\s+review\b|\balbum\s+review\b|\bculinary\s+festival\b|\bhoroscope\b|\bzodiac\s+sign\b)",
    re.IGNORECASE,
)

TRAFFIC_MUNICIPAL_AD_PATTERN = re.compile(
    r"(تنويه\s+مروري|مديرية\s+المرور|شرطة\s+المرور|مفارز\s+المرور|غرامة\s+مرورية|"
    r"رادار(?:ات)?\s+(?:المرور|السرعة)|كاميرات\s+(?:المراقبة\s+المرورية|السرعة|المرور)|"
    r"ازدحام\s+مروري|[کك]امێرای\s+(?:خێرایی|هاتووچۆ)|بەڕێوەبەرایەتی\s+هاتووچۆ|"
    r"\bspeed\s+(?:radar\s+)?cameras?\b|\btraffic\s+(?:radar|police)\b|"
    r"جدول\s+(?:قطع\s+)?الكهرباء|انقطاع\s+التيار\s+الكهربائي|صيانة\s+شبكة\s+المياه|"
    r"أمانة\s+بغداد.{0,40}صيانة|ساعات\s+التشغيل\s+للمولدات|"
    r"\bwater\s+and\s+electricity\s+outage\b|\boutage\s+maintenance\s+schedule\b|"
    r"للاعلان(?:ات)?|للتواصل\s+والإعلان|عرض\s+خاص|تخفيضات|خصم\s+\d+%|"
    r"اشترك(?:وا)?\s+(?:في|عبر)\s+قناتنا|انضموا\s+إلى\s+قناتنا|ڕیکلام|"
    r"\bsponsored\s+discount\b|\bfollow\s+us\s+on\s+telegram\b)",
    re.IGNORECASE,
)

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
    "- NEVER classify an article as \"kurdistan\" merely because it was published by a Kurdish/regional outlet (e.g., North Press Agency, Rudaw, Kurdistan24), "
    "because of an agency dateline (e.g., 'Qamishli, Syria (North Press) –', 'Erbil -'), or because it mentions ISIS/داعش in a foreign/non-Kurdistan context.\n"
    "- Classify strictly by actual subject, NOT the publisher name (e.g., Nvidia stock news from Al Jazeera is \"economy_markets\" or \"ai_tech\"; US court/FBI news published by North Press is NOT \"kurdistan\").\n\n"
    "STRICT Breaking News Rule (`is_breaking`):\n"
    "- Set \"is_breaking\": true ONLY for sudden, unfolding, emergency, high-impact events happening right now (and ONLY when score >= 9): "
    "active military strikes, explosions, missile/drone attacks, assassinations, sudden declarations of war or ceasefires, major earthquakes/natural disasters, "
    "emergency government or court decrees taking immediate effect, or major critical-infrastructure cyberattacks.\n"
    "- Explicitly set \"is_breaking\": false for ALL routine political statements, interviews, speeches, party-leader commentary, diplomatic meetings, "
    "opinions, analyses, forecasts, routine economic updates, or reactions — EVEN IF the article is high-priority Kurdistan news with score 8 or 9.\n"
    "- High importance alone does NOT make an article breaking news.\n\n"
    "Scoring Criteria (1 to 10):\n"
    "- 9 to 10 (HIGHEST PRIORITY): Major Southern Kurdistan (Bashur — Erbil, Sulaymaniyah, Duhok, Halabja, Kirkuk, KRG, Peshmerga, Erbil-Baghdad) politics, "
    "war, security/drone/missile incidents; or major wars, military escalations, world leader summits, and severe crises.\n"
    "- 8 (HIGH PASS): Genuine political, security, economic (salaries/مووچە, oil/energy, official decrees), or human-rights development taking place in ANY of the 4 parts of Kurdistan; "
    "regional geopolitics (Iraq, Iran, Turkey, Syria, US, Middle East); major global economy/sanctions, AI breakthroughs, and critical cyberattacks.\n"
    "- 7 (PASS): Notable international diplomacy, cross-border crises, ICJ/ICC international court rulings, or significant space/science milestones.\n"
    "- 1 to 6 (REJECT): Domestic crime, FBI arrests, individual extremism/propaganda trials, local police reports, personal prison sentences (EVEN if ISIS/terrorism is mentioned in a foreign domestic court); "
    "ALL sports/tournaments (football/soccer, AFCON, World Cup, Olympics, matches, players, teams — even from general feeds like NPR); "
    "culture, lifestyle, celebrity gossip, entertainment, routine business licensing disputes, or minor fluff.\n\n"
    "Respond ONLY with valid JSON (no markdown fences, no commentary):\n"
    '{"score": <integer 1-10>, "is_breaking": <true or false>, "category": "<one of the 7 keys>", "reason": "<short one-sentence justification>"}'
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


def clean_article_event_text(article: Article) -> str:
    """
    Extract normalized event text from an article after stripping leading agency
    datelines (e.g., 'Qamishli, Syria (North Press) –') so datelines and publisher
    headers cannot spoof geographic classification.
    """
    clean_title = strip_news_dateline(article.original_title or "")
    clean_summary = strip_news_dateline(article.original_summary or "")
    return f"{clean_title} {clean_summary}".strip()


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
    sports/tournaments, and culture/lifestyle fluff before classification or publishing.
    Returns rejection reason string if blocked, else None.
    """
    cat = (getattr(article, "category", "") or "").strip().lower()
    if cat in BLOCKED_CATEGORY_NAMES:
        return f"Blocked category: {article.category}"

    clean_text = clean_article_event_text(article)
    if not clean_text:
        return None

    if SPORTS_PATTERN.search(clean_text):
        return "Blocked topic: sports/tournament news excluded from channel"

    if DOMESTIC_CRIME_COURT_PATTERN.search(clean_text) and not INTERNATIONAL_COURT_OR_STATE_CRISIS_PATTERN.search(clean_text):
        return "Blocked topic: foreign/domestic crime or individual court sentencing"

    if CULTURE_LIFESTYLE_PATTERN.search(clean_text):
        return "Blocked topic: culture/lifestyle/entertainment fluff"

    if TRAFFIC_MUNICIPAL_AD_PATTERN.search(clean_text):
        return "Blocked topic: traffic/municipal-service notice or advertisement"

    return None


def normalize_category(raw_category: Optional[str], text_hint: str = "") -> str:
    """Map any category string strictly onto one of the 7 canonical category keys."""
    cleaned_hint = strip_news_dateline(text_hint or "")
    if raw_category:
        cleaned = raw_category.strip().lower()
        if cleaned == "kurdistan" or "kurd" in cleaned:
            # If event text is supplied, keep 'kurdistan' ONLY when actual content involves Kurdistan
            if not cleaned_hint or KURDISTAN_PATTERN.search(cleaned_hint):
                return "kurdistan"
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
        if "econom" in cleaned or "market" in cleaned or "business" in cleaned or "oil" in cleaned:
            return "economy_markets"
        if "science" in cleaned or "space" in cleaned:
            return "science_space"

    if cleaned_hint and KURDISTAN_PATTERN.search(cleaned_hint):
        return "kurdistan"
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
    ) -> None:
        self._min_relevance_score = (
            MIN_GLOBAL_RELEVANCE_SCORE if min_relevance_score is None else min_relevance_score
        )
        self._gemini = GeminiClient(
            api_key=GEMINI_API_KEY if api_key is None else api_key,
            model=GEMINI_MODEL if model is None else model,
            session=session,  # type: ignore[arg-type]
            semaphore=semaphore if semaphore is not None else asyncio.Semaphore(2),
            base_backoff=base_backoff,
        )

    async def cleanup(self) -> None:
        """Close the underlying Gemini HTTP session."""
        await self._gemini.cleanup()

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
            cat = "politics_world"

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

        if not self._gemini.has_keys:
            return self._rule_based_fallback(article, "GEMINI_API_KEY not configured")

        clean_title = strip_news_dateline(article.original_title or "")
        snippet = strip_news_dateline(article.original_summary or "").strip()[:500]
        # Only forward Region Tag if stripped event content genuinely pertains to Kurdistan
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
            if "score" not in data:
                raise GeminiAPIError("Missing 'score' field in Gemini classifier response.")

            score = max(1, min(10, int(round(float(data["score"])))))
            raw_is_breaking = bool(data.get("is_breaking", False))
            raw_category = str(data.get("category", "")).strip()
            clean_hint = f"{clean_title} {snippet}".strip()
            category = normalize_category(raw_category, text_hint=clean_hint)

            # Enforce high priority ONLY for genuine Kurdistan events scoring >= threshold
            if category == "kurdistan" and is_genuine_kurdistan_event(article) and score >= threshold and score < 8:
                score = 8

            # Strict breaking gate: only True if Gemini returned True AND score >= 9
            is_breaking = bool(raw_is_breaking and score >= 9)

            reason = str(data.get("reason", "")).strip() or "Classified by Gemini relevance gatekeeper"
            return RelevanceEvaluation(
                score=score,
                is_breaking=is_breaking,
                category=category,
                reason=reason,
                passed=(score >= threshold),
                used_fallback=False,
            )
        except Exception as e:
            err_msg = str(e)[:150]
            logger.warning(
                "Gemini relevance classifier error (%s); failing open to rule-based score for '%s'.",
                err_msg,
                article.original_title[:60],
            )
            return self._rule_based_fallback(article, err_msg)

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
