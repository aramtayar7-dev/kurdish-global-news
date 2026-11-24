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
    "IMPORTANT Categorization Accuracy:\n"
    "- Classify strictly by the article's actual subject, NOT the publisher name (e.g., Nvidia stock news from Al Jazeera is \"economy_markets\" or \"ai_tech\", NOT \"military_conflict\"; UK royal news is \"politics_world\", NOT \"military_conflict\").\n\n"
    "Scoring Criteria (1 to 10):\n"
    "- 8 to 10 (HIGH PASS): Any significant political, security, economic, or human-rights development in ANY of the 4 parts of Kurdistan "
    "(MUST get score >= 8 and category=\"kurdistan\"); OR major wars, military escalations, world leader summits, global economy/sanctions, major AI/tech releases, and high-impact cyberattacks.\n"
    "- 7 (PASS): Notable international diplomacy, cross-border crises, significant space/science milestones.\n"
    "- 1 to 6 (REJECT): Hyper-local domestic news in foreign countries (e.g., individual US state lawsuits, municipal/city politics, "
    "local crime, routine business/licensing disputes, local court cases), celebrity gossip, sports trivia, clickbait, or minor opinion fluff.\n\n"
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


def normalize_category(raw_category: Optional[str], text_hint: str = "") -> str:
    """Map any category string strictly onto one of the 7 canonical category keys."""
    if raw_category:
        cleaned = raw_category.strip().lower()
        if cleaned in CANONICAL_CATEGORIES:
            return cleaned
        if cleaned in LEGACY_TO_CANONICAL_CATEGORY:
            return LEGACY_TO_CANONICAL_CATEGORY[cleaned]
        # Partial keyword heuristics if model slightly varied format
        if "kurd" in cleaned:
            return "kurdistan"
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

    if text_hint and KURDISTAN_PATTERN.search(text_hint):
        return "kurdistan"
    return "politics_world"


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
        rule_score, priority = self.score_article(article)
        credibility = ArticleExtractor.get_source_credibility(article.source_name)
        passed = self.is_eligible(rule_score, credibility)
        scaled_score = max(1, min(10, round(rule_score * 10)))
        text_hint = f"{article.original_title} {article.original_summary} {article.region}"
        cat = normalize_category(article.category, text_hint=text_hint)
        if cat == "kurdistan":
            scaled_score = max(8, scaled_score)
            passed = True
        return RelevanceEvaluation(
            score=scaled_score,
            is_breaking=(priority == Priority.CRITICAL),
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

        if not self._gemini.has_keys:
            return self._rule_based_fallback(article, "GEMINI_API_KEY not configured")

        snippet = (article.original_summary or "").strip()[:500]
        region_line = f"Region Tag: {article.region}\n" if article.region else ""
        user_content = (
            f"Title: {article.original_title}\n"
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
            is_breaking = bool(data.get("is_breaking", False))
            raw_category = str(data.get("category", "")).strip()
            text_hint = f"{article.original_title} {snippet} {article.region}"
            category = normalize_category(raw_category, text_hint=text_hint)

            # Enforce high priority if Gemini classified as Kurdistan (all 4 parts)
            if category == "kurdistan" and score < 8:
                score = 8

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
        text = f"{article.original_title} {article.original_summary}".lower()

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

    def is_eligible(self, score: float, credibility_score: float | None = None) -> bool:
        """Check if an article meets the minimum thresholds for publication."""
        if score < MIN_IMPORTANCE_SCORE:
            return False
        if credibility_score is not None and credibility_score < MIN_SOURCE_CREDIBILITY:
            return False
        return True

    def _score_keywords(self, article: Article) -> float:
        text = f"{article.original_title} {article.original_summary}".lower()
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
        text = f"{article.original_title} {article.original_summary}".lower()
        critical_hits = sum(1 for kw in CRITICAL_KEYWORDS if kw in text)
        if critical_hits >= 2 or score >= 0.85:
            return Priority.CRITICAL
        if critical_hits == 1 or score >= 0.70:
            return Priority.HIGH
        if score >= MIN_IMPORTANCE_SCORE:
            return Priority.NORMAL
        return Priority.LOW
