"""
Kurdish Global News Bot — Importance Ranker

Scores and classifies articles by global significance, source credibility,
recency, completeness, and other quality factors.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Optional

from storage.models import Article
from processing.extractor import ArticleExtractor
from config import (
    MIN_IMPORTANCE_SCORE, MIN_SOURCE_CREDIBILITY, Priority,
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
]


class ImportanceRanker:
    """Scores and classifies article importance."""

    def score_article(self, article: Article) -> tuple[float, str]:
        """
        Calculate importance score (0.0 to 1.0) and priority level.
        Returns (score, priority).
        """
        scores: list[tuple[float, float]] = []  # (score, weight)

        # 1. Source credibility (weight: 0.20)
        credibility = ArticleExtractor.get_source_credibility(article.source_name)
        scores.append((credibility, 0.20))

        # 2. Keyword significance (weight: 0.30)
        keyword_score = self._score_keywords(article)
        scores.append((keyword_score, 0.30))

        # 3. Content completeness (weight: 0.15)
        completeness = self._score_completeness(article)
        scores.append((completeness, 0.15))

        # 4. Recency (weight: 0.15)
        recency = self._score_recency(article)
        scores.append((recency, 0.15))

        # 5. Title quality (weight: 0.10)
        title_quality = self._score_title_quality(article)
        scores.append((title_quality, 0.10))

        # 6. Category relevance (weight: 0.10)
        category_score = self._score_category(article)
        scores.append((category_score, 0.10))

        # Weighted average
        total_score = sum(s * w for s, w in scores)
        total_weight = sum(w for _, w in scores)
        base_score = total_score / total_weight if total_weight > 0 else 0.0

        # Apply topic / category priority boost (+0.20 Primary, +0.10 key Secondary)
        focus_boost = self._calculate_focus_boost(article)
        final_score = base_score + focus_boost

        # Clamp to [0.0, 1.0]
        final_score = max(0.0, min(1.0, final_score))

        # Determine priority
        priority = self._determine_priority(final_score, article)

        return final_score, priority

    def _calculate_focus_boost(self, article: Article) -> float:
        """
        Calculate priority score boost:
        - +0.20 for Primary focus (war, military strikes, conflicts, world leaders,
          presidents, prime ministers, NATO, UN Security Council, geopolitical escalations).
        - +0.10 for key Secondary focus (AI breakthroughs, major cyberattacks, ransomware,
          zero-day hacks, state-sponsored hacking, major economic shifts).
        """
        text = f"{article.original_title} {article.original_summary}".lower()

        # Check Primary Focus (+0.20)
        if is_primary_category(article.category):
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

        # Check key Secondary Focus (+0.10)
        secondary_categories = {
            "AI & Technology", "Cybersecurity", "Global Economy & Financial Markets"
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
        """Score based on presence of significance-indicating keywords."""
        text = f"{article.original_title} {article.original_summary}".lower()

        # Check critical keywords
        critical_hits = sum(1 for kw in CRITICAL_KEYWORDS if kw in text)
        if critical_hits >= 2:
            return 1.0
        if critical_hits == 1:
            return 0.85

        # Check high-importance keywords
        high_hits = sum(1 for kw in HIGH_KEYWORDS if kw in text)
        if high_hits >= 2:
            return 0.75
        if high_hits == 1:
            return 0.60

        return 0.40

    def _score_completeness(self, article: Article) -> float:
        """Score based on how complete the article data is."""
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
        """Score based on how recent the article is."""
        if not article.publication_date:
            return 0.5  # Unknown date gets middle score

        try:
            from dateutil import parser as dateparser
            pub_dt = dateparser.parse(article.publication_date)
            if not pub_dt:
                return 0.5

            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=timezone.utc)

            age = datetime.now(timezone.utc) - pub_dt
            hours = age.total_seconds() / 3600

            if hours < 1:
                return 1.0
            elif hours < 3:
                return 0.9
            elif hours < 6:
                return 0.75
            elif hours < 12:
                return 0.6
            elif hours < 24:
                return 0.4
            elif hours < 48:
                return 0.25
            else:
                return 0.1
        except Exception:
            return 0.5

    def _score_title_quality(self, article: Article) -> float:
        """Score based on title quality indicators."""
        title = article.original_title
        if not title:
            return 0.0

        score = 0.5  # Base score

        # Good length (not too short, not too long)
        if 20 <= len(title) <= 150:
            score += 0.2

        # Contains specific details (numbers, proper nouns)
        if re.search(r"\d", title):
            score += 0.1  # Contains numbers
        if re.search(r"[A-Z][a-z]+", title):
            score += 0.1  # Contains proper nouns

        # Penalty for all-caps clickbait
        if title.isupper():
            score -= 0.3

        return max(0.0, min(1.0, score))

    def _score_category(self, article: Article) -> float:
        """Score based on category relevance for international news."""
        high_priority_categories = {
            "Wars & Armed Conflicts", "International Politics",
            "Natural Disasters & Emergencies", "World Leaders & Governments",
            "Military & Defense", "Diplomacy & International Relations",
        }
        medium_priority_categories = {
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
        """Determine priority level from score and content analysis."""
        text = f"{article.original_title} {article.original_summary}".lower()

        # Check for critical indicators
        critical_hits = sum(1 for kw in CRITICAL_KEYWORDS if kw in text)
        if critical_hits >= 2 or score >= 0.85:
            return Priority.CRITICAL
        if critical_hits == 1 or score >= 0.70:
            return Priority.HIGH
        if score >= MIN_IMPORTANCE_SCORE:
            return Priority.NORMAL
        return Priority.LOW
