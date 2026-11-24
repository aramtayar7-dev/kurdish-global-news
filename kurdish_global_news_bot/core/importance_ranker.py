"""
Kurdish Global News Bot — Importance Ranker

Scores and classifies articles by global significance, source credibility,
recency, completeness, and other quality factors.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Optional

import aiohttp

from storage.models import Article
from processing.extractor import ArticleExtractor
from config import (
    MIN_IMPORTANCE_SCORE, MIN_SOURCE_CREDIBILITY, MIN_GLOBAL_RELEVANCE_SCORE,
    GROQ_API_KEY, GROQ_MODEL, REQUEST_TIMEOUT, Priority,
    is_primary_category,
)

logger = logging.getLogger("core.importance")

GROQ_API_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

GLOBAL_RELEVANCE_SYSTEM_PROMPT = (
    "You are the chief international news editor for a Central Kurdish (Sorani) global breaking news channel.\n"
    "Evaluate the global importance and relevance of the given article for a Kurdish and international audience.\n\n"
    "Scoring Criteria (1 to 10):\n"
    "- 8 to 10 (PASS): Major geopolitical events, Middle East / Kurdistan / Iraq / Iran / Turkey / Syria developments, "
    "active military conflicts, wars, strikes, defense breakthroughs, UN Security Council / NATO decisions, "
    "major global economic, oil, energy, or sanctions shifts, major AI/tech releases, and high-impact global cybersecurity attacks.\n"
    "- 7 (PASS): Notable international diplomacy, cross-border crises, major national shifts with global consequences, or significant space/science milestones.\n"
    "- 1 to 6 (REJECT): Hyper-local domestic news from foreign countries (e.g., individual US state lawsuits, municipal politics, "
    "local crime, routine administrative or licensing disputes, local court cases), celebrity gossip, entertainment, lifestyle/clickbait, or minor opinion pieces.\n\n"
    "Respond ONLY with valid JSON matching this schema (no markdown fences, no commentary):\n"
    '{"score": <integer 1-10>, "is_breaking": <true or false>, "reason": "<short one-sentence justification>"}'
)


@dataclass
class RelevanceEvaluation:
    """Result of the LLM Global Importance & Relevance Gatekeeper check."""
    score: int
    is_breaking: bool
    reason: str
    passed: bool
    used_fallback: bool = False

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
    """Scores and classifies article importance using rule-based + LLM relevance checks."""

    def __init__(
        self,
        api_key: Optional[str | list[str]] = None,
        model: Optional[str] = None,
        min_relevance_score: Optional[int] = None,
        session: Optional[aiohttp.ClientSession] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> None:
        raw_key = GROQ_API_KEY if api_key is None else api_key
        if isinstance(raw_key, list):
            self._api_keys = [k.strip() for k in raw_key if k.strip()]
        elif isinstance(raw_key, str):
            self._api_keys = [k.strip() for k in raw_key.split(",") if k.strip()]
        else:
            self._api_keys = []

        self._model = (GROQ_MODEL if model is None else model).strip() or "llama-3.3-70b-versatile"
        self._min_relevance_score = (
            MIN_GLOBAL_RELEVANCE_SCORE if min_relevance_score is None else min_relevance_score
        )
        self._session = session
        self._semaphore = semaphore if semaphore is not None else asyncio.Semaphore(2)
        self._current_key_index: int = 0

    async def _get_session(self) -> aiohttp.ClientSession:
        loop = asyncio.get_running_loop()
        if (
            self._session is None
            or self._session.closed
            or getattr(self._session, "_loop", None) != loop
        ):
            timeout = aiohttp.ClientTimeout(total=min(REQUEST_TIMEOUT, 20))
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def cleanup(self) -> None:
        """Close the underlying aiohttp session if open."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _rule_based_fallback(
        self, article: Article, err_reason: str
    ) -> RelevanceEvaluation:
        """Fail-open safely using rule-based importance scoring when LLM classifier errors out."""
        rule_score, priority = self.score_article(article)
        credibility = ArticleExtractor.get_source_credibility(article.source_name)
        passed = self.is_eligible(rule_score, credibility)
        scaled_score = max(1, min(10, round(rule_score * 10)))
        return RelevanceEvaluation(
            score=scaled_score,
            is_breaking=(priority == Priority.CRITICAL),
            reason=f"Rule-based fallback ({err_reason}; rule_score={rule_score:.2f})",
            passed=passed,
            used_fallback=True,
        )

    @staticmethod
    def _parse_relevance_json(raw_text: str) -> tuple[int, bool, str]:
        """Parse and validate JSON returned by the Groq relevance classifier."""
        clean = raw_text.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        elif clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()

        data = json.loads(clean)
        if not isinstance(data, dict) or "score" not in data:
            raise ValueError("Missing 'score' field in classifier JSON response")

        score_val = int(round(float(data["score"])))
        score_clamped = max(1, min(10, score_val))
        is_breaking = bool(data.get("is_breaking", False))
        reason = str(data.get("reason", "")).strip() or "Evaluated by global relevance classifier"
        return score_clamped, is_breaking, reason

    async def evaluate_global_relevance(
        self,
        article: Article,
        min_score: Optional[int] = None,
    ) -> RelevanceEvaluation:
        """
        Run lightweight Groq LLM global importance & relevance classification
        on article title, source, and snippet.
        Fails open to rule-based scoring if the API key is missing or API errors out.
        """
        threshold = self._min_relevance_score if min_score is None else min_score

        if not self._api_keys:
            return self._rule_based_fallback(article, "GROQ_API_KEY not configured")

        snippet = (article.original_summary or "").strip()[:500]
        user_content = (
            f"Title: {article.original_title}\n"
            f"Source: {article.source_name or 'Unknown'}\n"
            f"Category: {article.category or 'General'}\n"
            f"Snippet: {snippet}"
        )

        total_keys = len(self._api_keys)
        last_error = ""

        async with self._semaphore:
            session = await self._get_session()

            for attempt in range(total_keys):
                key = self._api_keys[self._current_key_index]
                headers = {
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "User-Agent": "KurdishGlobalNewsBot/1.0",
                }
                payload = {
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": GLOBAL_RELEVANCE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 120,
                    "response_format": {"type": "json_object"},
                }

                try:
                    async with session.post(
                        GROQ_API_ENDPOINT,
                        json=payload,
                        headers=headers,
                    ) as response:
                        if response.status == 200:
                            data = await response.json(content_type=None)
                            choices = data.get("choices", [])
                            if not choices:
                                raise ValueError("No choices in Groq classifier response")
                            content = choices[0].get("message", {}).get("content", "").strip()
                            score, is_breaking, reason = self._parse_relevance_json(content)
                            return RelevanceEvaluation(
                                score=score,
                                is_breaking=is_breaking,
                                reason=reason,
                                passed=(score >= threshold),
                                used_fallback=False,
                            )

                        err_text = await response.text()
                        last_error = f"HTTP {response.status}: {err_text[:150]}"
                        if (
                            response.status in (401, 403, 429)
                            or "rate" in err_text.lower()
                            or "quota" in err_text.lower()
                        ):
                            old_idx = self._current_key_index
                            self._current_key_index = (self._current_key_index + 1) % total_keys
                            logger.warning(
                                "Groq classifier key [%d] hit HTTP %d. Rotating to key [%d].",
                                old_idx, response.status, self._current_key_index,
                            )
                            if attempt < total_keys - 1:
                                continue
                        break

                except asyncio.TimeoutError:
                    last_error = "Groq classifier request timed out"
                    if attempt < total_keys - 1:
                        self._current_key_index = (self._current_key_index + 1) % total_keys
                        continue
                    break
                except Exception as e:
                    last_error = str(e)[:150]
                    if attempt < total_keys - 1 and "json" not in last_error.lower():
                        self._current_key_index = (self._current_key_index + 1) % total_keys
                        continue
                    break

        logger.warning(
            "Global relevance classifier error (%s); failing open to rule-based score for '%s'.",
            last_error, article.original_title[:60],
        )
        return self._rule_based_fallback(article, last_error or "Classifier API error")

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
