"""
Kurdish Global News Bot — Cross-Source Semantic Event Deduplicator

Two-stage deduplication engine that runs AFTER relevance classification
and BEFORE expensive web scraping / translation / card rendering:
  - Stage A (Zero-cost fast filter): Normalized title similarity against
    PUBLISHED headlines from the last DEDUP_WINDOW_HOURS (default 12h).
    If similarity >= 0.60, flags immediately as duplicate.
  - Stage B (Gemini semantic event match): Sends up to 25 recent published
    headlines (prioritizing the same category) to Gemini (`gemini-2.5-flash`,
    temperature=0.0, max_tokens=100) to detect the same underlying real-world
    event across different outlets, phrasing, or languages. Fails open safely
    on transient Gemini errors so publishing never stalls.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from rapidfuzz import fuzz

from config import DEDUP_WINDOW_HOURS, GEMINI_API_KEY, GEMINI_MODEL
from processing.gemini_client import GeminiAPIError, GeminiClient
from storage.database import Database
from storage.models import Article

logger = logging.getLogger("processing.deduplicator")

_DEDUP_SYSTEM_PROMPT = """You are a news wire deduplication editor.
Compare the CANDIDATE article against the list of RECENTLY_PUBLISHED headlines from the last 12 hours.
Determine if the candidate covers the EXACT SAME real-world news event or announcement as one of the published headlines, even if reported by a different outlet, phrased differently, or in a different language.

Do NOT mark different follow-up events on a broad topic (e.g., an earthquake yesterday vs a peace summit today) as duplicates.
Only mark `is_duplicate: true` if they are substantially reporting the SAME specific event.

Return ONLY valid JSON with no markdown or extra text:
{"is_duplicate": true/false, "matched_id": int_or_null, "reason": "brief explanation"}"""


@dataclass
class DedupResult:
    """Outcome of a two-stage semantic deduplication check."""
    is_duplicate: bool
    matched_id: Optional[int] = None
    reason: str = ""
    stage: str = "none"
    used_fallback: bool = False


def _normalize_for_match(text: str) -> str:
    """Lowercase and strip punctuation/extra whitespace for fast title comparison."""
    cleaned = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


class SemanticDeduplicator:
    """Two-stage cross-source event deduplicator powered by fast similarity + Gemini."""

    def __init__(
        self,
        db: Database,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        window_hours: Optional[int] = None,
        fast_threshold: float = 0.60,
        max_candidates: int = 25,
        base_backoff: float = 1.5,
    ) -> None:
        self._db = db
        self._window_hours = window_hours if window_hours is not None else DEDUP_WINDOW_HOURS
        self._fast_threshold = fast_threshold
        self._max_candidates = max_candidates
        self._gemini = GeminiClient(
            api_key=GEMINI_API_KEY if api_key is None else api_key,
            model=model or GEMINI_MODEL,
            base_backoff=base_backoff,
            max_rate_limit_retries=2,
        )

    async def cleanup(self) -> None:
        """Close the underlying Gemini HTTP session cleanly."""
        await self._gemini.cleanup()

    @staticmethod
    def _compute_title_similarity(title_a: str, title_b: str) -> float:
        """Return normalized [0.0, 1.0] lexical similarity between two titles."""
        norm_a = _normalize_for_match(title_a)
        norm_b = _normalize_for_match(title_b)
        if not norm_a or not norm_b:
            return 0.0
        ratio_seq = fuzz.ratio(norm_a, norm_b) / 100.0
        ratio_tok = fuzz.token_sort_ratio(norm_a, norm_b) / 100.0
        return max(ratio_seq, ratio_tok)

    def _fast_similarity_check(
        self,
        article: Article,
        recent_headlines: list[dict[str, Any]],
    ) -> Optional[DedupResult]:
        """Stage A: zero-cost lexical/token similarity check (>= 0.60)."""
        candidate_titles = [
            t for t in (article.original_title, article.translated_title) if t and t.strip()
        ]
        if not candidate_titles:
            return None

        best_score = 0.0
        best_item: Optional[dict[str, Any]] = None

        for item in recent_headlines:
            ref_titles = [
                t for t in (item.get("original_title", ""), item.get("translated_title", ""))
                if t and t.strip()
            ]
            for cand in candidate_titles:
                for ref in ref_titles:
                    sim = self._compute_title_similarity(cand, ref)
                    if sim > best_score:
                        best_score = sim
                        best_item = item

        if best_item is not None and best_score >= self._fast_threshold:
            return DedupResult(
                is_duplicate=True,
                matched_id=int(best_item["id"]) if best_item.get("id") is not None else None,
                reason=f"Fast title similarity {best_score:.2f} >= {self._fast_threshold:.2f}",
                stage="fast",
            )
        return None

    def _select_gemini_candidates(
        self,
        article: Article,
        recent_headlines: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Select up to `max_candidates` recent headlines, prioritizing the same category."""
        target_cat = (article.category or "").strip().lower()
        same_cat: list[dict[str, Any]] = []
        other_cat: list[dict[str, Any]] = []

        for item in recent_headlines:
            item_cat = (item.get("category") or "").strip().lower()
            if target_cat and item_cat == target_cat:
                same_cat.append(item)
            else:
                other_cat.append(item)

        return (same_cat + other_cat)[: self._max_candidates]

    @staticmethod
    def _build_gemini_payload(
        article: Article,
        candidates: list[dict[str, Any]],
    ) -> str:
        """Format candidate and recent published headlines cleanly for Gemini JSON check."""
        recent_payload = [
            {
                "id": item["id"],
                "title": item.get("original_title", ""),
                "translated_title": item.get("translated_title", ""),
                "category": item.get("category", ""),
            }
            for item in candidates
        ]
        return (
            f"CANDIDATE:\n"
            f"- title: {article.original_title}\n"
            f"- category: {article.category or 'unknown'}\n"
            f"- source: {article.source_name or 'unknown'}\n\n"
            f"RECENTLY_PUBLISHED ({len(recent_payload)} headlines):\n"
            f"{json.dumps(recent_payload, ensure_ascii=False)}"
        )

    async def check_duplicate(
        self,
        article: Article,
        extra_recent: Optional[list[dict[str, Any]]] = None,
    ) -> DedupResult:
        """
        Check whether `article` duplicates an event published in the last
        `DEDUP_WINDOW_HOURS` hours (or already accepted in `extra_recent` for this batch).
        Fails open on Gemini errors.
        """
        recent_headlines = await self._db.get_recent_published_headlines(hours=self._window_hours)
        if extra_recent:
            recent_headlines = list(extra_recent) + list(recent_headlines)
        if not recent_headlines:
            return DedupResult(
                is_duplicate=False,
                reason="No recent published headlines in lookback window",
                stage="none",
            )

        # Stage A: zero-cost normalized similarity check
        fast_match = self._fast_similarity_check(article, recent_headlines)
        if fast_match is not None:
            return fast_match

        if not self._gemini.has_keys:
            logger.debug("Gemini API key not configured for semantic dedup; failing open.")
            return DedupResult(
                is_duplicate=False,
                matched_id=None,
                reason="Fail-open: GEMINI_API_KEY not configured",
                stage="fail_open",
                used_fallback=True,
            )

        # Stage B: Gemini semantic event matching (up to 25 recent headlines)
        candidates = self._select_gemini_candidates(article, recent_headlines)
        user_content = self._build_gemini_payload(article, candidates)

        try:
            data = await self._gemini.generate_json(
                system_prompt=_DEDUP_SYSTEM_PROMPT,
                user_content=user_content,
                temperature=0.0,
                max_tokens=100,
            )
            is_dup = bool(data.get("is_duplicate", False))
            raw_matched = data.get("matched_id")
            matched_id: Optional[int] = None
            if raw_matched is not None and str(raw_matched).strip() != "":
                try:
                    matched_id = int(raw_matched)
                except (ValueError, TypeError):
                    matched_id = None
            reason = str(data.get("reason") or "").strip()
            return DedupResult(
                is_duplicate=is_dup,
                matched_id=matched_id if is_dup else None,
                reason=reason or ("Semantic event match" if is_dup else "Distinct event"),
                stage="gemini",
            )
        except (GeminiAPIError, Exception) as exc:
            logger.warning(
                "Gemini semantic deduplication fail-open for '%s': %s",
                (article.original_title or "")[:80],
                str(exc)[:200],
            )
            return DedupResult(
                is_duplicate=False,
                matched_id=None,
                reason=f"Fail-open after Gemini dedup error: {str(exc)[:120]}",
                stage="fail_open",
                used_fallback=True,
            )
