"""
Kurdish Global News Bot — Importance Ranker & Gemini Global Relevance Classifier

Re-exports symbols from core.importance_ranker so both processing.importance_ranker
and core.importance_ranker remain accessible cleanly.
"""

from core.importance_ranker import (
    CRITICAL_KEYWORDS,
    GLOBAL_RELEVANCE_SYSTEM_PROMPT,
    HIGH_KEYWORDS,
    ImportanceRanker,
    KURDISTAN_PATTERN,
    LEGACY_TO_CANONICAL_CATEGORY,
    RelevanceEvaluation,
    normalize_category,
)

__all__ = [
    "CRITICAL_KEYWORDS",
    "GLOBAL_RELEVANCE_SYSTEM_PROMPT",
    "HIGH_KEYWORDS",
    "ImportanceRanker",
    "KURDISTAN_PATTERN",
    "LEGACY_TO_CANONICAL_CATEGORY",
    "RelevanceEvaluation",
    "normalize_category",
]
