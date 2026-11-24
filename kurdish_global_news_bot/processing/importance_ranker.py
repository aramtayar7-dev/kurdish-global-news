"""
Kurdish Global News Bot — Importance Ranker & Global Relevance Gatekeeper

Re-exports ImportanceRanker and RelevanceEvaluation from core.importance_ranker
so both processing.importance_ranker and core.importance_ranker remain accessible.
"""

from core.importance_ranker import (
    CRITICAL_KEYWORDS,
    GLOBAL_RELEVANCE_SYSTEM_PROMPT,
    GROQ_API_ENDPOINT,
    HIGH_KEYWORDS,
    ImportanceRanker,
    RelevanceEvaluation,
)

__all__ = [
    "CRITICAL_KEYWORDS",
    "GLOBAL_RELEVANCE_SYSTEM_PROMPT",
    "GROQ_API_ENDPOINT",
    "HIGH_KEYWORDS",
    "ImportanceRanker",
    "RelevanceEvaluation",
]
