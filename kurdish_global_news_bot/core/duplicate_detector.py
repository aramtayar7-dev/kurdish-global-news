"""
Kurdish Global News Bot — Duplicate Detector

Multi-layer duplicate and event detection using URL canonicalization,
content hashing, fuzzy title matching, and event fingerprinting.
"""

from __future__ import annotations

import logging
from typing import Optional

from rapidfuzz import fuzz

from storage.models import Article
from storage.database import Database

logger = logging.getLogger("core.duplicates")

# Fuzzy match threshold (0-100)
FUZZY_TITLE_THRESHOLD = 78
# Event fingerprint similarity threshold
EVENT_SIMILARITY_THRESHOLD = 72


class DuplicateDetector:
    """Multi-layer duplicate and event detection."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def is_duplicate(self, article: Article) -> tuple[bool, str]:
        """
        Check if an article is a duplicate.
        Returns (is_duplicate, reason).
        """
        # Layer 1: Exact canonical URL match (excluding itself)
        existing = await self._db.get_article_by_canonical_url(article.canonical_url)
        if existing and (article.id is None or existing.id != article.id):
            return True, f"Exact URL match: {article.canonical_url[:80]}"

        # Layer 2: Content hash match (excluding itself)
        if article.content_hash:
            recent_hashes = await self._db.get_recent_content_hashes(hours=48, exclude_id=article.id)
            if article.content_hash in recent_hashes:
                return True, f"Content hash match: {article.content_hash[:16]}"

        # Layer 3: Fuzzy title matching (excluding itself)
        if article.normalized_title:
            is_fuzzy_dup, fuzzy_reason = await self._check_fuzzy_title(article)
            if is_fuzzy_dup:
                return True, fuzzy_reason

        return False, ""

    async def is_same_event_already_published(self, article: Article) -> tuple[bool, str]:
        """
        Check if the same EVENT (not just article) has already been published.
        Different headlines may describe the same event.

        Returns (is_same_event, reason).
        """
        if not article.event_fingerprint:
            return False, ""

        published_fps = await self._db.get_published_event_fingerprints(hours=72)
        if article.event_fingerprint in published_fps:
            return True, f"Same event fingerprint already published: {article.event_fingerprint[:16]}"

        # Also check fuzzy event similarity against published titles
        recent_titles = await self._db.get_recent_normalized_titles(hours=48, exclude_id=article.id, only_published=True)
        for _, existing_title in recent_titles:
            if not existing_title:
                continue
            similarity = fuzz.token_sort_ratio(article.normalized_title, existing_title)
            if similarity >= EVENT_SIMILARITY_THRESHOLD:
                return True, f"Similar event already covered (similarity={similarity}%)"

        return False, ""

    async def _check_fuzzy_title(self, article: Article) -> tuple[bool, str]:
        """Check for fuzzy title duplicates."""
        recent_titles = await self._db.get_recent_normalized_titles(hours=48, exclude_id=article.id)

        for existing_id, existing_title in recent_titles:
            if not existing_title or (article.id is not None and existing_id == article.id):
                continue
            similarity = fuzz.token_sort_ratio(article.normalized_title, existing_title)
            if similarity >= FUZZY_TITLE_THRESHOLD:
                return True, (
                    f"Fuzzy title match (similarity={similarity}%): "
                    f"existing_id={existing_id}"
                )

        return False, ""

    async def is_meaningful_update(self, article: Article) -> bool:
        """
        Check if an article about an existing event contains genuinely new information.
        This allows publishing updates about developing stories.
        """
        if not article.event_fingerprint:
            return False

        event = await self._db.get_event_by_fingerprint(article.event_fingerprint)
        if not event:
            return True  # New event entirely

        # Check if this article has substantially different content
        recent_titles = await self._db.get_recent_normalized_titles(hours=72)
        for _, existing_title in recent_titles:
            if not existing_title:
                continue
            similarity = fuzz.token_sort_ratio(article.normalized_title, existing_title)
            # If similarity is moderate (between 40% and 72%), it might be a meaningful update
            if 40 <= similarity < EVENT_SIMILARITY_THRESHOLD:
                # Check that the new title has significant unique words
                existing_words = set(existing_title.split())
                new_words = set(article.normalized_title.split())
                unique_words = new_words - existing_words
                if len(unique_words) >= 3:
                    logger.info("Meaningful update detected: %d unique words.", len(unique_words))
                    return True

        return False
