"""
Kurdish Global News Bot — Recovery Module

Handles crash recovery, interrupted job resumption, and state restoration.
"""

from __future__ import annotations

import logging
from typing import Optional

from storage.database import Database
from config import ArticleStatus, MAX_RETRIES

logger = logging.getLogger("core.recovery")


class RecoveryManager:
    """Manages crash recovery and interrupted job resumption."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def recover_interrupted_jobs(self) -> list[int]:
        """
        Find articles that were in-progress when the system last stopped.
        Reset them to appropriate states for reprocessing.
        Returns list of recovered article IDs.
        """
        recovered_ids: list[int] = []
        interrupted = await self._db.get_interrupted_articles()

        if not interrupted:
            logger.info("No interrupted jobs found.")
            return recovered_ids

        logger.info("Found %d interrupted jobs to recover.", len(interrupted))

        for article in interrupted:
            try:
                new_status = self._determine_recovery_status(article.status)
                await self._db.update_article_status(
                    article.id,  # type: ignore
                    new_status,
                    error_message=f"Recovered from interrupted state: {article.status}",
                )
                recovered_ids.append(article.id)  # type: ignore
                logger.info(
                    "Recovered article %d: %s -> %s (title: '%s')",
                    article.id, article.status, new_status,
                    article.original_title[:60],
                )
            except Exception as e:
                logger.error("Failed to recover article %d: %s", article.id, str(e)[:200])

        return recovered_ids

    async def recover_retry_pending(self) -> list[int]:
        """
        Find articles in RETRY_PENDING state and return their IDs.
        These will be re-fed into the pipeline.
        """
        retry_articles = await self._db.get_retry_pending_articles(max_retries=MAX_RETRIES)
        ids = [a.id for a in retry_articles if a.id is not None]
        if ids:
            logger.info("Found %d articles pending retry.", len(ids))
        return ids

    def _determine_recovery_status(self, interrupted_status: str) -> str:
        """Determine the appropriate recovery status based on where processing was interrupted."""
        if interrupted_status == ArticleStatus.VALIDATING:
            return ArticleStatus.DISCOVERED
        elif interrupted_status == ArticleStatus.PROCESSING:
            return ArticleStatus.DISCOVERED
        elif interrupted_status == ArticleStatus.TRANSLATING:
            return ArticleStatus.PROCESSING
        elif interrupted_status == ArticleStatus.READY_TO_PUBLISH:
            return ArticleStatus.READY_TO_PUBLISH  # Keep — just needs publishing
        elif interrupted_status == ArticleStatus.PUBLISHING:
            # Ambiguous: we don't know if Telegram received it
            # Mark as RETRY_PENDING for safe re-attempt
            return ArticleStatus.RETRY_PENDING
        else:
            return ArticleStatus.DISCOVERED

    async def run_startup_recovery(self) -> dict[str, int]:
        """
        Run all recovery procedures at startup.
        Returns a summary of recovered items.
        """
        logger.info("═══ Starting crash recovery ═══")

        interrupted = await self.recover_interrupted_jobs()
        retry_pending = await self.recover_retry_pending()

        summary = {
            "interrupted_recovered": len(interrupted),
            "retry_pending": len(retry_pending),
        }

        logger.info("Recovery complete: %s", summary)
        return summary
