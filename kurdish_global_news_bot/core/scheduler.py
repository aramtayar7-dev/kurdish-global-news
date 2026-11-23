"""
Kurdish Global News Bot — Scheduler

Runs news discovery every 60 seconds. Discovery and publication are independent:
articles are published IMMEDIATELY when ready, NOT on a schedule.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional, Callable, Awaitable

from config import NEWS_CHECK_INTERVAL_SECONDS

logger = logging.getLogger("core.scheduler")


class DiscoveryScheduler:
    """
    Schedules news discovery cycles every 60 seconds.

    IMPORTANT:
    - Discovery runs every NEWS_CHECK_INTERVAL_SECONDS (default: 60).
    - There is NO hourly publishing scheduler.
    - There is NO publishing batch or cooldown.
    - Articles are published IMMEDIATELY by the pipeline, not by this scheduler.
    - This scheduler only triggers DISCOVERY of new articles.
    """

    def __init__(self) -> None:
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._cycle_count = 0
        self._last_cycle_time: float = 0
        self._discovery_callback: Optional[Callable[[], Awaitable[None]]] = None
        self._interval = NEWS_CHECK_INTERVAL_SECONDS

    def set_discovery_callback(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Set the async function to call each discovery cycle."""
        self._discovery_callback = callback

    async def start(self) -> None:
        """Start the discovery scheduler."""
        if self._running:
            logger.warning("Scheduler already running.")
            return
        if not self._discovery_callback:
            raise RuntimeError("No discovery callback set. Call set_discovery_callback() first.")

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "Discovery scheduler started. Interval: %d seconds. "
            "NO hourly publishing schedule exists.",
            self._interval,
        )

    async def stop(self) -> None:
        """Stop the discovery scheduler gracefully."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Discovery scheduler stopped after %d cycles.", self._cycle_count)

    async def _run_loop(self) -> None:
        """Main discovery loop — runs every 60 seconds."""
        while self._running:
            cycle_start = time.time()
            self._cycle_count += 1

            try:
                logger.info("═══ Discovery cycle #%d starting ═══", self._cycle_count)
                assert self._discovery_callback is not None
                await self._discovery_callback()
                self._last_cycle_time = time.time() - cycle_start
                logger.info(
                    "═══ Discovery cycle #%d completed in %.1fs ═══",
                    self._cycle_count, self._last_cycle_time,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(
                    "Discovery cycle #%d error: %s",
                    self._cycle_count, str(e)[:300],
                )

            # Wait for the next cycle
            elapsed = time.time() - cycle_start
            wait_time = max(0, self._interval - elapsed)
            if wait_time > 0:
                try:
                    await asyncio.sleep(wait_time)
                except asyncio.CancelledError:
                    break

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def interval_seconds(self) -> int:
        return self._interval
