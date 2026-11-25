"""
Kurdish Global News Bot — Main Engine

Orchestrates all components: source adapters, pipeline, scheduler, 
database, Telegram publisher, monitoring, and recovery.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from datetime import datetime, timezone
from typing import Optional

from storage import create_database, get_kurdistan_date_str
from storage.migrations import run_migrations
from core.scheduler import DiscoveryScheduler
from core.importance_ranker import (
    check_blocked_topic,
    is_strict_breaking,
    select_top_candidates_for_gemini,
)
from core.pipeline import ArticlePipeline
from core.recovery import RecoveryManager
from sources.base import BaseSource, RawArticle
from sources.duckduckgo_source import DuckDuckGoSource
from sources.rss_source import RSSSource
from sources.gdelt_source import GDELTSource
from sources.official_sources import OfficialSourcesAdapter
from sources.kurdistan_source import KurdistanNewsSource
from sources.telegram_source import TelegramSource
from processing.extractor import ArticleExtractor
from processing.card_generator import NewsCardGenerator
from processing.digest_generator import DIGEST_HEADER_PREFIX, DailyDigestGenerator
from tg_bot.publisher import TelegramPublisher
from monitoring.health import HealthMonitor
from monitoring.metrics import Metrics, Timer
from config import (
    ENABLE_DDG, ENABLE_RSS, ENABLE_GDELT, ENABLE_TELEGRAM, CHANNEL_HANDLE,
    MAX_CONCURRENT_SOURCES, ArticleStatus, BACKUP_INTERVAL_HOURS,
    MAX_ARTICLE_AGE_MINUTES, BATCH_PUBLISH_LIMIT, MAX_GEMINI_CANDIDATES_PER_CYCLE,
    DEDUP_HISTORY_HOURS,
)

logger = logging.getLogger("core.engine")


class NewsEngine:
    """
    Main application engine. Coordinates:
    - Source discovery every 60 seconds
    - Immediate article processing and publishing
    - Database persistence
    - Health monitoring and metrics
    - Crash recovery
    """

    def __init__(self) -> None:
        self._db = create_database()
        self._publisher = TelegramPublisher()
        self._health = HealthMonitor()
        self._metrics = Metrics()
        self._scheduler = DiscoveryScheduler()
        self._extractor = ArticleExtractor()
        self._pipeline: Optional[ArticlePipeline] = None
        self._recovery: Optional[RecoveryManager] = None
        self._digest_generator: Optional[DailyDigestGenerator] = None
        self._card_generator: Optional[NewsCardGenerator] = None
        self._sources: list[BaseSource] = []
        self._running = False
        self._source_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SOURCES)
        self._backup_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Initialize all components and start the engine."""
        logger.info("╔══════════════════════════════════════════╗")
        logger.info("║   Kurdish Global News Bot — Starting     ║")
        logger.info("╚══════════════════════════════════════════╝")

        # 1. Connect database
        await self._db.connect()
        if getattr(self._db, "conn", None) is not None:
            await run_migrations(self._db.conn)

        # 2. Initialize components
        self._recovery = RecoveryManager(self._db)
        self._pipeline = ArticlePipeline(
            db=self._db,
            publisher=self._publisher,
            health=self._health,
            metrics=self._metrics,
        )

        # 3. Register health components
        for name in ["discovery", "pipeline", "publishing", "translation", "database"]:
            self._health.register_component(name)

        # 4. Initialize Telegram publisher
        try:
            await self._publisher.initialize()
            channel_ok = await self._publisher.test_connection()
            if not channel_ok:
                logger.error("Cannot access Telegram channel. Check TELEGRAM_CHANNEL_ID and bot permissions.")
                raise RuntimeError("Telegram channel not accessible.")
        except Exception as e:
            logger.error("Telegram initialization failed: %s", str(e))
            raise

        # 5. Initialize sources
        await self._init_sources()

        # 6. Run crash recovery
        recovery_summary = await self._recovery.run_startup_recovery()
        if recovery_summary["interrupted_recovered"] > 0:
            # Re-process recovered articles
            await self._process_recovered_articles()

        # 7. Start scheduler
        self._scheduler.set_discovery_callback(self._discovery_cycle)
        await self._scheduler.start()

        # 8. Start background tasks
        self._backup_task = asyncio.create_task(self._backup_loop())
        self._metrics_task = asyncio.create_task(self._metrics_loop())

        # Check if database started empty (cold start / ephemeral filesystem reset)
        db_empty = await self._db.is_empty()
        self._pipeline.set_cold_start(db_empty)
        if db_empty:
            logger.info(
                "Cold start detected: database is empty. Startup safety guard active (ignoring articles older than %d minutes).",
                MAX_ARTICLE_AGE_MINUTES,
            )

        self._running = True
        logger.info("Engine started. Discovery every %d seconds. Publishing: IMMEDIATE.",
                     self._scheduler.interval_seconds)

        # Log database stats
        stats = await self._db.get_stats()
        logger.info("Database stats: %s", stats)

    async def run_once(self) -> None:
        """
        Execute one complete discovery -> filter -> translate -> publish cycle,
        flush/close the database cleanly, and exit.
        """
        logger.info("╔══════════════════════════════════════════╗")
        logger.info("║   Kurdish Global News Bot — Single Run   ║")
        logger.info("╚══════════════════════════════════════════╝")

        # 1. Connect database & run migrations (SQLite only)
        await self._db.connect()
        if getattr(self._db, "conn", None) is not None:
            await run_migrations(self._db.conn)

        # 2. Initialize components
        self._recovery = RecoveryManager(self._db)
        self._pipeline = ArticlePipeline(
            db=self._db,
            publisher=self._publisher,
            health=self._health,
            metrics=self._metrics,
        )

        for name in ["discovery", "pipeline", "publishing", "translation", "database"]:
            self._health.register_component(name)

        try:
            # 3. Initialize Telegram publisher
            await self._publisher.initialize()
            channel_ok = await self._publisher.test_connection()
            if not channel_ok:
                logger.error("Cannot access Telegram channel. Check TELEGRAM_CHANNEL_ID and bot permissions.")
                raise RuntimeError("Telegram channel not accessible.")

            # 4. Initialize sources
            await self._init_sources()

            # 5. Cold start check
            db_empty = await self._db.is_empty()
            self._pipeline.set_cold_start(db_empty)
            if db_empty:
                logger.info(
                    "Cold start detected: database is empty. Startup safety guard active (ignoring articles older than %d minutes).",
                    MAX_ARTICLE_AGE_MINUTES,
                )

            # 6. Run crash recovery if any interrupted articles exist
            recovery_summary = await self._recovery.run_startup_recovery()
            if recovery_summary["interrupted_recovered"] > 0:
                await self._process_recovered_articles()

            # 7. Execute ONE complete discovery & publishing cycle
            logger.info("Executing single discovery and publishing cycle...")
            await self._discovery_cycle()
        finally:
            # 8. Clean shutdown & close DB
            await self.stop()
            logger.info("Single run completed cleanly.")

    async def run_daily_digest(self, force: bool = False) -> bool:
        """
        Execute the Automated Nightly Kurdish News Digest:
          1. Fetch top 10 published articles from last 24h (Kurdistan prioritized)
          2. Check duplicate digest lock for today's Kurdistan date (UTC+3) unless force=True
          3. Skip cleanly if < 3 articles exist in the last 24h
          4. Generate Gemini Sorani roundup & 1280x720 nightly cover card
          5. Publish to Telegram, mark digest date in SQLite, clean up temp card & close DB
        Returns True if digest was published, False if skipped or failed.
        """
        logger.info("╔══════════════════════════════════════════╗")
        logger.info("║  Kurdish Global News — Nightly Digest    ║")
        logger.info("╚══════════════════════════════════════════╝")

        await self._db.connect()
        if getattr(self._db, "conn", None) is not None:
            await run_migrations(self._db.conn)

        digest_gen = self._digest_generator or DailyDigestGenerator()
        card_gen = self._card_generator or NewsCardGenerator()
        card_path = None

        try:
            digest_date = get_kurdistan_date_str()
            if not force and await self._db.is_digest_published_for_date(digest_date):
                logger.info(
                    "SKIPPED: Daily digest for %s already published (pass --force to re-publish).",
                    digest_date,
                )
                return False

            candidates = await self._db.get_published_articles_last_24h(limit=10)
            if len(candidates) < 3:
                logger.info(
                    "SKIPPED: fewer than 3 articles in the last 24h (found %d).",
                    len(candidates),
                )
                return False

            digest = await digest_gen.generate_digest(candidates, digest_date=digest_date)
            if not digest:
                return False

            await self._publisher.initialize()
            channel_ok = await self._publisher.test_connection()
            if not channel_ok:
                raise RuntimeError("Telegram channel not accessible.")

            card_path = await card_gen.generate_digest_card(
                date_label=digest.date_label,
                story_count=digest.story_count,
            )
            header_caption = (
                f"<b>{DIGEST_HEADER_PREFIX}</b>\n"
                f"📅 {digest.date_label}\n"
                f"📡 {CHANNEL_HANDLE}"
            )
            msg_id = await self._publisher.publish_daily_digest(
                digest_text=digest.digest_text,
                card_path=card_path,
                header_caption=header_caption,
            )
            if msg_id is None:
                logger.error("Nightly digest publishing returned no Telegram message_id.")
                return False

            await self._db.mark_digest_published(
                digest_date=digest.date_str,
                story_count=digest.story_count,
                digest_text=digest.digest_text,
                telegram_message_id=msg_id,
            )
            logger.info(
                "Nightly digest published cleanly for %s (%d stories, msg_id=%d).",
                digest.date_str,
                digest.story_count,
                msg_id,
            )
            return True
        finally:
            if card_path is not None:
                try:
                    card_path.unlink(missing_ok=True)
                except Exception:
                    pass
            try:
                await digest_gen.cleanup()
            except Exception:
                pass
            try:
                await self._publisher.cleanup()
            except Exception:
                pass
            try:
                await self._db.backup()
            except Exception:
                pass
            await self._db.close()

    async def stop(self) -> None:
        """Gracefully shut down all components."""
        logger.info("Engine shutting down...")
        self._running = False

        # Stop scheduler
        await self._scheduler.stop()

        # Cancel background tasks
        for task in [self._backup_task, self._metrics_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Cleanup sources
        for source in self._sources:
            try:
                await source.cleanup()
            except Exception as e:
                logger.warning("Source cleanup error (%s): %s", source.name, str(e)[:100])

        # Cleanup pipeline (ranker & translator HTTP sessions)
        if self._pipeline:
            try:
                await self._pipeline.cleanup()
            except Exception as e:
                logger.warning("Pipeline cleanup error: %s", str(e)[:100])

        # Cleanup publisher
        await self._publisher.cleanup()

        # Final backup
        try:
            await self._db.backup()
        except Exception as e:
            logger.warning("Final backup failed: %s", str(e)[:100])

        # Close database
        await self._db.close()

        # Final metrics
        self._metrics.log_report()

        logger.info("Engine stopped. Total discovery cycles: %d", self._scheduler.cycle_count)

    async def run_forever(self) -> None:
        """Run the engine until interrupted."""
        await self.start()

        # Set up signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        def _handle_signal() -> None:
            logger.info("Shutdown signal received.")
            stop_event.set()

        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler in some contexts
            pass

        try:
            await stop_event.wait()
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            await self.stop()

    # ───────────────────────────────────────────────
    # Source Management
    # ───────────────────────────────────────────────

    async def _init_sources(self) -> None:
        """Initialize all enabled news source adapters."""
        if ENABLE_DDG:
            ddg = DuckDuckGoSource()
            await ddg.initialize()
            self._sources.append(ddg)

        if ENABLE_RSS:
            rss = RSSSource()
            await rss.initialize()
            self._sources.append(rss)

        if ENABLE_GDELT:
            gdelt = GDELTSource()
            await gdelt.initialize()
            self._sources.append(gdelt)

        official = OfficialSourcesAdapter()
        await official.initialize()
        self._sources.append(official)

        kurdistan = KurdistanNewsSource()
        await kurdistan.initialize()
        self._sources.append(kurdistan)

        if ENABLE_TELEGRAM:
            telegram_src = TelegramSource()
            await telegram_src.initialize()
            self._sources.append(telegram_src)

        # Register sources in database
        for source in self._sources:
            from storage.models import Source
            await self._db.upsert_source(Source(
                name=source.name,
                source_type=source.source_type,
                enabled=source.enabled,
            ))

        logger.info("Initialized %d news sources: %s",
                     len(self._sources), [s.name for s in self._sources])

    # ───────────────────────────────────────────────
    # Discovery Cycle
    # ───────────────────────────────────────────────

    async def _discovery_cycle(self) -> None:
        """
        Run one discovery cycle:
        1. Load last 48-72h published history from persistent DB (Supabase / SQLite)
        2. Fetch articles from all sources concurrently
        3. Extract and deduplicate across URLs, events, and post-translation Sorani
        4. Feed into the pipeline for IMMEDIATE processing and publishing
        """
        with Timer(self._metrics, "discovery_cycle_time"):
            try:
                recent_history = await self._db.get_recent_published_headlines(hours=DEDUP_HISTORY_HOURS)
                if self._pipeline is not None:
                    self._pipeline.set_cycle_published_history(recent_history)
                logger.info(
                    "Loaded %d published story records from the last %dh for cross-run deduplication.",
                    len(recent_history),
                    DEDUP_HISTORY_HOURS,
                )
            except Exception as hist_exc:
                logger.warning(
                    "Failed to preload %dh published history: %s",
                    DEDUP_HISTORY_HOURS,
                    str(hist_exc)[:150],
                )
                if self._pipeline is not None:
                    self._pipeline.set_cycle_published_history(None)

            try:
                # Fetch from all sources concurrently
                all_raw_articles: list[RawArticle] = []
                fetch_tasks = [
                    self._fetch_from_source(source) for source in self._sources if source.enabled
                ]
                results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

                for i, result in enumerate(results):
                    source_name = self._sources[i].name if i < len(self._sources) else "unknown"
                    if isinstance(result, Exception):
                        logger.error("Source '%s' error: %s", source_name, str(result)[:200])
                        self._health.record_failure("discovery", f"{source_name}: {result}")
                        continue
                    if isinstance(result, list):
                        all_raw_articles.extend(result)
                        self._health.record_success("discovery")

                self._metrics.increment("total_raw_articles", len(all_raw_articles))
                logger.info("Discovered %d raw articles from %d sources.",
                             len(all_raw_articles), len(self._sources))

                published_this_cycle = 0
                if all_raw_articles:
                    # Extract and prepare articles
                    new_articles = await self._extract_and_insert(all_raw_articles)
                    logger.info("%d new articles to process.", len(new_articles))

                    if new_articles:
                        # Pre-filter obvious off-topic items & cap fresh relevant candidates BEFORE Gemini
                        capped_articles, locally_rejected = select_top_candidates_for_gemini(
                            new_articles, max_candidates=MAX_GEMINI_CANDIDATES_PER_CYCLE
                        )
                        for rej_art, rej_reason in locally_rejected:
                            try:
                                await self._db.update_article_status(
                                    rej_art.id,
                                    ArticleStatus.SKIPPED,
                                    error_message=f"SKIPPED [Pre-Filter]: {rej_reason}",
                                )
                                if rej_reason.startswith(("Blocked", "Pre-filtered")):
                                    await self._db.mark_url_skipped(
                                        rej_art.canonical_url or rej_art.original_url,
                                        title=rej_art.original_title,
                                        score=2.0,
                                        reason=rej_reason,
                                    )
                            except Exception:
                                pass

                        if len(locally_rejected) > 0:
                            logger.info(
                                "Local pre-filter/cap: %d -> %d candidates forwarded to pipeline (%d filtered locally with 0 Gemini calls).",
                                len(new_articles),
                                len(capped_articles),
                                len(locally_rejected),
                            )

                        if capped_articles:
                            # Enforce 5-of-8 Bashur/Global quota & publish ordered winners
                            pipeline_results = await self._pipeline.process_batch(
                                capped_articles, max_publish=BATCH_PUBLISH_LIMIT
                            )
                            published_this_cycle = pipeline_results.get("published", 0)
                            logger.info("Pipeline results: %s", pipeline_results)

                # Check and publish held articles within remaining cycle capacity
                remaining_capacity = max(0, BATCH_PUBLISH_LIMIT - published_this_cycle)
                if remaining_capacity > 0:
                    await self._publish_held_articles(max_additional=remaining_capacity)

                # Cold start protection completed after first discovery pass
                self._pipeline.set_cold_start(False)
            finally:
                if self._pipeline is not None:
                    self._pipeline.set_cycle_published_history(None)

    async def _publish_held_articles(self, max_additional: int = BATCH_PUBLISH_LIMIT) -> None:
        """Evaluate held READY_TO_PUBLISH articles against 5-of-8 balancer and publish if eligible."""
        if max_additional <= 0:
            return
        held_articles = await self._db.get_articles_by_status(ArticleStatus.READY_TO_PUBLISH, limit=20)
        if not held_articles:
            return

        recent_published = await self._db.get_recent_published_articles(limit=8)
        plan = self._pipeline._quota_balancer.select_batch(
            held_articles,
            limit=max_additional,
            recent_published=recent_published,
        )
        for held in plan.selected:
            blocked_reason = check_blocked_topic(held)
            if blocked_reason:
                await self._pipeline._reject_blocked_topic(held, blocked_reason)
                continue
            held.is_breaking = is_strict_breaking(held)
            logger.info(
                "[BALANCER] Held article %s ('%s') eligible under 5-of-8 plan: %s",
                held.id, held.original_title[:60], plan.reason,
            )
            await self._pipeline._publish_immediately(held)

    async def _fetch_from_source(self, source: BaseSource) -> list[RawArticle]:
        """Fetch articles from a single source with bounded concurrency."""
        async with self._source_semaphore:
            try:
                articles = await source.fetch_articles()
                await self._db.update_source_status(source.name, success=True)
                return articles
            except Exception as e:
                await self._db.update_source_status(source.name, success=False, error=str(e))
                raise

    async def _extract_and_insert(self, raw_articles: list[RawArticle]) -> list:
        """Extract article metadata and insert into database. Returns list of new articles."""
        from storage.models import Article as ArticleModel
        new_articles = []

        for raw in raw_articles:
            try:
                article = self._extractor.extract(raw)

                # Quick pre-filter: skip obviously bad articles
                if not article.canonical_url or not article.original_title:
                    continue

                # Skip URLs already rejected for low global relevance in previous runs
                if await self._db.is_url_skipped(article.canonical_url):
                    logger.debug("Skipping previously rejected URL: %s", article.canonical_url[:80])
                    continue

                # Try inserting (will fail silently for duplicates due to UNIQUE constraint)
                article_id = await self._db.insert_article(article)
                if article_id is not None:
                    article.id = article_id
                    new_articles.append(article)
                    logger.info(
                        "[DISCOVERED] Article %d from %s: '%s' (pub_date: %s)",
                        article.id, article.source_name, article.original_title[:80],
                        article.publication_date or "N/A",
                    )
                    self._metrics.increment("new_articles_inserted")

            except Exception as e:
                logger.debug("Article extraction error: %s", str(e)[:200])

        return new_articles

    # ───────────────────────────────────────────────
    # Recovery
    # ───────────────────────────────────────────────

    async def _process_recovered_articles(self) -> None:
        """Re-process articles that were recovered from interrupted states."""
        # Get articles in DISCOVERED or PROCESSING states
        discovered = await self._db.get_articles_by_status(ArticleStatus.DISCOVERED, limit=20)
        ready = await self._db.get_articles_by_status(ArticleStatus.READY_TO_PUBLISH, limit=20)
        retry = await self._db.get_articles_by_status(ArticleStatus.RETRY_PENDING, limit=10)

        all_to_process = discovered + ready + retry
        if all_to_process:
            logger.info("Processing %d recovered articles.", len(all_to_process))
            await self._pipeline.process_batch(all_to_process)

    # ───────────────────────────────────────────────
    # Background Tasks
    # ───────────────────────────────────────────────

    async def _backup_loop(self) -> None:
        """Periodic database backup."""
        while self._running:
            try:
                await asyncio.sleep(BACKUP_INTERVAL_HOURS * 3600)
                await self._db.backup()
                self._health.record_success("database")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Backup failed: %s", str(e)[:200])
                self._health.record_failure("database", str(e))

    async def _metrics_loop(self) -> None:
        """Log metrics periodically."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                self._metrics.log_report()
                stats = await self._db.get_stats()
                logger.info("DB stats: %s", stats)
                health = self._health.get_status()
                logger.info("Health: %s", health)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Metrics error: %s", str(e)[:100])
