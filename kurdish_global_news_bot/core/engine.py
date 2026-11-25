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
from storage.models import Article
from storage.utils import (
    normalize_event_entities,
    normalize_event_key,
    pack_event_metadata,
)
from core.scheduler import DiscoveryScheduler
from core.importance_ranker import (
    article_selection_sort_key,
    check_blocked_topic,
    is_bashur_article,
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
from sources.x_video_source import XVideoSource
from processing.deduplicator import OUTCOME_THREAD_UPDATE
from processing.extractor import ArticleExtractor
from processing.card_generator import NewsCardGenerator
from processing.digest_generator import DIGEST_HEADER_PREFIX, DailyDigestGenerator
from processing.translator import is_valid_kurdish_sorani
from tg_bot.publisher import TelegramPublisher
from monitoring.health import HealthMonitor
from monitoring.metrics import Metrics, Timer
from monitoring.watchdog import AdminWatchdog, WatchdogCycleMetrics, redact_secrets
import config
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
        self._watchdog: Optional[AdminWatchdog] = None
        self._supabase_fallback_active: bool = False
        self._supabase_fallback_error: str = ""

    async def _connect_db_with_fallback(self) -> None:
        """Connect configured database, falling back cleanly to local SQLite if Supabase fails."""
        from storage.database import Database
        from storage.supabase_db import SupabaseDatabase

        self._supabase_fallback_active = False
        self._supabase_fallback_error = ""
        if isinstance(self._db, SupabaseDatabase):
            try:
                await self._db.connect()
                return
            except Exception as exc:
                self._supabase_fallback_active = True
                self._supabase_fallback_error = redact_secrets(str(exc))[:300]
                logger.error(
                    "Supabase connection failed (%s); falling back cleanly to local SQLite (%s).",
                    self._supabase_fallback_error,
                    config.DATABASE_PATH,
                )
                try:
                    await self._db.close()
                except Exception:
                    pass
                self._db = Database(db_path=config.DATABASE_PATH)
                await self._db.connect()
                return

        await self._db.connect()

    def _active_storage_backend_label(self) -> str:
        from storage.supabase_db import SupabaseDatabase
        if self._supabase_fallback_active:
            return "SQLite fallback"
        if isinstance(self._db, SupabaseDatabase):
            return "Supabase"
        return "SQLite"

    async def start(self) -> None:
        """Initialize all components and start the engine."""
        logger.info("╔══════════════════════════════════════════╗")
        logger.info("║   Kurdish Global News Bot — Starting     ║")
        logger.info("╚══════════════════════════════════════════╝")

        # 1. Connect database
        await self._connect_db_with_fallback()
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
        self._watchdog = AdminWatchdog(db=self._db, publisher=self._publisher)

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
        await self._connect_db_with_fallback()
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
        self._watchdog = AdminWatchdog(db=self._db, publisher=self._publisher)

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
        except Exception as exc:
            if self._watchdog is not None:
                await self._watchdog.notify_pipeline_crash(exc)
            raise
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

        if getattr(config, "ENABLE_X_VIDEO_SOURCE", True):
            x_video_src = XVideoSource()
            await x_video_src.initialize()
            self._sources.append(x_video_src)

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
            recent_history: list[dict] = []
            try:
                recent_history = await self._db.get_recent_published_headlines(hours=DEDUP_HISTORY_HOURS)
                if self._pipeline is not None and hasattr(self._pipeline, "set_cycle_published_history"):
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
                if self._pipeline is not None and hasattr(self._pipeline, "set_cycle_published_history"):
                    self._pipeline.set_cycle_published_history(None)

            try:
                published_this_cycle = 0
                raw_discovered_count = 0

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

                raw_discovered_count = len(all_raw_articles)
                self._metrics.increment("total_raw_articles", raw_discovered_count)
                logger.info("Discovered %d raw articles from %d sources.",
                             raw_discovered_count, len(self._sources))

                if config.ENABLE_SMART_PACING and isinstance(self._pipeline, ArticlePipeline):
                    pacing_results = await self._run_smart_pacing_cycle(
                        all_raw_articles,
                        recent_history=recent_history,
                    )
                    if isinstance(pacing_results, dict):
                        published_this_cycle = int(pacing_results.get("published", 0) or 0)
                    logger.info("Smart pacing cycle results: %s", pacing_results)
                else:
                    if all_raw_articles:
                        new_articles = await self._extract_and_insert(all_raw_articles)
                        logger.info("%d new articles to process.", len(new_articles))

                        if new_articles:
                            if self._pipeline is not None and hasattr(self._pipeline, "set_cycle_corroboration_pool"):
                                self._pipeline.set_cycle_corroboration_pool(new_articles)
                            capped_articles, _ = await self._apply_local_prefilter(new_articles)
                            if capped_articles:
                                # Enforce 5-of-8 Bashur/Global quota & publish ordered winners
                                pipeline_results = await self._pipeline.process_batch(
                                    capped_articles, max_publish=BATCH_PUBLISH_LIMIT
                                )
                                if isinstance(pipeline_results, dict):
                                    published_this_cycle = int(pipeline_results.get("published", 0) or 0)
                                logger.info("Pipeline results: %s", pipeline_results)

                    # Check and publish held articles within remaining cycle capacity
                    remaining_capacity = max(0, BATCH_PUBLISH_LIMIT - published_this_cycle)
                    if remaining_capacity > 0:
                        await self._publish_held_articles(max_additional=remaining_capacity)

                # Cold start protection completed after first discovery pass
                if self._pipeline is not None and hasattr(self._pipeline, "set_cold_start"):
                    self._pipeline.set_cold_start(False)

                # Private Admin Watchdog alerts & daily health report (ZERO Gemini calls, non-blocking)
                if self._watchdog is not None:
                    raw_gem_stats = (
                        self._pipeline.consume_gemini_cycle_stats()
                        if self._pipeline is not None and hasattr(self._pipeline, "consume_gemini_cycle_stats")
                        else {}
                    )
                    gem_stats = raw_gem_stats if isinstance(raw_gem_stats, dict) else {}
                    wd_metrics = WatchdogCycleMetrics(
                        discovered=raw_discovered_count,
                        published=published_this_cycle,
                        gemini_429_hits=int(gem_stats.get("http_429_hits", 0) or 0),
                        gemini_exhausted=int(gem_stats.get("quota_exhausted_events", 0) or 0),
                        supabase_fallback=self._supabase_fallback_active,
                        supabase_error=self._supabase_fallback_error,
                        storage_backend=self._active_storage_backend_label(),
                        discovery_ran=True,
                    )
                    await self._watchdog.run_cycle_checks(wd_metrics)
            finally:
                if self._pipeline is not None and hasattr(self._pipeline, "set_cycle_published_history"):
                    self._pipeline.set_cycle_published_history(None)
                if self._pipeline is not None and hasattr(self._pipeline, "set_cycle_corroboration_pool"):
                    self._pipeline.set_cycle_corroboration_pool(None)

    async def _apply_local_prefilter(self, new_articles: list[Article]) -> tuple[list[Article], int]:
        """Pre-filter obvious off-topic items & cap fresh relevant candidates before Gemini."""
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
        return capped_articles, len(locally_rejected)

    @staticmethod
    def _build_queue_dedup_ref(art: Article) -> dict:
        """Build a normalized dedup reference dictionary for a pre-translated queued article."""
        ev_key = normalize_event_key(getattr(art, "event_key", ""))
        ev_entities = normalize_event_entities(getattr(art, "event_entities", []))
        return {
            "id": art.id,
            "source_name": art.source_name or "",
            "original_title": art.original_title or "",
            "translated_title": art.translated_title or "",
            "translated_summary": art.translated_summary or "",
            "original_summary": art.original_summary or "",
            "category": art.category or "",
            "published_at": art.processed_at or art.discovered_at or "",
            "telegram_message_id": getattr(art, "telegram_message_id", None),
            "thread_root_message_id": getattr(art, "thread_root_message_id", None),
            "is_published": False,
            "event_key": ev_key,
            "event_entities": ev_entities,
            "named_entities": art.named_entities or pack_event_metadata(ev_key, ev_entities),
        }

    async def _enqueue_article(self, article: Article) -> None:
        """Persist a prepared article as QUEUED without resetting an already-QUEUED timestamp."""
        if article.status == ArticleStatus.QUEUED or article.id is None:
            return
        ev_key = normalize_event_key(getattr(article, "event_key", ""))
        ev_entities = normalize_event_entities(getattr(article, "event_entities", []))
        update_fields: dict[str, object] = {
            "translated_title": article.translated_title,
            "translated_summary": article.translated_summary,
            "original_summary": article.original_summary,
            "category": article.category,
            "region": article.region,
            "importance_score": article.importance_score,
            "priority": article.priority,
            "is_breaking": bool(getattr(article, "is_breaking", False)),
            "image_url": article.image_url,
            "named_entities": article.named_entities or pack_event_metadata(ev_key, ev_entities),
            "event_key": ev_key,
            "event_entities": ev_entities,
            "thread_root_message_id": article.thread_root_message_id,
        }
        if getattr(article, "video_url", None):
            update_fields["video_url"] = article.video_url
        await self._db.update_article_status(
            article.id,
            ArticleStatus.QUEUED,
            error_message="Queued by Smart Pacing",
            **update_fields,
        )
        article.status = ArticleStatus.QUEUED

    async def _validate_queued_candidates(
        self,
        raw_queued: list[Article],
        recent_history: list[dict],
    ) -> tuple[list[Article], list[dict]]:
        """
        Re-verify queued (and any held READY_TO_PUBLISH) articles using 0 Gemini calls:
        blocked topic, Sorani Kurdish gate, and post-queue duplicate/thread-cap re-check.
        """
        assert self._pipeline is not None
        valid_queued: list[Article] = []
        queued_refs: list[dict] = []

        for q_art in raw_queued:
            if q_art.id is None:
                continue
            blocked_reason = check_blocked_topic(q_art)
            if blocked_reason:
                await self._pipeline._reject_blocked_topic(q_art, blocked_reason)
                continue

            if not is_valid_kurdish_sorani(q_art.translated_title or "") or (
                (q_art.translated_summary or "").strip()
                and not is_valid_kurdish_sorani(q_art.translated_summary or "")
            ):
                await self._db.update_article_status(
                    q_art.id,
                    ArticleStatus.EXPIRED,
                    error_message="Expired from queue: failed strict Sorani gate",
                )
                logger.info("[QUEUE EXPIRED] Article %s failed Sorani gate; skipped.", q_art.id)
                continue

            q_art.is_breaking = is_strict_breaking(q_art)
            q_art.is_thread_update = (q_art.thread_root_message_id is not None)

            q_dup = await self._pipeline._semantic_dedup.check_post_translation_duplicate(
                q_art,
                extra_recent=queued_refs,
                recent_history=recent_history,
            )
            if q_dup.is_duplicate:
                if q_dup.stage == "thread_cooldown":
                    queued_refs.append(self._build_queue_dedup_ref(q_art))
                    continue
                await self._db.update_article_status(
                    q_art.id,
                    ArticleStatus.EXPIRED,
                    error_message=f"Expired from queue: {q_dup.reason[:400]}",
                )
                logger.info(
                    "[QUEUE EXPIRED] Article %s invalidated while queued (%s).",
                    q_art.id,
                    q_dup.reason[:100],
                )
                continue

            if q_dup.classification == OUTCOME_THREAD_UPDATE and q_dup.thread_root_message_id is not None:
                q_art.thread_root_message_id = int(q_dup.thread_root_message_id)
                q_art.is_thread_update = True

            ref = self._build_queue_dedup_ref(q_art)
            queued_refs.append(ref)
            self._pipeline._cycle_accepted_refs.append(ref)
            valid_queued.append(q_art)

        return valid_queued, queued_refs

    async def _run_smart_pacing_cycle(
        self,
        all_raw_articles: list[RawArticle],
        recent_history: list[dict],
    ) -> dict[str, int]:
        """
        Execute Smart Pacing & Persistent Queue orchestration:
          1. Expire stale queued articles (> MAX_QUEUED_ARTICLE_AGE_HOURS) & re-validate backlog
          2. Discover, evaluate, and translate eligible new stories (queued URLs skipped automatically)
          3. Publish breaking emergencies immediately (0 delay, bypass caps/cooldown)
          4. Publish up to MAX_NORMAL_POSTS_PER_CYCLE balanced normal posts with intra-cycle delay
          5. Persist remaining normal posts as QUEUED (capped at MAX_QUEUE_SIZE) for next cycle
        """
        assert isinstance(self._pipeline, ArticlePipeline)
        counts = {"published": 0, "queued": 0, "expired": 0, "rejected": 0}

        # 1. Queue maintenance at cycle start
        counts["expired"] += await self._db.expire_stale_queued_articles(
            config.MAX_QUEUED_ARTICLE_AGE_HOURS
        )
        raw_queued = await self._db.get_queued_articles(
            max_age_hours=config.MAX_QUEUED_ARTICLE_AGE_HOURS
        )
        held_ready = await self._db.get_articles_by_status(ArticleStatus.READY_TO_PUBLISH, limit=20)
        seen_ids = {a.id for a in raw_queued if a.id is not None}
        combined_backlog = list(raw_queued) + [
            a for a in held_ready if a.id is not None and a.id not in seen_ids
        ]
        valid_queued, queued_refs = await self._validate_queued_candidates(
            combined_backlog,
            recent_history=recent_history,
        )

        # 2. Discover & prepare eligible newly discovered articles
        newly_ready: list[Article] = []
        if all_raw_articles:
            new_articles = await self._extract_and_insert(all_raw_articles)
            logger.info("%d new articles to process.", len(new_articles))
            if new_articles:
                self._pipeline.set_cycle_corroboration_pool(new_articles)
                capped_articles, rej_count = await self._apply_local_prefilter(new_articles)
                counts["rejected"] += rej_count
                if capped_articles:
                    eligible_new, eval_counts = await self._pipeline.evaluate_batch_candidates(
                        capped_articles,
                        extra_queue_refs=queued_refs,
                    )
                    counts["rejected"] += eval_counts.get("rejected", 0)

                    sorted_eligible = sorted(
                        eligible_new,
                        key=lambda a: (0 if is_strict_breaking(a) else 1, *article_selection_sort_key(a)),
                    )
                    non_breaking_prep_cap = max(
                        BATCH_PUBLISH_LIMIT,
                        config.MAX_NORMAL_POSTS_PER_CYCLE + config.MAX_QUEUE_SIZE - len(valid_queued),
                    )
                    to_prepare: list[Article] = []
                    non_breaking_seen = 0
                    for art in sorted_eligible:
                        if is_strict_breaking(art):
                            to_prepare.append(art)
                        elif non_breaking_seen < non_breaking_prep_cap:
                            to_prepare.append(art)
                            non_breaking_seen += 1
                        else:
                            await self._db.update_article_status(
                                art.id,
                                ArticleStatus.SKIPPED,
                                error_message="Deferred: cycle queue capacity full",
                                importance_score=art.importance_score,
                                priority=art.priority,
                            )
                            counts["rejected"] += 1

                    for art in to_prepare:
                        try:
                            if await self._pipeline.prepare_article_for_publishing(art):
                                art.is_breaking = is_strict_breaking(art)
                                newly_ready.append(art)
                            else:
                                counts["rejected"] += 1
                        except Exception as exc:
                            logger.error("Prepare error for article %s: %s", art.id, str(exc)[:200])

        # 3. Merge & partition publishing pool (Breaking fast-lane vs Normal paced posts)
        combined_pool = valid_queued + newly_ready
        if not combined_pool:
            return counts

        breaking_pool = sorted(
            [a for a in combined_pool if is_strict_breaking(a)],
            key=lambda a: (
                0 if is_bashur_article(a) else 1,
                -float(getattr(a, "importance_score", 0.0) or 0.0),
                a.published_at or a.discovered_at or "",
            ),
        )
        normal_pool = [a for a in combined_pool if not is_strict_breaking(a)]

        # 4a. Breaking Fast-Lane: publish immediately with ZERO delay, bypassing cycle cap & cooldown
        for brk_art in breaking_pool:
            if not config.AUTO_PUBLISH:
                await self._enqueue_article(brk_art)
                counts["queued"] += 1
                continue
            if await self._pipeline._publish_immediately(brk_art):
                counts["published"] += 1

        # 4b. Paced Normal Posts (and non-breaking THREAD_UPDATEs)
        if normal_pool:
            to_queue: list[Article] = []
            selected_normals: list[Article] = []

            if not config.AUTO_PUBLISH:
                to_queue = list(normal_pool)
            else:
                last_normal_dt = await self._db.get_last_normal_published_at()
                cooldown_active = False
                elapsed_min = 0.0
                if last_normal_dt is not None:
                    elapsed_min = max(
                        0.0, (datetime.now(timezone.utc) - last_normal_dt).total_seconds() / 60.0
                    )
                    cooldown_active = elapsed_min < config.MIN_MINUTES_BETWEEN_NORMAL_POSTS

                if cooldown_active:
                    logger.info(
                        "[PACING] Cross-cycle cooldown active (%.1fm < %.1fm since last normal post); "
                        "deferring %d normal article(s) to persistent queue.",
                        elapsed_min,
                        config.MIN_MINUTES_BETWEEN_NORMAL_POSTS,
                        len(normal_pool),
                    )
                    to_queue = list(normal_pool)
                else:
                    recent_published = await self._db.get_recent_published_articles(limit=8)
                    plan = self._pipeline._quota_balancer.select_batch(
                        normal_pool,
                        limit=config.MAX_NORMAL_POSTS_PER_CYCLE,
                        recent_published=recent_published,
                    )
                    logger.info("[BALANCER/PACING] %s", plan.reason)
                    selected_normals = list(plan.selected)
                    to_queue = list(plan.deferred)

            for idx, norm_art in enumerate(selected_normals):
                if idx > 0 and config.INTRA_CYCLE_POST_DELAY_SECONDS > 0:
                    logger.info(
                        "[PACING] Delaying %.0fs before normal post %d/%d...",
                        config.INTRA_CYCLE_POST_DELAY_SECONDS,
                        idx + 1,
                        len(selected_normals),
                    )
                    await asyncio.sleep(config.INTRA_CYCLE_POST_DELAY_SECONDS)
                published = await self._pipeline._publish_immediately(norm_art)
                if published:
                    counts["published"] += 1
                elif norm_art.status in (ArticleStatus.READY_TO_PUBLISH, ArticleStatus.QUEUED):
                    to_queue.append(norm_art)

            for q_art in to_queue:
                await self._enqueue_article(q_art)
                if q_art.status == ArticleStatus.QUEUED:
                    counts["queued"] += 1

            counts["expired"] += await self._db.trim_queue(config.MAX_QUEUE_SIZE)

        return counts

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
            max_rec = (
                config.MAX_NORMAL_POSTS_PER_CYCLE
                if config.ENABLE_SMART_PACING
                else BATCH_PUBLISH_LIMIT
            )
            await self._pipeline.process_batch(all_to_process, max_publish=max_rec)

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
