"""
Kurdish Global News Bot — Article Processing Pipeline

Processes articles through validation, importance scoring, duplicate detection,
translation, and formatting — then immediately publishes eligible stories.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from datetime import datetime, timezone
from storage.database import Database
from storage.models import Article, Event
from core.importance_ranker import (
    ImportanceRanker,
    article_selection_sort_key,
    classify_priority_tier,
    is_bashur_article,
)
from core.balancer import QuotaBalancer
from core.duplicate_detector import DuplicateDetector
from processing.deduplicator import SemanticDeduplicator
from processing.extractor import ArticleExtractor
from processing.validator import ArticleValidator
from processing.translator import TranslationEngine, TranslationError
from processing.content_formatter import ContentFormatter
from tg_bot.publisher import TelegramPublisher
from monitoring.health import HealthMonitor
from monitoring.metrics import Metrics, Timer
from config import (
    ArticleStatus, Priority, MAX_RETRIES, AUTO_PUBLISH,
    MAX_CONCURRENT_ARTICLES, PRIMARY_CATEGORIES,
    PRIMARY_CATEGORY_RATIO, SECONDARY_CATEGORY_RATIO,
    FALLBACK_WINDOW_MINUTES, MAX_ARTICLE_AGE_MINUTES,
    BATCH_PUBLISH_LIMIT, is_primary_category,
)

logger = logging.getLogger("core.pipeline")


class ArticlePipeline:
    """
    Processes each article independently through the full pipeline:
    Validate → Score → Dedup → Translate → Publish (immediately).

    Each article is processed and published independently.
    No batching. No hourly schedule. No cooldowns.
    """

    def __init__(
        self,
        db: Database,
        publisher: TelegramPublisher,
        health: HealthMonitor,
        metrics: Metrics,
    ) -> None:
        self._db = db
        self._publisher = publisher
        self._health = health
        self._metrics = metrics
        self._extractor = ArticleExtractor()
        self._validator = ArticleValidator()
        self._ranker = ImportanceRanker()
        self._dedup = DuplicateDetector(db)
        self._semantic_dedup = SemanticDeduplicator(db)
        self._quota_balancer = QuotaBalancer(batch_limit=BATCH_PUBLISH_LIMIT)
        self._translator = TranslationEngine()
        self._formatter = ContentFormatter()
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_ARTICLES)
        self._cold_start: Optional[bool] = None

    def set_cold_start(self, enabled: bool) -> None:
        """Set whether cold start protection is active."""
        self._cold_start = enabled

    async def cleanup(self) -> None:
        """Clean up pipeline resources (ranker, semantic deduplicator, and translator sessions)."""
        await self._ranker.cleanup()
        await self._semantic_dedup.cleanup()
        await self._translator.cleanup()

    async def _check_cold_start(self) -> bool:
        """Check if cold start protection is active."""
        if self._cold_start is None:
            self._cold_start = False
        return self._cold_start

    async def process_article(self, article: Article) -> bool:
        """
        Process a single article through the full pipeline.
        Returns True if the article was successfully published.

        This runs IMMEDIATELY — does not wait for any batch or schedule.
        """
        async with self._semaphore:
            return await self._process_single(article)

    async def _evaluate_candidate(
        self,
        article: Article,
        extra_recent: Optional[list[dict]] = None,
    ) -> bool:
        """
        Run validation, scoring, local + semantic cross-source dedup, and Gemini
        global relevance classification BEFORE scraping/translation or slot allocation.
        """
        article_id = article.id
        logger.info(
            "[DISCOVERED] Processing article %s: '%s' (source: %s, pub_date: %s)",
            article_id, article.original_title[:80], article.source_name,
            article.publication_date or "N/A",
        )

        # ── Step 1: Validate ──
        await self._db.update_article_status(article_id, ArticleStatus.VALIDATING)
        is_cold = await self._check_cold_start()
        is_valid, rejection_reason = self._validator.validate(article, is_cold_start=is_cold)
        if not is_valid:
            await self._db.update_article_status(
                article_id, ArticleStatus.FAILED,
                error_message=f"Validation failed: {rejection_reason}",
            )
            logger.info("[FILTERED] Article %s rejected (%s): '%s'",
                        article_id, rejection_reason, article.original_title[:80])
            self._metrics.increment("articles_rejected")
            return False

        # ── Step 2: Score importance ──
        await self._db.update_article_status(article_id, ArticleStatus.PROCESSING)
        score, priority = self._ranker.score_article(article)
        article.importance_score = score
        article.priority = priority

        credibility = ArticleExtractor.get_source_credibility(article.source_name)
        if not self._ranker.is_eligible(score, credibility):
            await self._db.update_article_status(
                article_id, ArticleStatus.FAILED,
                error_message=f"Below importance threshold: score={score:.2f}",
                importance_score=score, priority=priority,
            )
            logger.info("[FILTERED] Article %s below threshold (score=%.2f, cred=%.2f): '%s'",
                        article_id, score, credibility, article.original_title[:80])
            self._metrics.increment("articles_below_threshold")
            return False

        # ── Step 3: Duplicate detection ──
        is_dup, dup_reason = await self._dedup.is_duplicate(article)
        if is_dup:
            await self._db.update_article_status(
                article_id, ArticleStatus.REJECTED_DUPLICATE,
                error_message=dup_reason,
                importance_score=score, priority=priority,
            )
            logger.info("[FILTERED] Article %s duplicate (%s): '%s'",
                        article_id, dup_reason, article.original_title[:80])
            self._metrics.increment("articles_duplicate")
            return False

        same_event, event_reason = await self._dedup.is_same_event_already_published(article)
        if same_event:
            is_update = await self._dedup.is_meaningful_update(article)
            if not is_update:
                await self._db.update_article_status(
                    article_id, ArticleStatus.REJECTED_DUPLICATE,
                    error_message=f"Same event: {event_reason}",
                    importance_score=score, priority=priority,
                )
                logger.info("[FILTERED] Article %s same event (%s): '%s'",
                            article_id, event_reason, article.original_title[:80])
                self._metrics.increment("articles_same_event")
                return False
            else:
                logger.info("Article %s is a meaningful update to existing event.", article_id)

        # ── Step 4: Global Importance & Relevance Gatekeeper (LLM Check) ──
        relevance = await self._ranker.evaluate_global_relevance(article)
        if not relevance.passed:
            reason_text = relevance.reason or "Below global relevance threshold"
            await self._db.update_article_status(
                article_id, ArticleStatus.SKIPPED,
                error_message=f"SKIPPED [Low Global Relevance {relevance.score}/10]: {reason_text}",
                importance_score=relevance.score / 10.0,
                priority=Priority.LOW,
            )
            await self._db.mark_url_skipped(
                article.canonical_url or article.original_url,
                title=article.original_title,
                score=float(relevance.score),
                reason=reason_text,
            )
            logger.info(
                "SKIPPED [Low Global Relevance %d/10]: %s — %s",
                relevance.score,
                article.original_title[:100],
                reason_text,
            )
            self._metrics.increment("articles_low_global_relevance")
            return False

        # Strict breaking gate: only breaking when BOTH is_breaking is True AND score >= 9.0
        article.importance_score = float(relevance.score)
        if not relevance.used_fallback:
            article.category = relevance.category
        article.is_breaking = bool(relevance.is_breaking and article.importance_score >= 9.0)

        if article.is_breaking and not relevance.used_fallback:
            priority = Priority.CRITICAL
            article.priority = Priority.CRITICAL
        elif not relevance.used_fallback and article.category == "kurdistan":
            if article.priority in (Priority.CRITICAL, Priority.NORMAL):
                priority = Priority.HIGH
                article.priority = Priority.HIGH
        elif not article.is_breaking and article.priority == Priority.CRITICAL and not relevance.used_fallback:
            priority = Priority.HIGH
            article.priority = Priority.HIGH

        # ── Step 4.5: Cross-Source Semantic Event Deduplication (After Classification, Before Scrape/Translate) ──
        dedup_res = await self._semantic_dedup.check_duplicate(article, extra_recent=extra_recent)
        if dedup_res.is_duplicate:
            matched_ref = f"#{dedup_res.matched_id}" if dedup_res.matched_id is not None else "#unknown"
            dedup_reason = dedup_res.reason or "Duplicate event"
            await self._db.update_article_status(
                article_id,
                ArticleStatus.REJECTED_DUPLICATE,
                error_message=f"SKIPPED [Duplicate Event]: matches {matched_ref} ({dedup_reason})",
                importance_score=relevance.score / 10.0,
                priority=priority,
            )
            await self._db.mark_url_skipped(
                article.canonical_url or article.original_url,
                title=article.original_title,
                score=float(relevance.score),
                reason=f"Duplicate Event {matched_ref}: {dedup_reason}",
            )
            logger.info(
                "SKIPPED [Duplicate Event]: %s — matches %s (%s)",
                article.original_title[:100],
                matched_ref,
                dedup_reason,
            )
            self._metrics.increment("articles_semantic_duplicate")
            return False

        return True

    async def _scrape_translate_and_publish(
        self,
        article: Article,
        check_category_balancer: bool = True,
    ) -> bool:
        """Run full-article scraping, translation, DB ready update, and Telegram delivery."""
        article_id = article.id
        priority = article.priority

        # ── Step 5: Extract full article text & og:image, verify depth ──
        full_text, og_image = await self._extractor.fetch_article_content_and_image(article.original_url)
        if not article.image_url and og_image:
            article.image_url = og_image

        if full_text and len(full_text) >= 160:
            article_content = full_text[:3000]
            article.original_summary = article_content
            logger.info(
                "Extracted %d chars of full article text from %s",
                len(article_content),
                article.original_url[:70],
            )
        else:
            fallback = (article.original_summary or "").strip()
            insufficient, reason = ArticleExtractor.is_insufficient_detail(
                article.original_title,
                fallback,
                min_len=160,
                max_similarity=70.0,
            )
            if insufficient:
                await self._db.update_article_status(
                    article_id, ArticleStatus.FAILED,
                    error_message=f"SKIPPED: insufficient article detail ({reason})",
                    importance_score=article.importance_score, priority=priority,
                )
                logger.info(
                    "[FILTERED] SKIPPED: insufficient article detail for article %s ('%s'): %s",
                    article_id, article.original_title[:80], reason,
                )
                self._metrics.increment("articles_insufficient_detail")
                return False
            article_content = fallback[:3000]

        # ── Step 5b: Translate to Kurdish Sorani & Gemini Proofread ──
        await self._db.update_article_status(article_id, ArticleStatus.TRANSLATING)
        try:
            with Timer(self._metrics, "translation_time"):
                article.translated_title, article.translated_summary = (
                    await self._translator.translate_article(
                        title=article.original_title,
                        summary=article_content,
                    )
                )

            title_ok = await self._translator.validate_translation(
                article.translated_title, article.original_title
            )
            if not title_ok:
                raise TranslationError("Title translation validation failed.")

            logger.info("[TRANSLATED] Article %s translated: '%s' -> '%s'",
                        article_id, article.original_title[:60], article.translated_title[:60])

        except TranslationError as e:
            retry_count = await self._db.increment_retry(article_id, str(e))
            if retry_count >= MAX_RETRIES:
                await self._db.update_article_status(
                    article_id, ArticleStatus.FAILED,
                    error_message=f"Translation failed after {MAX_RETRIES} retries: {e}",
                )
            self._health.record_failure("translation", str(e))
            self._metrics.increment("translation_failures")
            logger.warning("Translation failed for article %s: %s", article_id, str(e)[:200])
            return False

        # ── Step 6: Update article with all computed fields ──
        await self._db.update_article_status(
            article_id, ArticleStatus.READY_TO_PUBLISH,
            translated_title=article.translated_title,
            translated_summary=article.translated_summary,
            original_summary=article.original_summary,
            category=article.category,
            image_url=article.image_url,
            importance_score=article.importance_score,
            priority=priority,
        )

        # ── Step 6b: Publish immediately ──
        if AUTO_PUBLISH:
            if check_category_balancer:
                category_group = "PRIMARY" if is_primary_category(article.category) else "SECONDARY"
                publish_now, reason = await self.should_publish_now(article)
                if publish_now:
                    logger.info("[BALANCER] [%s] Publishing article %s ('%s'): %s",
                                category_group, article_id, article.category, reason)
                    return await self._publish_immediately(article)
                else:
                    logger.info("[BALANCER] [%s] Holding article %s ('%s'): %s",
                                category_group, article_id, article.category, reason)
                    return False
            return await self._publish_immediately(article)
        else:
            logger.info("Article %s ready but AUTO_PUBLISH is off.", article_id)
            return False

    async def _process_single(self, article: Article) -> bool:
        """Full pipeline for a single article."""
        start_time = time.time()
        article_id = article.id

        try:
            eligible = await self._evaluate_candidate(article)
            if not eligible:
                return False
            return await self._scrape_translate_and_publish(article, check_category_balancer=True)

        except Exception as e:
            logger.error("Pipeline error for article %s: %s", article_id, str(e)[:300])
            try:
                await self._db.update_article_status(
                    article_id, ArticleStatus.FAILED,
                    error_message=f"Pipeline error: {str(e)[:500]}",
                )
            except Exception:
                pass
            self._health.record_failure("pipeline", str(e))
            self._metrics.increment("pipeline_errors")
            return False
        finally:
            duration = time.time() - start_time
            self._metrics.record_timing("article_pipeline_time", duration)

    async def should_publish_now(self, article: Article) -> tuple[bool, str]:
        """
        Determine if an article should publish immediately based on the 60/40 balancer:
        1. CRITICAL priority breaking news from ANY category always publishes immediately.
        2. Primary focus articles (war, military, world leaders, geopolitics) always publish immediately.
        3. Secondary focus articles publish immediately if:
           a. Secondary share in the recent 20-article window is <= 40% (or < 5 recent articles exist), OR
           b. Automatic Fallback is active (no primary article published within FALLBACK_WINDOW_MINUTES).
        Returns (should_publish, reason_description).
        """
        # 1. CRITICAL breaking news from ANY category always passes immediately
        if article.priority == Priority.CRITICAL:
            return True, "CRITICAL breaking news bypasses category ratio check"

        # 2. Primary category articles always publish immediately
        if is_primary_category(article.category):
            return True, f"PRIMARY focus category ('{article.category}') publishes immediately"

        # 3. Secondary category articles: check rolling window balance (last 20 published)
        recent_published = await self._db.get_recent_published_articles(limit=20)
        total_recent = len(recent_published)

        # Small sample (< 5 published articles in DB): allow publishing
        if total_recent < 5:
            return True, f"Recent publishing history sample small ({total_recent} < 5 articles); secondary permitted"

        secondary_count = sum(1 for a in recent_published if not is_primary_category(a.category))
        secondary_ratio = secondary_count / total_recent

        if secondary_ratio <= SECONDARY_CATEGORY_RATIO:
            return True, (
                f"Secondary share ({secondary_count}/{total_recent} = {secondary_ratio:.1%}) "
                f"<= target threshold {SECONDARY_CATEGORY_RATIO:.0%}"
            )

        # 4. Secondary share is > 40%: check automatic fallback window
        last_primary_dt = await self._db.get_last_published_primary_time(PRIMARY_CATEGORIES)
        now = datetime.now(timezone.utc)

        if last_primary_dt is None:
            return True, "FALLBACK ACTIVE: No primary news in history; relaxing ratio for secondary news"

        elapsed_minutes = (now - last_primary_dt).total_seconds() / 60.0
        if elapsed_minutes >= FALLBACK_WINDOW_MINUTES:
            return True, (
                f"FALLBACK ACTIVE: No primary news published in the last {elapsed_minutes:.1f}m "
                f"(>= {FALLBACK_WINDOW_MINUTES}m threshold); relaxing ratio for secondary news"
            )

        return False, (
            f"RATIO HELD: Secondary share is {secondary_ratio:.1%} (> {SECONDARY_CATEGORY_RATIO:.0%}) "
            f"and primary news was published {elapsed_minutes:.1f}m ago (< {FALLBACK_WINDOW_MINUTES}m)"
        )

    async def _publish_immediately(self, article: Article) -> bool:
        """Publish an article to Telegram IMMEDIATELY. No batching. No waiting."""
        article_id = article.id

        try:
            await self._db.update_article_status(article_id, ArticleStatus.PUBLISHING)

            with Timer(self._metrics, "publish_time"):
                message_id = await self._publisher.publish_article(article)

            if message_id is not None:
                # Success — Telegram confirmed delivery
                await self._db.update_article_status(
                    article_id, ArticleStatus.PUBLISHED,
                    telegram_message_id=message_id,
                )

                # Track event
                if article.event_fingerprint:
                    event = Event(
                        event_fingerprint=article.event_fingerprint,
                        event_title=article.original_title[:200],
                    )
                    await self._db.upsert_event(event)

                self._health.record_success("publishing")
                self._metrics.increment("articles_published")
                logger.info(
                    "[PUBLISHED] Article %d published to Telegram: msg_id=%d, title='%s' (score=%.2f, priority=%s)",
                    article_id, message_id, article.original_title[:60],
                    article.importance_score, article.priority,
                )
                return True
            else:
                # Publish failed — no message_id returned
                retry_count = await self._db.increment_retry(
                    article_id, "Telegram publish returned no message_id"
                )
                self._health.record_failure("publishing", "No message_id returned")
                self._metrics.increment("publish_failures")
                return False

        except Exception as e:
            logger.error("Publish error for article %d: %s", article_id, str(e)[:200])
            await self._db.increment_retry(article_id, str(e)[:500])
            self._health.record_failure("publishing", str(e))
            self._metrics.increment("publish_failures")
            return False

    async def process_batch(
        self,
        articles: list[Article],
        max_publish: Optional[int] = None,
    ) -> dict[str, int]:
        """
        Evaluate candidates (filtering + cross-source batch dedup), enforce the
        elastic 5-of-8 Southern Kurdistan (Bashur) / Global quota, and publish
        selected winners in strict order:
          1. Genuine breaking emergencies
          2. Southern Kurdistan (Bashur) — Politics & War first
          3. Global & regional news
        """
        results = {"published": 0, "rejected": 0, "failed": 0}
        if not articles:
            return results

        effective_limit = BATCH_PUBLISH_LIMIT if max_publish is None else max(0, max_publish)

        # Evaluate highest-priority candidates first so duplicate outlets lose cleanly
        prioritized = sorted(articles, key=article_selection_sort_key)
        eligible: list[Article] = []
        accepted_batch_refs: list[dict] = []

        for art in prioritized:
            try:
                ok = await self._evaluate_candidate(art, extra_recent=accepted_batch_refs)
                if ok:
                    eligible.append(art)
                    accepted_batch_refs.append({
                        "id": art.id,
                        "original_title": art.original_title or "",
                        "translated_title": art.translated_title or "",
                        "category": art.category or "",
                        "published_at": art.published_at or art.discovered_at or "",
                    })
                else:
                    results["rejected"] += 1
            except Exception as exc:
                logger.error("Batch evaluation error for '%s': %s", art.original_title[:60], str(exc)[:200])
                results["failed"] += 1

        if not eligible:
            return results

        recent_published = await self._db.get_recent_published_articles(limit=8)
        plan = self._quota_balancer.select_batch(
            eligible,
            limit=effective_limit,
            recent_published=recent_published,
        )
        logger.info("[BALANCER] %s", plan.reason)

        reserves = sorted(plan.deferred, key=article_selection_sort_key)
        target_slots = len(plan.selected)

        for selected_art in plan.selected:
            try:
                published = await self._scrape_translate_and_publish(
                    selected_art, check_category_balancer=False
                )
                if published:
                    results["published"] += 1
                else:
                    results["rejected"] += 1
                    # Backfill empty slot elastically if reserve candidates exist
                    while reserves and results["published"] < target_slots:
                        backup = reserves.pop(0)
                        try:
                            if await self._scrape_translate_and_publish(backup, check_category_balancer=False):
                                results["published"] += 1
                                break
                            results["rejected"] += 1
                        except Exception as exc:
                            logger.error("Backup publish error: %s", str(exc)[:200])
                            results["failed"] += 1
            except Exception as exc:
                logger.error("Batch publish error for article %s: %s", selected_art.id, str(exc)[:200])
                results["failed"] += 1

        for leftover in reserves:
            try:
                await self._db.update_article_status(
                    leftover.id,
                    ArticleStatus.SKIPPED,
                    error_message="Deferred: outside 5-of-8 batch quota slots",
                    importance_score=leftover.importance_score,
                    priority=leftover.priority,
                )
            except Exception:
                pass
            results["rejected"] += 1

        return results
