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
from urllib.parse import urlparse
from storage.database import Database
from storage.models import Article, Event
from storage.utils import (
    normalize_event_entities,
    normalize_event_key,
    pack_event_metadata,
)
from core.importance_ranker import (
    ImportanceRanker,
    RelevanceEvaluation,
    article_selection_sort_key,
    check_blocked_topic,
    check_pre_filter_off_topic,
    classify_priority_tier,
    is_bashur_article,
    is_genuine_kurdistan_event,
    select_top_candidates_for_gemini,
)
from core.balancer import QuotaBalancer
from core.duplicate_detector import DuplicateDetector
from core.source_verifier import (
    TIER_3,
    apply_verification_guard,
    compute_corroboration,
    get_source_tier,
)
from processing.deduplicator import (
    OUTCOME_DUPLICATE,
    OUTCOME_NEW_STORY,
    OUTCOME_THREAD_UPDATE,
    SemanticDeduplicator,
)
from processing.extractor import ArticleExtractor
from processing.validator import ArticleValidator
from processing.translator import TranslationEngine, TranslationError, is_valid_kurdish_sorani
from processing.content_formatter import (
    ContentFormatter,
    normalize_neutral_terminology,
    prepend_tier3_attribution,
    strip_rss_boilerplate,
)
from tg_bot.publisher import TelegramPublisher
from monitoring.health import HealthMonitor
from monitoring.metrics import Metrics, Timer
import config
from config import (
    ArticleStatus, Priority, MAX_RETRIES, AUTO_PUBLISH,
    MAX_CONCURRENT_ARTICLES, PRIMARY_CATEGORIES,
    PRIMARY_CATEGORY_RATIO, SECONDARY_CATEGORY_RATIO,
    FALLBACK_WINDOW_MINUTES, MAX_ARTICLE_AGE_MINUTES,
    BATCH_PUBLISH_LIMIT, GEMINI_RANK_BATCH_SIZE, MAX_GEMINI_CANDIDATES_PER_CYCLE,
    DEDUP_HISTORY_HOURS,
    is_primary_category,
)

logger = logging.getLogger("core.pipeline")


def _is_telegram_channel_url(url: str) -> bool:
    """Return True for public Telegram post links so watermarked preview screenshots are never scraped."""
    try:
        host = (urlparse((url or "").strip()).netloc or "").lower()
    except Exception:
        return False
    if host.startswith("www."):
        host = host[4:]
    return host in {"t.me", "telegram.me", "telegram.dog"}


def _is_transient_quota_or_sorani_error(err_msg: str) -> bool:
    """Return True when translation failed due to API 429/quota exhaustion or temporary timeout."""
    lower = (err_msg or "").lower()
    return any(
        tok in lower
        for tok in ("429", "quota", "cooldown", "exhausted", "rate limit", "timed out", "timeout")
    )


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
        # Share one per-key 429 cooldown map between Gemini ranking and translation
        self._gemini_cooldowns: dict[str, float] = {}
        self._ranker = ImportanceRanker(cooldown_store=self._gemini_cooldowns)
        self._dedup = DuplicateDetector(db)
        self._semantic_dedup = SemanticDeduplicator(db)
        self._quota_balancer = QuotaBalancer(batch_limit=BATCH_PUBLISH_LIMIT)
        self._translator = TranslationEngine(cooldown_store=self._gemini_cooldowns)
        self._formatter = ContentFormatter()
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_ARTICLES)
        self._cold_start: Optional[bool] = None
        self._cycle_published_history: Optional[list[dict]] = None
        self._cycle_accepted_refs: list[dict] = []
        self._cycle_corroboration_pool: Optional[list[Article]] = None

    def set_cold_start(self, enabled: bool) -> None:
        """Set whether cold start protection is active."""
        self._cold_start = enabled

    def set_cycle_published_history(self, history: Optional[list[dict]]) -> None:
        """Cache the 48-72h persistent published history for the active cycle."""
        self._cycle_published_history = list(history) if history is not None else None
        self._cycle_accepted_refs = []

    def set_cycle_corroboration_pool(self, articles: Optional[list[Article]]) -> None:
        """Cache full pre-dedup candidate articles for zero-API cross-source corroboration."""
        self._cycle_corroboration_pool = list(articles) if articles is not None else None

    async def _get_published_history(self) -> list[dict]:
        """Return cached cycle history or load last 72h published history from storage."""
        if self._cycle_published_history is not None:
            return self._cycle_published_history
        return await self._db.get_recent_published_headlines(hours=DEDUP_HISTORY_HOURS)

    def _record_cycle_published_article(self, article: Article, is_published: bool = True) -> None:
        """Record a prepared or published article into same-cycle dedup state."""
        ev_key = normalize_event_key(getattr(article, "event_key", ""))
        ev_entities = normalize_event_entities(getattr(article, "event_entities", []))
        ref = {
            "id": article.id,
            "source_name": getattr(article, "source_name", "") or "",
            "original_title": article.original_title or "",
            "translated_title": article.translated_title or "",
            "translated_summary": article.translated_summary or "",
            "original_summary": article.original_summary or "",
            "category": article.category or "",
            "published_at": article.published_at or datetime.now(timezone.utc).isoformat(),
            "telegram_message_id": getattr(article, "telegram_message_id", None),
            "thread_root_message_id": getattr(article, "thread_root_message_id", None),
            "is_published": is_published,
            "event_key": ev_key,
            "event_entities": ev_entities,
            "named_entities": article.named_entities or pack_event_metadata(ev_key, ev_entities),
        }
        if article.id is not None:
            self._cycle_accepted_refs = [
                r for r in self._cycle_accepted_refs if r.get("id") != article.id
            ]
        self._cycle_accepted_refs.append(ref)
        if is_published and self._cycle_published_history is not None:
            self._cycle_published_history.insert(0, ref)

    async def _reject_duplicate_event(
        self,
        article: Article,
        matched_id: Optional[int],
        reason: str,
        score: float = 0.0,
        matched_title: str = "",
    ) -> bool:
        """Mark article as REJECTED_DUPLICATE, log cleanly, and record canonical URL in skipped_urls."""
        matched_ref = f"#{matched_id}" if matched_id is not None else "#unknown"
        dedup_reason = reason or "Duplicate event"
        canonical_target = Article.canonicalize_url(article.canonical_url or article.original_url)
        await self._db.update_article_status(
            article.id,
            ArticleStatus.REJECTED_DUPLICATE,
            error_message=f"SKIPPED [Duplicate Event]: matches {matched_ref} ({dedup_reason})",
            importance_score=float(score or article.importance_score or 0.0),
            priority=article.priority,
        )
        await self._db.mark_url_skipped(
            canonical_target,
            title=article.original_title,
            score=float(score or 0.0),
            reason=f"Duplicate Event {matched_ref}: {dedup_reason}"[:500],
        )
        logger.info(
            "SKIPPED [Duplicate Event]: %s — matches %s (%s)",
            article.original_title[:100],
            matched_ref,
            dedup_reason,
        )
        logger.info(
            "Duplicate story blocked: matched '%s' (%s)",
            matched_title or matched_ref,
            dedup_reason,
        )
        self._metrics.increment("articles_semantic_duplicate")
        return False

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

    async def _reject_blocked_topic(self, article: Article, reason_text: str) -> bool:
        """Mark an article SKIPPED and record its URL hash in skipped_urls."""
        await self._db.update_article_status(
            article.id,
            ArticleStatus.SKIPPED,
            error_message=f"SKIPPED [Blocked Topic]: {reason_text}",
            importance_score=0.0,
            priority=Priority.LOW,
        )
        await self._db.mark_url_skipped(
            article.canonical_url or article.original_url,
            title=article.original_title,
            score=2.0,
            reason=reason_text,
        )
        logger.info(
            "SKIPPED [Blocked Topic]: %s — %s",
            article.original_title[:100],
            reason_text,
        )
        self._metrics.increment("articles_blocked_topic")
        return False

    async def _local_pre_screen_candidate(
        self,
        article: Article,
        extra_recent: Optional[list[dict]] = None,
        use_off_topic_prefilter: bool = True,
    ) -> bool:
        """
        Zero-API local validation, topic pre-filtering, heuristic scoring, and
        cross-source deduplication BEFORE sending any candidate to Gemini.
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

        # ── Deterministic Blocked / Off-Topic Guard (Pre-LLM) ──
        blocked_reason = (
            check_pre_filter_off_topic(article)
            if use_off_topic_prefilter
            else check_blocked_topic(article)
        )
        if blocked_reason:
            return await self._reject_blocked_topic(article, blocked_reason)

        # ── Step 2: Score importance ──
        await self._db.update_article_status(article_id, ArticleStatus.PROCESSING)
        score, priority = self._ranker.score_article(article)
        article.importance_score = score
        article.priority = priority

        credibility = ArticleExtractor.get_source_credibility(article.source_name)
        if not self._ranker.is_eligible(score, credibility, article=article):
            await self._db.update_article_status(
                article_id, ArticleStatus.FAILED,
                error_message=f"Below importance threshold: score={score:.2f}",
                importance_score=score, priority=priority,
            )
            logger.info("[FILTERED] Article %s below threshold (score=%.2f, cred=%.2f): '%s'",
                        article_id, score, credibility, article.original_title[:80])
            self._metrics.increment("articles_below_threshold")
            return False

        # ── Step 3: Duplicate detection (Zero-API local + cross-source) ──
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

        recent_history = await self._get_published_history()
        combined_extra = list(extra_recent or []) + list(self._cycle_accepted_refs)
        dedup_res = await self._semantic_dedup.check_duplicate(
            article,
            extra_recent=combined_extra,
            recent_history=recent_history,
            allow_pending_cooldown=True,
        )

        same_event, event_reason = await self._dedup.is_same_event_already_published(article)
        if same_event and dedup_res.classification != OUTCOME_THREAD_UPDATE:
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

        if dedup_res.is_duplicate:
            return await self._reject_duplicate_event(
                article,
                matched_id=dedup_res.matched_id,
                reason=dedup_res.reason,
                score=float(score),
                matched_title=dedup_res.matched_title,
            )

        if dedup_res.classification == OUTCOME_THREAD_UPDATE and dedup_res.thread_root_message_id is not None:
            article.thread_root_message_id = int(dedup_res.thread_root_message_id)
            article.is_thread_update = True
            logger.info(
                "[THREAD_UPDATE] Candidate %s earmarked as reply to root msg_id=%s (%s)",
                article_id,
                article.thread_root_message_id,
                dedup_res.delta_reason or dedup_res.reason,
            )

        return True

    @staticmethod
    def _populate_relevance_fields(article: Article, relevance: RelevanceEvaluation) -> None:
        """Populate score, strict breaking gate, event metadata, and priority from relevance output."""
        if getattr(article, "_relevance_populated", False):
            return
        article.importance_score = float(relevance.score)
        is_kurd_content = is_genuine_kurdistan_event(article)
        if not relevance.used_fallback:
            article.category = relevance.category
        elif (article.category or "").strip().lower() in {"kurdistan", "کوردستان", "كردستان"} and not is_kurd_content:
            # Never retain a Kurdistan feed tag on non-Kurdistan international stories during fallback
            article.category = relevance.category
        if not is_kurd_content and (article.region or "").strip().lower() in {
            "south kurdistan", "kurdistan", "kurdistan_bashur", "کوردستان", "کوردستان (باشوور)",
        }:
            article.region = ""
        article.is_breaking = bool(relevance.is_breaking and article.importance_score >= 9.0)

        ev_key = normalize_event_key(getattr(relevance, "event_key", ""))
        ev_entities = normalize_event_entities(getattr(relevance, "event_entities", []))
        if ev_key:
            article.event_key = ev_key
        if ev_entities:
            article.event_entities = ev_entities
        if getattr(relevance, "update_type", ""):
            article.update_type = relevance.update_type  # type: ignore[attr-defined]
        if article.event_key or article.event_entities:
            article.named_entities = pack_event_metadata(article.event_key, article.event_entities)

        if article.is_breaking and not relevance.used_fallback:
            article.priority = Priority.CRITICAL
        elif not relevance.used_fallback and article.category == "kurdistan":
            if article.priority in (Priority.CRITICAL, Priority.NORMAL):
                article.priority = Priority.HIGH
        elif not article.is_breaking and article.priority == Priority.CRITICAL and not relevance.used_fallback:
            article.priority = Priority.HIGH
        article._relevance_populated = True  # type: ignore[attr-defined]

    async def _apply_relevance_evaluation(
        self,
        article: Article,
        relevance: RelevanceEvaluation,
        extra_recent: Optional[list[dict]] = None,
    ) -> bool:
        """Apply Gemini relevance threshold, event_key metadata, verification guard, and blocked-topic check."""
        article_id = article.id

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

        self._populate_relevance_fields(article, relevance)

        # Second-layer deterministic blocked topic guard behind Gemini & before scraping
        post_blocked = check_blocked_topic(article)
        if post_blocked:
            return await self._reject_blocked_topic(article, post_blocked)

        # Single-source verification guard (blocks uncorroborated Tier-3 breaking alerts, ZERO API calls)
        if getattr(config, "ENABLE_SOURCE_VERIFICATION", True) and not getattr(article, "_verification_checked", False):
            recent_hist = await self._get_published_history()
            combined_hist = list(extra_recent or []) + list(self._cycle_accepted_refs) + recent_hist
            pool = list(self._cycle_corroboration_pool or [article])
            corrob = compute_corroboration(article, batch_articles=pool, recent_history=combined_hist)
            apply_verification_guard(article, corrob)

        # Zero-extra-API three-way check (NEW_STORY | THREAD_UPDATE | DUPLICATE) using
        # Gemini's English event_key/event_entities & final score/is_breaking cooldown rules
        if (
            article.event_key
            or article.event_entities
            or article.is_thread_update
            or article.thread_root_message_id is not None
        ):
            recent_history = await self._get_published_history()
            combined_extra = list(extra_recent or []) + list(self._cycle_accepted_refs)
            ev_dup = await self._semantic_dedup.check_event_match(
                article, extra_recent=combined_extra, recent_history=recent_history
            )
            if ev_dup.is_duplicate:
                article.thread_root_message_id = None
                article.is_thread_update = False
                return await self._reject_duplicate_event(
                    article,
                    matched_id=ev_dup.matched_id,
                    reason=ev_dup.reason,
                    score=float(article.importance_score),
                    matched_title=ev_dup.matched_title,
                )
            if ev_dup.classification == OUTCOME_THREAD_UPDATE and ev_dup.thread_root_message_id is not None:
                article.thread_root_message_id = int(ev_dup.thread_root_message_id)
                article.is_thread_update = True
                logger.info(
                    "[THREAD_UPDATE] Post-scoring confirmed reply to root msg_id=%s (%s)",
                    article.thread_root_message_id,
                    ev_dup.delta_reason or ev_dup.reason,
                )
            elif ev_dup.stage == "thread_window_expired":
                article.thread_root_message_id = None
                article.is_thread_update = False

        return True

    async def _evaluate_candidate(
        self,
        article: Article,
        extra_recent: Optional[list[dict]] = None,
    ) -> bool:
        """
        Single-article evaluation path: local pre-screening + zero-API cross-source
        dedup + Gemini relevance check.
        """
        local_ok = await self._local_pre_screen_candidate(
            article, extra_recent=extra_recent, use_off_topic_prefilter=False
        )
        if not local_ok:
            return False

        relevance = await self._ranker.evaluate_global_relevance(article)
        return await self._apply_relevance_evaluation(article, relevance)

    async def prepare_article_for_publishing(self, article: Article) -> bool:
        """
        Run full-article scraping, Sorani Kurdish translation, neutrality normalization,
        post-translation dedup/threading checks, and persist READY_TO_PUBLISH state.
        Returns True if the article is ready to publish/queue cleanly.
        """
        article_id = article.id
        priority = article.priority

        # ── Step 5: Extract full article text & og:image, verify depth ──
        full_text, og_image = await self._extractor.fetch_article_content_and_image(article.original_url)
        if _is_telegram_channel_url(article.original_url):
            # Ignore watermarked static screenshots / og:image previews on Telegram source channels
            article.image_url = ""
        elif not article.image_url and og_image:
            article.image_url = og_image

        if full_text and len(full_text) >= 160:
            article_content = strip_rss_boilerplate(full_text)[:3000]
            article.original_summary = article_content
            logger.info(
                "Extracted %d chars of full article text from %s",
                len(article_content),
                article.original_url[:70],
            )
        else:
            fallback = strip_rss_boilerplate((article.original_summary or "").strip())
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

        # Second-layer pre-publish guard after full-article scrape
        scrape_blocked = check_blocked_topic(article)
        if scrape_blocked:
            return await self._reject_blocked_topic(article, scrape_blocked)

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
            if not title_ok or not is_valid_kurdish_sorani(article.translated_title or ""):
                raise TranslationError("Headline failed strict Central Kurdish (Sorani) gate.")
            if (article.translated_summary or "").strip() and not is_valid_kurdish_sorani(article.translated_summary):
                raise TranslationError("Summary failed strict Central Kurdish (Sorani) gate.")

            article.translated_title = normalize_neutral_terminology(article.translated_title)
            if article.translated_summary:
                article.translated_summary = normalize_neutral_terminology(
                    strip_rss_boilerplate(article.translated_summary)
                )

            # Uncorroborated Tier-3 preliminary attribution (only when Tier-3 has no corroboration)
            if getattr(config, "ENABLE_SOURCE_VERIFICATION", True):
                if not getattr(article, "_verification_checked", False) and get_source_tier(article.source_name) == TIER_3:
                    recent_hist = await self._get_published_history()
                    pool = list(self._cycle_corroboration_pool or [article])
                    corrob = compute_corroboration(article, batch_articles=pool, recent_history=recent_hist)
                    apply_verification_guard(article, corrob)
                if getattr(article, "_is_uncorroborated_tier3", False) and article.translated_summary:
                    article.translated_summary = prepend_tier3_attribution(article.translated_summary)

            # Final Sorani re-check after terminology normalization
            if not is_valid_kurdish_sorani(article.translated_title or "") or (
                (article.translated_summary or "").strip()
                and not is_valid_kurdish_sorani(article.translated_summary)
            ):
                raise TranslationError("Post-normalization text failed strict Sorani gate.")

            logger.info("[TRANSLATED] Article %s translated: '%s' -> '%s'",
                        article_id, article.original_title[:60], article.translated_title[:60])

            # Merge optional event_key / event_entities / update_type emitted by Gemini translation JSON
            tr_meta = getattr(self._translator, "last_event_metadata", None) or {}
            tr_key = normalize_event_key(tr_meta.get("event_key", ""))
            tr_ents = normalize_event_entities(tr_meta.get("event_entities"))
            if tr_key and not article.event_key:
                article.event_key = tr_key
            if tr_ents and not article.event_entities:
                article.event_entities = tr_ents
            if tr_meta.get("update_type") and not getattr(article, "update_type", ""):
                article.update_type = str(tr_meta["update_type"])  # type: ignore[attr-defined]
            if article.event_key or article.event_entities:
                article.named_entities = pack_event_metadata(article.event_key, article.event_entities)

        except TranslationError as e:
            err_str = str(e)
            if _is_transient_quota_or_sorani_error(err_str):
                # Preserve RETRY_PENDING cleanly so rate-limited stories retry next cycle (never skipped/published)
                await self._db.update_article_status(
                    article_id,
                    ArticleStatus.RETRY_PENDING,
                    error_message=f"Retry pending (quota/translation hold): {err_str[:400]}",
                    translated_title="",
                    translated_summary="",
                )
            else:
                retry_count = await self._db.increment_retry(article_id, err_str)
                if retry_count >= MAX_RETRIES:
                    await self._db.update_article_status(
                        article_id, ArticleStatus.FAILED,
                        error_message=f"Translation failed after {MAX_RETRIES} retries: {e}",
                        translated_title="",
                        translated_summary="",
                    )
            self._health.record_failure("translation", err_str)
            self._metrics.increment("translation_failures")
            logger.warning("Translation aborted/deferred for article %s: %s", article_id, err_str[:200])
            return False

        # ── Step 5c: Post-Translation Sorani Kurdish Deduplication & Threading Safety Gate ──
        recent_history = await self._get_published_history()
        post_dup = await self._semantic_dedup.check_post_translation_duplicate(
            article,
            extra_recent=self._cycle_accepted_refs,
            recent_history=recent_history,
        )
        if post_dup.is_duplicate:
            article.thread_root_message_id = None
            article.is_thread_update = False
            return await self._reject_duplicate_event(
                article,
                matched_id=post_dup.matched_id,
                reason=post_dup.reason,
                score=float(article.importance_score),
                matched_title=post_dup.matched_title,
            )
        if post_dup.classification == OUTCOME_THREAD_UPDATE and post_dup.thread_root_message_id is not None:
            article.thread_root_message_id = int(post_dup.thread_root_message_id)
            article.is_thread_update = True
        elif post_dup.classification == OUTCOME_NEW_STORY:
            article.thread_root_message_id = None
            article.is_thread_update = False

        # ── Step 6: Update article with all computed fields ──
        update_fields: dict[str, object] = {
            "translated_title": article.translated_title,
            "translated_summary": article.translated_summary,
            "original_summary": article.original_summary,
            "category": article.category,
            "image_url": article.image_url,
            "importance_score": article.importance_score,
            "priority": priority,
            "is_breaking": bool(getattr(article, "is_breaking", False)),
            "region": article.region,
            "named_entities": article.named_entities,
            "event_key": article.event_key,
            "event_entities": article.event_entities,
            "thread_root_message_id": article.thread_root_message_id,
        }
        if getattr(article, "video_url", None):
            update_fields["video_url"] = article.video_url
        await self._db.update_article_status(
            article_id,
            ArticleStatus.READY_TO_PUBLISH,
            **update_fields,
        )
        article.status = ArticleStatus.READY_TO_PUBLISH
        self._record_cycle_published_article(article, is_published=False)
        return True

    async def _scrape_translate_and_publish(
        self,
        article: Article,
        check_category_balancer: bool = True,
    ) -> bool:
        """Run full-article scraping, translation, DB ready update, and Telegram delivery."""
        article_id = article.id
        prepared = await self.prepare_article_for_publishing(article)
        if not prepared:
            return False

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
        """Publish an article to Telegram IMMEDIATELY after final topic & Sorani verification."""
        article_id = article.id

        # Never publish blocked topics or non-Sorani/untranslated text (e.g. from older held DB rows)
        blocked_reason = check_blocked_topic(article)
        if blocked_reason:
            return await self._reject_blocked_topic(article, blocked_reason)

        if not is_valid_kurdish_sorani(article.translated_title or "") or (
            (article.translated_summary or "").strip()
            and not is_valid_kurdish_sorani(article.translated_summary)
        ):
            logger.warning(
                "[BLOCKED PUBLISH] Article %s failed final Sorani publish gate ('%s'); moving to RETRY_PENDING.",
                article_id, article.original_title[:60],
            )
            await self._db.update_article_status(
                article_id,
                ArticleStatus.RETRY_PENDING,
                error_message="Aborted publish: non-Sorani or incomplete translation",
                translated_title="",
                translated_summary="",
            )
            return False

        try:
            await self._db.update_article_status(article_id, ArticleStatus.PUBLISHING)

            with Timer(self._metrics, "publish_time"):
                message_id = await self._publisher.publish_article(article)

            if message_id is not None:
                # Success — Telegram confirmed delivery; record message_id & optional thread root anchor
                article.telegram_message_id = message_id
                article.published_at = datetime.now(timezone.utc).isoformat()
                article.status = ArticleStatus.PUBLISHED
                await self._db.update_article_status(
                    article_id,
                    ArticleStatus.PUBLISHED,
                    published_at=article.published_at,
                    telegram_message_id=message_id,
                    thread_root_message_id=article.thread_root_message_id,
                    is_breaking=bool(getattr(article, "is_breaking", False)),
                )

                # Track published story & message_id in current cycle history for zero-duplicate & live threading
                self._record_cycle_published_article(article)

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
                    "[PUBLISHED] Article %d published to Telegram: msg_id=%d, reply_to=%s, title='%s' (score=%.2f, priority=%s)",
                    article_id,
                    message_id,
                    article.thread_root_message_id,
                    article.original_title[:60],
                    article.importance_score,
                    article.priority,
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

    async def evaluate_batch_candidates(
        self,
        articles: list[Article],
        extra_queue_refs: Optional[list[dict]] = None,
    ) -> tuple[list[Article], dict[str, int]]:
        """
        Pass 1 (zero-API validation, topic pre-filter, cross-source dedup) +
        Pass 2 (batched Gemini relevance scoring & three-way event matching).
        Returns `(eligible_articles, status_counts)`.
        """
        results = {"published": 0, "rejected": 0, "failed": 0}
        if not articles:
            return [], results

        seed_refs = list(extra_queue_refs or [])
        if getattr(config, "ENABLE_SOURCE_VERIFICATION", True):
            prioritized = sorted(
                articles,
                key=lambda a: (get_source_tier(a.source_name), *article_selection_sort_key(a)),
            )
        else:
            prioritized = sorted(articles, key=article_selection_sort_key)
        pre_screened: list[Article] = []
        accepted_batch_refs: list[dict] = []

        # Pass 1: Zero-API validation, local topic pre-filtering & cross-source dedup
        for art in prioritized:
            try:
                ok = await self._local_pre_screen_candidate(
                    art,
                    extra_recent=seed_refs + accepted_batch_refs,
                    use_off_topic_prefilter=True,
                )
                if ok:
                    pre_screened.append(art)
                    accepted_batch_refs.append({
                        "id": art.id,
                        "source_name": art.source_name or "",
                        "original_title": art.original_title or "",
                        "original_summary": art.original_summary or "",
                        "translated_title": art.translated_title or "",
                        "category": art.category or "",
                        "published_at": art.published_at or art.discovered_at or "",
                        "telegram_message_id": getattr(art, "telegram_message_id", None),
                        "thread_root_message_id": getattr(art, "thread_root_message_id", None),
                        "is_published": False,
                    })
                else:
                    results["rejected"] += 1
            except Exception as exc:
                logger.error("Batch pre-screen error for '%s': %s", art.original_title[:60], str(exc)[:200])
                results["failed"] += 1

        if not pre_screened:
            return [], results

        # Cap candidates sent to Gemini to freshest & highest-priority items
        gemini_candidates, capped_overflow = select_top_candidates_for_gemini(
            pre_screened, max_candidates=MAX_GEMINI_CANDIDATES_PER_CYCLE
        )
        for overflow_art, cap_reason in capped_overflow:
            try:
                await self._db.update_article_status(
                    overflow_art.id,
                    ArticleStatus.SKIPPED,
                    error_message=cap_reason,
                    importance_score=overflow_art.importance_score,
                    priority=overflow_art.priority,
                )
            except Exception:
                pass
            results["rejected"] += 1

        if not gemini_candidates:
            return [], results

        # Pass 2: Batched Gemini Importance Evaluation (10-15 articles per single JSON call)
        eligible: list[Article] = []
        accepted_event_refs: list[dict] = []
        try:
            evaluations = await self._ranker.evaluate_batch_relevance(
                gemini_candidates, batch_size=GEMINI_RANK_BATCH_SIZE
            )
            evaluated_pairs = list(zip(gemini_candidates, evaluations))

            if getattr(config, "ENABLE_SOURCE_VERIFICATION", True):
                for art, relevance in evaluated_pairs:
                    if relevance.passed:
                        self._populate_relevance_fields(art, relevance)
                recent_hist = await self._get_published_history()
                batch_pool = list(self._cycle_corroboration_pool or articles)
                combined_hist = seed_refs + list(self._cycle_accepted_refs) + recent_hist
                for art, relevance in evaluated_pairs:
                    if relevance.passed:
                        corrob = compute_corroboration(
                            art,
                            batch_articles=batch_pool,
                            recent_history=combined_hist,
                        )
                        apply_verification_guard(art, corrob)
                evaluated_pairs.sort(
                    key=lambda pair: (get_source_tier(pair[0].source_name), *article_selection_sort_key(pair[0]))
                )

            for art, relevance in evaluated_pairs:
                try:
                    if await self._apply_relevance_evaluation(
                        art,
                        relevance,
                        extra_recent=seed_refs + accepted_event_refs,
                    ):
                        eligible.append(art)
                        accepted_event_refs.append({
                            "id": art.id,
                            "source_name": art.source_name or "",
                            "original_title": art.original_title or "",
                            "translated_title": art.translated_title or "",
                            "translated_summary": art.translated_summary or "",
                            "original_summary": art.original_summary or "",
                            "category": art.category or "",
                            "event_key": art.event_key or "",
                            "event_entities": list(art.event_entities or []),
                            "named_entities": art.named_entities or "",
                            "telegram_message_id": getattr(art, "telegram_message_id", None),
                            "thread_root_message_id": getattr(art, "thread_root_message_id", None),
                            "is_published": False,
                        })
                    else:
                        results["rejected"] += 1
                except Exception as exc:
                    logger.error("Relevance gate error for '%s': %s", art.original_title[:60], str(exc)[:200])
                    results["failed"] += 1
        except Exception as exc:
            logger.error("Batch relevance evaluation failed: %s", str(exc)[:200])
            results["failed"] += len(gemini_candidates)
            return [], results

        return eligible, results

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
        if not articles:
            return {"published": 0, "rejected": 0, "failed": 0}

        self._cycle_accepted_refs = []
        effective_limit = BATCH_PUBLISH_LIMIT if max_publish is None else max(0, max_publish)

        eligible, results = await self.evaluate_batch_candidates(articles)
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
