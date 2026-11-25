"""
Kurdish Global News Bot — Supabase PostgREST Storage Backend

HTTPS/IPv4-safe persistent database backend talking to Supabase PostgREST
(`${SUPABASE_URL}/rest/v1/`) via lightweight async `httpx` calls.

Drop-in compatible with `storage.database.Database`, enabling persistent state
across stateless GitHub Actions runs (articles, URL hashes, skipped URLs,
12h dedup headlines, 24h digest candidates, daily digest date lock, events,
and sources) without PostgreSQL TCP/5432 or IPv6 dependencies.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx

from config import (
    ArticleStatus,
    DEDUP_HISTORY_HOURS,
    MAX_QUEUE_SIZE,
    MAX_QUEUED_ARTICLE_AGE_HOURS,
    SUPABASE_KEY,
    SUPABASE_URL,
)
from storage.models import Article, Event, Source
from storage.utils import (
    compute_url_hash as _compute_url_hash,
    filter_recent_headlines,
    normalize_event_entities,
    normalize_event_key,
    pack_event_metadata,
    parse_utc_timestamp,
    prioritize_digest_articles,
    select_queue_overflow,
    split_queued_by_age,
    unpack_event_metadata,
)

logger = logging.getLogger("storage.supabase_db")

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class SupabaseAPIError(RuntimeError):
    """Raised when Supabase PostgREST returns an unrecoverable error."""


class SupabaseDatabase:
    """Async persistent database manager backed by Supabase PostgREST over HTTPS."""

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
        max_retries: int = 3,
        base_backoff: float = 0.5,
    ) -> None:
        raw_url = SUPABASE_URL if url is None else url
        raw_key = SUPABASE_KEY if key is None else key
        self.url = raw_url.strip().rstrip("/")
        self.key = raw_key.strip()
        if not self.url or not self.key:
            raise ValueError("Both SUPABASE_URL and SUPABASE_KEY are required for SupabaseDatabase.")

        self.rest_url = f"{self.url}/rest/v1"
        self._client: Optional[httpx.AsyncClient] = client
        self._owns_client: bool = client is None
        self._max_retries = max(1, max_retries)
        self._base_backoff = max(0.0, base_backoff)
        self._lock = asyncio.Lock()
        # Tracks whether remote Supabase `articles` has optional columns
        self._supports_event_columns: Optional[bool] = None
        self._supports_thread_columns: Optional[bool] = None
        self._warned_missing_thread_cols: bool = False

    def _build_default_headers(self) -> dict[str, str]:
        return {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            transport = httpx.AsyncHTTPTransport(local_address="0.0.0.0", retries=0)
            self._client = httpx.AsyncClient(
                base_url=self.rest_url,
                headers=self._build_default_headers(),
                timeout=httpx.Timeout(20.0, connect=10.0),
                transport=transport,
            )
            self._owns_client = True
        return self._client

    async def connect(self) -> None:
        """Initialize the persistent IPv4-safe httpx async client."""
        self._ensure_client()
        logger.info("Supabase PostgREST client initialized: %s", self.rest_url)

    async def close(self) -> None:
        """Close the owned httpx async client cleanly."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None
            logger.info("Supabase PostgREST client closed.")

    @staticmethod
    def compute_url_hash(url: str) -> str:
        """Compute deterministic SHA-256 hash of a canonicalized URL."""
        return _compute_url_hash(url)

    async def _request(
        self,
        method: str,
        table_path: str,
        *,
        params: Optional[dict[str, str]] = None,
        json_body: Any = None,
        extra_headers: Optional[dict[str, str]] = None,
        allow_conflict: bool = False,
    ) -> httpx.Response:
        """Execute a PostgREST HTTP call with exponential backoff retries on transient/5xx errors."""
        client = self._ensure_client()
        req_headers = self._build_default_headers()
        if extra_headers:
            req_headers.update(extra_headers)

        url = f"{self.rest_url}/{table_path.lstrip('/')}"
        last_exc: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_body,
                    headers=req_headers,
                )
                if response.status_code in _RETRYABLE_STATUS_CODES and attempt < self._max_retries:
                    delay = self._base_backoff * (2 ** (attempt - 1))
                    logger.warning(
                        "Supabase %s %s returned HTTP %d (attempt %d/%d); retrying in %.2fs",
                        method, table_path, response.status_code, attempt, self._max_retries, delay,
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)
                    continue

                if allow_conflict and response.status_code == 409:
                    return response

                if response.status_code >= 400:
                    raise SupabaseAPIError(
                        f"Supabase {method} {table_path} failed with HTTP {response.status_code}: "
                        f"{response.text[:300]}"
                    )
                return response

            except (httpx.TimeoutException, httpx.TransportError, OSError) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    delay = self._base_backoff * (2 ** (attempt - 1))
                    logger.warning(
                        "Supabase transport error on %s %s (attempt %d/%d): %s; retrying in %.2fs",
                        method, table_path, attempt, self._max_retries, str(exc)[:150], delay,
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)
                    continue
                raise SupabaseAPIError(
                    f"Supabase {method} {table_path} transport error after {self._max_retries} attempts: {exc}"
                ) from exc

        raise SupabaseAPIError(f"Supabase {method} {table_path} failed: {last_exc}")

    # ───────────────────────────────────────────────
    # Article CRUD
    # ───────────────────────────────────────────────

    async def insert_article(self, article: Article) -> Optional[int]:
        """Insert a new article. Returns the assigned ID or None if canonical_url already exists."""
        canonical = Article.canonicalize_url(article.canonical_url or article.original_url)
        if canonical:
            article.canonical_url = canonical
        url_hash = self.compute_url_hash(article.canonical_url or article.original_url)

        ev_key = normalize_event_key(getattr(article, "event_key", ""))
        ev_entities_list = normalize_event_entities(getattr(article, "event_entities", []))
        named_entities_val = article.named_entities or pack_event_metadata(ev_key, ev_entities_list)

        payload: dict[str, Any] = {
            "canonical_url": article.canonical_url,
            "url_hash": url_hash,
            "original_url": article.original_url,
            "original_title": article.original_title,
            "normalized_title": article.normalized_title,
            "translated_title": article.translated_title,
            "original_summary": article.original_summary,
            "translated_summary": article.translated_summary,
            "source_name": article.source_name,
            "category": article.category,
            "priority": article.priority,
            "importance_score": float(article.importance_score or 0.0),
            "publication_date": article.publication_date,
            "discovered_at": article.discovered_at,
            "processed_at": article.processed_at,
            "published_at": article.published_at,
            "status": article.status,
            "content_hash": article.content_hash,
            "event_fingerprint": article.event_fingerprint,
            "retry_count": int(article.retry_count or 0),
            "error_message": article.error_message,
            "image_url": article.image_url,
            "country": article.country,
            "region": article.region,
            "named_entities": named_entities_val,
            "is_breaking": bool(getattr(article, "is_breaking", False)),
        }
        if self._supports_thread_columns is not False:
            payload["telegram_message_id"] = article.telegram_message_id
            if getattr(article, "thread_root_message_id", None) is not None:
                payload["thread_root_message_id"] = int(article.thread_root_message_id)
        if getattr(article, "video_url", None):
            payload["video_url"] = article.video_url
        if self._supports_event_columns is not False:
            if ev_key:
                payload["event_key"] = ev_key
            if ev_entities_list:
                payload["event_entities"] = json.dumps(ev_entities_list, ensure_ascii=True)

        resp = None
        for _ in range(4):
            try:
                resp = await self._request(
                    "POST",
                    "articles",
                    params={"on_conflict": "canonical_url"},
                    json_body=payload,
                    extra_headers={"Prefer": "resolution=ignore-duplicates,return=representation"},
                    allow_conflict=True,
                )
                break
            except SupabaseAPIError as exc:
                err_txt = str(exc)
                stripped = False
                if "video_url" in payload and "video_url" in err_txt:
                    logger.warning("Supabase schema missing video_url column; retrying insert without video_url.")
                    payload.pop("video_url", None)
                    stripped = True
                if ("thread_root_message_id" in payload or "telegram_message_id" in payload) and (
                    "thread_root_message_id" in err_txt
                    or "telegram_message_id" in err_txt
                    or (
                        ("HTTP 400" in err_txt or "PGRST" in err_txt)
                        and "event_key" not in err_txt
                        and "event_entities" not in err_txt
                    )
                ):
                    self._supports_thread_columns = False
                    if "thread_root_message_id" in err_txt or "thread_root_message_id" in payload:
                        payload.pop("thread_root_message_id", None)
                    if "telegram_message_id" in err_txt:
                        payload.pop("telegram_message_id", None)
                    self._log_missing_threading_columns_once()
                    stripped = True
                if ("event_key" in payload or "event_entities" in payload) and (
                    "event_key" in err_txt or "event_entities" in err_txt or "HTTP 400" in err_txt or "PGRST" in err_txt
                ):
                    self._supports_event_columns = False
                    payload.pop("event_key", None)
                    payload.pop("event_entities", None)
                    logger.warning(
                        "Supabase schema missing optional event_key/event_entities columns; using named_entities fallback."
                    )
                    stripped = True
                if not stripped:
                    raise
        assert resp is not None
        if resp.status_code == 409:
            logger.debug("Duplicate article URL in Supabase: %s", article.canonical_url)
            return None
        rows = resp.json() if resp.content else []
        if not rows:
            logger.debug("Duplicate article URL ignored in Supabase: %s", article.canonical_url)
            return None
        article.id = rows[0].get("id")
        return article.id

    def _log_missing_threading_columns_once(self) -> None:
        if not self._warned_missing_thread_cols:
            self._warned_missing_thread_cols = True
            logger.warning(
                "Supabase `articles` table is missing Live Event Threading columns "
                "(telegram_message_id / thread_root_message_id). Run "
                "storage/sql/add_event_threading_columns.sql in Supabase SQL Editor; "
                "falling back to non-threaded publishing cleanly."
            )

    async def update_article_status(
        self,
        article_id: int,
        status: str,
        error_message: str = "",
        **extra_fields: object,
    ) -> None:
        """Update an article's status and optional extra fields."""
        patch_data: dict[str, Any] = {
            "status": status,
            "error_message": error_message,
        }
        now_iso = datetime.now(timezone.utc).isoformat()
        normalized_extra = dict(extra_fields)
        if status == ArticleStatus.PUBLISHED and "published_at" not in normalized_extra:
            patch_data["published_at"] = now_iso
        if status in (ArticleStatus.PROCESSING, ArticleStatus.QUEUED) and "processed_at" not in normalized_extra:
            patch_data["processed_at"] = now_iso
        if "event_entities" in normalized_extra:
            ents = normalize_event_entities(normalized_extra["event_entities"])
            normalized_extra["event_entities"] = json.dumps(ents, ensure_ascii=True) if ents else ""
        if "event_key" in normalized_extra:
            normalized_extra["event_key"] = normalize_event_key(normalized_extra["event_key"])
        if self._supports_event_columns is False:
            normalized_extra.pop("event_key", None)
            normalized_extra.pop("event_entities", None)
        if self._supports_thread_columns is False:
            normalized_extra.pop("thread_root_message_id", None)
            normalized_extra.pop("telegram_message_id", None)

        patch_data.update(normalized_extra)

        for _ in range(4):
            try:
                await self._request(
                    "PATCH",
                    "articles",
                    params={"id": f"eq.{article_id}"},
                    json_body=patch_data,
                    extra_headers={"Prefer": "return=minimal"},
                )
                return
            except SupabaseAPIError as exc:
                err_txt = str(exc)
                stripped = False
                if "video_url" in patch_data and "video_url" in err_txt:
                    logger.warning("Supabase schema missing video_url column; retrying update without video_url.")
                    patch_data.pop("video_url", None)
                    stripped = True
                if ("thread_root_message_id" in patch_data or "telegram_message_id" in patch_data) and (
                    "thread_root_message_id" in err_txt
                    or "telegram_message_id" in err_txt
                    or (
                        ("HTTP 400" in err_txt or "PGRST" in err_txt)
                        and "event_key" not in err_txt
                        and "event_entities" not in err_txt
                    )
                ):
                    self._supports_thread_columns = False
                    if "thread_root_message_id" in err_txt or "thread_root_message_id" in patch_data:
                        patch_data.pop("thread_root_message_id", None)
                    if "telegram_message_id" in err_txt:
                        patch_data.pop("telegram_message_id", None)
                    self._log_missing_threading_columns_once()
                    stripped = True
                if ("event_key" in patch_data or "event_entities" in patch_data) and (
                    "event_key" in err_txt or "event_entities" in err_txt or "HTTP 400" in err_txt or "PGRST" in err_txt
                ):
                    self._supports_event_columns = False
                    patch_data.pop("event_key", None)
                    patch_data.pop("event_entities", None)
                    logger.warning(
                        "Supabase schema missing optional event_key/event_entities columns during PATCH; falling back safely."
                    )
                    stripped = True
                if not stripped:
                    raise

    async def get_article_by_canonical_url(self, canonical_url: str) -> Optional[Article]:
        """Fetch an article by its normalized canonical URL."""
        normalized_url = Article.canonicalize_url(canonical_url) if canonical_url else canonical_url
        resp = await self._request(
            "GET",
            "articles",
            params={"canonical_url": f"eq.{normalized_url}", "limit": "1"},
        )
        rows = resp.json() if resp.content else []
        return self._row_to_article(rows[0]) if rows else None

    async def get_articles_by_status(self, status: str, limit: int = 50) -> list[Article]:
        """Fetch articles with a given status ordered by importance score."""
        resp = await self._request(
            "GET",
            "articles",
            params={
                "status": f"eq.{status}",
                "order": "importance_score.desc",
                "limit": str(limit),
            },
        )
        return [self._row_to_article(r) for r in (resp.json() if resp.content else [])]

    async def get_recent_content_hashes(
        self, hours: int = 48, exclude_id: Optional[int] = None
    ) -> set[str]:
        """Get content hashes from recent articles for deduplication."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        params: dict[str, str] = {
            "select": "id,content_hash",
            "discovered_at": f"gte.{cutoff}",
            "content_hash": "neq.",
        }
        if exclude_id is not None:
            params["id"] = f"neq.{exclude_id}"
        resp = await self._request("GET", "articles", params=params)
        rows = resp.json() if resp.content else []
        return {r["content_hash"] for r in rows if r.get("content_hash")}

    async def get_recent_normalized_titles(
        self,
        hours: int = 48,
        exclude_id: Optional[int] = None,
        only_published: bool = False,
    ) -> list[tuple[int, str]]:
        """Get normalized titles from recent articles for fuzzy dedup."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        params: dict[str, str] = {
            "select": "id,normalized_title,status",
            "discovered_at": f"gte.{cutoff}",
            "normalized_title": "neq.",
        }
        if only_published:
            params["status"] = "eq.PUBLISHED"
        else:
            params["status"] = "not.in.(DISCOVERED,REJECTED_DUPLICATE,FAILED,SKIPPED,EXPIRED)"
        if exclude_id is not None:
            params["id"] = f"neq.{exclude_id}"
        resp = await self._request("GET", "articles", params=params)
        rows = resp.json() if resp.content else []
        return [
            (int(r["id"]), r["normalized_title"])
            for r in rows
            if r.get("id") is not None and r.get("normalized_title")
        ]

    async def get_published_event_fingerprints(self, hours: int = 72) -> set[str]:
        """Get event fingerprints of recently published articles."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        resp = await self._request(
            "GET",
            "articles",
            params={
                "select": "event_fingerprint",
                "status": "eq.PUBLISHED",
                "published_at": f"gte.{cutoff}",
                "event_fingerprint": "neq.",
            },
        )
        rows = resp.json() if resp.content else []
        return {r["event_fingerprint"] for r in rows if r.get("event_fingerprint")}

    async def is_empty(self) -> bool:
        """Check if the articles table is currently empty."""
        resp = await self._request(
            "GET",
            "articles",
            params={"select": "id", "limit": "1"},
        )
        rows = resp.json() if resp.content else []
        return len(rows) == 0

    async def mark_url_skipped(
        self, url: str, title: str = "", score: float = 0.0, reason: str = ""
    ) -> str:
        """Idempotently upsert a skipped URL hash into Supabase `skipped_urls`."""
        canonical = Article.canonicalize_url(url)
        url_hash = self.compute_url_hash(canonical)
        payload = {
            "url_hash": url_hash,
            "canonical_url": canonical,
            "title": title[:300],
            "score": float(score or 0.0),
            "reason": reason[:500],
            "skipped_at": datetime.now(timezone.utc).isoformat(),
        }
        await self._request(
            "POST",
            "skipped_urls",
            params={"on_conflict": "url_hash"},
            json_body=payload,
            extra_headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
        )
        return url_hash

    async def is_url_skipped(self, url: str) -> bool:
        """Check if a URL hash or canonical URL has already been marked skipped."""
        if not url:
            return False
        canonical = Article.canonicalize_url(url)
        url_hash = self.compute_url_hash(canonical)

        resp = await self._request(
            "GET",
            "skipped_urls",
            params={"select": "url_hash", "url_hash": f"eq.{url_hash}", "limit": "1"},
        )
        if resp.content and resp.json():
            return True

        art_resp = await self._request(
            "GET",
            "articles",
            params={
                "select": "id",
                "canonical_url": f"eq.{canonical}",
                "status": "eq.SKIPPED",
                "limit": "1",
            },
        )
        return bool(art_resp.content and art_resp.json())

    async def get_recent_published_articles(self, limit: int = 20) -> list[Article]:
        """Fetch the most recently published articles."""
        resp = await self._request(
            "GET",
            "articles",
            params={
                "status": "eq.PUBLISHED",
                "order": "published_at.desc,id.desc",
                "limit": str(limit),
            },
        )
        return [self._row_to_article(r) for r in (resp.json() if resp.content else [])]

    async def get_recent_published_headlines(self, hours: int = DEDUP_HISTORY_HOURS) -> list[dict]:
        """
        Fetch id, titles, Kurdish summaries, category, and optional event metadata of all
        PUBLISHED articles from the last `hours` hours for cross-run deduplication.
        Gracefully degrades if remote Supabase schema lacks `event_key`/`event_entities`.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, hours))
        full_select = (
            "id,source_name,original_title,translated_title,translated_summary,category,"
            "published_at,discovered_at,telegram_message_id,thread_root_message_id,"
            "named_entities,event_key,event_entities"
        )
        no_thread_select = (
            "id,source_name,original_title,translated_title,translated_summary,category,"
            "published_at,discovered_at,named_entities,event_key,event_entities"
        )
        fallback_select = (
            "id,source_name,original_title,translated_title,translated_summary,category,"
            "published_at,discovered_at,named_entities"
        )
        minimal_select = (
            "id,source_name,original_title,translated_title,category,published_at,discovered_at"
        )

        if self._supports_thread_columns is False and self._supports_event_columns is False:
            select_candidates = [fallback_select, minimal_select]
        elif self._supports_thread_columns is False:
            select_candidates = [no_thread_select, fallback_select, minimal_select]
        elif self._supports_event_columns is False:
            select_candidates = [fallback_select, minimal_select]
        else:
            select_candidates = [full_select, no_thread_select, fallback_select, minimal_select]

        rows: list[dict] = []
        for idx, select_expr in enumerate(select_candidates):
            try:
                resp = await self._request(
                    "GET",
                    "articles",
                    params={
                        "select": select_expr,
                        "status": "eq.PUBLISHED",
                        "published_at": f"gte.{cutoff.isoformat()}",
                        "order": "published_at.desc,id.desc",
                        "limit": "300",
                    },
                )
                rows = resp.json() if resp.content else []
                break
            except SupabaseAPIError as exc:
                err_txt = str(exc)
                if "thread_root_message_id" in err_txt or "telegram_message_id" in err_txt:
                    self._supports_thread_columns = False
                    self._log_missing_threading_columns_once()
                if "event_key" in err_txt or "event_entities" in err_txt:
                    self._supports_event_columns = False
                is_schema_err = (
                    "thread_root_message_id" in err_txt
                    or "telegram_message_id" in err_txt
                    or "event_key" in err_txt
                    or "event_entities" in err_txt
                    or "translated_summary" in err_txt
                    or "named_entities" in err_txt
                    or "HTTP 400" in err_txt
                    or "PGRST" in err_txt
                )
                if is_schema_err and idx < len(select_candidates) - 1:
                    logger.warning(
                        "Supabase articles query falling back to compatible columns (%s).",
                        err_txt[:140],
                    )
                    continue
                raise

        return filter_recent_headlines(rows, cutoff)

    async def get_published_articles_last_24h(self, limit: int = 10) -> list[Article]:
        """Fetch top published articles from the last 24 hours prioritizing Kurdistan stories."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        resp = await self._request(
            "GET",
            "articles",
            params={
                "status": "eq.PUBLISHED",
                "published_at": f"gte.{cutoff.isoformat()}",
                "order": "published_at.desc,id.desc",
                "limit": "150",
            },
        )
        rows = resp.json() if resp.content else []
        recent_24h: list[Article] = []
        for row in rows:
            art = self._row_to_article(row)
            dt = parse_utc_timestamp(art.published_at or art.discovered_at)
            if dt is not None and dt >= cutoff:
                recent_24h.append(art)
        return prioritize_digest_articles(recent_24h, limit=limit)

    async def is_digest_published_for_date(self, digest_date: str) -> bool:
        """Return True if a daily digest has already been recorded for `digest_date`."""
        resp = await self._request(
            "GET",
            "daily_digests",
            params={"select": "digest_date", "digest_date": f"eq.{digest_date}", "limit": "1"},
        )
        rows = resp.json() if resp.content else []
        return len(rows) > 0

    async def mark_digest_published(
        self,
        digest_date: str,
        story_count: int,
        digest_text: str,
        telegram_message_id: Optional[int] = None,
    ) -> None:
        """Idempotently upsert a daily digest record in Supabase `daily_digests`."""
        payload = {
            "digest_date": digest_date,
            "story_count": int(story_count),
            "digest_text": digest_text,
            "telegram_message_id": telegram_message_id,
            "published_at": datetime.now(timezone.utc).isoformat(),
        }
        await self._request(
            "POST",
            "daily_digests",
            params={"on_conflict": "digest_date"},
            json_body=payload,
            extra_headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
        )

    async def get_last_published_primary_time(
        self, primary_categories: set[str] | list[str]
    ) -> Optional[datetime]:
        """Get the publication timestamp of the most recent primary-focus article."""
        if not primary_categories:
            return None
        quoted = ",".join(f'"{cat}"' for cat in sorted(primary_categories))
        resp = await self._request(
            "GET",
            "articles",
            params={
                "select": "published_at",
                "status": "eq.PUBLISHED",
                "category": f"in.({quoted})",
                "published_at": "not.is.null",
                "order": "published_at.desc,id.desc",
                "limit": "1",
            },
        )
        rows = resp.json() if resp.content else []
        if rows and rows[0].get("published_at"):
            return parse_utc_timestamp(rows[0]["published_at"])
        return None

    async def increment_retry(self, article_id: int, error: str) -> int:
        """Increment retry count and set article status to RETRY_PENDING."""
        resp = await self._request(
            "GET",
            "articles",
            params={"select": "retry_count", "id": f"eq.{article_id}", "limit": "1"},
        )
        rows = resp.json() if resp.content else []
        current = int(rows[0].get("retry_count", 0)) if rows else 0
        new_count = current + 1

        await self._request(
            "PATCH",
            "articles",
            params={"id": f"eq.{article_id}"},
            json_body={
                "retry_count": new_count,
                "status": ArticleStatus.RETRY_PENDING,
                "error_message": error,
            },
            extra_headers={"Prefer": "return=minimal"},
        )
        return new_count

    async def get_retry_pending_articles(self, max_retries: int = 3) -> list[Article]:
        """Fetch articles pending retry."""
        resp = await self._request(
            "GET",
            "articles",
            params={
                "status": "eq.RETRY_PENDING",
                "retry_count": f"lt.{max_retries}",
                "order": "importance_score.desc",
            },
        )
        return [self._row_to_article(r) for r in (resp.json() if resp.content else [])]

    async def get_interrupted_articles(self) -> list[Article]:
        """Fetch articles stuck in intermediate processing states."""
        resp = await self._request(
            "GET",
            "articles",
            params={
                "status": "in.(VALIDATING,PROCESSING,TRANSLATING,READY_TO_PUBLISH,PUBLISHING)",
                "order": "importance_score.desc",
            },
        )
        return [self._row_to_article(r) for r in (resp.json() if resp.content else [])]

    # ───────────────────────────────────────────────
    # Smart Pacing & Persistent Queue
    # ───────────────────────────────────────────────

    async def get_queued_articles(
        self, max_age_hours: int = MAX_QUEUED_ARTICLE_AGE_HOURS
    ) -> list[Article]:
        """Fetch non-expired QUEUED articles ordered by priority and recency."""
        all_queued = await self.get_articles_by_status(ArticleStatus.QUEUED, limit=200)
        fresh, _ = split_queued_by_age(all_queued, max_age_hours=max_age_hours)
        return fresh

    async def expire_stale_queued_articles(
        self, max_age_hours: int = MAX_QUEUED_ARTICLE_AGE_HOURS
    ) -> int:
        """Mark QUEUED articles older than `max_age_hours` as EXPIRED and return count."""
        all_queued = await self.get_articles_by_status(ArticleStatus.QUEUED, limit=200)
        _, stale = split_queued_by_age(all_queued, max_age_hours=max_age_hours)
        expired_count = 0
        for art, age_hours in stale:
            if art.id is None:
                continue
            await self.update_article_status(
                art.id,
                ArticleStatus.EXPIRED,
                error_message=f"Expired from queue (age {age_hours:.1f}h > {max_age_hours}h)",
            )
            expired_count += 1
        return expired_count

    async def trim_queue(self, max_size: int = MAX_QUEUE_SIZE) -> int:
        """Keep top `max_size` highest-priority/freshest QUEUED articles and expire the rest."""
        all_queued = await self.get_articles_by_status(ArticleStatus.QUEUED, limit=200)
        overflow = select_queue_overflow(all_queued, max_size=max_size)
        trimmed_count = 0
        for art in overflow:
            if art.id is None:
                continue
            await self.update_article_status(
                art.id,
                ArticleStatus.EXPIRED,
                error_message=f"Expired from queue: exceeded MAX_QUEUE_SIZE={max_size}",
            )
            trimmed_count += 1
        return trimmed_count

    async def get_last_normal_published_at(self) -> Optional[datetime]:
        """Return UTC timestamp of the most recent non-breaking PUBLISHED article, or None."""
        try:
            resp = await self._request(
                "GET",
                "articles",
                params={
                    "select": "published_at,is_breaking",
                    "status": "eq.PUBLISHED",
                    "is_breaking": "eq.false",
                    "published_at": "not.is.null",
                    "order": "published_at.desc,id.desc",
                    "limit": "20",
                },
            )
            rows = resp.json() if resp.content else []
        except SupabaseAPIError as exc:
            if "is_breaking" not in str(exc) and "HTTP 400" not in str(exc):
                raise
            # Fallback if an older schema lacks is_breaking
            resp = await self._request(
                "GET",
                "articles",
                params={
                    "select": "published_at,priority",
                    "status": "eq.PUBLISHED",
                    "published_at": "not.is.null",
                    "order": "published_at.desc,id.desc",
                    "limit": "20",
                },
            )
            rows = [
                r for r in (resp.json() if resp.content else [])
                if (r.get("priority") or "").upper() != "CRITICAL"
            ]

        timestamps = [
            dt
            for row in rows
            if (dt := parse_utc_timestamp(row.get("published_at"))) is not None
        ]
        return max(timestamps) if timestamps else None

    # ───────────────────────────────────────────────
    # Event & Source CRUD
    # ───────────────────────────────────────────────

    async def upsert_event(self, event: Event) -> int:
        """Insert or increment an event in Supabase `events`."""
        existing = await self.get_event_by_fingerprint(event.event_fingerprint)
        now_iso = datetime.now(timezone.utc).isoformat()
        if existing and existing.id is not None:
            await self._request(
                "PATCH",
                "events",
                params={"id": f"eq.{existing.id}"},
                json_body={
                    "last_updated_at": now_iso,
                    "article_count": int(existing.article_count) + 1,
                },
                extra_headers={"Prefer": "return=minimal"},
            )
            return existing.id

        resp = await self._request(
            "POST",
            "events",
            params={"on_conflict": "event_fingerprint"},
            json_body={
                "event_fingerprint": event.event_fingerprint,
                "event_title": event.event_title,
                "first_seen_at": event.first_seen_at,
                "last_updated_at": event.last_updated_at,
                "article_count": int(event.article_count or 1),
            },
            extra_headers={"Prefer": "resolution=merge-duplicates,return=representation"},
        )
        rows = resp.json() if resp.content else []
        return int(rows[0]["id"]) if rows else 0

    async def get_event_by_fingerprint(self, fingerprint: str) -> Optional[Event]:
        """Fetch an event by its fingerprint."""
        resp = await self._request(
            "GET",
            "events",
            params={"event_fingerprint": f"eq.{fingerprint}", "limit": "1"},
        )
        rows = resp.json() if resp.content else []
        if not rows:
            return None
        row = rows[0]
        return Event(
            id=row.get("id"),
            event_fingerprint=row.get("event_fingerprint", ""),
            event_title=row.get("event_title", ""),
            first_seen_at=row.get("first_seen_at", ""),
            last_updated_at=row.get("last_updated_at", ""),
            article_count=int(row.get("article_count", 1) or 1),
        )

    async def upsert_source(self, source: Source) -> int:
        """Idempotently upsert a source in Supabase `sources`."""
        resp = await self._request(
            "POST",
            "sources",
            params={"on_conflict": "name"},
            json_body={
                "name": source.name,
                "source_type": source.source_type,
                "url": source.url,
                "category": source.category,
                "enabled": bool(source.enabled),
            },
            extra_headers={"Prefer": "resolution=merge-duplicates,return=representation"},
        )
        rows = resp.json() if resp.content else []
        return int(rows[0]["id"]) if rows else 0

    async def update_source_status(self, name: str, success: bool, error: str = "") -> None:
        """Update source check status in Supabase."""
        now_iso = datetime.now(timezone.utc).isoformat()
        if success:
            patch_data = {
                "last_checked_at": now_iso,
                "last_success_at": now_iso,
                "last_error": "",
                "consecutive_failures": 0,
            }
        else:
            resp = await self._request(
                "GET",
                "sources",
                params={"select": "consecutive_failures", "name": f"eq.{name}", "limit": "1"},
            )
            rows = resp.json() if resp.content else []
            failures = int(rows[0].get("consecutive_failures", 0) or 0) if rows else 0
            patch_data = {
                "last_checked_at": now_iso,
                "last_error": error[:500],
                "consecutive_failures": failures + 1,
            }
        await self._request(
            "PATCH",
            "sources",
            params={"name": f"eq.{name}"},
            json_body=patch_data,
            extra_headers={"Prefer": "return=minimal"},
        )

    # ───────────────────────────────────────────────
    # System Logs, Stats & Backup
    # ───────────────────────────────────────────────

    async def log_system_event(
        self,
        level: str,
        component: str,
        message: str,
        timestamp: Optional[str] = None,
    ) -> None:
        """Write a system log entry to Supabase `system_logs`."""
        try:
            await self._request(
                "POST",
                "system_logs",
                json_body={
                    "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
                    "level": level,
                    "component": component,
                    "message": message[:2000],
                },
                extra_headers={"Prefer": "return=minimal"},
            )
        except Exception as exc:
            logger.error("Failed to write system log to Supabase: %s", exc)

    async def get_recent_system_logs(
        self,
        component: str,
        limit: int = 50,
        hours: Optional[int] = None,
        now: Optional[datetime] = None,
    ) -> list[dict]:
        """Fetch recent system_logs entries for a component from Supabase."""
        params: dict[str, str] = {
            "select": "id,timestamp,level,component,message",
            "component": f"eq.{component}",
            "order": "timestamp.desc,id.desc",
            "limit": str(max(1, int(limit))),
        }
        if hours is not None:
            ref_now = now or datetime.now(timezone.utc)
            if ref_now.tzinfo is None:
                ref_now = ref_now.replace(tzinfo=timezone.utc)
            cutoff = ref_now - timedelta(hours=max(1, int(hours)))
            params["timestamp"] = f"gte.{cutoff.isoformat()}"
        resp = await self._request("GET", "system_logs", params=params)
        return resp.json() if resp.content else []

    async def get_last_batch_completed_at(self) -> Optional[datetime]:
        """Return UTC timestamp of the most recently completed paced batch from Supabase."""
        rows = await self.get_recent_system_logs(component="BATCH_COOLDOWN", limit=1)
        if not rows:
            return None
        return parse_utc_timestamp(rows[0].get("timestamp"))

    async def record_batch_completed_at(
        self,
        timestamp: Optional[datetime] = None,
        published_count: int = 0,
    ) -> datetime:
        """Persist completion timestamp of a paced batch into Supabase `system_logs`."""
        dt = timestamp or datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        await self.log_system_event(
            level="INFO",
            component="BATCH_COOLDOWN",
            message=f"Batch completed ({int(published_count)} post(s))",
            timestamp=dt.isoformat(),
        )
        return dt

    async def get_sources_status(self) -> list[dict]:
        """Fetch enabled sources and failure counts from Supabase `sources`."""
        resp = await self._request(
            "GET",
            "sources",
            params={
                "select": "name,source_type,enabled,consecutive_failures,last_error,last_checked_at,last_success_at",
                "enabled": "eq.true",
                "order": "consecutive_failures.desc,name.asc",
            },
        )
        return resp.json() if resp.content else []

    async def get_watchdog_article_metrics(
        self,
        hours: int = 24,
        now: Optional[datetime] = None,
    ) -> dict[str, object]:
        """Compute 24h article metrics, backlog count, and last published timestamp from Supabase."""
        ref_now = now or datetime.now(timezone.utc)
        if ref_now.tzinfo is None:
            ref_now = ref_now.replace(tzinfo=timezone.utc)
        cutoff = ref_now - timedelta(hours=max(1, int(hours)))

        pub_resp = await self._request(
            "GET",
            "articles",
            params={
                "status": "eq.PUBLISHED",
                "order": "published_at.desc,id.desc",
                "limit": "300",
            },
        )
        pub_rows = pub_resp.json() if pub_resp.content else []
        published_recent: list[Article] = []
        all_pub_timestamps: list[datetime] = []
        for row in pub_rows:
            art = self._row_to_article(row)
            dt = parse_utc_timestamp(art.published_at or art.discovered_at)
            if dt is not None:
                all_pub_timestamps.append(dt)
                if dt >= cutoff:
                    published_recent.append(art)

        eff_resp = await self._request(
            "GET",
            "articles",
            params={
                "select": "status,discovered_at",
                "status": "in.(REJECTED_DUPLICATE,SKIPPED)",
                "discovered_at": f"gte.{cutoff.isoformat()}",
                "limit": "1000",
            },
        )
        eff_rows = eff_resp.json() if eff_resp.content else []
        duplicates_recent = sum(1 for r in eff_rows if r.get("status") == ArticleStatus.REJECTED_DUPLICATE)
        skipped_recent = sum(1 for r in eff_rows if r.get("status") == ArticleStatus.SKIPPED)

        stats = await self.get_stats()
        return {
            "published_articles": published_recent,
            "duplicates_count": duplicates_recent,
            "skipped_count": skipped_recent,
            "queued_count": int(stats.get(ArticleStatus.QUEUED, 0)),
            "total_articles": int(stats.get("total", 0)),
            "last_published_at": max(all_pub_timestamps) if all_pub_timestamps else None,
        }

    async def get_stats(self) -> dict[str, int]:
        """Get article counts grouped by status."""
        resp = await self._request("GET", "articles", params={"select": "status"})
        rows = resp.json() if resp.content else []
        stats: dict[str, int] = {}
        for row in rows:
            st = row.get("status")
            if st:
                stats[st] = stats.get(st, 0) + 1
        stats["total"] = len(rows)
        return stats

    async def backup(self) -> str:
        """No-op for Supabase (data is already persisted in managed cloud Postgres)."""
        return ""

    # ───────────────────────────────────────────────
    # Helpers
    # ───────────────────────────────────────────────

    @staticmethod
    def _row_to_article(row: dict[str, Any]) -> Article:
        """Convert a PostgREST JSON object into an `Article` dataclass instance."""
        ev_key, ev_entities = unpack_event_metadata(
            row.get("named_entities") or "",
            row.get("event_key") or "",
            row.get("event_entities"),
        )
        return Article(
            id=row.get("id"),
            canonical_url=row.get("canonical_url", ""),
            original_url=row.get("original_url", ""),
            original_title=row.get("original_title", ""),
            normalized_title=row.get("normalized_title", ""),
            translated_title=row.get("translated_title") or "",
            original_summary=row.get("original_summary") or "",
            translated_summary=row.get("translated_summary") or "",
            source_name=row.get("source_name") or "",
            category=row.get("category") or "",
            priority=row.get("priority") or "NORMAL",
            importance_score=float(row.get("importance_score") or 0.0),
            publication_date=row.get("publication_date"),
            discovered_at=row.get("discovered_at") or datetime.now(timezone.utc).isoformat(),
            processed_at=row.get("processed_at"),
            published_at=row.get("published_at"),
            status=row.get("status") or "DISCOVERED",
            content_hash=row.get("content_hash") or "",
            event_fingerprint=row.get("event_fingerprint") or "",
            retry_count=int(row.get("retry_count") or 0),
            error_message=row.get("error_message") or "",
            telegram_message_id=row.get("telegram_message_id"),
            thread_root_message_id=row.get("thread_root_message_id"),
            image_url=row.get("image_url") or "",
            video_url=row.get("video_url") or None,
            country=row.get("country") or "",
            region=row.get("region") or "",
            named_entities=row.get("named_entities") or "",
            event_key=ev_key,
            event_entities=ev_entities,
            is_breaking=bool(row.get("is_breaking", False)),
            is_thread_update=(row.get("thread_root_message_id") is not None),
        )
