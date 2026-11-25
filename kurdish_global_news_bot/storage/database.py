"""
Kurdish Global News Bot — Database Module

Async SQLite database layer with connection pooling, migrations, and CRUD operations.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import aiosqlite

import config
from config import (
    ArticleStatus,
    BACKUP_DIR,
    DATABASE_PATH,
    DEDUP_HISTORY_HOURS,
    MAX_QUEUE_SIZE,
    MAX_QUEUED_ARTICLE_AGE_HOURS,
)
from storage.models import Article, Event, Source
from storage.utils import (
    compute_url_hash as _compute_url_hash,
    filter_recent_headlines as _filter_recent_headlines,
    normalize_event_entities as _normalize_event_entities,
    normalize_event_key as _normalize_event_key,
    pack_event_metadata as _pack_event_metadata,
    parse_utc_timestamp as _parse_utc_timestamp_util,
    prioritize_digest_articles as _prioritize_digest_articles,
    select_queue_overflow as _select_queue_overflow,
    split_queued_by_age as _split_queued_by_age,
    unpack_event_metadata as _unpack_event_metadata,
)

logger = logging.getLogger("database")
KURDISTAN_TZ = timezone(timedelta(hours=3))


def get_kurdistan_date_str(dt: Optional[datetime] = None) -> str:
    """Return current local date string YYYY-MM-DD in Kurdistan time (UTC+3)."""
    ref = dt or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    return ref.astimezone(KURDISTAN_TZ).strftime("%Y-%m-%d")


class Database:
    """Async SQLite database manager."""

    def __init__(self, db_path: str = DATABASE_PATH) -> None:
        self.db_path = db_path
        self._connection: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Open database connection and initialize schema."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(
            self.db_path,
            timeout=30.0,
        )
        self._connection.row_factory = aiosqlite.Row
        await self._connection.execute("PRAGMA journal_mode=WAL")
        await self._connection.execute("PRAGMA busy_timeout=10000")
        await self._connection.execute("PRAGMA foreign_keys=ON")
        await self._connection.execute("PRAGMA synchronous=NORMAL")
        await self._init_schema()
        logger.info("Database connected: %s", self.db_path)

    async def close(self) -> None:
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("Database connection closed.")

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._connection

    async def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        await self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_url TEXT NOT NULL,
                original_url TEXT NOT NULL,
                original_title TEXT NOT NULL DEFAULT '',
                normalized_title TEXT NOT NULL DEFAULT '',
                translated_title TEXT DEFAULT '',
                original_summary TEXT DEFAULT '',
                translated_summary TEXT DEFAULT '',
                source_name TEXT DEFAULT '',
                category TEXT DEFAULT '',
                priority TEXT DEFAULT 'NORMAL',
                importance_score REAL DEFAULT 0.0,
                publication_date TEXT,
                discovered_at TEXT NOT NULL,
                processed_at TEXT,
                published_at TEXT,
                status TEXT NOT NULL DEFAULT 'DISCOVERED',
                content_hash TEXT DEFAULT '',
                event_fingerprint TEXT DEFAULT '',
                retry_count INTEGER DEFAULT 0,
                error_message TEXT DEFAULT '',
                telegram_message_id INTEGER,
                thread_root_message_id INTEGER DEFAULT NULL,
                image_url TEXT DEFAULT '',
                video_url TEXT DEFAULT NULL,
                country TEXT DEFAULT '',
                region TEXT DEFAULT '',
                named_entities TEXT DEFAULT '',
                event_key TEXT DEFAULT '',
                event_entities TEXT DEFAULT '',
                is_breaking INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                source_type TEXT NOT NULL DEFAULT '',
                url TEXT DEFAULT '',
                category TEXT DEFAULT '',
                enabled INTEGER DEFAULT 1,
                last_checked_at TEXT,
                last_success_at TEXT,
                last_error TEXT DEFAULT '',
                consecutive_failures INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_fingerprint TEXT NOT NULL UNIQUE,
                event_title TEXT DEFAULT '',
                first_seen_at TEXT NOT NULL,
                last_updated_at TEXT NOT NULL,
                article_count INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL DEFAULT 'INFO',
                component TEXT DEFAULT '',
                message TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS skipped_urls (
                url_hash TEXT PRIMARY KEY,
                canonical_url TEXT NOT NULL,
                title TEXT DEFAULT '',
                score REAL DEFAULT 0.0,
                reason TEXT DEFAULT '',
                skipped_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS daily_digests (
                digest_date TEXT PRIMARY KEY,
                story_count INTEGER DEFAULT 0,
                digest_text TEXT DEFAULT '',
                telegram_message_id INTEGER,
                published_at TEXT NOT NULL
            );

            -- Indexes for performance
            CREATE INDEX IF NOT EXISTS idx_articles_canonical_url ON articles(canonical_url);
            CREATE INDEX IF NOT EXISTS idx_articles_content_hash ON articles(content_hash);
            CREATE INDEX IF NOT EXISTS idx_articles_event_fingerprint ON articles(event_fingerprint);
            CREATE INDEX IF NOT EXISTS idx_articles_status ON articles(status);
            CREATE INDEX IF NOT EXISTS idx_articles_discovered_at ON articles(discovered_at);
            CREATE INDEX IF NOT EXISTS idx_articles_normalized_title ON articles(normalized_title);
            CREATE INDEX IF NOT EXISTS idx_events_fingerprint ON events(event_fingerprint);
            CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp);
            CREATE INDEX IF NOT EXISTS idx_skipped_urls_canonical ON skipped_urls(canonical_url);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_articles_canonical_url_unique
                ON articles(canonical_url);
        """)
        # Safe lightweight column migrations for pre-existing SQLite databases
        async with self.conn.execute("PRAGMA table_info(articles)") as cursor:
            existing_cols = {row["name"] for row in await cursor.fetchall()}
        if "video_url" not in existing_cols:
            await self.conn.execute("ALTER TABLE articles ADD COLUMN video_url TEXT DEFAULT NULL")
        if "event_key" not in existing_cols:
            await self.conn.execute("ALTER TABLE articles ADD COLUMN event_key TEXT DEFAULT ''")
        if "event_entities" not in existing_cols:
            await self.conn.execute("ALTER TABLE articles ADD COLUMN event_entities TEXT DEFAULT ''")
        if "thread_root_message_id" not in existing_cols:
            await self.conn.execute("ALTER TABLE articles ADD COLUMN thread_root_message_id INTEGER DEFAULT NULL")
        if "is_breaking" not in existing_cols:
            await self.conn.execute("ALTER TABLE articles ADD COLUMN is_breaking INTEGER DEFAULT 0")
        await self.conn.commit()
        logger.info("Database schema initialized.")

    # ───────────────────────────────────────────────
    # Article CRUD
    # ───────────────────────────────────────────────

    async def insert_article(self, article: Article) -> Optional[int]:
        """Insert a new article. Returns the ID or None if duplicate."""
        ev_key = _normalize_event_key(getattr(article, "event_key", ""))
        ev_entities_list = _normalize_event_entities(getattr(article, "event_entities", []))
        named_entities_val = article.named_entities or _pack_event_metadata(ev_key, ev_entities_list)
        ev_entities_json = json.dumps(ev_entities_list, ensure_ascii=True) if ev_entities_list else ""

        async with self._lock:
            try:
                cursor = await self.conn.execute("""
                    INSERT INTO articles (
                        canonical_url, original_url, original_title, normalized_title,
                        translated_title, original_summary, translated_summary,
                        source_name, category, priority, importance_score,
                        publication_date, discovered_at, processed_at, published_at,
                        status, content_hash, event_fingerprint, retry_count,
                        error_message, telegram_message_id, thread_root_message_id,
                        image_url, video_url,
                        country, region, named_entities, event_key, event_entities,
                        is_breaking
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    article.canonical_url, article.original_url, article.original_title,
                    article.normalized_title, article.translated_title, article.original_summary,
                    article.translated_summary, article.source_name, article.category,
                    article.priority, article.importance_score, article.publication_date,
                    article.discovered_at, article.processed_at, article.published_at,
                    article.status, article.content_hash, article.event_fingerprint,
                    article.retry_count, article.error_message, article.telegram_message_id,
                    article.thread_root_message_id,
                    article.image_url, article.video_url, article.country, article.region,
                    named_entities_val, ev_key, ev_entities_json,
                    int(bool(getattr(article, "is_breaking", False))),
                ))
                await self.conn.commit()
                article.id = cursor.lastrowid
                return article.id
            except aiosqlite.IntegrityError:
                logger.debug("Duplicate article URL: %s", article.canonical_url)
                return None

    async def update_article_status(self, article_id: int, status: str,
                                     error_message: str = "",
                                     **extra_fields: object) -> None:
        """Update an article's status and optional extra fields."""
        normalized_extra = dict(extra_fields)
        if "event_entities" in normalized_extra:
            ents = _normalize_event_entities(normalized_extra["event_entities"])
            normalized_extra["event_entities"] = json.dumps(ents, ensure_ascii=True) if ents else ""
        if "event_key" in normalized_extra:
            normalized_extra["event_key"] = _normalize_event_key(normalized_extra["event_key"])

        async with self._lock:
            sets = ["status = ?", "error_message = ?"]
            vals: list[object] = [status, error_message]
            if status == "PUBLISHED" and "published_at" not in normalized_extra:
                sets.append("published_at = ?")
                vals.append(datetime.now(timezone.utc).isoformat())
            if status in ("PROCESSING", ArticleStatus.QUEUED) and "processed_at" not in normalized_extra:
                sets.append("processed_at = ?")
                vals.append(datetime.now(timezone.utc).isoformat())
            if "is_breaking" in normalized_extra:
                normalized_extra["is_breaking"] = int(bool(normalized_extra["is_breaking"]))
            for key, value in normalized_extra.items():
                sets.append(f"{key} = ?")
                vals.append(value)
            vals.append(article_id)
            await self.conn.execute(
                f"UPDATE articles SET {', '.join(sets)} WHERE id = ?",
                vals,
            )
            await self.conn.commit()

    async def get_article_by_canonical_url(self, canonical_url: str) -> Optional[Article]:
        """Fetch an article by its canonical URL."""
        normalized_url = Article.canonicalize_url(canonical_url) if canonical_url else ""
        async with self.conn.execute(
            "SELECT * FROM articles WHERE canonical_url = ? OR canonical_url = ?",
            (normalized_url, canonical_url),
        ) as cursor:
            row = await cursor.fetchone()
            return self._row_to_article(row) if row else None

    async def get_articles_by_status(self, status: str, limit: int = 50) -> list[Article]:
        """Fetch articles with a given status."""
        async with self.conn.execute(
            "SELECT * FROM articles WHERE status = ? ORDER BY importance_score DESC LIMIT ?",
            (status, limit),
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_article(row) for row in rows]

    async def get_recent_content_hashes(self, hours: int = 48, exclude_id: Optional[int] = None) -> set[str]:
        """Get content hashes from recent articles for deduplication."""
        cutoff = datetime.now(timezone.utc).isoformat()
        query = """SELECT content_hash FROM articles
                   WHERE discovered_at >= datetime(?, '-' || ? || ' hours')
                   AND content_hash != ''"""
        params: list[object] = [cutoff, hours]
        if exclude_id is not None:
            query += " AND id != ?"
            params.append(exclude_id)
        async with self.conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return {row["content_hash"] for row in rows}

    async def get_recent_normalized_titles(
        self, hours: int = 48, exclude_id: Optional[int] = None, only_published: bool = False
    ) -> list[tuple[int, str]]:
        """Get normalized titles from recent articles for fuzzy dedup."""
        cutoff = datetime.now(timezone.utc).isoformat()
        query = """SELECT id, normalized_title FROM articles
                   WHERE discovered_at >= datetime(?, '-' || ? || ' hours')
                   AND normalized_title != ''"""
        params: list[object] = [cutoff, hours]
        if only_published:
            query += " AND status = 'PUBLISHED'"
        else:
            query += " AND status NOT IN ('DISCOVERED', 'REJECTED_DUPLICATE', 'FAILED', 'SKIPPED', 'EXPIRED')"
        if exclude_id is not None:
            query += " AND id != ?"
            params.append(exclude_id)
        async with self.conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [(row["id"], row["normalized_title"]) for row in rows]

    async def get_published_event_fingerprints(self, hours: int = 72) -> set[str]:
        """Get event fingerprints of recently published articles."""
        cutoff = datetime.now(timezone.utc).isoformat()
        async with self.conn.execute(
            """SELECT DISTINCT event_fingerprint FROM articles
               WHERE status = 'PUBLISHED'
               AND published_at >= datetime(?, '-' || ? || ' hours')
               AND event_fingerprint != ''""",
            (cutoff, hours),
        ) as cursor:
            rows = await cursor.fetchall()
            return {row["event_fingerprint"] for row in rows}

    async def is_empty(self) -> bool:
        """Check if articles table is currently empty."""
        async with self.conn.execute("SELECT COUNT(*) AS cnt FROM articles") as cursor:
            row = await cursor.fetchone()
            return bool(row["cnt"] == 0) if row else True

    @staticmethod
    def compute_url_hash(url: str) -> str:
        """Compute deterministic SHA-256 hash of a canonicalized URL."""
        return _compute_url_hash(url)

    async def mark_url_skipped(
        self, url: str, title: str = "", score: float = 0.0, reason: str = ""
    ) -> str:
        """Persist a skipped URL hash in SQLite so it is never re-evaluated."""
        canonical = Article.canonicalize_url(url)
        url_hash = self.compute_url_hash(canonical)
        now = datetime.now(timezone.utc).isoformat()
        async with self._lock:
            await self.conn.execute(
                """INSERT OR REPLACE INTO skipped_urls
                   (url_hash, canonical_url, title, score, reason, skipped_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (url_hash, canonical, title[:300], score, reason[:500], now),
            )
            await self.conn.commit()
        return url_hash

    async def is_url_skipped(self, url: str) -> bool:
        """Check if a URL hash or canonical URL has already been marked as skipped."""
        if not url:
            return False
        canonical = Article.canonicalize_url(url)
        url_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        async with self.conn.execute(
            "SELECT 1 FROM skipped_urls WHERE url_hash = ? LIMIT 1",
            (url_hash,),
        ) as cursor:
            if await cursor.fetchone():
                return True
        async with self.conn.execute(
            "SELECT 1 FROM articles WHERE canonical_url = ? AND status = 'SKIPPED' LIMIT 1",
            (canonical,),
        ) as cursor:
            return (await cursor.fetchone()) is not None

    async def get_recent_published_articles(self, limit: int = 20) -> list[Article]:
        """Fetch the most recently published articles."""
        async with self.conn.execute(
            """SELECT * FROM articles WHERE status = 'PUBLISHED'
               ORDER BY published_at DESC, id DESC LIMIT ?""",
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_article(row) for row in rows]

    @staticmethod
    def _parse_utc_timestamp(ts_str: Optional[str]) -> Optional[datetime]:
        """Parse an ISO/RFC timestamp string into a UTC-aware datetime, or None."""
        return _parse_utc_timestamp_util(ts_str)

    async def get_recent_published_headlines(self, hours: int = DEDUP_HISTORY_HOURS) -> list[dict]:
        """
        Fetch id, titles, Kurdish summaries, category, and stored event metadata
        for all PUBLISHED articles from the last `hours` hours (newest first).
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, hours))
        async with self.conn.execute(
            """SELECT id, source_name, original_title, translated_title, translated_summary,
                      original_summary, category, published_at, discovered_at,
                      telegram_message_id, thread_root_message_id,
                      named_entities, event_key, event_entities
               FROM articles WHERE status = 'PUBLISHED'
               ORDER BY published_at DESC, id DESC LIMIT 300"""
        ) as cursor:
            rows = await cursor.fetchall()

        return _filter_recent_headlines([dict(row) for row in rows], cutoff)

    async def get_published_articles_last_24h(self, limit: int = 10) -> list[Article]:
        """
        Fetch published articles from the last 24 hours ordered by importance/relevance
        score descending (prioritizing `kurdistan` + major global/military/tech stories).
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        async with self.conn.execute(
            """SELECT * FROM articles WHERE status = 'PUBLISHED'
               ORDER BY published_at DESC, id DESC LIMIT 150"""
        ) as cursor:
            rows = await cursor.fetchall()

        recent_24h: list[Article] = []
        for row in rows:
            art = self._row_to_article(row)
            dt = self._parse_utc_timestamp(art.published_at or art.discovered_at)
            if dt is not None and dt >= cutoff:
                recent_24h.append(art)

        return _prioritize_digest_articles(recent_24h, limit=limit)

    async def get_global_posts_count_last_24h(self) -> int:
        """
        Count the number of published articles with region == 'global' in the rolling 24 hours.
        Fail-closed to MAX_GLOBAL_POSTS_PER_DAY (4) on error to protect 90/10 Middle East ratio.
        """
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
            async with self.conn.execute(
                """SELECT region, published_at, discovered_at FROM articles
                   WHERE status = 'PUBLISHED'
                   ORDER BY published_at DESC, id DESC LIMIT 200"""
            ) as cursor:
                rows = await cursor.fetchall()

            global_count = 0
            for row in rows:
                dt = self._parse_utc_timestamp(row["published_at"] or row["discovered_at"])
                if dt is not None and dt >= cutoff:
                    reg = str(row["region"] or "").strip().lower()
                    if reg == "global":
                        global_count += 1
            return global_count
        except Exception as e:
            logger.error("Failed to query global posts count last 24h: %s; failing closed to max daily limit.", e)
            return int(getattr(config, "MAX_GLOBAL_POSTS_PER_DAY", 4))

    async def is_digest_published_for_date(self, digest_date: str) -> bool:
        """Return True if a daily digest has already been recorded for `digest_date` (YYYY-MM-DD)."""
        async with self.conn.execute(
            "SELECT 1 FROM daily_digests WHERE digest_date = ? LIMIT 1",
            (digest_date,),
        ) as cursor:
            return (await cursor.fetchone()) is not None

    async def mark_digest_published(
        self,
        digest_date: str,
        story_count: int,
        digest_text: str,
        telegram_message_id: Optional[int] = None,
    ) -> None:
        """Record a sent daily digest in SQLite so the same day is never duplicated."""
        now = datetime.now(timezone.utc).isoformat()
        async with self._lock:
            await self.conn.execute(
                """INSERT OR REPLACE INTO daily_digests
                   (digest_date, story_count, digest_text, telegram_message_id, published_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (digest_date, story_count, digest_text, telegram_message_id, now),
            )
            await self.conn.commit()

    async def get_last_published_primary_time(
        self, primary_categories: set[str] | list[str]
    ) -> Optional[datetime]:
        """Get the publication timestamp of the most recent primary focus article."""
        if not primary_categories:
            return None
        cats = list(primary_categories)
        placeholders = ",".join("?" for _ in cats)
        query = f"""SELECT published_at FROM articles
                    WHERE status = 'PUBLISHED'
                    AND category IN ({placeholders})
                    AND published_at IS NOT NULL
                    ORDER BY published_at DESC, id DESC LIMIT 1"""
        async with self.conn.execute(query, cats) as cursor:
            row = await cursor.fetchone()
            if row and row["published_at"]:
                try:
                    from dateutil import parser as dateparser
                    dt = dateparser.parse(row["published_at"])
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
                except Exception:
                    return None
        return None

    async def increment_retry(self, article_id: int, error: str) -> int:
        """Increment retry count and return the new count."""
        async with self._lock:
            await self.conn.execute(
                """UPDATE articles SET retry_count = retry_count + 1,
                   status = 'RETRY_PENDING', error_message = ? WHERE id = ?""",
                (error, article_id),
            )
            await self.conn.commit()
            async with self.conn.execute(
                "SELECT retry_count FROM articles WHERE id = ?", (article_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return row["retry_count"] if row else 0

    async def get_retry_pending_articles(self, max_retries: int = 3) -> list[Article]:
        """Fetch articles pending retry."""
        async with self.conn.execute(
            """SELECT * FROM articles WHERE status = 'RETRY_PENDING'
               AND retry_count < ? ORDER BY importance_score DESC""",
            (max_retries,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_article(row) for row in rows]

    async def get_interrupted_articles(self) -> list[Article]:
        """Fetch articles in a processing state (interrupted by crash)."""
        async with self.conn.execute(
            """SELECT * FROM articles WHERE status IN
               ('VALIDATING', 'PROCESSING', 'TRANSLATING', 'READY_TO_PUBLISH', 'PUBLISHING')
               ORDER BY importance_score DESC"""
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_article(row) for row in rows]

    # ───────────────────────────────────────────────
    # Smart Pacing & Persistent Queue
    # ───────────────────────────────────────────────

    async def get_queued_articles(
        self, max_age_hours: int = MAX_QUEUED_ARTICLE_AGE_HOURS
    ) -> list[Article]:
        """Fetch non-expired QUEUED articles ordered by priority and recency."""
        all_queued = await self.get_articles_by_status(ArticleStatus.QUEUED, limit=200)
        fresh, _ = _split_queued_by_age(all_queued, max_age_hours=max_age_hours)
        return fresh

    async def expire_stale_queued_articles(
        self, max_age_hours: int = MAX_QUEUED_ARTICLE_AGE_HOURS
    ) -> int:
        """Mark QUEUED articles older than `max_age_hours` as EXPIRED and return count."""
        all_queued = await self.get_articles_by_status(ArticleStatus.QUEUED, limit=200)
        _, stale = _split_queued_by_age(all_queued, max_age_hours=max_age_hours)
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
        overflow = _select_queue_overflow(all_queued, max_size=max_size)
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
        async with self.conn.execute(
            """SELECT published_at FROM articles
               WHERE status = 'PUBLISHED'
               AND COALESCE(is_breaking, 0) = 0
               AND published_at IS NOT NULL
               ORDER BY published_at DESC, id DESC LIMIT 20"""
        ) as cursor:
            rows = await cursor.fetchall()
        timestamps = [
            dt
            for row in rows
            if (dt := self._parse_utc_timestamp(row["published_at"])) is not None
        ]
        return max(timestamps) if timestamps else None

    # ───────────────────────────────────────────────
    # Event CRUD
    # ───────────────────────────────────────────────

    async def upsert_event(self, event: Event) -> int:
        """Insert or update an event. Returns the event ID."""
        async with self._lock:
            async with self.conn.execute(
                "SELECT id, article_count FROM events WHERE event_fingerprint = ?",
                (event.event_fingerprint,),
            ) as cursor:
                row = await cursor.fetchone()
            if row:
                await self.conn.execute(
                    """UPDATE events SET last_updated_at = ?, article_count = article_count + 1
                       WHERE id = ?""",
                    (datetime.now(timezone.utc).isoformat(), row["id"]),
                )
                await self.conn.commit()
                return row["id"]
            else:
                cursor = await self.conn.execute(
                    """INSERT INTO events (event_fingerprint, event_title, first_seen_at, last_updated_at)
                       VALUES (?, ?, ?, ?)""",
                    (event.event_fingerprint, event.event_title,
                     event.first_seen_at, event.last_updated_at),
                )
                await self.conn.commit()
                return cursor.lastrowid  # type: ignore

    async def get_event_by_fingerprint(self, fingerprint: str) -> Optional[Event]:
        """Fetch an event by its fingerprint."""
        async with self.conn.execute(
            "SELECT * FROM events WHERE event_fingerprint = ?", (fingerprint,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return Event(
                    id=row["id"],
                    event_fingerprint=row["event_fingerprint"],
                    event_title=row["event_title"],
                    first_seen_at=row["first_seen_at"],
                    last_updated_at=row["last_updated_at"],
                    article_count=row["article_count"],
                )
            return None

    # ───────────────────────────────────────────────
    # Source CRUD
    # ───────────────────────────────────────────────

    async def upsert_source(self, source: Source) -> int:
        """Insert or update a source. Returns the source ID."""
        async with self._lock:
            async with self.conn.execute(
                "SELECT id FROM sources WHERE name = ?", (source.name,)
            ) as cursor:
                row = await cursor.fetchone()
            if row:
                await self.conn.execute(
                    """UPDATE sources SET source_type = ?, url = ?, category = ?, enabled = ?
                       WHERE id = ?""",
                    (source.source_type, source.url, source.category, int(source.enabled), row["id"]),
                )
                await self.conn.commit()
                return row["id"]
            else:
                cursor = await self.conn.execute(
                    """INSERT INTO sources (name, source_type, url, category, enabled)
                       VALUES (?, ?, ?, ?, ?)""",
                    (source.name, source.source_type, source.url, source.category, int(source.enabled)),
                )
                await self.conn.commit()
                return cursor.lastrowid  # type: ignore

    async def update_source_status(self, name: str, success: bool, error: str = "") -> None:
        """Update a source's check status."""
        now = datetime.now(timezone.utc).isoformat()
        async with self._lock:
            if success:
                await self.conn.execute(
                    """UPDATE sources SET last_checked_at = ?, last_success_at = ?,
                       last_error = '', consecutive_failures = 0 WHERE name = ?""",
                    (now, now, name),
                )
            else:
                await self.conn.execute(
                    """UPDATE sources SET last_checked_at = ?, last_error = ?,
                       consecutive_failures = consecutive_failures + 1 WHERE name = ?""",
                    (now, error[:500], name),
                )
            await self.conn.commit()

    # ───────────────────────────────────────────────
    # System Logs
    # ───────────────────────────────────────────────

    async def log_system_event(
        self,
        level: str,
        component: str,
        message: str,
        timestamp: Optional[str] = None,
    ) -> None:
        """Write a system log entry to the database."""
        now = timestamp or datetime.now(timezone.utc).isoformat()
        try:
            async with self._lock:
                await self.conn.execute(
                    """INSERT INTO system_logs (timestamp, level, component, message)
                       VALUES (?, ?, ?, ?)""",
                    (now, level, component, message[:2000]),
                )
                await self.conn.commit()
        except Exception as e:
            logger.error("Failed to write system log: %s", e)

    async def get_recent_system_logs(
        self,
        component: str,
        limit: int = 50,
        hours: Optional[int] = None,
        now: Optional[datetime] = None,
    ) -> list[dict]:
        """Fetch recent system_logs entries for a component (newest first)."""
        async with self.conn.execute(
            """SELECT id, timestamp, level, component, message
               FROM system_logs WHERE component = ?
               ORDER BY timestamp DESC, id DESC LIMIT ?""",
            (component, max(1, int(limit))),
        ) as cursor:
            rows = [dict(r) for r in await cursor.fetchall()]

        if hours is not None:
            ref_now = now or datetime.now(timezone.utc)
            if ref_now.tzinfo is None:
                ref_now = ref_now.replace(tzinfo=timezone.utc)
            cutoff = ref_now - timedelta(hours=max(1, int(hours)))
            filtered: list[dict] = []
            for row in rows:
                dt = self._parse_utc_timestamp(row.get("timestamp"))
                if dt is not None and dt >= cutoff:
                    filtered.append(row)
            return filtered
        return rows

    async def get_last_batch_completed_at(self) -> Optional[datetime]:
        """Return UTC timestamp of the most recently completed paced batch, or None."""
        rows = await self.get_recent_system_logs(component="BATCH_COOLDOWN", limit=1)
        if not rows:
            return None
        return self._parse_utc_timestamp(rows[0].get("timestamp"))

    async def record_batch_completed_at(
        self,
        timestamp: Optional[datetime] = None,
        published_count: int = 0,
    ) -> datetime:
        """Persist completion timestamp of a paced batch into existing `system_logs` table."""
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
        """Return enabled sources and their failure counters for watchdog health monitoring."""
        async with self.conn.execute(
            """SELECT name, source_type, enabled, consecutive_failures,
                      last_error, last_checked_at, last_success_at
               FROM sources WHERE COALESCE(enabled, 1) = 1
               ORDER BY consecutive_failures DESC, name ASC"""
        ) as cursor:
            return [dict(r) for r in await cursor.fetchall()]

    async def get_watchdog_article_metrics(
        self,
        hours: int = 24,
        now: Optional[datetime] = None,
    ) -> dict[str, object]:
        """Compute 24h article metrics, backlog count, and last published time for watchdog."""
        ref_now = now or datetime.now(timezone.utc)
        if ref_now.tzinfo is None:
            ref_now = ref_now.replace(tzinfo=timezone.utc)
        cutoff = ref_now - timedelta(hours=max(1, int(hours)))

        async with self.conn.execute(
            """SELECT * FROM articles WHERE status = 'PUBLISHED'
               ORDER BY published_at DESC, id DESC LIMIT 300"""
        ) as cursor:
            pub_rows = await cursor.fetchall()

        published_recent: list[Article] = []
        all_pub_timestamps: list[datetime] = []
        for row in pub_rows:
            art = self._row_to_article(row)
            dt = self._parse_utc_timestamp(art.published_at or art.discovered_at)
            if dt is not None:
                all_pub_timestamps.append(dt)
                if dt >= cutoff:
                    published_recent.append(art)

        last_published_at = max(all_pub_timestamps) if all_pub_timestamps else None

        async with self.conn.execute(
            """SELECT status, discovered_at FROM articles
               WHERE status IN ('REJECTED_DUPLICATE', 'SKIPPED')
               ORDER BY id DESC LIMIT 1000"""
        ) as cursor:
            eff_rows = await cursor.fetchall()

        duplicates_recent = 0
        skipped_recent = 0
        for row in eff_rows:
            dt = self._parse_utc_timestamp(row["discovered_at"])
            if dt is None or dt >= cutoff:
                if row["status"] == ArticleStatus.REJECTED_DUPLICATE:
                    duplicates_recent += 1
                elif row["status"] == ArticleStatus.SKIPPED:
                    skipped_recent += 1

        stats = await self.get_stats()
        return {
            "published_articles": published_recent,
            "duplicates_count": duplicates_recent,
            "skipped_count": skipped_recent,
            "queued_count": int(stats.get(ArticleStatus.QUEUED, 0)),
            "total_articles": int(stats.get("total", 0)),
            "last_published_at": last_published_at,
        }

    # ───────────────────────────────────────────────
    # Statistics
    # ───────────────────────────────────────────────

    async def get_stats(self) -> dict[str, int]:
        """Get article counts by status."""
        stats: dict[str, int] = {}
        async with self.conn.execute(
            "SELECT status, COUNT(*) as cnt FROM articles GROUP BY status"
        ) as cursor:
            rows = await cursor.fetchall()
            for row in rows:
                stats[row["status"]] = row["cnt"]
        async with self.conn.execute("SELECT COUNT(*) as cnt FROM articles") as cursor:
            row = await cursor.fetchone()
            stats["total"] = row["cnt"] if row else 0
        return stats

    # ───────────────────────────────────────────────
    # Backup
    # ───────────────────────────────────────────────

    async def backup(self) -> str:
        """Create a database backup. Returns the backup path."""
        backup_dir = Path(BACKUP_DIR)
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"news_bot_backup_{timestamp}.db"

        # Use SQLite backup API via a sync copy (safe with WAL mode)
        await self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        await asyncio.to_thread(shutil.copy2, self.db_path, str(backup_path))
        logger.info("Database backed up to: %s", backup_path)
        return str(backup_path)

    # ───────────────────────────────────────────────
    # Helpers
    # ───────────────────────────────────────────────

    @staticmethod
    def _row_to_article(row: aiosqlite.Row) -> Article:
        """Convert a database row to an Article object."""
        row_keys = set(row.keys())
        ev_key, ev_entities = _unpack_event_metadata(
            row["named_entities"] if "named_entities" in row_keys else "",
            row["event_key"] if "event_key" in row_keys else "",
            row["event_entities"] if "event_entities" in row_keys else None,
        )
        thread_root_val = (
            row["thread_root_message_id"]
            if "thread_root_message_id" in row_keys
            else None
        )
        return Article(
            id=row["id"],
            canonical_url=row["canonical_url"],
            original_url=row["original_url"],
            original_title=row["original_title"],
            normalized_title=row["normalized_title"],
            translated_title=row["translated_title"],
            original_summary=row["original_summary"] or "",
            translated_summary=row["translated_summary"] or "",
            source_name=row["source_name"],
            category=row["category"],
            priority=row["priority"],
            importance_score=row["importance_score"],
            publication_date=row["publication_date"],
            discovered_at=row["discovered_at"],
            processed_at=row["processed_at"],
            published_at=row["published_at"],
            status=row["status"],
            content_hash=row["content_hash"],
            event_fingerprint=row["event_fingerprint"],
            retry_count=row["retry_count"],
            error_message=row["error_message"] or "",
            telegram_message_id=row["telegram_message_id"],
            thread_root_message_id=thread_root_val,
            image_url=row["image_url"] or "",
            video_url=(row["video_url"] or None) if "video_url" in row_keys else None,
            country=row["country"] or "",
            region=row["region"] or "",
            named_entities=row["named_entities"] or "",
            event_key=ev_key,
            event_entities=ev_entities,
            is_breaking=bool(row["is_breaking"]) if "is_breaking" in row_keys else False,
            is_thread_update=(thread_root_val is not None),
        )
