"""
Kurdish Global News Bot — Database Module

Async SQLite database layer with connection pooling, migrations, and CRUD operations.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import aiosqlite

from config import DATABASE_PATH, BACKUP_DIR
from storage.models import Article, Source, Event

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
                image_url TEXT DEFAULT '',
                country TEXT DEFAULT '',
                region TEXT DEFAULT '',
                named_entities TEXT DEFAULT ''
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
        await self.conn.commit()
        logger.info("Database schema initialized.")

    # ───────────────────────────────────────────────
    # Article CRUD
    # ───────────────────────────────────────────────

    async def insert_article(self, article: Article) -> Optional[int]:
        """Insert a new article. Returns the ID or None if duplicate."""
        async with self._lock:
            try:
                cursor = await self.conn.execute("""
                    INSERT INTO articles (
                        canonical_url, original_url, original_title, normalized_title,
                        translated_title, original_summary, translated_summary,
                        source_name, category, priority, importance_score,
                        publication_date, discovered_at, processed_at, published_at,
                        status, content_hash, event_fingerprint, retry_count,
                        error_message, telegram_message_id, image_url, country,
                        region, named_entities
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    article.canonical_url, article.original_url, article.original_title,
                    article.normalized_title, article.translated_title, article.original_summary,
                    article.translated_summary, article.source_name, article.category,
                    article.priority, article.importance_score, article.publication_date,
                    article.discovered_at, article.processed_at, article.published_at,
                    article.status, article.content_hash, article.event_fingerprint,
                    article.retry_count, article.error_message, article.telegram_message_id,
                    article.image_url, article.country, article.region, article.named_entities,
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
        async with self._lock:
            sets = ["status = ?", "error_message = ?"]
            vals: list[object] = [status, error_message]
            if status == "PUBLISHED":
                sets.append("published_at = ?")
                vals.append(datetime.now(timezone.utc).isoformat())
            if status == "PROCESSING":
                sets.append("processed_at = ?")
                vals.append(datetime.now(timezone.utc).isoformat())
            for key, value in extra_fields.items():
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
        async with self.conn.execute(
            "SELECT * FROM articles WHERE canonical_url = ?", (canonical_url,)
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
            query += " AND status NOT IN ('REJECTED_DUPLICATE', 'FAILED', 'SKIPPED')"
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
        canonical = Article.canonicalize_url(url) if url else ""
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    async def mark_url_skipped(
        self, url: str, title: str = "", score: float = 0.0, reason: str = ""
    ) -> str:
        """Persist a skipped URL hash in SQLite so it is never re-evaluated."""
        canonical = Article.canonicalize_url(url)
        url_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
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

        from dateutil import parser as dateparser
        recent_24h: list[Article] = []
        for row in rows:
            art = self._row_to_article(row)
            ts_str = art.published_at or art.discovered_at
            if not ts_str:
                continue
            try:
                dt = dateparser.parse(ts_str)
                if not dt:
                    continue
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt >= cutoff:
                    recent_24h.append(art)
            except Exception:
                continue

        # Sort descending by Kurdistan priority + importance_score, then recency
        def _sort_key(a: Article) -> tuple[float, int, float, str]:
            is_kurd = 1 if (a.category or "").strip().lower() == "kurdistan" else 0
            boosted = (a.importance_score or 0.0) + (0.20 if is_kurd else 0.0)
            return (boosted, is_kurd, a.importance_score or 0.0, a.published_at or "")

        recent_24h.sort(key=_sort_key, reverse=True)
        return recent_24h[:limit]

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

    async def log_system_event(self, level: str, component: str, message: str) -> None:
        """Write a system log entry to the database."""
        now = datetime.now(timezone.utc).isoformat()
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
            image_url=row["image_url"] or "",
            country=row["country"] or "",
            region=row["region"] or "",
            named_entities=row["named_entities"] or "",
        )
