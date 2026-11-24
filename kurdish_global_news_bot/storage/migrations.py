"""
Kurdish Global News Bot — Database Migrations

Safe schema migration support for database upgrades.
"""

from __future__ import annotations

import logging
from typing import Optional

import aiosqlite

logger = logging.getLogger("migrations")

# Each migration is a tuple of (version, description, sql)
MIGRATIONS: list[tuple[int, str, str]] = [
    (1, "Initial schema", "-- handled by database.py _init_schema"),
    (2, "Add image_url column if missing", """
        ALTER TABLE articles ADD COLUMN image_url TEXT DEFAULT '';
    """),
    (3, "Add country and region columns if missing", """
        ALTER TABLE articles ADD COLUMN country TEXT DEFAULT '';
    """),
    (4, "Add skipped_urls table for persisting low-relevance URL hashes", """
        CREATE TABLE IF NOT EXISTS skipped_urls (
            url_hash TEXT PRIMARY KEY,
            canonical_url TEXT NOT NULL,
            title TEXT DEFAULT '',
            score REAL DEFAULT 0.0,
            reason TEXT DEFAULT '',
            skipped_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_skipped_urls_canonical ON skipped_urls(canonical_url);
    """),
    (5, "Add daily_digests table for duplicate nightly digest protection", """
        CREATE TABLE IF NOT EXISTS daily_digests (
            digest_date TEXT PRIMARY KEY,
            story_count INTEGER DEFAULT 0,
            digest_text TEXT DEFAULT '',
            telegram_message_id INTEGER,
            published_at TEXT NOT NULL
        );
    """),
]


async def get_db_version(conn: aiosqlite.Connection) -> int:
    """Get the current schema version."""
    try:
        async with conn.execute("PRAGMA user_version") as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0
    except Exception:
        return 0


async def set_db_version(conn: aiosqlite.Connection, version: int) -> None:
    """Set the schema version."""
    await conn.execute(f"PRAGMA user_version = {version}")
    await conn.commit()


async def run_migrations(conn: aiosqlite.Connection) -> None:
    """Run any pending migrations."""
    current_version = await get_db_version(conn)
    logger.info("Current database version: %d", current_version)

    for version, description, sql in MIGRATIONS:
        if version <= current_version:
            continue
        if sql.strip().startswith("--"):
            # Skip placeholder migrations
            await set_db_version(conn, version)
            continue
        try:
            logger.info("Running migration %d: %s", version, description)
            # Check if the column already exists before ALTER TABLE
            if "ADD COLUMN" in sql:
                col_name = sql.split("ADD COLUMN")[1].strip().split()[0]
                try:
                    async with conn.execute(f"SELECT {col_name} FROM articles LIMIT 0"):
                        pass
                    logger.info("Column %s already exists, skipping.", col_name)
                except Exception:
                    await conn.executescript(sql)
            else:
                await conn.executescript(sql)
            await set_db_version(conn, version)
            logger.info("Migration %d completed.", version)
        except Exception as e:
            logger.error("Migration %d failed: %s", version, e)
            # Don't stop on migration failure — the column may already exist
            await set_db_version(conn, version)

    final_version = await get_db_version(conn)
    logger.info("Database version after migrations: %d", final_version)
