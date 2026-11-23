"""Tests for database operations."""

import asyncio
import os
import pytest
import tempfile
from storage.database import Database
from storage.models import Article, Source, Event
from config import ArticleStatus


@pytest.fixture
async def db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    database = Database(db_path=db_path)
    await database.connect()
    yield database
    await database.close()

    # Cleanup
    try:
        os.unlink(db_path)
    except Exception:
        pass


class TestDatabase:
    """Test database CRUD operations."""

    @pytest.mark.asyncio
    async def test_connect_and_init(self, db):
        """TEST 9: Database initializes correctly."""
        stats = await db.get_stats()
        assert stats["total"] == 0

    @pytest.mark.asyncio
    async def test_insert_article(self, db):
        """Test article insertion."""
        article = Article(
            canonical_url="https://test.com/article1",
            original_url="https://test.com/article1?ref=x",
            original_title="Test Article",
            normalized_title="test article",
            source_name="TestSource",
            status=ArticleStatus.DISCOVERED,
        )
        article.compute_content_hash()
        article_id = await db.insert_article(article)
        assert article_id is not None
        assert article_id > 0

    @pytest.mark.asyncio
    async def test_duplicate_url_rejected(self, db):
        """TEST 6: Duplicate URLs should be rejected by the database."""
        article1 = Article(
            canonical_url="https://test.com/same-url",
            original_url="https://test.com/same-url",
            original_title="First Article",
            normalized_title="first article",
        )
        article2 = Article(
            canonical_url="https://test.com/same-url",
            original_url="https://test.com/same-url",
            original_title="Duplicate Article",
            normalized_title="duplicate article",
        )

        id1 = await db.insert_article(article1)
        id2 = await db.insert_article(article2)

        assert id1 is not None
        assert id2 is None  # Should be rejected as duplicate

    @pytest.mark.asyncio
    async def test_update_article_status(self, db):
        """Test status transitions."""
        article = Article(
            canonical_url="https://test.com/status-test",
            original_url="https://test.com/status-test",
            original_title="Status Test",
            normalized_title="status test",
            status=ArticleStatus.DISCOVERED,
        )
        article_id = await db.insert_article(article)
        assert article_id is not None

        await db.update_article_status(article_id, ArticleStatus.PROCESSING)
        fetched = await db.get_article_by_canonical_url("https://test.com/status-test")
        assert fetched is not None
        assert fetched.status == ArticleStatus.PROCESSING

    @pytest.mark.asyncio
    async def test_get_articles_by_status(self, db):
        """Test fetching articles by status."""
        for i in range(3):
            article = Article(
                canonical_url=f"https://test.com/article-{i}",
                original_url=f"https://test.com/article-{i}",
                original_title=f"Article {i}",
                normalized_title=f"article {i}",
                status=ArticleStatus.DISCOVERED,
            )
            await db.insert_article(article)

        discovered = await db.get_articles_by_status(ArticleStatus.DISCOVERED)
        assert len(discovered) == 3

    @pytest.mark.asyncio
    async def test_database_survives_restart(self, db):
        """TEST 9: The application restarts without losing database history."""
        # Insert an article
        article = Article(
            canonical_url="https://test.com/persist-test",
            original_url="https://test.com/persist-test",
            original_title="Persistence Test",
            normalized_title="persistence test",
            status=ArticleStatus.PUBLISHED,
        )
        await db.insert_article(article)

        # "Restart" — close and reopen
        db_path = db.db_path
        await db.close()

        db2 = Database(db_path=db_path)
        await db2.connect()

        fetched = await db2.get_article_by_canonical_url("https://test.com/persist-test")
        assert fetched is not None
        assert fetched.original_title == "Persistence Test"
        assert fetched.status == ArticleStatus.PUBLISHED

        await db2.close()

    @pytest.mark.asyncio
    async def test_interrupted_articles_recovery(self, db):
        """TEST 10: Interrupted jobs can be recovered."""
        # Insert articles in various processing states
        for status in [ArticleStatus.VALIDATING, ArticleStatus.TRANSLATING, ArticleStatus.PUBLISHING]:
            article = Article(
                canonical_url=f"https://test.com/interrupted-{status}",
                original_url=f"https://test.com/interrupted-{status}",
                original_title=f"Interrupted {status}",
                normalized_title=f"interrupted {status}",
                status=status,
            )
            await db.insert_article(article)

        interrupted = await db.get_interrupted_articles()
        assert len(interrupted) == 3

    @pytest.mark.asyncio
    async def test_retry_increment(self, db):
        """Test retry counter."""
        article = Article(
            canonical_url="https://test.com/retry-test",
            original_url="https://test.com/retry-test",
            original_title="Retry Test",
            normalized_title="retry test",
            status=ArticleStatus.DISCOVERED,
        )
        article_id = await db.insert_article(article)

        count = await db.increment_retry(article_id, "Test error")
        assert count == 1

        count = await db.increment_retry(article_id, "Second error")
        assert count == 2

    @pytest.mark.asyncio
    async def test_upsert_event(self, db):
        """Test event tracking."""
        event = Event(
            event_fingerprint="fp_test_123",
            event_title="Test Event",
        )
        event_id = await db.upsert_event(event)
        assert event_id > 0

        # Upsert again should update
        event_id2 = await db.upsert_event(event)
        assert event_id2 == event_id

    @pytest.mark.asyncio
    async def test_source_tracking(self, db):
        """Test source status tracking."""
        source = Source(name="TestSource", source_type="rss")
        source_id = await db.upsert_source(source)
        assert source_id > 0

        await db.update_source_status("TestSource", success=True)
        await db.update_source_status("TestSource", success=False, error="Timeout")

    @pytest.mark.asyncio
    async def test_system_log(self, db):
        """Test system log entries."""
        await db.log_system_event("INFO", "test", "Test log message")
        # Should not raise

    @pytest.mark.asyncio
    async def test_backup(self, db):
        """Test database backup."""
        backup_path = await db.backup()
        assert os.path.exists(backup_path)
        # Cleanup
        try:
            os.unlink(backup_path)
        except Exception:
            pass
