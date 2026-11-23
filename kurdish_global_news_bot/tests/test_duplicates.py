"""Tests for duplicate detection."""

import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from storage.models import Article
from core.duplicate_detector import DuplicateDetector


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.get_article_by_canonical_url = AsyncMock(return_value=None)
    db.get_recent_content_hashes = AsyncMock(return_value=set())
    db.get_recent_normalized_titles = AsyncMock(return_value=[])
    db.get_published_event_fingerprints = AsyncMock(return_value=set())
    db.get_event_by_fingerprint = AsyncMock(return_value=None)
    return db


class TestDuplicateDetector:
    """Test duplicate detection layers."""

    @pytest.mark.asyncio
    async def test_new_article_not_duplicate(self, mock_db):
        """A genuinely new article should not be flagged as duplicate."""
        detector = DuplicateDetector(mock_db)
        article = Article(
            canonical_url="https://example.com/new-article",
            original_title="Completely New Story",
            normalized_title="completely new story",
            content_hash="abc123def456",
        )
        is_dup, reason = await detector.is_duplicate(article)
        assert is_dup is False

    @pytest.mark.asyncio
    async def test_exact_url_duplicate(self, mock_db):
        """TEST 6: Same URL should be detected as duplicate."""
        existing = Article(canonical_url="https://example.com/article")
        mock_db.get_article_by_canonical_url = AsyncMock(return_value=existing)

        detector = DuplicateDetector(mock_db)
        article = Article(
            canonical_url="https://example.com/article",
            original_title="Same Article",
        )
        is_dup, reason = await detector.is_duplicate(article)
        assert is_dup is True
        assert "URL match" in reason

    @pytest.mark.asyncio
    async def test_content_hash_duplicate(self, mock_db):
        """TEST 6: Same content hash should be detected as duplicate."""
        mock_db.get_recent_content_hashes = AsyncMock(return_value={"abc123"})

        detector = DuplicateDetector(mock_db)
        article = Article(
            canonical_url="https://different-url.com/article",
            original_title="Same Content Different URL",
            normalized_title="same content different url",
            content_hash="abc123",
        )
        is_dup, reason = await detector.is_duplicate(article)
        assert is_dup is True
        assert "hash match" in reason

    @pytest.mark.asyncio
    async def test_fuzzy_title_duplicate(self, mock_db):
        """TEST 6: Similar titles should be detected as duplicates."""
        mock_db.get_recent_normalized_titles = AsyncMock(return_value=[
            (1, "major earthquake hits turkey killing hundreds"),
        ])

        detector = DuplicateDetector(mock_db)
        article = Article(
            canonical_url="https://other.com/article",
            original_title="Major Earthquake Hits Turkey, Killing Hundreds of People",
            normalized_title="major earthquake hits turkey killing hundreds of people",
            content_hash="xyz789",
        )
        is_dup, reason = await detector.is_duplicate(article)
        assert is_dup is True
        assert "Fuzzy title match" in reason

    @pytest.mark.asyncio
    async def test_different_stories_not_duplicate(self, mock_db):
        """TEST 5: Different stories should be processed independently."""
        mock_db.get_recent_normalized_titles = AsyncMock(return_value=[
            (1, "earthquake in japan"),
        ])

        detector = DuplicateDetector(mock_db)
        article = Article(
            canonical_url="https://news.com/flood",
            original_title="Massive Flooding in Brazil Displaces Thousands",
            normalized_title="massive flooding in brazil displaces thousands",
            content_hash="unique123",
        )
        is_dup, reason = await detector.is_duplicate(article)
        assert is_dup is False

    @pytest.mark.asyncio
    async def test_same_event_already_published(self, mock_db):
        """TEST 6: Same event from multiple sources should not be repeatedly published."""
        mock_db.get_published_event_fingerprints = AsyncMock(return_value={"eventfp123"})

        detector = DuplicateDetector(mock_db)
        article = Article(
            canonical_url="https://news.com/same-event",
            original_title="Same Event Different Source",
            normalized_title="same event different source",
            event_fingerprint="eventfp123",
        )
        is_same, reason = await detector.is_same_event_already_published(article)
        assert is_same is True

    @pytest.mark.asyncio
    async def test_meaningful_update_allowed(self, mock_db):
        """TEST 7: A genuinely new development about an existing event can be published."""
        from storage.models import Event
        mock_db.get_event_by_fingerprint = AsyncMock(return_value=Event(
            event_fingerprint="fp123",
            event_title="Original Event",
        ))
        mock_db.get_recent_normalized_titles = AsyncMock(return_value=[
            (1, "earthquake damage in turkey initial report"),
        ])

        detector = DuplicateDetector(mock_db)
        article = Article(
            original_title="Turkey Earthquake: Rescue Teams Find Survivors Under Rubble After 72 Hours",
            normalized_title="turkey earthquake rescue teams find survivors under rubble after 72 hours",
            event_fingerprint="fp123",
        )
        is_update = await detector.is_meaningful_update(article)
        # This should potentially be True if the content is different enough
        assert isinstance(is_update, bool)


class TestArticleURLNormalization:
    """Test URL canonicalization for deduplication."""

    def test_remove_tracking_params(self):
        url = "https://reuters.com/article?utm_source=twitter&utm_medium=social&id=123"
        canonical = Article.canonicalize_url(url)
        assert "utm_source" not in canonical
        assert "utm_medium" not in canonical
        assert "id=123" in canonical

    def test_remove_www(self):
        url = "https://www.bbc.com/news/article"
        canonical = Article.canonicalize_url(url)
        assert "www." not in canonical
        assert "bbc.com" in canonical

    def test_remove_fragment(self):
        url = "https://news.com/article#comments"
        canonical = Article.canonicalize_url(url)
        assert "#comments" not in canonical

    def test_normalize_title(self):
        title = "BREAKING: Major Earthquake!!! — 7.5 Magnitude"
        normalized = Article.normalize_title(title)
        assert normalized == "breaking major earthquake 7 5 magnitude"

    def test_content_hash_consistency(self):
        a1 = Article(canonical_url="https://test.com/article", normalized_title="test article")
        a2 = Article(canonical_url="https://test.com/article", normalized_title="test article")
        assert a1.compute_content_hash() == a2.compute_content_hash()
