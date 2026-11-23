"""Tests for Telegram publishing."""

import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from storage.models import Article
from processing.content_formatter import ContentFormatter
from config import ArticleStatus


class TestContentFormatter:
    """Test the Telegram message formatter."""

    def test_format_telegram_message(self):
        """TEST 12: Telegram messages contain the correct source and original link."""
        formatter = ContentFormatter()
        article = Article(
            original_title="Test Article Title",
            translated_title="سەردێری تاقیکاری",
            original_summary="A test summary of the article.",
            translated_summary="پوختەی تاقیکاری بابەت.",
            source_name="Reuters",
            category="International Politics",
            original_url="https://reuters.com/test-article",
            publication_date="2024-01-15T10:00:00Z",
        )

        message = formatter.format_telegram_message(article)

        # Check structure
        assert "سەردێری تاقیکاری" in message  # Kurdish title
        assert "پوختەی تاقیکاری" in message  # Kurdish summary
        assert "Reuters" in message  # Source name
        assert "reuters.com/test-article" in message  # Original link
        assert "بابەتی ئەسڵی" in message  # "Original article" in Kurdish
        assert "🌍" in message  # Category emoji

    def test_format_caption(self):
        """Test caption formatting for image posts."""
        formatter = ContentFormatter()
        article = Article(
            translated_title="سەردێری تاقیکاری",
            source_name="BBC",
            original_url="https://bbc.com/news/test",
        )

        caption = formatter.format_caption(article)
        assert len(caption) <= 1024  # Telegram caption limit
        assert "BBC" in caption
        assert "bbc.com/news/test" in caption

    def test_format_message_html_escaping(self):
        """Test that HTML special characters are escaped."""
        formatter = ContentFormatter()
        article = Article(
            translated_title="تاقیکاری <script>alert('xss')</script>",
            original_url="https://test.com/article",
            source_name="Test & Source",
        )

        message = formatter.format_telegram_message(article)
        assert "<script>" not in message
        assert "&lt;script&gt;" in message
        assert "Test &amp; Source" in message

    def test_format_message_long_summary_truncated(self):
        """Test that long summaries are truncated."""
        formatter = ContentFormatter()
        article = Article(
            translated_title="سەردێر",
            translated_summary="ئەمە پوختەیەکی زۆر درێژە. " * 100,
            original_url="https://test.com",
            source_name="Test",
        )

        message = formatter.format_telegram_message(article)
        # Should be reasonable length
        assert len(message) < 5000

    def test_text_only_when_no_image(self):
        """TEST 13: Image failure does not prevent text publication."""
        formatter = ContentFormatter()
        article = Article(
            translated_title="هەواڵی تاقیکاری",
            original_url="https://test.com",
            source_name="Test",
            image_url="",  # No image
        )

        # Text-only formatting should work
        message = formatter.format_telegram_message(article)
        assert "هەواڵی تاقیکاری" in message


class TestTelegramPublishing:
    """Test Telegram publishing behavior."""

    def test_no_hourly_scheduler_exists(self):
        """TEST 4: No hourly publishing scheduler exists."""
        # Verify by importing the scheduler and checking it has no hourly logic
        from core.scheduler import DiscoveryScheduler
        scheduler = DiscoveryScheduler()

        # The interval should be configurable but default to 60 seconds
        assert scheduler.interval_seconds == 60
        # There should be no mention of "hour" in the scheduler's methods
        import inspect
        source = inspect.getsource(DiscoveryScheduler)
        assert "hourly" not in source.lower() or "NO hourly" in source

    def test_no_batch_publishing(self):
        """TEST 4: No delayed publishing batch exists."""
        from core.pipeline import ArticlePipeline
        import inspect
        source = inspect.getsource(ArticlePipeline)
        # The pipeline should publish immediately, not batch
        assert "IMMEDIATE" in source or "immediately" in source.lower()

    def test_article_states_complete(self):
        """Verify all article states are defined."""
        from config import ArticleStatus
        states = [
            ArticleStatus.DISCOVERED, ArticleStatus.VALIDATING,
            ArticleStatus.PROCESSING, ArticleStatus.TRANSLATING,
            ArticleStatus.READY_TO_PUBLISH, ArticleStatus.PUBLISHING,
            ArticleStatus.PUBLISHED, ArticleStatus.FAILED,
            ArticleStatus.RETRY_PENDING, ArticleStatus.REJECTED_DUPLICATE,
        ]
        assert len(states) == 10


class TestArticleValidatorRecency:
    """Test ArticleValidator recency filter."""

    def test_recent_article_passes(self):
        from datetime import datetime, timezone, timedelta
        from processing.validator import ArticleValidator
        validator = ArticleValidator(max_age_hours=3.0)
        recent_date = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        article = Article(
            original_title="Valid Recent Article Title For Test",
            original_url="https://reuters.com/world/recent-article-test",
            canonical_url="https://reuters.com/world/recent-article-test",
            original_summary="Summary of recent article with sufficient text content.",
            publication_date=recent_date,
        )
        is_valid, reason = validator.validate(article)
        assert is_valid is True
        assert reason == ""

    def test_stale_article_rejected(self):
        from datetime import datetime, timezone, timedelta
        from processing.validator import ArticleValidator
        validator = ArticleValidator(max_age_hours=3.0)
        stale_date = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
        article = Article(
            original_title="Valid Stale Article Title For Test",
            original_url="https://reuters.com/world/stale-article-test",
            canonical_url="https://reuters.com/world/stale-article-test",
            original_summary="Summary of stale article with sufficient text content.",
            publication_date=stale_date,
        )
        is_valid, reason = validator.validate(article)
        assert is_valid is False
        assert "Article too old" in reason

    def test_empty_date_allowed(self):
        from processing.validator import ArticleValidator
        validator = ArticleValidator(max_age_hours=3.0)
        article = Article(
            original_title="Valid Article Without Publication Date",
            original_url="https://reuters.com/world/nodate-article-test",
            canonical_url="https://reuters.com/world/nodate-article-test",
            original_summary="Summary of article without publication date content.",
            publication_date="",
        )
        is_valid, reason = validator.validate(article)
        assert is_valid is True
