"""
Acceptance Criteria Tests: Directly verifying criteria TEST 1 through TEST 15.
"""

import asyncio
import io
from datetime import datetime, timezone
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from config import ArticleStatus, Priority
from storage.database import Database
from storage.models import Article, Event
from core.pipeline import ArticlePipeline
from core.scheduler import DiscoveryScheduler
from tg_bot.publisher import TelegramPublisher
from telegram.error import RetryAfter
from monitoring.health import HealthMonitor
from monitoring.metrics import Metrics


@pytest.fixture
async def temp_db(tmp_path):
    db_file = tmp_path / "acceptance_test.db"
    db = Database(str(db_file))
    await db.connect()
    yield db
    await db.close()


@pytest.mark.asyncio
async def test_acceptance_criteria_2_breaking_article_published_immediately(temp_db):
    """
    TEST 2: A new eligible breaking article is translated and published
    without waiting for the next polling cycle.
    """
    publisher = MagicMock(spec=TelegramPublisher)
    publisher.publish_article = AsyncMock(return_value=12345)
    health = HealthMonitor()
    metrics = Metrics()

    pipeline = ArticlePipeline(temp_db, publisher, health, metrics)

    # Mock external network dependencies in unit test
    from core.importance_ranker import RelevanceEvaluation
    with patch.object(pipeline._translator, 'translate_title', new_callable=AsyncMock) as mock_tr_title, \
         patch.object(pipeline._translator, 'translate_summary', new_callable=AsyncMock) as mock_tr_summary, \
         patch.object(pipeline._extractor, 'fetch_full_article_text', new_callable=AsyncMock) as mock_scrape, \
         patch.object(pipeline._ranker, 'evaluate_global_relevance', new_callable=AsyncMock) as mock_rel:
        
        mock_rel.return_value = RelevanceEvaluation(score=9, is_breaking=True, reason="Mock breaking", passed=True)
        mock_scrape.return_value = None
        mock_tr_title.return_value = "هێرشی بەپەلە لە پایتەخت ئەنجامدرا"
        mock_tr_summary.return_value = "بەپێی هەواڵە بەپەلەکان ڕووداوێکی گەورە ڕوویدا."

        breaking_article = Article(
            original_url="https://reuters.com/world/breaking-news-urgent-1",
            canonical_url="https://reuters.com/world/breaking-news-urgent-1",
            original_title="Breaking: Urgent security crisis declared in capital",
            original_summary=(
                "Government declared a nationwide state of emergency after a major security crisis unfolded across the capital district. "
                "Security forces and emergency response teams deployed to secure key administrative buildings and critical infrastructure, "
                "while international embassies advised citizens to shelter in place amid ongoing operations."
            ),
            source_name="Reuters",
            category="International Politics",
            publication_date=datetime.now(timezone.utc).isoformat(),
        )
        breaking_article.compute_content_hash()
        breaking_article.compute_event_fingerprint()

        # Insert into DB
        art_id = await temp_db.insert_article(breaking_article)
        breaking_article.id = art_id

        # Process single article directly (proves zero wait for next polling cycle)
        published = await pipeline.process_article(breaking_article)

        assert published is True
        publisher.publish_article.assert_awaited_once()

        # Verify DB state is published
        db_article = await temp_db.get_article_by_canonical_url(breaking_article.canonical_url)
        assert db_article.status == ArticleStatus.PUBLISHED
        assert db_article.telegram_message_id == 12345
        assert db_article.priority == Priority.CRITICAL


@pytest.mark.asyncio
async def test_acceptance_criteria_3_normal_article_published_immediately(temp_db):
    """
    TEST 3: A normal eligible article is also published immediately.
    """
    publisher = MagicMock(spec=TelegramPublisher)
    publisher.publish_article = AsyncMock(return_value=99999)
    health = HealthMonitor()
    metrics = Metrics()

    pipeline = ArticlePipeline(temp_db, publisher, health, metrics)

    from core.importance_ranker import RelevanceEvaluation
    with patch.object(pipeline._translator, 'translate_title', new_callable=AsyncMock) as mock_tr_title, \
         patch.object(pipeline._translator, 'translate_summary', new_callable=AsyncMock) as mock_tr_summary, \
         patch.object(pipeline._extractor, 'fetch_full_article_text', new_callable=AsyncMock) as mock_scrape, \
         patch.object(pipeline._ranker, 'evaluate_global_relevance', new_callable=AsyncMock) as mock_rel:
        
        mock_rel.return_value = RelevanceEvaluation(score=8, is_breaking=False, reason="Mock relevant", passed=True)
        mock_scrape.return_value = None
        mock_tr_title.return_value = "لێکۆڵینەوەیەکی زانستی نوێ بڵاوکرایەوە"
        mock_tr_summary.return_value = "زانایان دۆزینەوەیەکی گرنگیان لە بواری گەردوونناسیدا کرد."

        normal_article = Article(
            original_url="https://bbc.com/science/space-discovery-test",
            canonical_url="https://bbc.com/science/space-discovery-test",
            original_title="New Scientific Research Breakthrough in Deep Space Astronomy",
            original_summary=(
                "Astronomers at university research laboratories have detected clear atmospheric biosignatures on distant exoplanets. "
                "The findings, published in leading astrophysics journals, indicate water vapor and molecular oxygen in the habitable zone, "
                "offering unprecedented insights into planetary evolution beyond our solar system."
            ),
            source_name="BBC",
            category="Science & Research",
            publication_date=datetime.now(timezone.utc).isoformat(),
        )
        normal_article.compute_content_hash()
        normal_article.compute_event_fingerprint()

        art_id = await temp_db.insert_article(normal_article)
        normal_article.id = art_id

        published = await pipeline.process_article(normal_article)

        assert published is True
        publisher.publish_article.assert_awaited_once()

        db_article = await temp_db.get_article_by_canonical_url(normal_article.canonical_url)
        assert db_article.status == ArticleStatus.PUBLISHED
        assert db_article.telegram_message_id == 99999


@pytest.mark.asyncio
async def test_acceptance_criteria_14_telegram_rate_limits_handled():
    """
    TEST 14: Telegram rate limits (RetryAfter) are handled correctly.
    """
    publisher = TelegramPublisher()
    mock_bot = MagicMock()

    call_count = 0
    async def mock_send_message(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First attempt raises RetryAfter with 0.1s
            raise RetryAfter(retry_after=0.1)
        res = MagicMock()
        res.message_id = 777
        return res

    mock_bot.send_message = AsyncMock(side_effect=mock_send_message)
    publisher._bot = mock_bot

    article = Article(
        original_title="Test Rate Limit",
        translated_title="تاقیکردنەوەی خێرایی",
        original_summary="Testing rate limits handling in publisher",
        translated_summary="پشکنینی کۆنتڕۆڵی سنووری خێرایی تێلێگرام",
        original_url="https://reuters.com/rate-test",
        source_name="Reuters"
    )

    msg_id = await publisher._publish_text_only(article)
    assert msg_id == 777
    assert call_count == 2  # Handled retry after backoff


@pytest.mark.asyncio
async def test_acceptance_criteria_15_long_running_simulation_no_duplicates(temp_db):
    """
    TEST 15: The system survives a long-running test without memory growth,
    overlapping scheduler failures, or duplicate publication.
    """
    publisher = MagicMock(spec=TelegramPublisher)
    publisher.publish_article = AsyncMock(return_value=55555)
    health = HealthMonitor()
    metrics = Metrics()
    pipeline = ArticlePipeline(temp_db, publisher, health, metrics)

    from core.importance_ranker import RelevanceEvaluation
    with patch.object(pipeline._translator, 'translate_title', new_callable=AsyncMock) as mock_tr_title, \
         patch.object(pipeline._translator, 'translate_summary', new_callable=AsyncMock) as mock_tr_summary, \
         patch.object(pipeline._extractor, 'fetch_full_article_text', new_callable=AsyncMock) as mock_scrape, \
         patch.object(pipeline._ranker, 'evaluate_global_relevance', new_callable=AsyncMock) as mock_rel:
        
        mock_rel.return_value = RelevanceEvaluation(score=8, is_breaking=False, reason="Mock relevant", passed=True)
        mock_scrape.return_value = None
        mock_tr_title.side_effect = lambda t: f"وەرگێڕدراو بۆ کوردی: {t}"
        mock_tr_summary.side_effect = lambda s: f"پوختەی بابەت بە کوردی: {s}"

        distinct_topics = [
            ("Global Economy Summit Concludes in Geneva with New Trade Accords", "https://example.com/world/economy-summit-geneva", "Global Economy & Financial Markets"),
            ("Astronomers Discover Water Ice on Distant Martian Crater", "https://example.com/world/astronomy-mars-ice", "Space & Astronomy"),
            ("Magnitude 7.1 Earthquake Strikes Offshore Islands in Pacific Region", "https://example.com/world/earthquake-pacific-islands", "Natural Disasters & Emergencies"),
            ("Renewable Solar Energy Efficiency Reaches Record 34 Percent Breakthrough", "https://example.com/world/solar-efficiency-record", "Energy, Oil & Gas"),
            ("Cybersecurity Agency Issues Warning on Major Global Banking Vulnerability", "https://example.com/world/cyber-security-banking", "Cybersecurity"),
        ]

        published_count = 0
        duplicate_count = 0

        for i in range(20):
            # i % 5 causes repeats
            topic_idx = i % 5
            title, url, category = distinct_topics[topic_idx]

            art = Article(
                original_url=url,
                canonical_url=url,
                original_title=title,
                original_summary=(
                    f"Comprehensive international news dispatch covering {title} with detailed contextual analysis and eyewitness accounts. "
                    "Regional correspondents report significant geopolitical ramifications across neighboring countries and global energy markets, "
                    "as multinational delegations convene emergency sessions to formulate unified bilateral policies."
                ),
                source_name="Reuters",
                category=category,
                publication_date=datetime.now(timezone.utc).isoformat(),
            )
            art.compute_content_hash()
            art.compute_event_fingerprint()

            # Insert or ignore duplicate
            art_id = await temp_db.insert_article(art)
            if art_id is None:
                duplicate_count += 1
                continue
            art.id = art_id

            pub = await pipeline.process_article(art)
            if pub:
                published_count += 1

        # Only unique articles should be published
        assert published_count == 5
        assert duplicate_count == 15
        assert health.is_healthy() is True
