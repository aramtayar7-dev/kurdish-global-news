"""
Tests for full article text scraping, journalistic Sorani prompts, and clean Kurdish Telegram layout.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from storage.models import Article
from config import ArticleStatus, Priority
from processing.extractor import ArticleExtractor
from processing.content_formatter import ContentFormatter, CATEGORY_LABELS_KURDISH
from processing.translator import GROQ_SYSTEM_PROMPT, GEMINI_PROOFREADER_SYSTEM_PROMPT, GroqTranslationProvider, GeminiProofreader
from core.pipeline import ArticlePipeline
from monitoring.health import HealthMonitor
from monitoring.metrics import Metrics
from tg_bot.publisher import TelegramPublisher
from telegram import LinkPreviewOptions


class TestArticleTextExtractor:
    """Test full article webpage extraction and detail filters."""

    def test_extract_article_body_from_html(self):
        extractor = ArticleExtractor()
        html = """
        <!DOCTYPE html>
        <html>
        <head><title>Test News Article</title></head>
        <body>
            <header><nav><a href="/">Home</a><p>Navigation item</p></nav></header>
            <article>
                <h1>Major Breakthrough in Renewable Energy</h1>
                <p>Scientists and engineers have announced a substantial advancement in photovoltaic solar cell efficiency.</p>
                <script>console.log("analytics");</script>
                <style>.ad { display: none; }</style>
                <div class="ad"><p>Subscribe to our newsletter for daily updates!</p></div>
                <p>The new silicon-perovskite tandem cells demonstrated a certified 34 percent conversion efficiency in laboratory trials.</p>
                <p>Commercial deployment is projected within the next two years according to the international consortium.</p>
                <aside><p>Related stories you might like</p></aside>
            </article>
            <footer><p>Copyright 2026. All rights reserved.</p></footer>
        </body>
        </html>
        """
        body = extractor.extract_article_body_from_html(html)
        assert body is not None
        assert "Scientists and engineers have announced" in body
        assert "34 percent conversion efficiency" in body
        assert "Commercial deployment is projected" in body
        assert "analytics" not in body
        assert "Navigation item" not in body
        assert "Subscribe to our newsletter" not in body
        assert "All rights reserved" not in body

    def test_extract_article_body_length_limit(self):
        extractor = ArticleExtractor()
        long_paragraph = "<p>" + ("Detailed news analysis sentence reporting on international affairs. " * 80) + "</p>"
        html = f"<article>{long_paragraph}</article>"
        body = extractor.extract_article_body_from_html(html)
        assert body is not None
        assert len(body) <= 3000

    def test_is_insufficient_detail_thin_text(self):
        title = "US and UK launch strikes on Houthi targets in Yemen"
        thin_text = "US forces conducted strikes in Yemen on Friday."  # 47 chars < 160
        insufficient, reason = ArticleExtractor.is_insufficient_detail(title, thin_text)
        assert insufficient is True
        assert "Text too thin" in reason

    def test_is_insufficient_detail_headline_repetition(self):
        title = (
            "United States and United Kingdom launch major military strikes on multiple Houthi "
            "underground facilities and command radar installations across western Yemen"
        )
        # Over 160 chars and >70% similar to title (headline repetition)
        repetitive_text = (
            "United States and United Kingdom launch major military strikes on multiple Houthi "
            "underground facilities and command radar installations across western Yemen on Friday."
        )
        assert len(repetitive_text) >= 160
        insufficient, reason = ArticleExtractor.is_insufficient_detail(title, repetitive_text)
        assert insufficient is True
        assert "similar to title" in reason

    def test_is_insufficient_detail_valid_rich_text(self):
        title = "US and UK launch strikes on Houthi targets in Yemen"
        rich_text = (
            "American and British military aircraft conducted multiple strikes against radar installations, "
            "missile depots, and coastal defense systems across western Yemen on Friday night. "
            "Defense officials confirmed the joint operation aimed to degrade capability to attack commercial shipping lanes in the Red Sea."
        )
        insufficient, reason = ArticleExtractor.is_insufficient_detail(title, rich_text)
        assert insufficient is False
        assert reason == ""


class TestJournalisticPromptsAndTokens:
    """Test updated prompts and 800 token caps."""

    def test_groq_prompt_rules(self):
        assert "3 to 5 sentences" in GROQ_SYSTEM_PROMPT
        assert "NEVER Repeat the Headline" in GROQ_SYSTEM_PROMPT or "NEVER repeat the headline" in GROQ_SYSTEM_PROMPT.lower()
        assert "5Ws" in GROQ_SYSTEM_PROMPT
        assert "ڕ, ڵ, ۆ, ێ, ە, ه" in GROQ_SYSTEM_PROMPT
        assert "ڕیگۆلیتی" in GROQ_SYSTEM_PROMPT

    def test_gemini_prompt_rules(self):
        assert "3 to 5 sentences" in GEMINI_PROOFREADER_SYSTEM_PROMPT
        assert "NEVER Repeat the Headline" in GEMINI_PROOFREADER_SYSTEM_PROMPT or "never repeat the headline" in GEMINI_PROOFREADER_SYSTEM_PROMPT.lower()
        assert "350–650 characters" in GEMINI_PROOFREADER_SYSTEM_PROMPT or "350 to 650 characters" in GEMINI_PROOFREADER_SYSTEM_PROMPT
        assert "ڕیگۆلیتی" in GEMINI_PROOFREADER_SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_groq_max_tokens_800(self):
        provider = GroqTranslationProvider(api_key="fake-groq-key")
        captured_payload = None

        class MockAsyncContextManager:
            def __init__(self, response):
                self._response = response

            async def __aenter__(self):
                return self._response

            async def __aexit__(self, exc_type, exc, tb):
                pass

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"choices": [{"message": {"content": "وەرگێڕانی کوردی"}}]})

        mock_session = MagicMock()
        def mock_post(url, json=None, headers=None):
            nonlocal captured_payload
            captured_payload = json
            return MockAsyncContextManager(mock_resp)

        mock_session.post = mock_post
        provider._get_session = AsyncMock(return_value=mock_session)

        res = await provider.translate("Breaking international news article text.")
        assert res == "وەرگێڕانی کوردی"
        assert captured_payload is not None
        assert captured_payload["max_tokens"] == 800

    @pytest.mark.asyncio
    async def test_gemini_max_output_tokens_800(self):
        proofreader = GeminiProofreader(api_key="fake-gemini-key")
        captured_payload = None

        class MockAsyncContextManager:
            def __init__(self, response):
                self._response = response

            async def __aenter__(self):
                return self._response

            async def __aexit__(self, exc_type, exc, tb):
                pass

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "candidates": [{
                "content": {
                    "parts": [{"text": '{"title": "سەردێر", "summary": "پوختە"}'}]
                }
            }]
        })

        mock_session = MagicMock()
        def mock_post(url, json=None, headers=None):
            nonlocal captured_payload
            captured_payload = json
            return MockAsyncContextManager(mock_resp)

        mock_session.post = mock_post
        proofreader._get_session = AsyncMock(return_value=mock_session)

        t, s = await proofreader.proofread("Eng Title", "Draft Title", "Eng Summary", "Draft Summary")
        assert t == "سەردێر"
        assert s == "پوختە"
        assert captured_payload is not None
        assert captured_payload["generationConfig"]["maxOutputTokens"] == 800


class TestKurdishPostLayoutAndPublisher:
    """Test Kurdish category translations and preview options."""

    def test_kurdish_category_dictionary(self):
        assert CATEGORY_LABELS_KURDISH["Military & Defense"] == "سەربازی و بەرگری"
        assert CATEGORY_LABELS_KURDISH["Science & Space"] == "زانست و بۆشایی ئاسمان"
        assert CATEGORY_LABELS_KURDISH["AI & Tech"] == "تەکنەلۆژیا و زیرەکیی دەستکرد"
        assert CATEGORY_LABELS_KURDISH["Cybersecurity"] == "ئاسایشی سایبەری"
        assert CATEGORY_LABELS_KURDISH["World News"] == "هەواڵی جیهانی"

    def test_format_telegram_message_uses_kurdish_category(self):
        formatter = ContentFormatter()
        article = Article(
            original_title="AI Model Released",
            translated_title="مۆدێلی نوێی زیرەکی دەستکرد ڕاگەیەندرا",
            translated_summary="پوختەی هەواڵ لەسەر تواناکانی ئەم مۆدێلە لە بوارە جیاوازەکاندا.",
            category="AI & Technology",
            source_name="TechSource",
            original_url="https://tech.com/model",
        )
        msg = formatter.format_telegram_message(article)
        assert "تەکنەلۆژیا و زیرەکیی دەستکرد" in msg
        assert "TechSource" in msg

    def test_format_caption_uses_kurdish_category(self):
        formatter = ContentFormatter()
        article = Article(
            original_title="Defense Agreement Signed",
            translated_title="ڕێککەوتنی نوێی بەرگری واژۆکرا",
            translated_summary="هەردوو وڵات ڕێککەوتن لەسەر زیادکردنی مەشقە هاوبەشە سەربازییەکان.",
            category="Military & Defense",
            source_name="Reuters",
            original_url="https://reuters.com/defense",
        )
        caption = formatter.format_caption(article)
        assert "سەربازی و بەرگری" in caption
        assert "Reuters" in caption

    @pytest.mark.asyncio
    async def test_publish_text_only_disables_link_preview(self):
        publisher = TelegramPublisher()
        mock_bot = MagicMock()
        mock_resp = MagicMock()
        mock_resp.message_id = 1234
        mock_bot.send_message = AsyncMock(return_value=mock_resp)
        publisher._bot = mock_bot

        article = Article(
            original_title="Test News Title",
            translated_title="سەردێری هەواڵ",
            translated_summary="پوختەی هەواڵەکە بە تەواوی لێرە دەنووسرێت.",
            original_url="https://example.com/test-preview",
            source_name="BBC",
        )

        msg_id = await publisher._publish_text_only(article)
        assert msg_id == 1234
        mock_bot.send_message.assert_awaited_once()
        _, kwargs = mock_bot.send_message.call_args
        assert "link_preview_options" in kwargs
        assert kwargs["link_preview_options"].is_disabled is True


class TestPipelineScrapingAndInsufficientDetail:
    """Test pipeline integration with web scraping and detail validation."""

    @pytest.mark.asyncio
    async def test_pipeline_skips_insufficient_article_detail(self, tmp_path):
        from storage.database import Database
        from datetime import datetime, timezone
        db = Database(str(tmp_path / "test_detail.db"))
        await db.connect()

        pipeline = ArticlePipeline(db, MagicMock(spec=TelegramPublisher), HealthMonitor(), Metrics())
        # Mock extractor to fail scraping and relevance check to pass deterministically offline
        from core.importance_ranker import RelevanceEvaluation
        pipeline._ranker.evaluate_global_relevance = AsyncMock(
            return_value=RelevanceEvaluation(score=8, is_breaking=False, reason="Mock pass", passed=True)
        )
        pipeline._extractor.fetch_full_article_text = AsyncMock(return_value=None)

        art = Article(
            original_url="https://reuters.com/thin-news",
            canonical_url="https://reuters.com/thin-news",
            original_title="Breaking: Brief Update from Geneva",
            original_summary="Brief update announced.",  # Only 24 chars < 160
            source_name="Reuters",
            category="International Politics",
            publication_date=datetime.now(timezone.utc).isoformat(),
        )
        art.compute_content_hash()
        art.compute_event_fingerprint()
        art_id = await db.insert_article(art)
        art.id = art_id

        published = await pipeline.process_article(art)
        assert published is False

        db_art = await db.get_article_by_canonical_url(art.canonical_url)
        assert db_art.status == ArticleStatus.FAILED
        assert "SKIPPED: insufficient article detail" in db_art.error_message
        await db.close()

    @pytest.mark.asyncio
    async def test_pipeline_uses_full_scraped_text(self, tmp_path):
        from storage.database import Database
        from datetime import datetime, timezone
        db = Database(str(tmp_path / "test_scraped.db"))
        await db.connect()

        publisher = MagicMock(spec=TelegramPublisher)
        publisher.publish_article = AsyncMock(return_value=8888)
        pipeline = ArticlePipeline(db, publisher, HealthMonitor(), Metrics())

        from core.importance_ranker import RelevanceEvaluation
        pipeline._ranker.evaluate_global_relevance = AsyncMock(
            return_value=RelevanceEvaluation(score=8, is_breaking=False, reason="Mock pass", passed=True)
        )

        rich_scraped_text = (
            "Delegations from thirty nations concluded negotiations on a new international treaty regulating satellite constellations. "
            "The agreement sets mandatory de-orbit standards for low Earth orbit satellites to mitigate orbital debris collisions. "
            "Space agencies welcomed the binding accord as a critical foundation for commercial orbital safety."
        )
        pipeline._extractor.fetch_full_article_text = AsyncMock(return_value=rich_scraped_text)

        translated_summary_received = None
        async def fake_translate(title, summary, target_lang="ckb"):
            nonlocal translated_summary_received
            translated_summary_received = summary
            return "ڕێککەوتنی بۆشایی ئاسمان واژۆکرا", "پوختەی ڕێککەوتنەکە بۆ ڕێکخستنی مانگە دەستکردەکان."

        pipeline._translator.translate_article = fake_translate

        art = Article(
            original_url="https://reuters.com/rich-space-news",
            canonical_url="https://reuters.com/rich-space-news",
            original_title="Space Treaty Regulating Satellites Signed in Vienna",
            original_summary="Short RSS snippet.",
            source_name="Reuters",
            category="Space & Astronomy",
            publication_date=datetime.now(timezone.utc).isoformat(),
        )
        art.compute_content_hash()
        art.compute_event_fingerprint()
        art_id = await db.insert_article(art)
        art.id = art_id

        published = await pipeline.process_article(art)
        assert published is True
        assert translated_summary_received == rich_scraped_text
        publisher.publish_article.assert_awaited_once()

        db_art = await db.get_article_by_canonical_url(art.canonical_url)
        assert db_art.status == ArticleStatus.PUBLISHED
        assert db_art.original_summary == rich_scraped_text
        await db.close()
