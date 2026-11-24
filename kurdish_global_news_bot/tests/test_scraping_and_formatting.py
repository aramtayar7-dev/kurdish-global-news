"""
Tests for full article text scraping, journalistic Sorani prompts, Kurdish category labels,
and clean sentence-boundary caption formatting.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import ArticleStatus
from core.importance_ranker import RelevanceEvaluation
from core.pipeline import ArticlePipeline
from monitoring.health import HealthMonitor
from monitoring.metrics import Metrics
from processing.content_formatter import CATEGORY_LABELS_KURDISH, ContentFormatter, resolve_category_display
from processing.extractor import ArticleExtractor
from processing.translator import GEMINI_TRANSLATION_SYSTEM_PROMPT, GeminiTranslationProvider
from storage.models import Article
from tg_bot.publisher import TelegramPublisher


class TestArticleTextExtractor:
    """Test full article webpage extraction, og:image extraction, and detail filters."""

    def test_extract_article_body_and_og_image_from_html(self):
        extractor = ArticleExtractor()
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test News Article</title>
            <meta property="og:image" content="/assets/hero-news.jpg" />
        </head>
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
            </article>
            <footer><p>Copyright 2026. All rights reserved.</p></footer>
        </body>
        </html>
        """
        body = extractor.extract_article_body_from_html(html)
        og_img = extractor.extract_og_image_from_html(html, base_url="https://example.com/news/1")

        assert body is not None
        assert "Scientists and engineers have announced" in body
        assert "34 percent conversion efficiency" in body
        assert "analytics" not in body
        assert "Subscribe to our newsletter" not in body
        assert og_img == "https://example.com/assets/hero-news.jpg"

    def test_extract_article_body_length_limit(self):
        extractor = ArticleExtractor()
        long_paragraph = "<p>" + ("Detailed news analysis sentence reporting on international affairs. " * 80) + "</p>"
        body = extractor.extract_article_body_from_html(f"<article>{long_paragraph}</article>")
        assert body is not None
        assert len(body) <= 3000

    def test_is_insufficient_detail_checks(self):
        title = "US and UK launch strikes on Houthi targets in Yemen"
        insufficient_thin, reason_thin = ArticleExtractor.is_insufficient_detail(title, "Too short.")
        assert insufficient_thin is True
        assert "Text too thin" in reason_thin

        long_title = (
            "United States and United Kingdom launch major military strikes on multiple Houthi "
            "underground facilities and command radar installations across western Yemen"
        )
        rep_text = (
            "United States and United Kingdom launch major military strikes on multiple Houthi "
            "underground facilities and command radar installations across western Yemen on Friday."
        )
        insufficient_rep, reason_rep = ArticleExtractor.is_insufficient_detail(long_title, rep_text)
        assert insufficient_rep is True
        assert "similar to title" in reason_rep


class TestGeminiJournalisticPromptsAndTokens:
    """Test single-pass Gemini prompts and maxOutputTokens=800."""

    def test_gemini_prompt_rules(self):
        assert "3 to 4 complete sentences" in GEMINI_TRANSLATION_SYSTEM_PROMPT
        assert "NEVER repeat the headline" in GEMINI_TRANSLATION_SYSTEM_PROMPT
        assert "350 to 650 characters" in GEMINI_TRANSLATION_SYSTEM_PROMPT
        assert "پیاوێکی تەمەن ٦٢ ساڵ" in GEMINI_TRANSLATION_SYSTEM_PROMPT
        assert "ئیفێمیزم" in GEMINI_TRANSLATION_SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_gemini_translation_max_tokens_800(self):
        provider = GeminiTranslationProvider(api_key="fake-gem-key", base_backoff=0.0)
        captured_payload = None

        class MockCM:
            def __init__(self, resp):
                self._resp = resp

            async def __aenter__(self):
                return self._resp

            async def __aexit__(self, exc_type, exc, tb):
                return False

        mock_resp = AsyncMock(
            status=200,
            json=AsyncMock(return_value={
                "candidates": [{"content": {"parts": [{"text": '{"title": "سەردێر", "summary": "ڕستەی یەکەم. ڕستەی دووەم."}'}]}}]
            }),
        )

        mock_session = MagicMock()
        def mock_post(url, json=None, headers=None):
            nonlocal captured_payload
            captured_payload = json
            return MockCM(mock_resp)

        mock_session.post = mock_post
        provider._client._get_session = AsyncMock(return_value=mock_session)

        t, s = await provider.translate_article("Eng Title", "Eng Summary facts.")
        assert t == "سەردێر"
        assert s == "ڕستەی یەکەم. ڕستەی دووەم."
        assert captured_payload["generationConfig"]["maxOutputTokens"] == 800


class TestKurdishCategoriesAndCleanCaption:
    """Test canonical 7 Kurdish categories, emojis, and zero mid-sentence truncation."""

    def test_seven_canonical_kurdish_categories_and_emojis(self):
        expected = {
            "kurdistan": ("کوردستان", "☀️"),
            "military_conflict": ("شەڕ و سەربازی", "⚔️"),
            "politics_world": ("سیاسەت و جیهان", "🌍"),
            "economy_markets": ("ئابووری و بازاڕ", "📈"),
            "ai_tech": ("تەکنەلۆژیا و زیرەکیی دەستکرد", "🤖"),
            "cybersecurity": ("ئاسایشی سایبەری", "🛡️"),
            "science_space": ("زانست و بۆشایی ئاسمان", "🚀"),
        }
        for key, (label, emoji) in expected.items():
            assert resolve_category_display(key) == (label, emoji)
            assert CATEGORY_LABELS_KURDISH[key] == label

    def test_format_caption_no_mid_sentence_ellipsis(self):
        formatter = ContentFormatter()
        # Generate multi-sentence summary totaling ~950 chars that needs trimming before caption overflow
        sentence = "زانایان توێژینەوەیەکی نوێیان لەسەر تەکنەلۆژیای بۆشایی ئاسمان ئەنجامدا و دەرئەنجامەکان بڵاوکرانەوە. "
        art = Article(
            translated_title="پێشکەوتنێکی نوێ لە بواری زانست و بۆشایی ئاسماندا ڕاگەیەندرا",
            translated_summary=sentence * 10,
            category="science_space",
            source_name="NASA",
            original_url="https://nasa.gov/mission/deep-space-research-report",
        )
        caption = formatter.format_caption(art)
        assert len(caption) <= 1024
        assert "زانست و بۆشایی ئاسمان" in caption
        assert "🚀" in caption
        assert "..." not in caption
        assert '<a href="https://nasa.gov/mission/deep-space-research-report">بابەتی ئەسڵی</a>' in caption

    @pytest.mark.asyncio
    async def test_publish_text_only_disables_link_preview(self):
        publisher = TelegramPublisher()
        mock_bot = MagicMock()
        mock_resp = MagicMock(message_id=1234)
        mock_bot.send_message = AsyncMock(return_value=mock_resp)
        publisher._bot = mock_bot

        article = Article(
            original_title="Test News Title",
            translated_title="سەردێری هەواڵ",
            translated_summary="پوختەی هەواڵەکە بە تەواوی لێرە دەنووسرێت.",
            category="politics_world",
            original_url="https://example.com/test-preview",
            source_name="BBC",
        )
        msg_id = await publisher._publish_text_only(article)
        assert msg_id == 1234
        _, kwargs = mock_bot.send_message.call_args
        assert kwargs["link_preview_options"].is_disabled is True


class TestPipelineScrapingAndInsufficientDetail:
    @pytest.mark.asyncio
    async def test_pipeline_skips_insufficient_article_detail(self, tmp_path):
        from storage.database import Database
        db = Database(str(tmp_path / "test_detail.db"))
        await db.connect()

        pipeline = ArticlePipeline(db, MagicMock(spec=TelegramPublisher), HealthMonitor(), Metrics())
        pipeline._ranker.evaluate_global_relevance = AsyncMock(
            return_value=RelevanceEvaluation(score=8, is_breaking=False, category="politics_world", reason="Mock pass", passed=True)
        )
        pipeline._extractor.fetch_article_content_and_image = AsyncMock(return_value=(None, None))

        art = Article(
            original_url="https://reuters.com/thin-news",
            canonical_url="https://reuters.com/thin-news",
            original_title="Breaking: Brief Update from Geneva",
            original_summary="Brief update announced.",
            source_name="Reuters",
            category="politics_world",
            publication_date=datetime.now(timezone.utc).isoformat(),
        )
        art.compute_content_hash()
        art.compute_event_fingerprint()
        art.id = await db.insert_article(art)

        published = await pipeline.process_article(art)
        assert published is False

        db_art = await db.get_article_by_canonical_url(art.canonical_url)
        assert db_art.status == ArticleStatus.FAILED
        assert "SKIPPED: insufficient article detail" in db_art.error_message
        await db.close()

    @pytest.mark.asyncio
    async def test_pipeline_uses_full_scraped_text_and_og_image(self, tmp_path):
        from storage.database import Database
        db = Database(str(tmp_path / "test_scraped.db"))
        await db.connect()

        publisher = MagicMock(spec=TelegramPublisher)
        publisher.publish_article = AsyncMock(return_value=8888)
        pipeline = ArticlePipeline(db, publisher, HealthMonitor(), Metrics())

        pipeline._ranker.evaluate_global_relevance = AsyncMock(
            return_value=RelevanceEvaluation(score=8, is_breaking=False, category="science_space", reason="Mock pass", passed=True)
        )
        rich_scraped_text = (
            "Delegations from thirty nations concluded negotiations on a new international treaty regulating satellite constellations. "
            "The agreement sets mandatory de-orbit standards for low Earth orbit satellites to mitigate orbital debris collisions. "
            "Space agencies welcomed the binding accord as a critical foundation for commercial orbital safety."
        )
        pipeline._extractor.fetch_article_content_and_image = AsyncMock(
            return_value=(rich_scraped_text, "https://reuters.com/space-cover.jpg")
        )

        translated_summary_received = None
        async def fake_translate(title, summary="", target_lang="ckb"):
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
            category="science_space",
            publication_date=datetime.now(timezone.utc).isoformat(),
        )
        art.compute_content_hash()
        art.compute_event_fingerprint()
        art.id = await db.insert_article(art)

        published = await pipeline.process_article(art)
        assert published is True
        assert translated_summary_received == rich_scraped_text
        db_art = await db.get_article_by_canonical_url(art.canonical_url)
        assert db_art.status == ArticleStatus.PUBLISHED
        assert db_art.image_url == "https://reuters.com/space-cover.jpg"
        await db.close()
