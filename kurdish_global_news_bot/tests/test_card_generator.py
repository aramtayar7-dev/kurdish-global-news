"""
Unit tests for Branded 1280x720 News Graphic Cards (`processing/card_generator.py`)
and `TelegramPublisher` temporary card lifecycle + cleanup.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from config import ASSETS_DIR, CARD_HEIGHT, CARD_WIDTH
import processing.card_generator as card_gen_mod
from processing.card_generator import MARGIN_X, NewsCardGenerator, _RTL_RESHAPER
from storage.models import Article
from tg_bot.publisher import TelegramPublisher


def _make_sample_photo_bytes(width: int = 1600, height: int = 900) -> bytes:
    img = Image.new("RGB", (width, height), (45, 95, 160))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestNewsCardGenerator:
    """Test Pillow card generation across all visual modes."""

    @pytest.mark.asyncio
    async def test_generate_card_with_article_image_is_1280x720(self):
        generator = NewsCardGenerator()
        art = Article(
            translated_title="سەرکردەکانی جیهان کۆبوونەوەی بەپەلە لەسەر پرسی ئەمنیی کوردستان سازدەکەن",
            category="kurdistan",
            source_name="Kurdistan24",
        )
        photo_bytes = _make_sample_photo_bytes(1920, 1080)
        card_bytes = await generator.generate_card_bytes(art, image_bytes=photo_bytes)

        with Image.open(io.BytesIO(card_bytes)) as result:
            assert result.size == (CARD_WIDTH, CARD_HEIGHT)
            assert result.size == (1280, 720)
            assert result.format == "JPEG"

    @pytest.mark.asyncio
    async def test_generate_card_fallback_background_without_image(self):
        generator = NewsCardGenerator()
        art = Article(
            translated_title="هێرشە ئاسمانییەکان پێگە سەربازییەکانیان کردە ئامانج لە ناوچەکەدا و گرژییەکان زیادیان کرد",
            category="military_conflict",
            source_name="Reuters",
        )
        card_bytes = await generator.generate_card_bytes(art, image_bytes=None)

        with Image.open(io.BytesIO(card_bytes)) as result:
            assert result.size == (1280, 720)
            assert result.format == "JPEG"

    @pytest.mark.asyncio
    async def test_generate_card_bundled_sorani_font_exists(self):
        generator = NewsCardGenerator()
        assert generator._font_path is not None
        assert generator._font_path.exists()

    def test_bundled_circular_logo_transparency(self):
        """Verify assets/logo.png is circular RGBA with transparent corners and opaque center."""
        logo_path = ASSETS_DIR / "logo.png"
        assert logo_path.exists(), f"Expected processed transparent logo at {logo_path}"

        with Image.open(logo_path) as raw_logo:
            logo = raw_logo.convert("RGBA")
            w, h = logo.size
            assert w == h
            assert logo.getpixel((0, 0))[3] == 0
            assert logo.getpixel((w - 1, 0))[3] == 0
            assert logo.getpixel((0, h - 1))[3] == 0
            assert logo.getpixel((w - 1, h - 1))[3] == 0
            assert logo.getpixel((w // 2, h // 2))[3] == 255

    @pytest.mark.asyncio
    async def test_card_composites_transparent_logo_without_white_box(self):
        """Verify top-right corner outside circular logo stays dark/transparent (no white square)."""
        generator = NewsCardGenerator()
        assert generator._logo_path == ASSETS_DIR / "logo.png"

        # Bright blue test image background -> top-right corner of the 76x76 logo box should show shaded blue, not white!
        photo_bytes = _make_sample_photo_bytes(1280, 720)
        art = Article(
            translated_title="تاقیکردنەوەی لۆگۆی بازنەیی بەبێ پاشبنەمای سپی",
            category="kurdistan",
            source_name="Kurdistan24",
        )
        card_bytes = await generator.generate_card_bytes(art, image_bytes=photo_bytes)

        with Image.open(io.BytesIO(card_bytes)) as card:
            # Top-right logo bounding box starts at (1280 - 56 - 76 = 1148, 36).
            # The very corner pixel (1148, 36) is outside the circular logo rim and MUST NOT be white!
            corner_r, corner_g, corner_b = card.getpixel((CARD_WIDTH - MARGIN_X - 76, 36))
            assert corner_r < 80
            assert corner_b > corner_r  # Background blue shows cleanly through transparent logo corner!

    @pytest.mark.asyncio
    async def test_publisher_sends_generated_card_and_cleans_up_temp_file(self):
        publisher = TelegramPublisher()
        mock_bot = MagicMock()
        mock_msg = MagicMock(message_id=4242)
        mock_bot.send_photo = AsyncMock(return_value=mock_msg)
        publisher._bot = mock_bot

        created_paths: list[Path] = []
        original_gen_file = publisher._card_generator.generate_card_file

        async def tracked_gen_file(article, image_bytes=None, is_breaking=None):
            p = await original_gen_file(article, image_bytes=image_bytes, is_breaking=is_breaking)
            created_paths.append(p)
            assert p.exists()
            return p

        publisher._card_generator.generate_card_file = tracked_gen_file
        publisher._image_handler.download_image = AsyncMock(return_value=_make_sample_photo_bytes())

        art = Article(
            original_title="Kurdistan Region Launches Infrastructure Plan",
            translated_title="حکومەتی هەرێمی کوردستان پرۆژەیەکی ستراتیژی ڕاگەیاند",
            translated_summary=(
                "پرۆژەکە کەرتەکانی وزە و ڕێگاوبان لە هەولێر و سلێمانی و دهۆک دەگرێتەوە. "
                "ئامانجی سەرەکی باشترکردنی خزمەتگوزارییە گشتییەکانە بۆ هاووڵاتیان. "
                "بڕیارە لە ماوەی شەش مانگدا قۆناغی یەکەمی تەواو بکرێت."
            ),
            category="kurdistan",
            source_name="Kurdistan24 EN",
            image_url="https://example.com/erbil.jpg",
            original_url="https://example.com/erbil-story",
        )

        msg_id = await publisher.publish_article(art)
        assert msg_id == 4242
        mock_bot.send_photo.assert_awaited_once()

        # Verify caption uses Kurdish category & emoji
        _, kwargs = mock_bot.send_photo.call_args
        assert "☀️" in kwargs["caption"]
        assert "کوردستان" in kwargs["caption"]
        assert "..." not in kwargs["caption"]

        # Verify temporary JPEG card file was deleted after send_photo
        assert len(created_paths) == 1
        assert not created_paths[0].exists()

    @pytest.mark.asyncio
    async def test_non_raqm_rtl_shapes_after_wrapping_and_preserves_raqm_behavior(self, monkeypatch):
        """
        Verify Windows non-raqm fallback shapes & bidi-reorders each wrapped Kurdish line
        and badge label cleanly (without passing direction='rtl'), while raqm=True keeps text untouched.
        """
        assert _RTL_RESHAPER.configuration.getboolean("support_ligatures") is True

        raw_headline = "حکومەتی هەرێمی کوردستان پرۆژەیەکی گەورەی وزە و ئابووری لە هەولێر و سلێمانی ڕاگەیاند"
        art = Article(
            translated_title=raw_headline,
            category="kurdistan",
            source_name="Kurdistan24",
            is_breaking=True,
            importance_score=9.5,
        )

        # 1. Simulate non-raqm (Windows) explicitly
        monkeypatch.setattr(card_gen_mod, "_HAS_RAQM", False)
        generator = NewsCardGenerator()

        prepared_inputs: list[str] = []
        orig_prepare = generator._prepare_rtl_text

        def spy_prepare(text: str) -> str:
            prepared_inputs.append(text)
            return orig_prepare(text)

        wrapped_logical_lines: list[list[str]] = []
        orig_wrap = generator._wrap_headline

        def spy_wrap(draw, headline, max_width):
            lines, font, lh = orig_wrap(draw, headline, max_width)
            wrapped_logical_lines.append(list(lines))
            # Logical lines are NOT reshaped prior to wrapping
            assert all(word in raw_headline for line in lines for word in line.split())
            return lines, font, lh

        monkeypatch.setattr(generator, "_prepare_rtl_text", spy_prepare)
        monkeypatch.setattr(generator, "_wrap_headline", spy_wrap)

        assert generator._rtl_kwargs() == {}
        shaped_sample = generator._prepare_rtl_text("کوردستان")
        assert shaped_sample != "کوردستان"
        prepared_inputs.clear()

        card_bytes = await generator.generate_card_bytes(art, is_breaking=True)
        with Image.open(io.BytesIO(card_bytes)) as img:
            assert img.size == (1280, 720)
            assert img.format == "JPEG"

        # Wrapping occurred before shaping; badge + breaking + wrapped logical lines were shaped afterward
        assert len(wrapped_logical_lines) == 1
        assert "کوردستان" in prepared_inputs
        assert "بەپەلە" in prepared_inputs
        for logical_line in wrapped_logical_lines[0]:
            assert logical_line in prepared_inputs

        # Digest card also shapes cleanly without raqm
        digest_bytes = await generator.generate_digest_card_bytes("٢٤ی تشرینی دووەم ٢٠٢٦", 5)
        with Image.open(io.BytesIO(digest_bytes)) as dimg:
            assert dimg.size == (1280, 720)

        # 2. Simulate raqm=True (Linux / GitHub Actions): logical text untouched and direction='rtl' enabled
        monkeypatch.setattr(card_gen_mod, "_HAS_RAQM", True)
        assert generator._rtl_kwargs() == {"direction": "rtl"}
        assert orig_prepare("کوردستان") == "کوردستان"
