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
from processing.card_generator import (
    MARGIN_X,
    NewsCardGenerator,
    _RTL_RESHAPER,
    detect_country_flag_filename,
)
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

    def test_sorani_kurdish_letters_reshape_joined_without_raqm(self, monkeypatch):
        """
        Verify Sorani letters (ە, ێ, ڵ, ک, گ, پ, چ, ژ, ڤ, ۆ, ڕ) reshape into
        connected Arabic Presentation Forms without unmapped PUA codepoints when
        libraqm is unavailable (e.g., Vercel Linux).
        """
        monkeypatch.setattr(card_gen_mod, "_HAS_RAQM", False)

        # Verify connected contextual joining in reshaped output (before bidi reversal)
        # 1. لەسەر -> ﻟ + ـە + سـ + ـە + ـر (all joined cleanly across ە)
        reshaped_lasar = _RTL_RESHAPER.reshape("لەسەر")
        assert reshaped_lasar == "\uFEDF\uFEEA\uFEB3\uFEEA\uFEAD"

        # 2. بەپەلە -> بـ + ـە + پـ + ـە + لـ + ـە
        reshaped_bapala = _RTL_RESHAPER.reshape("بەپەلە")
        assert reshaped_bapala == "\uFE91\uFEEA\uFB58\uFEEA\uFEDF\uFEEA"

        # 3. ئێران -> ئـ + ـێـ + ـر + ا + ن
        reshaped_iran = _RTL_RESHAPER.reshape("ئێران")
        assert reshaped_iran == "\uFE8B\uFBFF\u065A\uFEAE\uFE8D\uFEE5"

        # 4. کەشتییەک -> کـ + ـە + شـ + ـتـ + ـیـ + ـیـ + ـە + ک
        reshaped_kashtiyak = _RTL_RESHAPER.reshape("کەشتییەک")
        assert reshaped_kashtiyak == "\uFB90\uFEEA\uFEB7\uFE98\uFBFF\uFBFF\uFEEA\uFB8E"

        # 5. ڕۆژھەڵاتی -> ڕ + ۆ + ژ + ھـ + ـە + ڵـ + ـا + تـ + ـی
        reshaped_rozhalati = _RTL_RESHAPER.reshape("ڕۆژھەڵاتی")
        assert reshaped_rozhalati == "\u0695\uFBD9\uFB8A\uFBAC\uFEEA\uFEDF\u065A\uFE8E\uFE97\uFBFD"

        # Also verify _prepare_rtl_text produces non-empty visual strings with zero PUA chars (U+E000..U+F8FF)
        for word in ("لەسەر", "بەپەلە", "ئێران", "کەشتییەک", "ڕۆژھەڵاتی"):
            visual = NewsCardGenerator._prepare_rtl_text(word)
            assert visual
            assert not any(0xE000 <= ord(ch) <= 0xF8FF for ch in visual)

    @pytest.mark.parametrize(
        "text,expected_flag",
        [
            ("عومان، مەسقەت و سەڵتەنەتی عومان", "oman.png"),
            ("Oman talks held in Muscat", "oman.png"),
            ("بەحرەین لە مەنامە کۆبوونەوەی ئەنجامدا", "bahrain.png"),
            ("Bahrain security summit in Manama", "bahrain.png"),
            ("لەیەمەن و سەنعا حوسی هێرشی کردە سەر عەدەن", "yemen.png"),
            ("Yemen Houthi strike near Sanaa and Aden", "yemen.png"),
            ("لوبنان و بەیروت و هەڵوێستی حزبوڵڵا و حیزبوڵڵا", "lebanon.png"),
            ("Hezbollah statement from Beirut, Lebanon", "lebanon.png"),
            ("میسر و قاهیرە و سینا و کەناڵی سوێس", "egypt.png"),
            ("Egypt closes Suez transit near Cairo", "egypt.png"),
            ("ئوردن و عەممان پێشوازیان لە شاندەکە کرد", "jordan.png"),
            ("Jordan envoy arrives in Amman", "jordan.png"),
            ("قەتەر لە دەوحە ناوبژیوانی دەکات", "qatar.png"),
            ("Qatar hosts talks in Doha", "qatar.png"),
            ("ئیمارات و دوبەی و ئەبوزەبی و شاریقە", "uae.png"),
            ("UAE summit in Dubai and Abu Dhabi, Emirates", "uae.png"),
            ("کوەیت و کوێت ڕاگەیەندراوێکیان بڵاوکردەوە", "kuwait.png"),
            ("Kuwait parliament approves budget", "kuwait.png"),
            ("سعودیە و سعوودیە لە ڕیاز و جددە و جەددە", "saudi.png"),
            ("Saudi talks between Riyadh and Jeddah", "saudi.png"),
            # Negative / false-positive guards
            ("سیناریۆی نوێ لە عێراق و بەغدا تاوتوێ کرا", "iraq.png"),
            ("سیناریۆی نوێ لە بەرپرسیارێتی ناوچەیی تاوتوێ کرا", None),
            ("Woman speaks about diplomacy globally", None),
            ("Officials told us talks continue today", None),
            ("ئاڵۆزییەکانی ڕۆژهەڵاتی ناوەڕاست لە ئوردن باس کرا", "jordan.png"),
            ("کۆمپانیای نەوتی باکوور بەرهەمهێنانی زیاد کرد", None),
            ("هێزە ئەمنییەکان لە ناوچەی ئەسکەندەرییە لە پارێزگای بابل ئۆپەراسیۆنیان ئەنجامدا", "iraq.png"),
            ("ئەنبار و دیالە و نەینەوا و سەڵاحەدین و زیقار و میسان و واست و موسەننا و قادسیە", "iraq.png"),
            ("پەلاماری دانیشتوان بۆ سەر فەلەستینییەکان و موڵکەکانیان لە کەرتی ڕۆژئاوا", "palestine.png"),
            ("بەیاننامەیەک لە کەرتی رۆژئاوا و فەلەستینی بڵاوکرایەوە", "palestine.png"),
            # Kurdistan & higher priority over Iraq
            ("کوردستان و هەرێمی کوردستان لە هەولێر و سلێمانی و دهۆک و کەرکوک و هەڵەبجە", "kurdistan.png"),
            ("باشوری کوردستان و باکووری کوردستان و ڕۆژهەڵاتی کوردستان و ڕۆژئاوای کوردستان و پێشمەرگە", "kurdistan.png"),
            ("عێراق و بەغدا لەگەڵ هەولێر و کەرکوک ڕێککەوتن", "kurdistan.png"),
            ("Kurdistan forces in Erbil, Sulaymaniyah, Duhok, and Kirkuk", "kurdistan.png"),
            ("Baghdad talks with Erbil and Kurdistan", "kurdistan.png"),
            # 10 previous countries
            ("عێراق و بەغدا و بەسرە و نەجەف و کەربەلا و مووسڵ", "iraq.png"),
            ("Iraq summit in Baghdad", "iraq.png"),
            ("ئێران و تاران و تەهران و سپای پاسداران و ئەسفەهان و مەشهەد", "iran.png"),
            ("Iran envoy meets officials in Tehran", "iran.png"),
            ("تورکیا و ئەنقەرە و ئیستەنبوڵ و ئەنکەرە", "turkey.png"),
            ("Türkiye talks between Ankara and Istanbul", "turkey.png"),
            ("سووریا و سوریا لە دیمەشق و حەلەب و حومس", "syria.png"),
            ("Syria report from Damascus and Aleppo", "syria.png"),
            ("ئیسرائیل و تەلئەبیب و تەلئەڤیڤ", "israel.png"),
            ("Israel officials meet in Tel Aviv", "israel.png"),
            ("فەلەستین و کەرتی غەززە و غەززە و ڕامەڵڵا و ڕەفەح و خان یونس", "palestine.png"),
            ("Palestine, Gaza, and Ramallah update", "palestine.png"),
            ("ئەمریکا و ویلایەتە یەکگرتووەکان لە واشنتۆن و کۆشکی سپی و پێنتاگۆن", "usa.png"),
            ("US Pentagon and Washington announcement", "usa.png"),
            ("ڕووسیا و روسیا لە مۆسکۆ و کرێملین", "russia.png"),
            ("Russia and Moscow Kremlin briefing", "russia.png"),
            ("چین و پەکین و بێیجینگ", "china.png"),
            ("China trade talks in Beijing", "china.png"),
            ("بەریتانیا و لەندەن و شانشینی یەکگرتوو", "uk.png"),
            ("UK and Britain officials in London", "uk.png"),
            # 9 new global countries
            ("فەرەنسا و پاریس و لێدوانی ماکرۆن", "france.png"),
            ("France President Macron in Paris", "france.png"),
            ("ئەڵمانیا و بەرلین بڕیاریان دا", "germany.png"),
            ("Germany government statement in Berlin", "germany.png"),
            ("ئۆکراینا و کیێڤ و زیلینسکی", "ukraine.png"),
            ("Ukraine talks in Kyiv", "ukraine.png"),
            ("ئیتاڵیا و ڕۆما میوانداری دەکەن", "italy.png"),
            ("Italy conference in Rome", "italy.png"),
            ("ئیسپانیا و مەدرید ڕاگەیەندراویان دا", "spain.png"),
            ("Spain summit in Madrid", "spain.png"),
            ("کەنەدا و ئۆتاوا هاوکاری دەنێرن", "canada.png"),
            ("Canada briefing from Ottawa", "canada.png"),
            ("ژاپۆن و تۆکیۆ پلانێکیان ڕاگەیاند", "japan.png"),
            ("Japan statement in Tokyo", "japan.png"),
            ("هیندستان و نیودەلهی ڕێککەوتن", "india.png"),
            ("India talks in New Delhi", "india.png"),
            ("پاکستان و ئیسلامئاباد کۆبوونەوە", "pakistan.png"),
            ("Pakistan officials in Islamabad", "pakistan.png"),
        ],
    )
    def test_detect_country_flag_all_10_countries_and_guards(self, text: str, expected_flag: str | None):
        assert detect_country_flag_filename(text) == expected_flag

    def test_all_30_country_flag_assets_exist(self):
        expected = {
            "oman.png", "bahrain.png", "yemen.png", "lebanon.png", "egypt.png",
            "jordan.png", "qatar.png", "uae.png", "kuwait.png", "saudi.png",
            "iraq.png", "iran.png", "turkey.png", "syria.png", "israel.png",
            "palestine.png", "usa.png", "russia.png", "china.png", "uk.png",
            "kurdistan.png", "france.png", "germany.png", "ukraine.png", "italy.png",
            "spain.png", "canada.png", "japan.png", "india.png", "pakistan.png",
        }
        flags_dir = ASSETS_DIR / "flags"
        for fname in expected:
            p = flags_dir / fname
            assert p.exists(), f"Missing flag asset: {p}"
            with Image.open(p) as im:
                assert im.width > 0 and im.height > 0

    @pytest.mark.asyncio
    async def test_card_renders_top_left_flag_badge_when_matched_and_clean_when_unmatched(self):
        generator = NewsCardGenerator()
        solid_dark_bg = Image.new("RGB", (1280, 720), (15, 25, 45))
        buf = io.BytesIO()
        solid_dark_bg.save(buf, format="JPEG")
        bg_bytes = buf.getvalue()

        art_no_flag = Article(
            translated_title="کۆبوونەوەی ئەمنی جیهانی دەستیپێکرد",
            category="politics_world",
            source_name="Reuters",
        )
        art_with_flag = Article(
            translated_title="کۆبوونەوەی ئەمنی لە دەوحە و قەتەر دەستیپێکرد",
            category="politics_world",
            source_name="Reuters",
        )

        card_plain_bytes = await generator.generate_card_bytes(art_no_flag, image_bytes=bg_bytes)
        card_flag_bytes = await generator.generate_card_bytes(art_with_flag, image_bytes=bg_bytes)

        with Image.open(io.BytesIO(card_plain_bytes)) as plain_img, Image.open(
            io.BytesIO(card_flag_bytes)
        ) as flag_img:
            assert plain_img.size == (1280, 720)
            assert flag_img.size == (1280, 720)

            # Plain card top-left corner at (MARGIN_X + 45, 36 + 28) has only dark background
            plain_px = plain_img.getpixel((MARGIN_X + 45, 36 + 28))
            assert max(plain_px) < 60

            # Qatar maroon/white flag badge renders in the top-left center pixel
            flag_px = flag_img.getpixel((MARGIN_X + 45, 36 + 28))
            assert flag_px != plain_px
            assert flag_px[0] > 90  # maroon or bright white flag pigment present

    @pytest.mark.asyncio
    async def test_admin_manual_photo_post_also_renders_kurdistan_corner_flag(self):
        """Confirm manual Telegram bot posts (admin_poster) use card_generator corner flags too."""
        from types import SimpleNamespace
        from tg_bot.admin_poster import publish_admin_manual_post

        solid_dark_bg = Image.new("RGB", (1280, 720), (15, 25, 45))
        bg_buf = io.BytesIO()
        solid_dark_bg.save(bg_buf, format="JPEG")
        photo_bytes = bg_buf.getvalue()

        captured_card_bytes: list[bytes] = []
        bot = MagicMock()
        bot.defaults = None
        tg_file = MagicMock()
        tg_file.download_as_bytearray = AsyncMock(return_value=bytearray(photo_bytes))
        bot.get_file = AsyncMock(return_value=tg_file)

        async def _capture_send_photo(*args, **kwargs):
            captured_card_bytes.append(kwargs["photo"].read())
            return SimpleNamespace(message_id=9001)

        bot.send_photo = AsyncMock(side_effect=_capture_send_photo)
        bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=9002))

        msg = SimpleNamespace(
            message_id=77,
            from_user=SimpleNamespace(id=111222333),
            chat=SimpleNamespace(id=111222333, type="private"),
            caption="عێراق و هەرێمی کوردستان لە هەولێر کۆبوونەوە",
            text=None,
            photo=[SimpleNamespace(file_id="hi_res_photo")],
        )

        result = await publish_admin_manual_post(
            bot,
            msg,
            channel_id="@KurdishGlobalNews",
            admin_chat_id=111222333,
            channel_handle="@KurdishGlobalNews",
        )
        assert result["status"] == "published"
        assert len(captured_card_bytes) == 1

        with Image.open(io.BytesIO(captured_card_bytes[0])) as rendered:
            # Kurdistan flag golden sun emblem sits centered at (MARGIN_X + 45, 36 + 28)
            center_px = rendered.getpixel((MARGIN_X + 45, 36 + 28))
            assert center_px[0] > 180 and center_px[1] > 140  # yellow sun emblem rendered over dark photo!

    def test_flag_detection_headline_priority_and_specific_cases(self):
        generator = NewsCardGenerator()

        # 1. Babel / Alexandria (Iraq) must resolve to iraq.png even if body mentions other places
        babil_art = Article(
            translated_title="هێزە ئەمنییەکان لە ناوچەی ئەسکەندەرییە لە پارێزگای بابل ئۆپەراسیۆنێکیان ئەنجامدا",
            translated_summary="لە هەولێر و واشنتۆن باس لە دۆخەکە کرا",
        )
        assert generator._resolve_article_flag_path(babil_art).name == "iraq.png"

        # 2. Palestinians in the West Bank must resolve to palestine.png, never kurdistan.png
        west_bank_art = Article(
            translated_title="پەلاماری دانیشتوان بۆ سەر فەلەستینییەکان و موڵکەکانیان لە کەرتی ڕۆژئاوا",
            translated_summary="بەپێی ڕاپۆرتەکان بارودۆخەکە لە تەلئەبیب و واشنتۆن تاوتوێ کرا",
        )
        assert generator._resolve_article_flag_path(west_bank_art).name == "palestine.png"

        # 3. Erbil / Kurdistan Region continues to resolve to kurdistan.png
        kurdistan_art = Article(
            translated_title="سەرۆکی حکومەتی هەرێمی کوردستان لە هەولێر پێشوازی لە شاندێک کرد",
            translated_summary="لە کۆبوونەوەکەدا باس لە پەیوەندییەکانی بەغدا و واشنتۆن کرا",
        )
        assert generator._resolve_article_flag_path(kurdistan_art).name == "kurdistan.png"

    def test_workflow_schedule_every_5_minutes_and_limit_5(self):
        workflow_path = ASSETS_DIR.parent.parent / ".github" / "workflows" / "news_bot.yml"
        workflow_text = workflow_path.read_text(encoding="utf-8")
        assert "cron: '*/5 * * * *'" in workflow_text
        assert "BATCH_TARGET_POSTS: '5'" in workflow_text
        assert "BATCH_PUBLISH_LIMIT: '5'" in workflow_text
        assert "MAX_NORMAL_POSTS_PER_CYCLE: '5'" in workflow_text

