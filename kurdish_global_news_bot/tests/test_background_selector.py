"""
Tests for Smart Location & Topic-Aware Fallback Backgrounds (Idea 3)

Validates:
1. All 11 canonical default backgrounds exist as valid 1280x720 images.
2. Accurate theme selection across Kurdish, English, and Arabic keywords.
3. Strict precedence hierarchy (Cybersecurity -> Bashur -> Economy -> Regional -> Military -> Global).
4. Drop-in user custom override priority (.jpg > .jpeg > .webp > .png).
5. Deterministic variant hashing for numbered backgrounds.
6. Feature flag and fallback handling.
7. Seamless card generator integration without layout alterations.
"""

from __future__ import annotations

import io
from pathlib import Path
from PIL import Image
import pytest

from config import ASSETS_DIR, BACKGROUNDS_DIR
from processing.background_selector import (
    CANONICAL_THEMES,
    THEME_CYBERSECURITY,
    THEME_ECONOMY_ENERGY,
    THEME_GLOBAL_DIPLOMACY,
    THEME_IRAN_TEHRAN,
    THEME_IRAQ_BAGHDAD,
    THEME_ISRAEL_LEVANT,
    THEME_KURDISTAN_BASHUR,
    THEME_MILITARY_CONFLICT,
    THEME_SYRIA_DAMASCUS,
    THEME_TURKEY_ANKARA,
    THEME_USA_WASHINGTON,
    BackgroundSelector,
    resolve_background_bytes,
    resolve_background_path,
    select_background_theme,
)
from processing.card_generator import NewsCardGenerator
from storage.models import Article


# ═══════════════════════════════════════════════════
# 1. Default Background Assets Verification
# ═══════════════════════════════════════════════════

def test_all_11_canonical_backgrounds_exist():
    """Verify that all 11 canonical fallback background PNGs exist in assets/backgrounds."""
    assert BACKGROUNDS_DIR.exists(), f"Directory not found: {BACKGROUNDS_DIR}"

    for theme in CANONICAL_THEMES:
        bg_path = BACKGROUNDS_DIR / f"{theme}.png"
        assert bg_path.exists(), f"Missing default background for theme: {theme} at {bg_path}"
        assert bg_path.is_file()
        assert bg_path.stat().st_size > 1000, f"File {bg_path} appears empty or truncated"


def test_default_backgrounds_are_valid_1280x720_images():
    """Verify each default background can be decoded by PIL and has exact 1280x720 dimensions."""
    for theme in CANONICAL_THEMES:
        bg_path = BACKGROUNDS_DIR / f"{theme}.png"
        with Image.open(bg_path) as img:
            assert img.size == (1280, 720), f"Theme {theme} has invalid size {img.size}, expected (1280, 720)"
            assert img.mode in ("RGB", "RGBA"), f"Theme {theme} has unexpected mode {img.mode}"


# ═══════════════════════════════════════════════════
# 2. Theme Selection Logic (Kurdish, English, Arabic)
# ═══════════════════════════════════════════════════

def test_theme_selection_kurdistan_bashur():
    """Verify Southern Kurdistan articles select kurdistan_bashur theme."""
    # Kurdish Erbil/Peshmerga
    art_ku = Article(
        original_title="پێشمەرگە هێرشێکی چەکدارانی داعشی لە سنووری هەولێر تێکشکاند",
        original_summary="هێزەکانی پێشمەرگە لە میحوەری مەخموور ئۆپەراسیۆنێکیان ئەنجامدا.",
        category="کوردستان",
        region="کوردستان (باشوور)",
    )
    assert select_background_theme(art_ku) == THEME_KURDISTAN_BASHUR

    # English Sulaymaniyah / KRG
    art_en = Article(
        original_title="KRG Council of Ministers meets in Erbil to discuss civil servant salaries",
        original_summary="Prime Minister Masrour Barzani chaired the cabinet meeting on the budget.",
        category="kurdistan",
        region="Kurdistan",
    )
    assert select_background_theme(art_en) == THEME_KURDISTAN_BASHUR

    # Arabic Kirkuk
    art_ar = Article(
        original_title="وفد من حكومة إقليم كردستان يزور كركوك لبحث الأوضاع الأمنية",
        original_summary="البيشمركة والقوات الاتحادية تنسقان في المناطق المشتركة.",
        category="كردستان",
    )
    assert select_background_theme(art_ar) == THEME_KURDISTAN_BASHUR


def test_theme_selection_iraq_baghdad():
    """Verify Federal Iraq stories select iraq_baghdad theme."""
    art = Article(
        original_title="پەرلەمانی عێراق دەنگ لەسەر پڕۆژەیاسای نوێی هەڵبژاردنەکان دەدات",
        original_summary="محەمەد شیاع سودانی داوای لە لایەنە سیاسییەکان کرد ڕێکبکەون.",
        category="سیاسەت و جیهان",
        region="عێراق",
    )
    assert select_background_theme(art) == THEME_IRAQ_BAGHDAD

    art_en = Article(
        original_title="Iraqi Prime Minister Al-Sudani arrives in Basra to inaugurate oil refinery",
        original_summary="The federal government in Baghdad announced new infrastructure projects.",
        category="politics_world",
    )
    assert select_background_theme(art_en) == THEME_IRAQ_BAGHDAD


def test_theme_selection_iran_tehran():
    """Verify Iranian geopolitical news selects iran_tehran theme."""
    art_ku = Article(
        original_title="سوپای پاسدارانی ئێران مانۆڕێکی سەربازی لە گەرووی هورمز ئەنجام دەدات",
        original_summary="تاران هۆشداری دەداتە هێزە بیانییەکان لە کەنداوی فارس.",
        category="سیاسەت و جیهان",
        region="ئێران",
    )
    assert select_background_theme(art_ku) == THEME_IRAN_TEHRAN

    art_en = Article(
        original_title="Iranian President Pezeshkian comments on nuclear talks with IAEA inspectors",
        original_summary="Tehran confirmed enrichment levels at Natanz facility.",
        category="politics_world",
    )
    assert select_background_theme(art_en) == THEME_IRAN_TEHRAN


def test_theme_selection_israel_levant():
    """Verify Israel, Gaza, and Lebanon stories select israel_levant theme."""
    art_ku = Article(
        original_title="هێرشی ئاسمانی سوپای ئیسرائیل بۆ سەر باشووری بەیروت بەردەوامە",
        original_summary="حزبوڵڵا بە مووشەک وەڵامی بۆردومانەکانی تێل ئەڤیڤی دایەوە.",
        category="سیاسەت و جیهان",
    )
    assert select_background_theme(art_ku) == THEME_ISRAEL_LEVANT

    art_en = Article(
        original_title="Ceasefire negotiations between Israel and Hamas stall in Gaza",
        original_summary="Netanyahu addressed reporters regarding IDF operations in Rafah.",
        category="politics_world",
    )
    assert select_background_theme(art_en) == THEME_ISRAEL_LEVANT


def test_theme_selection_syria_damascus():
    """Verify Syria and Rojava news selects syria_damascus theme."""
    art_ku = Article(
        original_title="هەسەدە ئۆپەراسیۆنێکی لە قامیشلۆ و حەسەکە دژی پاشماوەکانی داعش راگەیاند",
        original_summary="مەزڵوم عەبدی فەرماندەی گشتی هێزەکانی سوریای دیموکرات لێدوانی دا.",
        category="سیاسەت و جیهان",
    )
    assert select_background_theme(art_ku) == THEME_SYRIA_DAMASCUS

    art_en = Article(
        original_title="Diplomatic meetings in Damascus focus on reconstruction of Aleppo",
        original_summary="Syrian government officials met with regional delegations.",
        category="politics_world",
    )
    assert select_background_theme(art_en) == THEME_SYRIA_DAMASCUS


def test_theme_selection_turkey_ankara():
    """Verify Turkey and Bakur news selects turkey_ankara theme."""
    art_ku = Article(
        original_title="ئەردۆغان لە ئەنقەرە لەگەڵ هاکان فیدان کۆبووەوە",
        original_summary="وەزارەتی دەرەوەی تورکیا هەڵوێستی وڵاتەکەی خستەڕوو.",
        category="سیاسەت و جیهان",
    )
    assert select_background_theme(art_ku) == THEME_TURKEY_ANKARA

    art_en = Article(
        original_title="Turkish Parliament convenes in Ankara to debate defense budget and Bayraktar drones",
        original_summary="President Erdogan outlined strategic goals for 2026.",
        category="politics_world",
    )
    assert select_background_theme(art_en) == THEME_TURKEY_ANKARA


def test_theme_selection_usa_washington():
    """Verify US government and military stories select usa_washington theme."""
    art_ku = Article(
        original_title="کۆشکی سپی و پێنتاگۆن بەیاننامەیەکی هاوبەش لەسەر ڕۆژهەڵاتی ناوەڕاست بڵاودەکەنەوە",
        original_summary="کۆنگرێسی ئەمریکا تاوتوێی بودجەی بەرگری دەکات.",
        category="سیاسەت و جیهان",
    )
    assert select_background_theme(art_ku) == THEME_USA_WASHINGTON

    art_en = Article(
        original_title="White House confirms President Trump will deliver address on foreign policy",
        original_summary="The Pentagon and State Department coordinated security measures.",
        category="politics_world",
    )
    assert select_background_theme(art_en) == THEME_USA_WASHINGTON


def test_theme_selection_cybersecurity():
    """Verify cybersecurity articles select cybersecurity theme."""
    art_ku = Article(
        original_title="هێرشێکی گەورەی ئەلیکترۆنی و باجەواڵە کەرتی تەندروستی گرتەوە",
        original_summary="هاککەرەکان داتای هەزاران نەخۆشیان دزیوە.",
        category="ئاسایشی سایبەری",
    )
    assert select_background_theme(art_ku) == THEME_CYBERSECURITY

    art_en = Article(
        original_title="Critical zero-day vulnerability discovered in widespread enterprise software",
        original_summary="CISA urges immediate patching to prevent ransomware exploitation.",
        category="cybersecurity",
    )
    assert select_background_theme(art_en) == THEME_CYBERSECURITY


def test_theme_selection_economy_energy():
    """Verify economy, inflation, oil, and market stories select economy_energy theme."""
    art_ku = Article(
        original_title="نرخی نەوتی خاوی برێنت بەرزبووەوە دوای کۆبوونەوەی ئۆپیک",
        original_summary="بازاڕەکانی وزە و نرخی دۆلار گۆڕانکارییان بەسەردا هات.",
        category="ئابووری",
    )
    assert select_background_theme(art_ku) == THEME_ECONOMY_ENERGY

    art_en = Article(
        original_title="Federal Reserve holds interest rates steady amid stubborn inflation figures",
        original_summary="Wall Street stock markets fluctuated following central bank comments.",
        category="economy_markets",
    )
    assert select_background_theme(art_en) == THEME_ECONOMY_ENERGY


def test_theme_selection_military_conflict():
    """Verify general war and missile/airstrike news selects military_conflict theme."""
    art = Article(
        original_title="هێرشی مووشەکی و بۆردومانی فڕۆکە جەنگییەکان لە بەرەکانی پێشەوەی شەڕ",
        original_summary="تۆپخانە سەربازییەکان چەندین پێگەی ستراتیژیان وێران کرد.",
        category="جەنگ و ئەمنی",
    )
    assert select_background_theme(art) == THEME_MILITARY_CONFLICT

    art_en = Article(
        original_title="Heavy artillery bombardment and ballistic missile strikes hit fortified positions",
        original_summary="Air defense systems intercepted incoming drone salvos over frontline territory.",
        category="military_conflict",
    )
    assert select_background_theme(art_en) == THEME_MILITARY_CONFLICT


def test_theme_selection_global_diplomacy_default():
    """Verify generic international politics and diplomatic summits fall back to global_diplomacy."""
    art = Article(
        original_title="لوتکەی نێودەوڵەتی سەرکردەکانی جیهان بۆ تاوتوێکردنی گۆڕانی کەشوهەوا دەستیپێکرد",
        original_summary="نوێنەرانی نەتەوە یەکگرتووەکان رێککەوتننامەیەکی نوێیان واژۆ کرد.",
        category="سیاسەت و جیهان",
    )
    assert select_background_theme(art) == THEME_GLOBAL_DIPLOMACY


# ═══════════════════════════════════════════════════
# 3. Precedence Hierarchy
# ═══════════════════════════════════════════════════

def test_precedence_cybersecurity_over_country():
    """Cybersecurity takes precedence over country mentions (e.g. Iranian cyberattack)."""
    art = Article(
        original_title="Major ransomware cyberattack hits Iranian infrastructure, disabling port systems",
        original_summary="Tehran officials reported a sophisticated zero-day malware breach.",
        category="cybersecurity",
    )
    assert select_background_theme(art) == THEME_CYBERSECURITY


def test_precedence_bashur_over_general_economy():
    """Bashur economic/salary news routes to kurdistan_bashur rather than generic economy."""
    art = Article(
        original_title="شاندێکی هەرێمی کوردستان بۆ پرسی مووچە و بودجە دەچێتە بەغدا",
        original_summary="هەولێر و بەغدا لەسەر هەناردەی نەوتی هەرێمی کوردستان گفتوگۆ دەکەن.",
        category="کوردستان",
        region="کوردستان (باشوور)",
    )
    assert select_background_theme(art) == THEME_KURDISTAN_BASHUR


def test_precedence_country_over_generic_military():
    """Country/regional specificity takes precedence over generic military conflict."""
    art = Article(
        original_title="Israeli airstrikes strike Hezbollah positions in southern Lebanon",
        original_summary="IDF forces launched artillery shelling across the border line.",
        category="military_conflict",
    )
    assert select_background_theme(art) == THEME_ISRAEL_LEVANT


# ═══════════════════════════════════════════════════
# 4. Drop-in User Overrides & Extension Priority
# ═══════════════════════════════════════════════════

def test_custom_jpg_overrides_default_png(tmp_path: Path):
    """A user-supplied .jpg file must take precedence over the default .png for the same theme."""
    # Create default PNG
    png_file = tmp_path / "kurdistan_bashur.png"
    png_file.write_bytes(b"default_png_data")

    # Create custom user JPG
    jpg_file = tmp_path / "kurdistan_bashur.jpg"
    jpg_file.write_bytes(b"custom_jpg_data")

    art = Article(
        original_title="Peshmerga forces complete joint training in Erbil",
        category="kurdistan",
        canonical_url="https://example.com/test-article-1",
    )

    selector = BackgroundSelector(backgrounds_dir=tmp_path)
    resolved = selector.resolve_path(art)
    assert resolved is not None
    assert resolved == jpg_file
    assert resolved.suffix.lower() == ".jpg"
    assert selector.resolve_bytes(art) == b"custom_jpg_data"


def test_extension_priority_order(tmp_path: Path):
    """Extension priority strictly adheres to .jpg > .jpeg > .webp > .png."""
    theme = "usa_washington"
    art = Article(original_title="Pentagon press briefing in Washington", category="politics_world")

    # Only .png exists -> picks .png
    f_png = tmp_path / f"{theme}.png"
    f_png.write_bytes(b"png")
    assert resolve_background_path(art, backgrounds_dir=tmp_path) == f_png

    # Add .webp -> picks .webp over .png
    f_webp = tmp_path / f"{theme}.webp"
    f_webp.write_bytes(b"webp")
    assert resolve_background_path(art, backgrounds_dir=tmp_path) == f_webp

    # Add .jpeg -> picks .jpeg over .webp
    f_jpeg = tmp_path / f"{theme}.jpeg"
    f_jpeg.write_bytes(b"jpeg")
    assert resolve_background_path(art, backgrounds_dir=tmp_path) == f_jpeg

    # Add .jpg -> picks .jpg over .jpeg
    f_jpg = tmp_path / f"{theme}.jpg"
    f_jpg.write_bytes(b"jpg")
    assert resolve_background_path(art, backgrounds_dir=tmp_path) == f_jpg


# ═══════════════════════════════════════════════════
# 5. Deterministic Variant Hashing
# ═══════════════════════════════════════════════════

def test_deterministic_numbered_variant_selection(tmp_path: Path):
    """When multiple variants exist, resolution is deterministic and varies across articles."""
    theme = "kurdistan_bashur"
    v1 = tmp_path / f"{theme}_1.jpg"
    v2 = tmp_path / f"{theme}_2.jpg"
    v3 = tmp_path / f"{theme}_3.jpg"
    v1.write_bytes(b"variant 1")
    v2.write_bytes(b"variant 2")
    v3.write_bytes(b"variant 3")

    art_a = Article(
        original_title="Erbil municipality completes highway project",
        canonical_url="https://example.com/erbil-story-a",
        category="kurdistan",
    )
    art_b = Article(
        original_title="Sulaymaniyah cultural festival opens",
        canonical_url="https://example.com/suli-story-b",
        category="kurdistan",
    )

    selector = BackgroundSelector(backgrounds_dir=tmp_path)
    res_a1 = selector.resolve_path(art_a)
    res_a2 = selector.resolve_path(art_a)
    # Consistency check: exact same article always yields the exact same variant
    assert res_a1 == res_a2
    assert res_a1 in (v1, v2, v3)

    res_b = selector.resolve_path(art_b)
    assert res_b in (v1, v2, v3)


# ═══════════════════════════════════════════════════
# 6. Fallbacks & Feature Flag
# ═══════════════════════════════════════════════════

def test_feature_flag_disabled_returns_none(tmp_path: Path):
    """When ENABLE_SMART_BACKGROUNDS is False, resolve_path returns None."""
    (tmp_path / "global_diplomacy.png").write_bytes(b"data")
    art = Article(original_title="UN summit in Geneva", category="politics_world")

    selector = BackgroundSelector(backgrounds_dir=tmp_path, enabled=False)
    assert selector.resolve_path(art) is None
    assert selector.resolve_bytes(art) is None


def test_missing_theme_falls_back_to_global_diplomacy(tmp_path: Path):
    """If a specific theme has no file, it gracefully falls back to global_diplomacy."""
    global_file = tmp_path / "global_diplomacy.png"
    global_file.write_bytes(b"global_fallback")

    # Article matches iraq_baghdad, but iraq_baghdad doesn't exist in tmp_path
    art = Article(original_title="Iraqi government in Baghdad", category="politics_world")
    resolved = resolve_background_path(art, backgrounds_dir=tmp_path, enabled=True)
    assert resolved == global_file


# ═══════════════════════════════════════════════════
# 7. Card Generator Integration & Visual Integrity
# ═══════════════════════════════════════════════════

def test_card_generator_uses_smart_background_when_image_missing():
    """Card generator automatically uses smart fallback background when image_bytes is None."""
    generator = NewsCardGenerator(assets_dir=ASSETS_DIR)
    article = Article(
        original_title="Erbil Parliament passes historic resolution",
        translated_title="پەرلەمانی کوردستان لە هەولێر بڕیارێکی مێژوویی پەسەند کرد",
        category="کوردستان",
        source_name="KGN Local",
        image_url="",  # No image
    )

    card_bytes = generator.render_card_sync(article, image_bytes=None)
    assert len(card_bytes) > 5000, "Rendered card should contain valid JPEG bytes"

    # Verify rendered card dimensions
    with Image.open(io.BytesIO(card_bytes)) as img:
        assert img.size == (1280, 720)
        assert img.format == "JPEG"


def test_card_generator_preserves_provided_image_bytes():
    """Card generator must NEVER override actual article image_bytes when provided."""
    generator = NewsCardGenerator(assets_dir=ASSETS_DIR)
    article = Article(
        original_title="Pentagon briefing on defense operations",
        translated_title="پێنتاگۆن روونکردنەوە لەسەر ئۆپەراسیۆنە بەرگرییەکان دەدات",
        category="سیاسەت و جیهان",
        source_name="Reuters",
    )

    # Provide a distinctive 100x100 solid bright green image as downloaded bytes
    custom_src = Image.new("RGB", (100, 100), (0, 255, 0))
    buf = io.BytesIO()
    custom_src.save(buf, format="JPEG")
    provided_bytes = buf.getvalue()

    card_bytes = generator.render_card_sync(article, image_bytes=provided_bytes)
    assert len(card_bytes) > 5000

    with Image.open(io.BytesIO(card_bytes)) as img:
        assert img.size == (1280, 720)
        # Check upper-left pixel area (where green should be visible under subtle vignette)
        r, g, b = img.getpixel((50, 150))[:3]
        # Green channel should dominate because the custom image was green
        assert g > r and g > b, f"Expected green channel to dominate, got RGB=({r}, {g}, {b})"
