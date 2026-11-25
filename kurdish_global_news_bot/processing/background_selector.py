"""
Kurdish Global News Bot — Smart Location & Topic-Aware Fallback Backgrounds

Deterministically selects and resolves high-contrast, broadcast-quality fallback
backgrounds for news articles lacking a usable photo (e.g. stripped Telegram
screenshots, missing RSS images) based on their geographic region, key entities,
and thematic topic.

Zero Gemini API calls, zero external stock-photo dependencies, 100% offline.
Full support for drop-in user overrides (.jpg > .jpeg > .webp > .png) and
deterministic numbered variant selection.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

from config import BACKGROUNDS_DIR, ENABLE_SMART_BACKGROUNDS
from core.importance_ranker import (
    clean_article_event_text,
    is_bashur_article,
    is_cybersecurity_article,
    is_economy_article,
    normalize_category,
)
from storage.models import Article

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════
# 11 Canonical Themes & Extensions
# ═══════════════════════════════════════════════════
THEME_KURDISTAN_BASHUR = "kurdistan_bashur"
THEME_IRAQ_BAGHDAD = "iraq_baghdad"
THEME_IRAN_TEHRAN = "iran_tehran"
THEME_ISRAEL_LEVANT = "israel_levant"
THEME_SYRIA_DAMASCUS = "syria_damascus"
THEME_TURKEY_ANKARA = "turkey_ankara"
THEME_USA_WASHINGTON = "usa_washington"
THEME_CYBERSECURITY = "cybersecurity"
THEME_ECONOMY_ENERGY = "economy_energy"
THEME_MILITARY_CONFLICT = "military_conflict"
THEME_GLOBAL_DIPLOMACY = "global_diplomacy"

CANONICAL_THEMES = [
    THEME_KURDISTAN_BASHUR,
    THEME_IRAQ_BAGHDAD,
    THEME_IRAN_TEHRAN,
    THEME_ISRAEL_LEVANT,
    THEME_SYRIA_DAMASCUS,
    THEME_TURKEY_ANKARA,
    THEME_USA_WASHINGTON,
    THEME_CYBERSECURITY,
    THEME_ECONOMY_ENERGY,
    THEME_MILITARY_CONFLICT,
    THEME_GLOBAL_DIPLOMACY,
]

# Extension priority: user custom photos (.jpg/.jpeg) override default .png
EXTENSION_PRIORITY = [".jpg", ".jpeg", ".webp", ".png"]


# ═══════════════════════════════════════════════════
# Regex Patterns (Kurdish, Arabic, English)
# ═══════════════════════════════════════════════════

CYBERSECURITY_PATTERN = re.compile(
    r"(\bcybersecurity\b|\bcyber\s+security\b|\bcyberattack\b|\bcyber\s+attack\b|\bhacker\b|\bhacking\b|"
    r"\bmalware\b|\bransomware\b|\bdata\s+breach\b|\bzero-day\b|\bddos\b|\bphishing\b|\bspyware\b|"
    r"\binfosec\b|\btrojan\b|\bcisa\b|\bdark\s+web\b|bleepingcomputer|"
    r"ئاسایشی\s+سایبەری|\bهاک\b|\bهاککردن\b|\bهاککەر\b|\bهاککرا\b|هێرشی\s+ئەلیکترۆنی|باجەواڵە|دزەکردنی\s+زانیاری|فیشینگ|سیخوڕی\s+ئەلیکترۆنی|"
    r"الأمن\s+السيبراني|\bقرصنة\b|هجوم\s+سيبراني|برمجيات\s+خبيثة|\bفدية\b|\bاختراق\b|ثغرة\s+أمنية|تجسس\s+إلكتروني)",
    re.IGNORECASE,
)

BASHUR_EXPLICIT_PATTERN = re.compile(
    r"(\berbil\b|\bhewler\b|\bsulaymaniyah\b|\bslemani\b|\bsuli\b|\bduhok\b|\bhalabja\b|\bkirkuk\b|"
    r"\bpeshmerga\b|\bkrg\b|kurdistan\s+region|iraqi\s+kurdistan|\bbashur\b(?!\s+(?:beirut|lebanon|gaza|israel|syria|iraq|iran|damascus|tehran|baghdad))|"
    r"\bbarzani\b|\btalabani\b|\bzakho\b|\bsoran\b|\bchamchamal\b|\branya\b|\bqaladze\b|\bderbendikhan\b|"
    r"هەولێر|سلێمانی|دهۆک|هەڵەبجە|کەرکوک|پێشمەرگە|هەرێمی\s+کوردستان|حکومەتی\s+هەرێم|کوردستانی\s+باشوور|باشووری\s+کوردستان|"
    r"باشوور(?!ی\s+(?:بەیروت|لوبنان|غەززە|ئیسرائیل|سوریا|عێراق|تاران|بەغدا|دیمەشق|ئەمریکا|یەمان|سودان|کەرتی|ئێران|تەورێز|کەناری))|"
    r"بارزانی|تاڵەبانی|زاخۆ|سۆران|چەمچەماڵ|ڕانیە|قەڵادزێ|دەربەندیخان|مووچە|"
    r"أربيل|السليمانية|دهوك|حلبجة|كركوك|البيشمركة|إقليم\s+كردستان|حكومة\s+الإقليم)",
    re.IGNORECASE,
)

ECONOMY_ENERGY_PATTERN = re.compile(
    r"(\beconomy\b|\beconomic\b|\bcrude\s+oil\b|\boil\s+prices?\b|\bopec\b|\bbrent\b|natural\s+gas|"
    r"\binflation\b|interest\s+rates?|federal\s+reserve|central\s+bank|stock\s+market|wall\s+street|"
    r"\bdollar\b|\bpetroleum\b|\bgdp\b|trade\s+tariff|gas\s+pipeline|\benergy\s+crisis\b|"
    r"ئابووری|نەوت|گاز|ئۆپیک|دۆلار|هەڵاوسان|سوودی\s+بانکی|بۆرسە|نرخی\s+نەوت|هەناردە|بانکی\s+ناوەندی|وزە|"
    r"اقتصاد|نفط|برنت|أوبك|تضخم|سعر\s+الفائدة|أسعار\s+الفائدة|بنك\s+مركزي|أسواق|دولار|طاقة)",
    re.IGNORECASE,
)

ISRAEL_LEVANT_PATTERN = re.compile(
    r"(\bisrael\b|\bisraeli\b|tel\s+aviv|\bjerusalem\b|\bgaza\b|\bhamas\b|\bnetanyahu\b|\bidf\b|"
    r"\blebanon\b|\blebanese\b|\bbeirut\b|\bhezbollah\b|west\s+bank|\brafah\b|khan\s+younis|"
    r"\bpalestine\b|\bpalestinian\b|\bunifil\b|\bnasrallah\b|\bgolan\b|"
    r"ئیسرائیل|تێل\s+ئەڤیڤ|قودس|غەززە|حەماس|ناتانیاهۆ|لوبنان|بەیروت|حزبوڵڵا|کەرتی\s+غەززە|فەلەستین|سوپای\s+ئیسرائیل|رەفەح|خان\s+یونس|"
    r"إسرائيل|تل\s+أبيب|القدس|غزة|حماس|نتنياهو|لبنان|بيروت|حزب\s+الله|قطاع\s+غزة|فلسطين|رفح|خان\s+يونس)",
    re.IGNORECASE,
)

IRAN_TEHRAN_PATTERN = re.compile(
    r"(\biran\b|\biranian\b|\btehran\b|\birgc\b|revolutionary\s+guards?|\bkhamenei\b|\bpezeshkian\b|"
    r"\braisi\b|quds\s+force|\bnatanz\b|\bfordow\b|strait\s+of\s+hormuz|persian\s+gulf|\bisfahan\b|\bshiraz\b|\btabriz\b|"
    r"ئێران|تاران|خامنەیی|پزشکیان|سوپای\s+پاسداران|قودس|هورمز|کەنداوی\s+فارس|ئەتۆمی\s+ئێران|تەورێز|شیراز|ئیسفەهان|"
    r"إيران|طهران|خامنئي|بزشكيان|الحرس\s+الثوري|فيلق\s+القدس|مضيق\s+هرمز|الخليج\s+الفارسي)",
    re.IGNORECASE,
)

IRAQ_BAGHDAD_PATTERN = re.compile(
    r"(\biraq\b|\biraqi\b|\bbaghdad\b|\bbasra\b|\bmosul\b|\bnajaf\b|\bkarbala\b|\banbar\b|\bsudani\b|"
    r"al-sudani|iraqi\s+parliament|iraqi\s+army|iraqi\s+government|green\s+zone|\bhashd\b|popular\s+mobilization|federal\s+iraq|"
    r"بەغدا|بەسرە|مووسڵ|نەجەف|کەربەلا|ئەنبار|سودانی|حەشد|سوپای\s+عێراق|پەرلەمانی\s+عێراق|حکومەتی\s+عێراق|عێراق|"
    r"بغداد|البصرة|الموصل|النجف|كربلاء|الأنبار|السوداني|الحشد\s+الشعبي|الجيش\s+العراقي|البرلمان\s+العراقي|العراق)",
    re.IGNORECASE,
)

SYRIA_DAMASCUS_PATTERN = re.compile(
    r"(\bsyria\b|\bsyrian\b|\bdamascus\b|\baleppo\b|\bhoms\b|\blatakia\b|\bassad\b|\brojava\b|"
    r"\bqamishli\b|\bkobani\b|\bkobane\b|\bhasakah\b|\bsdf\b|syrian\s+democratic\s+forces|mazloum\s+abdi|"
    r"\bmanbij\b|\bidlib\b|deir\s+ez-zor|\baanes\b|"
    r"سوریا|دیمەشق|ئەسەد|حەلەب|لازقیە|ڕۆژئاوا|قامیشلۆ|کۆبانێ|حەسەکە|مەزڵوم\s+عەبدی|هەسەدە|منبج|دێرەزوور|ئیدلب|"
    r"سوريا|سورية|دمشق|الأسد|حلب|اللاذقية|روجآفا|القامشلي|كوباني|الحسكة|قسد|مظلوم\s+عبدي|منبج|إدلب)",
    re.IGNORECASE,
)

TURKEY_ANKARA_PATTERN = re.compile(
    r"(\bturkey\b|\bturkish\b|\btürkiye\b|\bankara\b|\bistanbul\b|\berdogan\b|\berdo\u011fan\b|"
    r"\bfidan\b|hakan\s+fidan|turkish\s+parliament|mit\s+intelligence|\btsk\b|\bbayraktar\b|\bbakur\b|"
    r"تورکیا|ئەنقەرە|ئەستەنبوڵ|ئەردۆغان|هاکان\s+فیدان|میت|سوپای\s+تورکیا|باکوور|"
    r"تركيا|أنقرة|إسطنبول|أردوغان|هاكان\s+فيدان|المخابرات\s+التركية|الجيش\s+التركي)",
    re.IGNORECASE,
)

USA_WASHINGTON_PATTERN = re.compile(
    r"(\bunited\s+states\b|\busa\b|\bu\.s\.?\b|\bwashington\b|white\s+house|\bpentagon\b|\bcongress\b|"
    r"\bcapitol\b|\btrump\b|\bbiden\b|\bblinken\b|\bcentcom\b|us\s+military|us\s+forces|state\s+department|"
    r"ئەمریکا|ویلایەتە\s+یەکگرتووەکان|واشنتن|کۆشکی\s+سپی|پێنتاگۆن|کۆنگرێس|ترەمپ|بایدن|بلینکن|سێنتکۆم|سوپای\s+ئەمریکا|"
    r"أمريكا|الولايات\s+المتحدة|واشنطن|البيت\s+الأبيض|البنتاغون|الكونغرس|ترامب|بايدن|بلينكن|سنتكوم)",
    re.IGNORECASE,
)

MILITARY_CONFLICT_PATTERN = re.compile(
    r"(\bmissile\s+strike\b|\bairstrike\b|\bair\s+strike\b|\bdrone\s+attack\b|\bballistic\s+missile\b|"
    r"\bartillery\b|\bshelling\b|armed\s+clashes|\bfirefight\b|\bcombat\b|air\s+defense|\binterception\b|"
    r"\bbombardment\b|\bwarfare\b|battlefield|\bdrone\s+strike\b|"
    r"مووشەک|هێرشی\s+مووشەکی|بۆردومان|درۆن|تۆپباران|شەڕ\s+و\s+پێکدادان|بەرگری\s+ئاسمانی|هێرشی\s+ئاسمانی|تەقینەوە|سەربازی|"
    r"صاروخ|ضربة\s+صاروخية|غارة\s+جوية|طائرة\s+مسيرة|قصف|اشتباكات|دفاع\s+جوي|معارك|انفجار)",
    re.IGNORECASE,
)


def select_background_theme(article: Article) -> str:
    """
    Select one of the 11 canonical fallback themes using strict precedence:
      1. Specialized Category: Cybersecurity
      2. Southern Kurdistan (Bashur): Erbil, Sulaymaniyah, Duhok, Kirkuk, KRG, Peshmerga
      3. Specialized Category: Economy & Energy (when not Bashur)
      4. Regional / Country:
         - Israel & Levant
         - Iran
         - Federal Iraq
         - Syria & Rojava
         - Turkey
         - USA
      5. Military / Conflict
      6. Global Diplomacy (Ultimate default)
    """
    clean_text = clean_article_event_text(article)
    # Use content-only text for Southern Kurdistan checks so feed region hints never misclassify global stories
    combined_text = f"{clean_text} {getattr(article, 'country', '')}"
    raw_cat = (getattr(article, "category", "") or "").strip().lower()
    norm_cat = normalize_category(raw_cat, text_hint=clean_text)

    # 1. Specialized: Cybersecurity
    if is_cybersecurity_article(article) or norm_cat == "cybersecurity" or CYBERSECURITY_PATTERN.search(combined_text):
        return THEME_CYBERSECURITY

    # 2. Southern Kurdistan (Bashur) — strictly content-based
    if is_bashur_article(article) or BASHUR_EXPLICIT_PATTERN.search(clean_text):
        return THEME_KURDISTAN_BASHUR

    # 3. Specialized: Economy & Energy
    if is_economy_article(article) or norm_cat == "economy_markets" or ECONOMY_ENERGY_PATTERN.search(combined_text):
        return THEME_ECONOMY_ENERGY

    # 4. Regional / Country Specific
    if ISRAEL_LEVANT_PATTERN.search(combined_text):
        return THEME_ISRAEL_LEVANT

    if IRAN_TEHRAN_PATTERN.search(combined_text):
        return THEME_IRAN_TEHRAN

    if IRAQ_BAGHDAD_PATTERN.search(combined_text):
        return THEME_IRAQ_BAGHDAD

    if SYRIA_DAMASCUS_PATTERN.search(combined_text):
        return THEME_SYRIA_DAMASCUS

    if TURKEY_ANKARA_PATTERN.search(combined_text):
        return THEME_TURKEY_ANKARA

    if USA_WASHINGTON_PATTERN.search(combined_text):
        return THEME_USA_WASHINGTON

    # 5. Military / Conflict
    if norm_cat == "military_conflict" or raw_cat in {"جەنگ و ئەمنی", "military_conflict"} or MILITARY_CONFLICT_PATTERN.search(combined_text):
        return THEME_MILITARY_CONFLICT

    # 6. Global Diplomacy (default)
    return THEME_GLOBAL_DIPLOMACY


def _gather_theme_candidates(theme: str, backgrounds_dir: Path) -> list[Path]:
    """Gather all image files matching theme or its numbered variants in backgrounds_dir."""
    if not backgrounds_dir.exists() or not backgrounds_dir.is_dir():
        return []

    valid_exts = {ext.lower() for ext in EXTENSION_PRIORITY}
    candidates: list[Path] = []

    for item in backgrounds_dir.iterdir():
        if not item.is_file():
            continue
        if item.suffix.lower() not in valid_exts:
            continue

        stem = item.stem.lower()
        # Matches e.g. kurdistan_bashur, kurdistan_bashur_1, kurdistan_bashur-2
        if stem == theme or stem.startswith(f"{theme}_") or stem.startswith(f"{theme}-"):
            candidates.append(item)

    return candidates


def _filter_by_extension_priority(candidates: list[Path]) -> list[Path]:
    """Filter candidates to keep only the highest-priority extension (.jpg > .jpeg > .webp > .png)."""
    if not candidates:
        return []

    for ext in EXTENSION_PRIORITY:
        matched = [c for c in candidates if c.suffix.lower() == ext]
        if matched:
            return matched

    return candidates


def resolve_background_path(
    article: Article,
    backgrounds_dir: Optional[Path] = None,
    enabled: Optional[bool] = None,
) -> Optional[Path]:
    """
    Resolve the absolute Path of the best matching background file for an article.
    Honors ENABLE_SMART_BACKGROUNDS feature flag, drop-in user overrides, and
    deterministic numbered variant selection.
    """
    is_enabled = ENABLE_SMART_BACKGROUNDS if enabled is None else enabled
    if not is_enabled:
        return None

    bg_dir = backgrounds_dir or BACKGROUNDS_DIR
    if not bg_dir.exists() or not bg_dir.is_dir():
        return None

    theme = select_background_theme(article)
    candidates = _gather_theme_candidates(theme, bg_dir)

    # Fallback to global_diplomacy if specific theme has no assets
    if not candidates and theme != THEME_GLOBAL_DIPLOMACY:
        candidates = _gather_theme_candidates(THEME_GLOBAL_DIPLOMACY, bg_dir)

    # Fallback to any file starting with 'default'
    if not candidates:
        candidates = _gather_theme_candidates("default", bg_dir)

    if not candidates:
        return None

    # Apply extension priority (.jpg overrides .png)
    candidates = _filter_by_extension_priority(candidates)
    if not candidates:
        return None

    # Deterministic variant selection using article identity
    candidates.sort(key=lambda p: p.name.lower())
    seed = (
        getattr(article, "canonical_url", "")
        or getattr(article, "original_url", "")
        or getattr(article, "event_key", "")
        or getattr(article, "original_title", "")
        or "default_seed"
    )
    hash_val = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16)
    selected = candidates[hash_val % len(candidates)]
    return selected


def resolve_background_bytes(
    article: Article,
    backgrounds_dir: Optional[Path] = None,
    enabled: Optional[bool] = None,
) -> Optional[bytes]:
    """Read and return JPEG/PNG/WebP bytes of the resolved background file, or None."""
    path = resolve_background_path(article, backgrounds_dir=backgrounds_dir, enabled=enabled)
    if path and path.is_file():
        try:
            return path.read_bytes()
        except Exception as e:
            logger.warning("Failed to read background image from %s: %s", path, e)
    return None


class BackgroundSelector:
    """Class wrapper providing dependency-injected background selection."""

    def __init__(
        self,
        backgrounds_dir: Optional[Path] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self.backgrounds_dir = backgrounds_dir or BACKGROUNDS_DIR
        self.enabled = ENABLE_SMART_BACKGROUNDS if enabled is None else enabled

    def select_theme(self, article: Article) -> str:
        return select_background_theme(article)

    def resolve_path(self, article: Article) -> Optional[Path]:
        return resolve_background_path(
            article,
            backgrounds_dir=self.backgrounds_dir,
            enabled=self.enabled,
        )

    def resolve_bytes(self, article: Article) -> Optional[bytes]:
        return resolve_background_bytes(
            article,
            backgrounds_dir=self.backgrounds_dir,
            enabled=self.enabled,
        )
