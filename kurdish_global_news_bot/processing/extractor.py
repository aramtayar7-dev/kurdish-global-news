"""
Kurdish Global News Bot — Article Extractor

Extracts and enriches article metadata from raw article data.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urljoin, urlparse

from sources.base import RawArticle
from storage.models import Article
from config import SOURCE_CREDIBILITY, NEWS_CATEGORIES

import aiohttp
from bs4 import BeautifulSoup
from rapidfuzz import fuzz

logger = logging.getLogger("processing.extractor")

SCRAPING_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
}

# Country detection from domains / keywords
COUNTRY_DOMAINS: dict[str, str] = {
    ".uk": "United Kingdom", ".co.uk": "United Kingdom",
    ".us": "United States", ".gov": "United States",
    ".de": "Germany", ".fr": "France", ".it": "Italy",
    ".es": "Spain", ".ru": "Russia", ".cn": "China",
    ".jp": "Japan", ".kr": "South Korea", ".in": "India",
    ".au": "Australia", ".ca": "Canada", ".br": "Brazil",
    ".tr": "Turkey", ".il": "Israel", ".sa": "Saudi Arabia",
    ".ir": "Iran", ".iq": "Iraq", ".eg": "Egypt",
    ".ua": "Ukraine", ".pk": "Pakistan",
}

# Category keywords for fallback classification onto the 7 canonical categories
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "kurdistan": [
        "kurdistan", "kurdish", "erbil", "hewler", "sulaymaniyah", "duhok", "kirkuk",
        "rojava", "qamishli", "kobani", "hasakah", "sdf", "peshmerga", "krg",
        "diyarbakir", "amed", "van ", "bakur", "rojhelat", "sanandaj", "mahabad",
        "urmia", "kermanshah", "هەولێر", "سلێمانی", "کەرکوک", "کوردستان", "ڕۆژئاوا", "دهۆک",
    ],
    "military_conflict": [
        "war", "attack", "airstrike", "bombing", "offensive", "ceasefire", "battle",
        "troops", "killed", "casualties", "military", "army", "navy", "missile",
        "weapon", "defense", "pentagon", "drone", "nuclear weapon",
    ],
    "economy_markets": [
        "economy", "stock market", "gdp", "inflation", "recession", "trade", "tariff",
        "bank", "federal reserve", "interest rate", "oil", "gas", "opec", "energy",
        "nvidia", "earnings", "ipo", "revenue", "profit",
    ],
    "ai_tech": [
        "artificial intelligence", "ai", "chatgpt", "openai", "anthropic", "gemini",
        "llm", "machine learning", "tech", "silicon valley", "semiconductor", "chip",
    ],
    "cybersecurity": [
        "hack", "cyber", "breach", "ransomware", "data leak", "malware",
        "cybersecurity", "phishing", "zero-day", "vulnerability",
    ],
    "science_space": [
        "space", "nasa", "spacex", "satellite", "asteroid", "mars", "moon", "orbit",
        "astronaut", "telescope", "research", "study", "scientist", "discovery",
    ],
    "politics_world": [
        "president", "prime minister", "parliament", "election", "government",
        "sanctions", "summit", "referendum", "legislation", "diplomat", "embassy",
        "treaty", "negotiation", "united nations", "nato", "eu",
    ],
}


class ArticleExtractor:
    """Extracts and enriches article metadata from raw discovery data."""

    def extract(self, raw: RawArticle) -> Article:
        """Convert a RawArticle to a fully enriched Article model."""
        article = Article()
        article.original_url = raw.url.strip()
        article.canonical_url = Article.canonicalize_url(raw.url)
        article.original_title = raw.title.strip()
        article.normalized_title = Article.normalize_title(raw.title)
        article.original_summary = raw.summary.strip()[:1000]
        article.source_name = raw.source_name.strip()
        article.publication_date = raw.publication_date or ""
        article.image_url = (raw.image_url or "").strip()
        article.video_url = raw.video_url.strip() if getattr(raw, "video_url", None) else None
        article.discovered_at = datetime.now(timezone.utc).isoformat()
        article.country = raw.country or self._detect_country(raw.url)
        article.region = raw.region.strip() if raw.region else ""
        article.category = raw.category or self._classify_category(raw.title, raw.summary)

        # Compute deduplication hashes
        article.compute_content_hash()
        article.compute_event_fingerprint()

        return article

    def _detect_country(self, url: str) -> str:
        """Attempt to detect country from URL domain."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            for suffix, country in COUNTRY_DOMAINS.items():
                if domain.endswith(suffix):
                    return country
        except Exception:
            pass
        return ""

    def _classify_category(self, title: str, summary: str) -> str:
        """Classify article into one of the canonical categories based on keyword matching."""
        text = f"{title} {summary}".lower()
        best_category = "politics_world"
        best_score = 0

        for category, keywords in CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > best_score:
                best_score = score
                best_category = category

        return best_category

    @staticmethod
    def get_source_credibility(source_name: str) -> float:
        """Look up the credibility score for a news source."""
        name = source_name.lower().strip()
        for key, score in SOURCE_CREDIBILITY.items():
            if key in name or name in key:
                return score
        return SOURCE_CREDIBILITY.get("default", 0.5)

    async def fetch_full_article_text(self, url: str) -> Optional[str]:
        """
        Asynchronously fetch the full webpage HTML from url using aiohttp
        and extract the main article paragraphs using beautifulsoup4.
        Returns up to the first ~3,000 characters of clean article text,
        or None if scraping failed or returned empty content.
        """
        text, _ = await self.fetch_article_content_and_image(url)
        return text

    async def fetch_article_content_and_image(self, url: str) -> tuple[Optional[str], Optional[str]]:
        """
        Asynchronously fetch the webpage HTML from url and extract BOTH:
        1. Main article body text (up to ~3,000 chars)
        2. Representative article image URL (og:image / twitter:image)
        """
        if not url or not url.startswith(("http://", "https://")):
            return None, None

        timeout = aiohttp.ClientTimeout(total=10.0)
        final_url = url
        try:
            async with aiohttp.ClientSession(timeout=timeout, headers=SCRAPING_HEADERS) as session:
                async with session.get(url, allow_redirects=True) as response:
                    if response.status != 200:
                        logger.warning("Scraping %s returned HTTP %d", url[:80], response.status)
                        return None, None

                    final_url = str(response.url)
                    content_type = response.headers.get("Content-Type", "").lower()
                    if content_type and "text/html" not in content_type and "application/xhtml" not in content_type:
                        logger.warning("Scraping %s non-HTML content type: %s", url[:80], content_type)
                        return None, None

                    html = await response.text(errors="replace")
        except Exception as e:
            logger.warning("Scraping failed for %s: %s", url[:80], str(e)[:150])
            return None, None

        body = self.extract_article_body_from_html(html)
        og_image = self.extract_og_image_from_html(html, base_url=final_url)
        return body, og_image

    @staticmethod
    def extract_og_image_from_html(html: str, base_url: str = "") -> Optional[str]:
        """Extract og:image or twitter:image URL from article HTML."""
        if not html or not html.strip():
            return None

        try:
            soup = BeautifulSoup(html, "html.parser")
            meta_selectors = [
                {"property": "og:image"},
                {"property": "og:image:secure_url"},
                {"name": "twitter:image"},
                {"name": "twitter:image:src"},
            ]
            for attrs in meta_selectors:
                tag = soup.find("meta", attrs=attrs)
                if tag and tag.get("content"):
                    raw_img = str(tag["content"]).strip()
                    if not raw_img:
                        continue
                    resolved = urljoin(base_url, raw_img) if base_url else raw_img
                    if resolved.startswith(("http://", "https://")):
                        return resolved
        except Exception as e:
            logger.debug("og:image extraction failed: %s", e)

        return None

    def extract_article_body_from_html(self, html: str) -> Optional[str]:
        """Extract clean article paragraphs from raw HTML (up to ~3,000 chars)."""
        if not html or not html.strip():
            return None

        try:
            soup = BeautifulSoup(html, "html.parser")

            # Strip non-content and layout elements
            for tag in soup([
                "script", "style", "nav", "header", "footer", "aside",
                "noscript", "form", "svg", "button", "figure", "figcaption", "iframe"
            ]):
                tag.decompose()

            # Find main article container or paragraphs
            container = soup.find("article") or soup.find("main")
            if not container:
                container = soup.find(class_=re.compile(
                    r"article[-_]?(body|content|text)|story[-_]?content|post[-_]?content",
                    re.I,
                ))
            if not container:
                container = soup.body

            if not container:
                return None

            paragraphs: list[str] = []
            for p in container.find_all("p"):
                text = p.get_text(strip=True)
                if len(text) < 35:
                    continue
                lower_text = text.lower()
                if any(bp in lower_text for bp in [
                    "cookie", "subscribe", "newsletter", "sign up", "terms of service",
                    "privacy policy", "all rights reserved", "advertisement", "copyright",
                    "follow us on", "read more:", "photo by", "getty images"
                ]):
                    continue
                paragraphs.append(text)

            if not paragraphs:
                return None

            combined = "\n\n".join(paragraphs).strip()
            cleaned = combined[:3000].strip()
            return cleaned if len(cleaned) >= 100 else None

        except Exception as e:
            logger.warning("HTML parsing error: %s", e)
            return None

    @staticmethod
    def is_insufficient_detail(
        title: str,
        text: str,
        min_len: int = 160,
        max_similarity: float = 70.0,
    ) -> tuple[bool, str]:
        """
        Check if article text is insufficient for publication:
        - Text is very thin (< min_len characters), or
        - Text is > max_similarity% similar to the headline (repeat of headline).
        Returns (is_insufficient, reason).
        """
        clean_text = (text or "").strip()
        if len(clean_text) < min_len:
            return True, f"Text too thin ({len(clean_text)} chars < {min_len} min limit)"

        clean_title = (title or "").strip()
        if clean_title:
            ratio = fuzz.ratio(clean_title.lower(), clean_text.lower())
            token_sort = fuzz.token_sort_ratio(clean_title.lower(), clean_text.lower())
            max_sim = max(ratio, token_sort)
            if max_sim > max_similarity:
                return True, f"Text is {max_sim:.1f}% similar to title (headline repetition)"

        return False, ""
