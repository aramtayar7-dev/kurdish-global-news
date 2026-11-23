"""
Kurdish Global News Bot — Article Extractor

Extracts and enriches article metadata from raw article data.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

from sources.base import RawArticle
from storage.models import Article
from config import SOURCE_CREDIBILITY, NEWS_CATEGORIES

logger = logging.getLogger("processing.extractor")

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

# Category keywords for classification
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Wars & Armed Conflicts": ["war", "attack", "airstrike", "bombing", "offensive", "ceasefire", "battle", "troops", "killed", "casualties"],
    "Military & Defense": ["military", "army", "navy", "missile", "weapon", "defense", "pentagon", "drone", "nuclear weapon"],
    "International Politics": ["president", "prime minister", "parliament", "election", "government", "sanctions", "summit", "referendum", "legislation"],
    "Diplomacy & International Relations": ["diplomat", "embassy", "treaty", "negotiation", "UN", "bilateral", "alliance", "accord"],
    "Global Economy & Financial Markets": ["economy", "stock market", "gdp", "inflation", "recession", "trade", "tariff", "bank", "federal reserve", "interest rate"],
    "AI & Technology": ["artificial intelligence", "ai", "chatgpt", "machine learning", "startup", "tech", "silicon valley", "chip", "semiconductor"],
    "Science & Research": ["research", "study", "scientist", "discovery", "journal", "university", "experiment", "breakthrough"],
    "Health & Medical": ["health", "vaccine", "disease", "pandemic", "who", "hospital", "drug", "medical", "cancer", "virus"],
    "Climate & Environment": ["climate", "carbon", "emissions", "global warming", "renewable", "pollution", "environment", "deforestation"],
    "Natural Disasters & Emergencies": ["earthquake", "hurricane", "flood", "tornado", "tsunami", "wildfire", "eruption", "emergency", "disaster"],
    "Energy, Oil & Gas": ["oil", "gas", "opec", "energy", "petroleum", "pipeline", "solar", "wind power", "nuclear energy"],
    "Cybersecurity": ["hack", "cyber", "breach", "ransomware", "data leak", "malware", "cybersecurity", "phishing"],
    "International Organizations": ["united nations", "un", "nato", "eu", "who", "imf", "world bank", "iaea", "g7", "g20"],
    "Human Rights & Humanitarian": ["human rights", "refugee", "humanitarian", "asylum", "persecution", "genocide", "discrimination"],
    "Criminal Investigations & Security": ["arrest", "investigation", "crime", "terror", "fbi", "interpol", "smuggling", "cartel"],
    "Space & Astronomy": ["space", "nasa", "spacex", "satellite", "asteroid", "mars", "moon", "orbit", "astronaut", "telescope"],
    "International Sports": ["olympics", "world cup", "fifa", "champion", "tournament", "medal", "athletics"],
    "Business & Corporate": ["merger", "acquisition", "ceo", "ipo", "revenue", "profit", "layoff", "startup"],
    "World Leaders & Governments": ["biden", "putin", "xi jinping", "macron", "modi", "erdogan", "zelensky", "leader"],
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
        article.image_url = raw.image_url.strip()
        article.discovered_at = datetime.now(timezone.utc).isoformat()
        article.country = raw.country or self._detect_country(raw.url)
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
        """Classify article into a category based on keyword matching."""
        text = f"{title} {summary}".lower()
        best_category = "Other Significant Events"
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
