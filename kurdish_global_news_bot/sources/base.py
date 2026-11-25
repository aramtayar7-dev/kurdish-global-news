"""
Kurdish Global News Bot — Base Source Adapter

Abstract interface for all news source adapters.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from storage.models import Article

logger = logging.getLogger("sources.base")


@dataclass
class RawArticle:
    """Raw article data as discovered from a source, before full processing."""
    url: str = ""
    title: str = ""
    summary: str = ""
    source_name: str = ""
    category: str = ""
    publication_date: Optional[str] = None
    image_url: Optional[str] = ""
    video_url: Optional[str] = None
    country: str = ""
    region: str = ""


class BaseSource(ABC):
    """Abstract base class for all news source adapters."""

    def __init__(self, name: str, source_type: str, enabled: bool = True) -> None:
        self.name = name
        self.source_type = source_type
        self.enabled = enabled
        self.logger = logging.getLogger(f"sources.{name}")

    @abstractmethod
    async def fetch_articles(self) -> list[RawArticle]:
        """
        Fetch new articles from this source.

        Returns a list of RawArticle objects. Each adapter must handle its own
        errors internally and return an empty list on failure rather than raising.
        """
        ...

    async def initialize(self) -> None:
        """Optional initialization hook (e.g., session setup)."""
        pass

    async def cleanup(self) -> None:
        """Optional cleanup hook (e.g., close sessions)."""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} enabled={self.enabled}>"
