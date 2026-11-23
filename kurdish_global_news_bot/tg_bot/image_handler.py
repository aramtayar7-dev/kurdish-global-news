"""
Kurdish Global News Bot — Image Handler

Downloads, validates, and prepares images for Telegram publication.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Optional
from urllib.parse import urlparse

import aiohttp

from config import MAX_IMAGE_SIZE_BYTES, REQUEST_TIMEOUT, USER_AGENT

logger = logging.getLogger("tg_bot.image")

# Allowed image content types
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

# Maximum dimension for Telegram photos
MAX_DIMENSION = 4096


class ImageHandler:
    """Downloads and validates images for Telegram posts."""

    def __init__(self) -> None:
        self._session: Optional[aiohttp.ClientSession] = None

    async def initialize(self) -> None:
        """Create aiohttp session for image downloads."""
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        self._session = aiohttp.ClientSession(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
        )

    async def cleanup(self) -> None:
        """Close aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def download_image(self, image_url: str) -> Optional[bytes]:
        """
        Download and validate an image.
        Returns image bytes or None if the image is unavailable or invalid.
        """
        if not image_url or not image_url.strip():
            return None

        # Validate URL
        if not self._is_safe_url(image_url):
            logger.debug("Unsafe image URL rejected: %s", image_url[:100])
            return None

        if not self._session:
            await self.initialize()

        try:
            assert self._session is not None
            async with self._session.get(image_url) as response:
                if response.status != 200:
                    logger.debug("Image download failed with status %d: %s",
                                 response.status, image_url[:100])
                    return None

                # Validate content type
                content_type = response.headers.get("Content-Type", "").lower().split(";")[0]
                if content_type not in ALLOWED_CONTENT_TYPES:
                    logger.debug("Invalid image content type '%s': %s",
                                 content_type, image_url[:100])
                    return None

                # Check content length
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > MAX_IMAGE_SIZE_BYTES:
                    logger.debug("Image too large (%s bytes): %s",
                                 content_length, image_url[:100])
                    return None

                # Download with size limit
                image_data = bytearray()
                async for chunk in response.content.iter_chunked(8192):
                    image_data.extend(chunk)
                    if len(image_data) > MAX_IMAGE_SIZE_BYTES:
                        logger.debug("Image exceeds size limit during download: %s",
                                     image_url[:100])
                        return None

                if len(image_data) < 1024:  # Too small to be a real image
                    logger.debug("Image too small (%d bytes): %s",
                                 len(image_data), image_url[:100])
                    return None

                # Validate it's actually an image
                if self._validate_image_bytes(bytes(image_data)):
                    logger.debug("Image downloaded: %d bytes from %s",
                                 len(image_data), image_url[:100])
                    return bytes(image_data)
                else:
                    logger.debug("Image validation failed: %s", image_url[:100])
                    return None

        except asyncio.TimeoutError:
            logger.debug("Image download timed out: %s", image_url[:100])
            return None
        except Exception as e:
            logger.debug("Image download error: %s — %s", str(e)[:100], image_url[:100])
            return None

    def _is_safe_url(self, url: str) -> bool:
        """Check if an image URL is safe to download."""
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return False
            if not parsed.netloc:
                return False
            hostname = parsed.netloc.lower().split(":")[0]
            if hostname in ("localhost", "127.0.0.1", "0.0.0.0"):
                return False
            return True
        except Exception:
            return False

    @staticmethod
    def _validate_image_bytes(data: bytes) -> bool:
        """Validate that the bytes represent a real image."""
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(data))
            img.verify()
            return True
        except Exception:
            return False
