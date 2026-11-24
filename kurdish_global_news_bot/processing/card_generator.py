"""
Kurdish Global News Bot — Branded News Graphic Card Generator

Generates clean 1280x720 broadcast-style news cards using Pillow (PIL):
  - Background image (article image / og:image center-cropped & LANCZOS resized)
    or a sleek dark slate/crimson geometric gradient fallback
  - Smooth dark bottom-55% gradient overlay for crisp legibility
  - Top-right channel logo (assets/logo.png if present, else red/white KGN badge)
  - Colored Kurdish category pill badge above the headline
  - Connected Kurdish Sorani RTL wrapped headline (max 3 lines) rendered via Pillow RAQM
  - Bottom-left subtle @KurdishGlobalNews watermark
"""

from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path
import tempfile
from typing import Optional

from PIL import Image, ImageDraw, ImageFont, ImageOps, features

from config import ASSETS_DIR, CARD_HEIGHT, CARD_WIDTH, CHANNEL_HANDLE
from processing.content_formatter import resolve_category_display
from core.importance_ranker import normalize_category
from storage.models import Article

logger = logging.getLogger("processing.card_generator")

_HAS_RAQM: bool = features.check("raqm")
if not _HAS_RAQM:
    logger.warning(
        "Pillow libraqm is not available in this environment; "
        "using standard font layout fallback (install libfribidi0 libharfbuzz0b libraqm0 for RTL shaping)."
    )

CATEGORY_COLORS: dict[str, tuple[int, int, int]] = {
    "kurdistan": (217, 119, 6),       # Warm gold / amber
    "military_conflict": (220, 38, 38),  # Crimson red
    "politics_world": (37, 99, 235),     # Broadcast blue
    "economy_markets": (5, 150, 105),    # Emerald green
    "ai_tech": (124, 58, 237),           # Deep violet
    "cybersecurity": (234, 88, 12),      # Alert orange
    "science_space": (8, 145, 178),      # Cosmic teal
}

MARGIN_X = 56
CONTENT_MAX_WIDTH = CARD_WIDTH - (MARGIN_X * 2)  # 1168px


class NewsCardGenerator:
    """Async Pillow graphic card generator for Kurdish Global News posts."""

    def __init__(
        self,
        assets_dir: Optional[Path] = None,
        channel_handle: str = CHANNEL_HANDLE,
    ) -> None:
        self._assets_dir = assets_dir or ASSETS_DIR
        self._channel_handle = channel_handle or "@KurdishGlobalNews"
        self._font_path = self._resolve_font_path()
        self._logo_path = self._resolve_logo_path()

    def _resolve_font_path(self) -> Optional[Path]:
        """Locate bundled Kurdish-compatible bold TTF font."""
        fonts_dir = self._assets_dir / "fonts"
        candidates = [
            fonts_dir / "Vazirmatn-Bold.ttf",
            fonts_dir / "NotoSansArabic-Bold.ttf",
            fonts_dir / "Vazirmatn.ttf",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _resolve_logo_path(self) -> Optional[Path]:
        """Locate optional PNG channel logo in assets/logo.png or project root logo.png."""
        for candidate in (self._assets_dir / "logo.png", self._assets_dir.parent / "logo.png"):
            if candidate.exists():
                return candidate
        return None

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """Load TrueType font with Pillow RAQM RTL layout engine when available."""
        if self._font_path is not None:
            try:
                if _HAS_RAQM:
                    return ImageFont.truetype(
                        str(self._font_path),
                        size,
                        layout_engine=ImageFont.Layout.RAQM,
                    )
                return ImageFont.truetype(str(self._font_path), size)
            except Exception as e:
                logger.warning("Failed to load bundled font (%s): %s", self._font_path, e)

        for fallback in ("arialbd.ttf", "tahoma.ttf", "DejaVuSans-Bold.ttf"):
            try:
                return ImageFont.truetype(fallback, size)
            except Exception:
                continue

        return ImageFont.load_default()

    @staticmethod
    def _rtl_kwargs() -> dict[str, str]:
        """Only pass direction='rtl' when libraqm is active so Windows tests never raise KeyError."""
        return {"direction": "rtl"} if _HAS_RAQM else {}

    def _measure_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    ) -> tuple[int, int]:
        """Return (width, height) bounding box for text."""
        rtl_kw = self._rtl_kwargs()
        bbox = draw.textbbox((0, 0), text, font=font, **rtl_kw)
        return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])

    def _create_fallback_background(self) -> Image.Image:
        """Generate sleek dark slate & crimson geometric broadcast background (1280x720)."""
        img = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), (12, 18, 32, 255))
        draw = ImageDraw.Draw(img)

        # Vertical slate gradient
        for y in range(CARD_HEIGHT):
            t = y / CARD_HEIGHT
            r = int(12 + (24 - 12) * t)
            g = int(18 + (28 - 18) * t)
            b = int(32 + (46 - 32) * t)
            draw.line([(0, y), (CARD_WIDTH, y)], fill=(r, g, b, 255))

        # Geometric crimson broadcast accents
        overlay = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        odraw.polygon(
            [(0, 0), (420, 0), (160, CARD_HEIGHT), (0, CARD_HEIGHT)],
            fill=(153, 27, 27, 45),
        )
        odraw.polygon(
            [(880, 0), (CARD_WIDTH, 0), (CARD_WIDTH, 380)],
            fill=(220, 38, 38, 35),
        )
        odraw.line([(0, 0), (CARD_WIDTH, 0)], fill=(220, 38, 38, 220), width=6)

        # Subtle horizontal broadcast grid lines
        for gx in range(0, CARD_WIDTH, 80):
            odraw.line([(gx, 0), (gx, CARD_HEIGHT)], fill=(255, 255, 255, 8), width=1)
        for gy in range(0, CARD_HEIGHT, 80):
            odraw.line([(0, gy), (CARD_WIDTH, gy)], fill=(255, 255, 255, 8), width=1)

        return Image.alpha_composite(img, overlay)

    def _prepare_background(self, image_bytes: Optional[bytes]) -> Image.Image:
        """Prepare 1280x720 center-cropped image or geometric fallback."""
        if image_bytes:
            try:
                src = Image.open(io.BytesIO(image_bytes))
                src = ImageOps.exif_transpose(src)
                src = src.convert("RGBA")
                fitted = ImageOps.fit(
                    src,
                    (CARD_WIDTH, CARD_HEIGHT),
                    method=Image.Resampling.LANCZOS,
                    centering=(0.5, 0.5),
                )
                return fitted
            except Exception as e:
                logger.warning("Background image decode failed (%s); using geometric fallback.", e)

        return self._create_fallback_background()

    @staticmethod
    def _apply_bottom_gradient(base: Image.Image) -> Image.Image:
        """Apply smooth dark gradient over bottom 55% so white headlines stand out cleanly."""
        gradient = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(gradient)

        start_y = int(CARD_HEIGHT * 0.45)  # Bottom 55% starts at y=324
        span = CARD_HEIGHT - start_y
        for y in range(start_y, CARD_HEIGHT):
            progress = (y - start_y) / max(span, 1)
            alpha = int(242 * (progress ** 1.25))
            draw.line([(0, y), (CARD_WIDTH, y)], fill=(8, 12, 22, alpha))

        # Top subtle vignette so top-right KGN logo/badge stays readable
        for y in range(0, 120):
            alpha = int(110 * (1.0 - (y / 120.0)))
            draw.line([(0, y), (CARD_WIDTH, y)], fill=(8, 12, 22, alpha))

        # Crisp crimson top border accent
        draw.rectangle([(0, 0), (CARD_WIDTH, 5)], fill=(220, 38, 38, 255))
        return Image.alpha_composite(base, gradient)

    def _draw_logo_or_badge(self, canvas: Image.Image) -> None:
        """Render assets/logo.png or a red/white KGN broadcast badge in the top-right corner."""
        if self._logo_path is not None:
            try:
                logo = Image.open(self._logo_path).convert("RGBA")
                max_h = 76
                if logo.height > max_h:
                    ratio = max_h / logo.height
                    logo = logo.resize(
                        (max(1, int(logo.width * ratio)), max_h),
                        Image.Resampling.LANCZOS,
                    )
                pos_x = CARD_WIDTH - MARGIN_X - logo.width
                pos_y = 36
                canvas.alpha_composite(logo, dest=(pos_x, pos_y))
                return
            except Exception as e:
                logger.warning("Failed to draw logo.png (%s); drawing KGN badge.", e)

        # Draw crisp red/white KGN broadcast badge
        draw = ImageDraw.Draw(canvas)
        badge_font = self._load_font(28)
        sub_font = self._load_font(18)
        kgn_text = "KGN"
        kgn_w, kgn_h = self._measure_text(draw, kgn_text, badge_font)
        news_text = "NEWS"
        news_w, news_h = self._measure_text(draw, news_text, sub_font)

        pad_x = 18
        kgn_box_w = kgn_w + (pad_x * 2)
        news_box_w = news_w + (pad_x * 2)
        box_h = 46
        right_edge = CARD_WIDTH - MARGIN_X
        top_y = 36

        # White NEWS box on left side of badge
        news_x0 = right_edge - kgn_box_w - news_box_w
        draw.rounded_rectangle(
            [(news_x0, top_y), (news_x0 + news_box_w + 8, top_y + box_h)],
            radius=8,
            fill=(248, 250, 252, 245),
        )
        draw.text(
            (news_x0 + pad_x, top_y + (box_h - news_h) // 2 - 2),
            news_text,
            font=sub_font,
            fill=(15, 23, 42, 255),
        )

        # Crimson KGN box at far top-right
        kgn_x0 = right_edge - kgn_box_w
        draw.rounded_rectangle(
            [(kgn_x0, top_y), (right_edge, top_y + box_h)],
            radius=8,
            fill=(220, 38, 38, 255),
        )
        draw.text(
            (kgn_x0 + pad_x, top_y + (box_h - kgn_h) // 2 - 4),
            kgn_text,
            font=badge_font,
            fill=(255, 255, 255, 255),
        )

    def _wrap_headline(
        self,
        draw: ImageDraw.ImageDraw,
        headline: str,
        max_width: int,
    ) -> tuple[list[str], ImageFont.FreeTypeFont | ImageFont.ImageFont, int]:
        """
        Wrap Sorani headline cleanly into at most 3 lines, stepping down font size
        (50 -> 44 -> 38) if needed to prevent truncation.
        """
        words = headline.split()
        if not words:
            font = self._load_font(46)
            return [""], font, 62

        for font_size in (50, 44, 38):
            font = self._load_font(font_size)
            lines: list[str] = []
            current = ""

            for word in words:
                candidate = f"{current} {word}".strip() if current else word
                w, _ = self._measure_text(draw, candidate, font)
                if w <= max_width:
                    current = candidate
                else:
                    if current:
                        lines.append(current)
                    current = word

            if current:
                lines.append(current)

            if len(lines) <= 3:
                return lines, font, int(font_size * 1.42)

        # Still >3 lines even at 38px: keep first 3 balanced lines
        font = self._load_font(38)
        lines = lines[:3]
        return lines, font, int(38 * 1.42)

    def render_card_sync(
        self,
        article: Article,
        image_bytes: Optional[bytes] = None,
    ) -> bytes:
        """Synchronously render 1280x720 branded JPEG news card and return bytes."""
        bg = self._prepare_background(image_bytes)
        canvas = self._apply_bottom_gradient(bg)
        self._draw_logo_or_badge(canvas)

        draw = ImageDraw.Draw(canvas)
        rtl_kw = self._rtl_kwargs()

        # Resolve category label & pill badge color
        canonical_cat = normalize_category(
            article.category,
            text_hint=f"{article.original_title} {article.region}",
        )
        kurdish_cat, _ = resolve_category_display(canonical_cat)
        pill_color = CATEGORY_COLORS.get(canonical_cat, (37, 99, 235))

        headline_text = (article.translated_title or article.original_title or "").strip()
        lines, headline_font, line_height = self._wrap_headline(
            draw, headline_text, max_width=CONTENT_MAX_WIDTH
        )

        # Compute bottom-up layout coordinates so badge sits cleanly above headline
        footer_y = CARD_HEIGHT - 46  # Bottom edge for watermark/source
        bottom_padding = 26
        total_headline_h = len(lines) * line_height
        headline_top_y = footer_y - bottom_padding - total_headline_h

        # 1. Category pill badge above headline (right-aligned)
        pill_font = self._load_font(25)
        cat_w, cat_h = self._measure_text(draw, kurdish_cat, pill_font)
        pill_pad_x = 20
        pill_h = 44
        pill_w = cat_w + (pill_pad_x * 2)
        right_edge = CARD_WIDTH - MARGIN_X
        pill_y = max(120, headline_top_y - pill_h - 18)
        pill_x0 = right_edge - pill_w

        draw.rounded_rectangle(
            [(pill_x0, pill_y), (right_edge, pill_y + pill_h)],
            radius=12,
            fill=(*pill_color, 245),
        )
        draw.text(
            (right_edge - pill_pad_x - cat_w, pill_y + (pill_h - cat_h) // 2 - 4),
            kurdish_cat,
            font=pill_font,
            fill=(255, 255, 255, 255),
            **rtl_kw,
        )

        # 2. Bold wrapped Sorani headline (right-aligned, max 3 lines)
        cur_y = headline_top_y
        for line in lines:
            if not line:
                cur_y += line_height
                continue
            line_w, _ = self._measure_text(draw, line, headline_font)
            line_x = max(MARGIN_X, right_edge - line_w)
            # Subtle drop shadow
            draw.text(
                (line_x + 2, cur_y + 2),
                line,
                font=headline_font,
                fill=(0, 0, 0, 185),
                **rtl_kw,
            )
            draw.text(
                (line_x, cur_y),
                line,
                font=headline_font,
                fill=(255, 255, 255, 255),
                **rtl_kw,
            )
            cur_y += line_height

        # 3. Bottom accent bar & @KurdishGlobalNews subtle branding (bottom-left)
        draw.line(
            [(MARGIN_X, footer_y - 12), (right_edge, footer_y - 12)],
            fill=(255, 255, 255, 45),
            width=1,
        )
        wm_font = self._load_font(22)
        draw.text(
            (MARGIN_X, footer_y),
            self._channel_handle,
            font=wm_font,
            fill=(226, 232, 240, 215),
        )

        # Source badge subtly on bottom-right
        if article.source_name:
            src_text = article.source_name[:40]
            src_w, _ = self._measure_text(draw, src_text, wm_font)
            draw.text(
                (right_edge - src_w, footer_y),
                src_text,
                font=wm_font,
                fill=(203, 213, 225, 200),
            )

        rgb = canvas.convert("RGB")
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=92, optimize=True)
        return buf.getvalue()

    async def generate_card_bytes(
        self,
        article: Article,
        image_bytes: Optional[bytes] = None,
    ) -> bytes:
        """Asynchronously generate a 1280x720 news graphic card as JPEG bytes."""
        return await asyncio.to_thread(self.render_card_sync, article, image_bytes)

    async def generate_card_file(
        self,
        article: Article,
        image_bytes: Optional[bytes] = None,
    ) -> Path:
        """Asynchronously generate a temporary JPEG card file and return its Path."""
        card_bytes = await self.generate_card_bytes(article, image_bytes=image_bytes)

        def _write_temp() -> Path:
            with tempfile.NamedTemporaryFile(
                prefix="kgn_card_", suffix=".jpg", delete=False
            ) as tmp:
                tmp.write(card_bytes)
                return Path(tmp.name)

        return await asyncio.to_thread(_write_temp)
