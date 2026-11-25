"""
Kurdish Global News Bot — Local Multi-Frame Video & Image Watermark Detector

100% local, zero Gemini/AI calls (FFmpeg + Pillow):
  - Extracts 3 evenly spaced frames (20%, 50%, 80% of video duration) into a temp directory.
  - Inspects all 4 corners (outer 18% width x 15% height) and bottom ticker strip.
  - Verifies center-frame motion so stationary-camera footage does not false-positive.
  - Flags corners that remain temporally static across all 3 frames while carrying
    high-contrast edge/overlay badge density.
  - Detects static-image corner watermarks & third-party repost markers for unofficial sources.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import io
import json
import logging
from pathlib import Path
import re
import shutil
import tempfile
from typing import Optional

from PIL import Image

logger = logging.getLogger("processing.watermark")

CORNER_PRIORITY: tuple[str, ...] = ("top_right", "top_left", "bottom_right", "bottom_left")

_THIRD_PARTY_REPOST_RE = re.compile(
    r"(?:tiktok\.com|snaptik|capcut|made\s+with\s+(?:capcut|kinemaster|inshot)|"
    r"snapchat|via\s+@\w{3,}|watermark)",
    re.IGNORECASE,
)


@dataclass
class WatermarkReport:
    """Result of local multi-frame video corner watermark detection."""
    has_watermark: bool = False
    occupied_corners: list[str] = field(default_factory=list)
    has_third_party_repost_marker: bool = False
    reason: str = ""


def choose_kgn_corner(occupied_corners: Optional[list[str]] = None) -> str:
    """
    Pick the first unoccupied corner for the KGN logo using priority order:
    top_right -> top_left -> bottom_right -> bottom_left.
    Defaults to top_right when all corners are free (or all are occupied).
    """
    occupied = {str(c).strip().lower() for c in (occupied_corners or []) if c}
    for corner in CORNER_PRIORITY:
        if corner not in occupied:
            return corner
    return "top_right"


def has_third_party_repost_marker(text_hint: str = "") -> bool:
    """Return True when caption/metadata explicitly marks a TikTok/CapCut/repost clip."""
    return bool(text_hint and _THIRD_PARTY_REPOST_RE.search(text_hint))


def _corner_box(width: int, height: int, corner: str) -> tuple[int, int, int, int]:
    cw = max(12, int(width * 0.18))
    ch = max(12, int(height * 0.15))
    if corner == "top_left":
        return (0, 0, cw, ch)
    if corner == "top_right":
        return (max(0, width - cw), 0, width, ch)
    if corner == "bottom_left":
        return (0, max(0, height - ch), cw, height)
    # bottom_right
    return (max(0, width - cw), max(0, height - ch), width, height)


def _mean_abs_diff(gray_a: Image.Image, gray_b: Image.Image) -> float:
    pa = list(gray_a.getdata())
    pb = list(gray_b.getdata())
    if not pa or len(pa) != len(pb):
        return 0.0
    total = sum(abs(a - b) for a, b in zip(pa, pb))
    return total / len(pa)


def _edge_density(gray: Image.Image, diff_thresh: int = 25) -> float:
    w, h = gray.size
    if w < 3 or h < 3:
        return 0.0
    px = list(gray.getdata())
    edges = 0
    samples = 0
    for y in range(h - 1):
        row = y * w
        next_row = (y + 1) * w
        for x in range(w - 1):
            cur = px[row + x]
            if abs(cur - px[row + x + 1]) >= diff_thresh or abs(cur - px[next_row + x]) >= diff_thresh:
                edges += 1
            samples += 1
    return (edges / samples) if samples else 0.0


def inspect_frames_for_watermarks(
    frames: list[Image.Image],
    text_hint: str = "",
) -> WatermarkReport:
    """
    Analyze 2+ RGB/Grayscale video frames locally to detect static corner overlays/watermarks
    and bottom ticker overlays. Runs cleanly with Pillow only (zero AI calls).
    """
    repost_marker = has_third_party_repost_marker(text_hint)
    if len(frames) < 2:
        return WatermarkReport(
            has_watermark=repost_marker,
            occupied_corners=[],
            has_third_party_repost_marker=repost_marker,
            reason="third_party_repost_marker" if repost_marker else "insufficient_frames",
        )

    # Downscale uniformly for fast deterministic pixel analysis
    norm_w, norm_h = 160, 90
    grays = [f.convert("L").resize((norm_w, norm_h), Image.Resampling.BILINEAR) for f in frames]

    # Measure center-region motion between frames so stationary cams do not false-positive
    cx0, cy0, cx1, cy1 = int(norm_w * 0.25), int(norm_h * 0.25), int(norm_w * 0.75), int(norm_h * 0.75)
    center_crops = [g.crop((cx0, cy0, cx1, cy1)) for g in grays]
    center_motion = max(
        _mean_abs_diff(center_crops[i], center_crops[j])
        for i in range(len(center_crops))
        for j in range(i + 1, len(center_crops))
    )

    occupied: list[str] = []
    # Check 4 corners
    for corner in CORNER_PRIORITY:
        box = _corner_box(norm_w, norm_h, corner)
        crops = [g.crop(box) for g in grays]
        corner_diff = max(
            _mean_abs_diff(crops[i], crops[j])
            for i in range(len(crops))
            for j in range(i + 1, len(crops))
        )
        avg_edges = sum(_edge_density(c, diff_thresh=24) for c in crops) / len(crops)

        # Static logo criteria: edge density >= 2.2% AND either pixel-locked while scene moves
        # or extremely strong graphic logo edges (>= 8%) with minimal corner change (<= 4.5)
        if avg_edges >= 0.022 and corner_diff <= 5.0 and (center_motion >= 2.0 or avg_edges >= 0.08):
            occupied.append(corner)

    # Check bottom ticker bar region if bottom corners weren't already flagged cleanly
    tx0, ty0, tx1, ty1 = int(norm_w * 0.20), int(norm_h * 0.88), int(norm_w * 0.80), norm_h
    ticker_crops = [g.crop((tx0, ty0, tx1, ty1)) for g in grays]
    ticker_diff = max(
        _mean_abs_diff(ticker_crops[i], ticker_crops[j])
        for i in range(len(ticker_crops))
        for j in range(i + 1, len(ticker_crops))
    )
    ticker_edges = sum(_edge_density(c, diff_thresh=24) for c in ticker_crops) / len(ticker_crops)
    has_ticker = bool(ticker_edges >= 0.03 and ticker_diff <= 4.5 and center_motion >= 2.0)

    has_wm = bool(occupied or has_ticker or repost_marker)
    reasons: list[str] = []
    if occupied:
        reasons.append(f"static_corner_logo:{','.join(occupied)}")
    if has_ticker:
        reasons.append("static_bottom_ticker")
    if repost_marker:
        reasons.append("third_party_repost_marker")

    return WatermarkReport(
        has_watermark=has_wm,
        occupied_corners=occupied,
        has_third_party_repost_marker=repost_marker,
        reason="; ".join(reasons) if reasons else "clean",
    )


def has_image_corner_watermark(
    image_source: str | Path | bytes | Image.Image,
    text_hint: str = "",
) -> tuple[bool, str]:
    """
    Inspect a static photo/thumbnail (from an unofficial/ordinary account) for corner
    watermarks/badges or repost text markers using Pillow edge & contrast isolation.
    Returns `(has_watermark, reason)`.
    """
    if has_third_party_repost_marker(text_hint):
        return True, "third_party_repost_marker"

    try:
        if isinstance(image_source, Image.Image):
            img = image_source.copy()
        elif isinstance(image_source, (bytes, bytearray)):
            img = Image.open(io.BytesIO(bytes(image_source)))
        else:
            img = Image.open(image_source)

        with img:
            norm_w, norm_h = 160, 90
            gray = img.convert("L").resize((norm_w, norm_h), Image.Resampling.BILINEAR)
            center = gray.crop((int(norm_w * 0.25), int(norm_h * 0.25), int(norm_w * 0.75), int(norm_h * 0.75)))
            center_edges = _edge_density(center, diff_thresh=30)

            for corner in CORNER_PRIORITY:
                c_crop = gray.crop(_corner_box(norm_w, norm_h, corner))
                c_edges = _edge_density(c_crop, diff_thresh=32)
                px = list(c_crop.getdata())
                bright_ratio = sum(1 for v in px if v >= 225) / max(1, len(px))
                # Distinct high-contrast overlaid badge/logo stamp concentrated in a corner
                if c_edges >= 0.055 and (c_edges >= max(0.055, center_edges * 1.45) or bright_ratio >= 0.12):
                    return True, f"static_image_corner_watermark:{corner}"
    except Exception as exc:
        logger.debug("Static image watermark check skipped: %s", str(exc)[:120])

    return False, ""


async def _probe_duration_seconds(video_path: str, ffprobe_bin: Optional[str]) -> float:
    """Return video duration in seconds using ffprobe, defaulting cleanly to 10.0s."""
    if not ffprobe_bin:
        return 10.0
    try:
        proc = await asyncio.create_subprocess_exec(
            ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            video_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        if proc.returncode == 0 and stdout_b:
            data = json.loads(stdout_b.decode("utf-8", errors="replace"))
            dur = float(data.get("format", {}).get("duration", 0.0) or 0.0)
            if dur > 0.5:
                return dur
    except Exception:
        pass
    return 10.0


async def detect_video_watermarks(
    video_path: str | Path,
    duration_sec: Optional[float] = None,
    text_hint: str = "",
) -> WatermarkReport:
    """
    Extract 3 evenly spaced frames (20%, 50%, 80%) via FFmpeg into a temporary folder
    and analyze all corners + ticker strip for static watermarks/logos.
    Always cleans up temporary extracted frames in `finally:`.
    """
    repost_marker = has_third_party_repost_marker(text_hint)
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return WatermarkReport(
            has_watermark=repost_marker,
            occupied_corners=[],
            has_third_party_repost_marker=repost_marker,
            reason="third_party_repost_marker" if repost_marker else "ffmpeg_unavailable",
        )

    ffprobe_bin = shutil.which("ffprobe")
    resolved_dur = (
        float(duration_sec)
        if duration_sec and duration_sec > 0.5
        else await _probe_duration_seconds(str(video_path), ffprobe_bin)
    )
    timestamps = [max(0.1, resolved_dur * frac) for frac in (0.20, 0.50, 0.80)]

    with tempfile.TemporaryDirectory(prefix="kgn_wm_frames_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        extracted_images: list[Image.Image] = []
        try:
            for idx, ts in enumerate(timestamps):
                frame_out = tmp_root / f"frame_{idx}.png"
                proc = await asyncio.create_subprocess_exec(
                    ffmpeg_bin,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{ts:.2f}",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    str(frame_out),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(proc.communicate(), timeout=15.0)
                if proc.returncode == 0 and frame_out.exists() and frame_out.stat().st_size > 0:
                    with Image.open(frame_out) as im:
                        extracted_images.append(im.convert("RGB"))
            return inspect_frames_for_watermarks(extracted_images, text_hint=text_hint)
        except Exception as exc:
            logger.debug("Video watermark frame extraction skipped: %s", str(exc)[:140])
            return WatermarkReport(
                has_watermark=repost_marker,
                occupied_corners=[],
                has_third_party_repost_marker=repost_marker,
                reason="third_party_repost_marker" if repost_marker else "extraction_skipped",
            )
        finally:
            for im in extracted_images:
                try:
                    im.close()
                except Exception:
                    pass
