"""
Kurdish Global News Bot — Async Video Downloader & KGN Logo Watermarker

Downloads direct MP4 video clips from public news sources with strict size/timeout
caps and overlays the transparent circular KGN logo (`assets/logo.png`) cleanly in
the top-right corner using non-blocking FFmpeg (`asyncio.create_subprocess_exec`).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import shutil
import tempfile
from typing import Optional
from urllib.parse import urlparse

import aiohttp

from config import ASSETS_DIR

logger = logging.getLogger("processing.video_processor")

DEFAULT_MAX_DOWNLOAD_MB = 30
DEFAULT_TIMEOUT_SEC = 90
MAX_TELEGRAM_VIDEO_BYTES = 48 * 1024 * 1024  # Strictly under Telegram Bot API 50 MB limit
_DOWNLOAD_CHUNK_BYTES = 64 * 1024

_DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "video/mp4,video/*;q=0.9,*/*;q=0.8",
}

_ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm"}


def resolve_logo_path(assets_dir: Optional[Path] = None) -> Optional[Path]:
    """Locate the existing transparent circular KGN logo PNG without modifying card_generator."""
    base = assets_dir or ASSETS_DIR
    for candidate in (base / "logo.png", base.parent / "logo.png"):
        if candidate.exists():
            return candidate
    return None


def cleanup_video_file(file_path: Optional[str | Path]) -> None:
    """Safely remove a temporary video file, suppressing OS cleanup errors."""
    if not file_path:
        return
    try:
        Path(file_path).unlink(missing_ok=True)
    except Exception as exc:
        logger.debug("Temp video cleanup skipped (%s): %s", file_path, exc)


def build_ffmpeg_watermark_command(
    input_path: str | Path,
    logo_path: str | Path,
    output_path: str | Path,
    ffmpeg_bin: str = "ffmpeg",
) -> list[str]:
    """
    Build the FFmpeg argv list that:
      - Scales the circular KGN logo proportionally to ~11% of video width.
      - Overlays the logo in the top-right corner with a balanced ~3.5% margin.
      - Ensures even output dimensions for libx264/yuv420p compatibility.
      - Preserves audio if present (`-map 0:a? -c:a aac -b:a 128k`) without failing on muted videos.
    """
    filter_complex = (
        "[1:v][0:v]scale2ref=w=trunc(main_w*0.11/2)*2:h=-2[logo][base];"
        "[base][logo]overlay=x=W-w-W*0.035:y=W*0.035:format=auto,"
        "scale=trunc(iw/2)*2:trunc(ih/2)*2[vout]"
    )
    return [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-i",
        str(logo_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "24",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(output_path),
    ]


def _guess_video_suffix(video_url: str) -> str:
    """Determine safe temp file extension from video URL."""
    path = urlparse(video_url).path
    ext = Path(path).suffix.lower() if path else ""
    return ext if ext in _ALLOWED_VIDEO_EXTENSIONS else ".mp4"


def _create_temp_path(prefix: str, suffix: str) -> Path:
    """Create a closed named temp file and return its Path."""
    with tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, delete=False) as tmp:
        return Path(tmp.name)


async def _download_video_to_temp(
    video_url: str,
    max_bytes: int,
    timeout_sec: float,
) -> Optional[Path]:
    """Stream-download video to a temp file while strictly enforcing max_bytes."""
    raw_path: Optional[Path] = None
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_sec)
        async with aiohttp.ClientSession(timeout=timeout, headers=_DOWNLOAD_HEADERS) as session:
            async with session.get(video_url, allow_redirects=True) as resp:
                if resp.status != 200:
                    logger.warning("Video download returned HTTP %d: %s", resp.status, video_url[:90])
                    return None

                content_length = resp.headers.get("Content-Length")
                if content_length and content_length.strip().isdigit():
                    declared_size = int(content_length.strip())
                    if declared_size > max_bytes:
                        logger.warning(
                            "Video Content-Length (%d bytes) exceeds %d byte cap; aborting download.",
                            declared_size,
                            max_bytes,
                        )
                        return None

                raw_path = await asyncio.to_thread(
                    _create_temp_path, "kgn_raw_video_", _guess_video_suffix(video_url)
                )

                downloaded = 0
                with open(raw_path, "wb") as out_file:
                    async for chunk in resp.content.iter_chunked(_DOWNLOAD_CHUNK_BYTES):
                        if not chunk:
                            continue
                        downloaded += len(chunk)
                        if downloaded > max_bytes:
                            logger.warning(
                                "Streamed video exceeded %d byte limit (%d bytes); aborting.",
                                max_bytes,
                                downloaded,
                            )
                            out_file.close()
                            cleanup_video_file(raw_path)
                            return None
                        out_file.write(chunk)

                if downloaded == 0:
                    logger.warning("Downloaded video is empty (0 bytes): %s", video_url[:90])
                    cleanup_video_file(raw_path)
                    return None

                return raw_path

    except asyncio.TimeoutError:
        logger.warning("Video download timed out after %ss: %s", timeout_sec, video_url[:90])
        cleanup_video_file(raw_path)
        return None
    except Exception as exc:
        logger.warning("Video download failed (%s): %s", video_url[:90], str(exc)[:180])
        cleanup_video_file(raw_path)
        return None


async def download_and_brand_video(
    video_url: str,
    max_download_mb: int = DEFAULT_MAX_DOWNLOAD_MB,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
) -> Optional[str]:
    """
    Download `video_url` asynchronously, watermark it with the KGN logo in the top-right corner
    via FFmpeg, verify the branded output is <= 48 MB, and return the temp file path (or None).
    Caller should clean up the returned temp file path after publishing via `cleanup_video_file`.
    """
    if not video_url or not video_url.strip().startswith(("http://", "https://")):
        logger.warning("Invalid video URL skipped: %s", str(video_url)[:80])
        return None

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        logger.warning("ffmpeg not found on PATH; skipping video branding and falling back to card.")
        return None

    logo_path = resolve_logo_path()
    if logo_path is None:
        logger.warning("KGN logo asset (assets/logo.png) not found; skipping video branding.")
        return None

    max_bytes = max(1, int(max_download_mb)) * 1024 * 1024
    raw_path: Optional[Path] = await _download_video_to_temp(
        video_url.strip(), max_bytes=max_bytes, timeout_sec=float(timeout_sec)
    )
    if raw_path is None:
        return None

    output_path: Optional[Path] = None
    proc: Optional[asyncio.subprocess.Process] = None
    try:
        output_path = await asyncio.to_thread(_create_temp_path, "kgn_branded_", ".mp4")
        cmd = build_ffmpeg_watermark_command(
            input_path=raw_path,
            logo_path=logo_path,
            output_path=output_path,
            ffmpeg_bin=ffmpeg_bin,
        )
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=float(timeout_sec))

        if proc.returncode != 0:
            err_msg = (stderr_bytes or b"").decode("utf-8", errors="replace").strip()
            logger.warning(
                "FFmpeg branding failed (exit=%s): %s",
                proc.returncode,
                err_msg[:220] or "unknown error",
            )
            cleanup_video_file(output_path)
            return None

        if not output_path.exists():
            logger.warning("FFmpeg produced no output file.")
            return None

        out_size = output_path.stat().st_size
        if out_size <= 0 or out_size > MAX_TELEGRAM_VIDEO_BYTES:
            logger.warning(
                "Branded video size (%d bytes) invalid or exceeds 48 MB Telegram limit; falling back.",
                out_size,
            )
            cleanup_video_file(output_path)
            return None

        return str(output_path)

    except asyncio.TimeoutError:
        logger.warning("FFmpeg video branding timed out after %ss.", timeout_sec)
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
        cleanup_video_file(output_path)
        return None
    except Exception as exc:
        logger.warning("Video branding error: %s", str(exc)[:200])
        cleanup_video_file(output_path)
        return None
    finally:
        cleanup_video_file(raw_path)
