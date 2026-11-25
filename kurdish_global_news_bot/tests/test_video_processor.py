"""
Unit tests for `processing/video_processor.py`:
- Exact KGN logo asset resolution (`assets/logo.png`) matching `card_generator.py`
- FFmpeg top-right overlay & libx264/aac/faststart command construction
- Streaming download size limits (Content-Length & mid-stream chunk cutoff)
- Async subprocess timeout/non-zero error handling & missing ffmpeg safeguard
- Temp file cleanup on both success and failure paths
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from processing.video_processor import (
    MAX_TELEGRAM_VIDEO_BYTES,
    build_ffmpeg_watermark_command,
    cleanup_video_file,
    download_and_brand_video,
    resolve_logo_path,
)


class _MockContentStream:
    """Async chunk iterator simulating streamed video download chunks."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_chunked(self, _chunk_size: int):
        for chunk in self._chunks:
            yield chunk


class _MockGetContext:
    def __init__(self, response=None, error: BaseException | None = None) -> None:
        self._response = response
        self._error = error

    async def __aenter__(self):
        if self._error is not None:
            raise self._error
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _MockClientSession:
    def __init__(self, response=None, error: BaseException | None = None) -> None:
        self._response = response
        self._error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    def get(self, *_args, **_kwargs):
        return _MockGetContext(response=self._response, error=self._error)


class TestVideoProcessor:
    """Test FFmpeg branding, safety limits, and cleanup."""

    def test_resolve_logo_path_uses_existing_assets_logo(self):
        logo = resolve_logo_path()
        assert logo is not None
        assert logo.exists()
        assert logo.name == "logo.png"

    def test_build_ffmpeg_watermark_command_positions_logo_top_right_and_telegram_safe(self):
        cmd = build_ffmpeg_watermark_command(
            input_path="/tmp/in.mp4",
            logo_path="/tmp/logo.png",
            output_path="/tmp/out.mp4",
            ffmpeg_bin="ffmpeg",
        )
        assert cmd[0] == "ffmpeg"
        assert "/tmp/in.mp4" in cmd
        assert "/tmp/logo.png" in cmd
        assert str(cmd[-1]) == "/tmp/out.mp4"

        fc_idx = cmd.index("-filter_complex")
        filter_str = cmd[fc_idx + 1]
        assert "scale2ref=w=trunc(main_w*0.11/2)*2:h=-2" in filter_str
        assert "overlay=x=W-w-W*0.035:y=W*0.035" in filter_str
        assert "libx264" in cmd
        assert "veryfast" in cmd
        assert "yuv420p" in cmd
        assert "+faststart" in cmd
        assert "0:a?" in cmd
        assert "aac" in cmd

    @pytest.mark.asyncio
    async def test_returns_none_when_ffmpeg_missing_or_invalid_url(self):
        assert await download_and_brand_video("ftp://invalid.com/v.mp4") is None

        with patch("processing.video_processor.shutil.which", return_value=None):
            res = await download_and_brand_video("https://cdn4.telegram-cdn.org/file/v1.mp4")
        assert res is None

        with patch("processing.video_processor.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("processing.video_processor.resolve_logo_path", return_value=None):
            res_no_logo = await download_and_brand_video("https://cdn4.telegram-cdn.org/file/v1.mp4")
        assert res_no_logo is None

    @pytest.mark.asyncio
    async def test_aborts_when_content_length_or_stream_exceeds_max_download_mb(self, tmp_path):
        created_temps: list[Path] = []

        def fake_create_temp(prefix: str, suffix: str) -> Path:
            p = tmp_path / f"{prefix}{len(created_temps)}{suffix}"
            p.touch()
            created_temps.append(p)
            return p

        # Case 1: Declared Content-Length exceeds 1 MB cap
        resp_hdr = MagicMock()
        resp_hdr.status = 200
        resp_hdr.headers = {"Content-Length": str(2 * 1024 * 1024)}
        resp_hdr.content = _MockContentStream([b"x" * 100])

        with patch("processing.video_processor.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("processing.video_processor.aiohttp.ClientSession", return_value=_MockClientSession(resp_hdr)), \
             patch("processing.video_processor._create_temp_path", side_effect=fake_create_temp):
            res1 = await download_and_brand_video(
                "https://cdn4.telegram-cdn.org/file/big_hdr.mp4",
                max_download_mb=1,
            )
        assert res1 is None
        assert created_temps == []

        # Case 2: No Content-Length, streamed bytes cross 1 MB mid-download -> aborts & cleans temp file
        resp_stream = MagicMock()
        resp_stream.status = 200
        resp_stream.headers = {}
        resp_stream.content = _MockContentStream([b"a" * (600 * 1024), b"b" * (600 * 1024)])

        with patch("processing.video_processor.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("processing.video_processor.aiohttp.ClientSession", return_value=_MockClientSession(resp_stream)), \
             patch("processing.video_processor._create_temp_path", side_effect=fake_create_temp):
            res2 = await download_and_brand_video(
                "https://cdn4.telegram-cdn.org/file/big_stream.mp4",
                max_download_mb=1,
            )
        assert res2 is None
        assert len(created_temps) == 1
        assert not created_temps[0].exists()

    @pytest.mark.asyncio
    async def test_download_and_brand_video_success_and_cleanup(self, tmp_path):
        created_paths: list[Path] = []

        def fake_create_temp(prefix: str, suffix: str) -> Path:
            p = tmp_path / f"{prefix}{len(created_paths)}{suffix}"
            p.touch()
            created_paths.append(p)
            return p

        resp_ok = MagicMock()
        resp_ok.status = 200
        resp_ok.headers = {"Content-Length": "1024"}
        resp_ok.content = _MockContentStream([b"\x00\x00\x00\x18ftypmp42" + b"0" * 512])

        recorded_cmd: list[str] = []

        async def fake_create_subproc(*cmd, **_kwargs):
            recorded_cmd.extend(cmd)
            out_file = Path(cmd[-1])
            out_file.write_bytes(b"BRANDED_MP4_DATA" * 100)
            proc = MagicMock()
            proc.returncode = 0
            proc.communicate = AsyncMock(return_value=(b"", b""))
            return proc

        with patch("processing.video_processor.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("processing.video_processor.aiohttp.ClientSession", return_value=_MockClientSession(resp_ok)), \
             patch("processing.video_processor._create_temp_path", side_effect=fake_create_temp), \
             patch("processing.video_processor.asyncio.create_subprocess_exec", side_effect=fake_create_subproc):
            branded = await download_and_brand_video("https://cdn4.telegram-cdn.org/file/clip.mp4")

        assert branded is not None
        branded_path = Path(branded)
        assert branded_path.exists()
        assert branded_path.read_bytes().startswith(b"BRANDED_MP4_DATA")

        # Raw downloaded file MUST be cleaned up already
        assert len(created_paths) == 2
        assert not created_paths[0].exists()

        # Branded command includes existing KGN logo PNG
        logo = resolve_logo_path()
        assert logo is not None and str(logo) in recorded_cmd

        cleanup_video_file(branded)
        assert not branded_path.exists()

    @pytest.mark.asyncio
    async def test_ffmpeg_failure_timeout_and_oversize_output_cleanup_all_temp_files(self, tmp_path):
        def make_temp_factory():
            paths: list[Path] = []

            def factory(prefix: str, suffix: str) -> Path:
                p = tmp_path / f"{prefix}{len(paths)}{suffix}"
                p.touch()
                paths.append(p)
                return p

            return paths, factory

        def ok_session():
            resp = MagicMock()
            resp.status = 200
            resp.headers = {"Content-Length": "256"}
            resp.content = _MockContentStream([b"video-bytes"])
            return _MockClientSession(resp)

        # Case A: FFmpeg non-zero exit
        paths_a, fact_a = make_temp_factory()

        async def subproc_fail(*_cmd, **_kwargs):
            proc = MagicMock()
            proc.returncode = 1
            proc.communicate = AsyncMock(return_value=(b"", b"codec error"))
            return proc

        with patch("processing.video_processor.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("processing.video_processor.aiohttp.ClientSession", return_value=ok_session()), \
             patch("processing.video_processor._create_temp_path", side_effect=fact_a), \
             patch("processing.video_processor.asyncio.create_subprocess_exec", side_effect=subproc_fail):
            assert await download_and_brand_video("https://cdn4.telegram-cdn.org/file/a.mp4") is None
        assert all(not p.exists() for p in paths_a)

        # Case B: FFmpeg timeout kills process and cleans temp files
        paths_b, fact_b = make_temp_factory()
        kill_mock = MagicMock()

        async def subproc_timeout(*_cmd, **_kwargs):
            proc = MagicMock()
            proc.returncode = None
            proc.kill = kill_mock
            proc.wait = AsyncMock()
            proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
            return proc

        with patch("processing.video_processor.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("processing.video_processor.aiohttp.ClientSession", return_value=ok_session()), \
             patch("processing.video_processor._create_temp_path", side_effect=fact_b), \
             patch("processing.video_processor.asyncio.create_subprocess_exec", side_effect=subproc_timeout):
            assert await download_and_brand_video("https://cdn4.telegram-cdn.org/file/b.mp4", timeout_sec=5) is None
        kill_mock.assert_called_once()
        assert all(not p.exists() for p in paths_b)

        # Case C: Output exceeds 48 MB Telegram cap -> rejected & cleaned up
        paths_c, fact_c = make_temp_factory()

        async def subproc_huge(*cmd, **_kwargs):
            Path(cmd[-1]).write_bytes(b"0" * 16)
            proc = MagicMock()
            proc.returncode = 0
            proc.communicate = AsyncMock(return_value=(b"", b""))
            return proc

        with patch("processing.video_processor.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("processing.video_processor.MAX_TELEGRAM_VIDEO_BYTES", 8), \
             patch("processing.video_processor.aiohttp.ClientSession", return_value=ok_session()), \
             patch("processing.video_processor._create_temp_path", side_effect=fact_c), \
             patch("processing.video_processor.asyncio.create_subprocess_exec", side_effect=subproc_huge):
            assert await download_and_brand_video("https://cdn4.telegram-cdn.org/file/c.mp4") is None
        assert all(not p.exists() for p in paths_c)
