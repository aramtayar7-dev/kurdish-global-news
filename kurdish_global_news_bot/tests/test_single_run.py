"""
Unit tests for Single-Run Mode (--once) and GitHub Actions workflow configuration.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from main import parse_args, main
from core.engine import NewsEngine


def test_parse_args_default():
    """Verify default arguments run in continuous mode."""
    args = parse_args([])
    assert args.once is False


def test_parse_args_once_flag():
    """Verify --once argument sets once=True."""
    args = parse_args(["--once"])
    assert args.once is True


@pytest.mark.asyncio
async def test_engine_run_once(tmp_path):
    """Verify NewsEngine.run_once executes one cycle and shuts down cleanly."""
    engine = NewsEngine()
    db_file = str(tmp_path / "test_once.db")
    engine._db.db_path = db_file

    # Mock external network-dependent components
    engine._publisher.initialize = AsyncMock()
    engine._publisher.test_connection = AsyncMock(return_value=True)
    engine._publisher.cleanup = AsyncMock()

    # Track discovery cycle invocation
    discovery_invoked = False

    async def fake_discovery_cycle():
        nonlocal discovery_invoked
        discovery_invoked = True

    engine._discovery_cycle = fake_discovery_cycle
    engine._init_sources = AsyncMock()

    # Execute run_once()
    await engine.run_once()

    # Verify discovery was executed exactly once
    assert discovery_invoked is True

    # Verify scheduler was NEVER started
    assert engine._scheduler._running is False
    assert engine._scheduler._task is None

    # Verify background loops were NEVER started
    assert engine._backup_task is None
    assert engine._metrics_task is None

    # Verify database was closed cleanly
    assert engine._db._connection is None

    # Verify publisher was initialized and cleaned up
    engine._publisher.initialize.assert_awaited_once()
    engine._publisher.cleanup.assert_awaited_once()


def test_main_cli_once_mode():
    """Verify main(['--once']) invokes run_single_cycle and returns 0."""
    with patch("main.validate_config", return_value=[]), \
         patch("main.run_single_cycle", new_callable=AsyncMock) as mock_run_single:
        exit_code = main(["--once"])
        assert exit_code == 0
        mock_run_single.assert_awaited_once()


def test_main_cli_continuous_mode():
    """Verify main([]) invokes run_bot_with_web_server."""
    with patch("main.validate_config", return_value=[]), \
         patch("main.run_bot_with_web_server", new_callable=AsyncMock) as mock_web_server:
        exit_code = main([])
        assert exit_code == 0
        mock_web_server.assert_awaited_once()


def test_github_actions_workflow_configuration():
    """Verify the GitHub Actions workflow YAML file exists and has all required directives."""
    # Find repository root (.github/workflows/news_bot.yml)
    current_file = Path(__file__).resolve()
    repo_root = current_file.parents[2]  # from tests/ -> kurdish_global_news_bot/ -> D:\guevara hawal
    workflow_path = repo_root / ".github" / "workflows" / "news_bot.yml"

    assert workflow_path.exists(), f"Workflow file not found at {workflow_path}"

    content = workflow_path.read_text(encoding="utf-8")

    # Triggers
    assert "cron: '*/30 * * * *'" in content
    assert "workflow_dispatch:" in content

    # Python & Cache setup
    assert "actions/checkout@v4" in content
    assert "actions/setup-python@v5" in content
    assert "actions/cache@v4" in content
    assert "bot-db-${{ github.run_id }}" in content
    assert "restore-keys: |" in content
    assert "bot-db-" in content

    # Execution & Directory
    assert "libfribidi0 libharfbuzz0b libraqm0" in content
    assert "Prefer IPv4 for Telegram API" in content
    assert "net.ipv6.conf.all.disable_ipv6=1" in content
    assert "net.ipv6.conf.default.disable_ipv6=1" in content
    assert "working-directory: kurdish_global_news_bot" in content
    assert "python main.py --once" in content

    # Environment variables & secrets (Groq stripped, Gemini primary)
    assert "TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}" in content
    assert "TELEGRAM_CHANNEL_ID: ${{ secrets.TELEGRAM_CHANNEL_ID }}" in content
    assert "GROQ_API_KEY" not in content
    assert "GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}" in content
    assert "ENABLE_LOCAL_LLM: 'false'" in content
    assert "AUTO_PUBLISH: 'true'" in content
    assert "MAX_ARTICLE_AGE_MINUTES: '60'" in content
