"""
Tests for 24/7 Cloud Deployment on Render.com:
- Embedded health check web server (/ and /health)
- Cloud-safe configuration defaults (ENABLE_LOCAL_LLM=False, PORT=10000)
- Cold-start safety guard (MAX_ARTICLE_AGE_MINUTES=30 on empty DB)
- Deployment configuration files (render.yaml, requirements.txt, .gitignore)
"""

import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest
from aiohttp.test_utils import TestClient, TestServer

from config import (
    PORT,
    ENABLE_LOCAL_LLM,
    MAX_ARTICLE_AGE_MINUTES,
    MAX_ARTICLE_AGE_HOURS,
)
from main import create_health_app
from storage.database import Database
from storage.models import Article
from processing.validator import ArticleValidator
from core.pipeline import ArticlePipeline
from monitoring.health import HealthMonitor
from monitoring.metrics import Metrics


class TestEmbeddedHealthServer:
    """Test the aiohttp health check web server."""

    @pytest.mark.asyncio
    async def test_health_root_endpoint(self):
        app = create_health_app()
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            resp = await client.get("/")
            assert resp.status == 200
            data = await resp.json()
            assert data == {"status": "ok", "service": "kurdish-global-news"}
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_health_check_endpoint(self):
        app = create_health_app()
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            resp = await client.get("/health")
            assert resp.status == 200
            data = await resp.json()
            assert data == {"status": "ok", "service": "kurdish-global-news"}
        finally:
            await client.close()


class TestCloudSafeConfig:
    """Verify cloud-safe default settings in config.py."""

    def test_enable_local_llm_defaults_to_false(self):
        """Cloud deployments must not try to reach localhost:8080."""
        assert ENABLE_LOCAL_LLM is False

    def test_default_port_is_10000(self):
        """Render Web Services default port is 10000."""
        assert PORT == 10000

    def test_max_article_age_minutes_default(self):
        """Startup cold start safety guard defaults to 30 minutes."""
        assert MAX_ARTICLE_AGE_MINUTES == 30


class TestColdStartSafetyGuard:
    """Test ephemeral database cold start protection against flooding old articles."""

    @pytest.fixture
    async def temp_db(self, tmp_path):
        db_path = tmp_path / "cold_start_test.db"
        db = Database(str(db_path))
        await db.connect()
        yield db
        await db.close()

    @pytest.mark.asyncio
    async def test_database_is_empty_check(self, temp_db):
        """Test Database.is_empty helper."""
        assert await temp_db.is_empty() is True

        # Insert one article
        art = Article(
            canonical_url="https://example.com/item1",
            original_url="https://example.com/item1",
            original_title="Sample news headline for test",
            source_name="Reuters",
            category="International Politics",
            publication_date=datetime.now(timezone.utc).isoformat(),
        )
        await temp_db.insert_article(art)
        assert await temp_db.is_empty() is False

    def test_validator_cold_start_rejects_older_than_30_minutes(self):
        validator = ArticleValidator(cold_start=True)
        now = datetime.now(timezone.utc)

        # Article published 45 minutes ago
        old_article = Article(
            canonical_url="https://example.com/old",
            original_url="https://example.com/old",
            original_title="Important political development announced",
            original_summary="Detailed analysis of the political news event.",
            source_name="Reuters",
            category="International Politics",
            publication_date=(now - timedelta(minutes=45)).isoformat(),
        )

        is_valid, reason = validator.validate(old_article)
        assert is_valid is False
        assert "Cold start recency limit" in reason
        assert "30.0m limit" in reason

    def test_validator_cold_start_accepts_under_30_minutes(self):
        validator = ArticleValidator(cold_start=True)
        now = datetime.now(timezone.utc)

        # Article published 10 minutes ago
        fresh_article = Article(
            canonical_url="https://example.com/fresh",
            original_url="https://example.com/fresh",
            original_title="Breaking international news event occurring",
            original_summary="Detailed summary of the breaking international story.",
            source_name="Reuters",
            category="Wars & Armed Conflicts",
            publication_date=(now - timedelta(minutes=10)).isoformat(),
        )

        is_valid, reason = validator.validate(fresh_article)
        assert is_valid is True
        assert reason == ""

    def test_validator_normal_mode_accepts_older_than_30_minutes(self):
        """When cold_start=False, normal MAX_ARTICLE_AGE_HOURS (3h) applies."""
        validator = ArticleValidator(cold_start=False)
        now = datetime.now(timezone.utc)

        # 45 minutes old should pass when not in cold start
        article_45m = Article(
            canonical_url="https://example.com/normal-mode",
            original_url="https://example.com/normal-mode",
            original_title="Diplomatic summit conclusions in Geneva",
            original_summary="Comprehensive report on the international summit.",
            source_name="Reuters",
            category="Diplomacy & International Relations",
            publication_date=(now - timedelta(minutes=45)).isoformat(),
        )

        is_valid, reason = validator.validate(article_45m)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_pipeline_cold_start_filter(self, temp_db):
        """Verify pipeline enforces cold start guard when activated."""
        from unittest.mock import MagicMock
        pipeline = ArticlePipeline(
            db=temp_db,
            publisher=MagicMock(),
            health=HealthMonitor(),
            metrics=Metrics(),
        )
        pipeline.set_cold_start(True)
        now = datetime.now(timezone.utc)

        # 45 minute old article should be rejected by cold start guard
        old_art = Article(
            canonical_url="https://example.com/cold-filter-1",
            original_url="https://example.com/cold-filter-1",
            original_title="Old article from 45 minutes ago in feed",
            original_summary="Summary of older article that would flood channel on redeploy.",
            source_name="Reuters",
            category="International Politics",
            publication_date=(now - timedelta(minutes=45)).isoformat(),
        )
        old_art.id = await temp_db.insert_article(old_art)

        result = await pipeline.process_article(old_art)
        assert result is False

        db_art = await temp_db.get_article_by_canonical_url(old_art.canonical_url)
        assert db_art.status == "FAILED"
        assert "Cold start recency limit" in db_art.error_message


class TestDeploymentFiles:
    """Verify render.yaml, requirements.txt, and .gitignore exist and are properly configured."""

    def test_render_yaml_exists_and_configured(self):
        root = Path(__file__).parent.parent
        render_path = root / "render.yaml"
        assert render_path.exists(), "render.yaml must exist at project root"

        content = render_path.read_text(encoding="utf-8")
        assert "type: web" in content
        assert "pip install -r requirements.txt" in content
        assert "python main.py" in content
        assert "healthCheckPath: /health" in content
        assert "PORT" in content

    def test_requirements_txt_minimal(self):
        root = Path(__file__).parent.parent
        req_path = root / "requirements.txt"
        assert req_path.exists(), "requirements.txt must exist at project root"

        content = req_path.read_text(encoding="utf-8")
        assert "python-telegram-bot" in content
        assert "aiohttp" in content
        assert "aiosqlite" in content
        assert "feedparser" in content
        assert "duckduckgo-search" in content
        assert "beautifulsoup4" in content
        # deep-translator should NOT be in minimal cloud requirements
        assert "deep-translator" not in content

    def test_gitignore_protects_secrets_and_databases(self):
        root = Path(__file__).parent.parent
        ignore_path = root / ".gitignore"
        assert ignore_path.exists(), ".gitignore must exist"

        content = ignore_path.read_text(encoding="utf-8")
        lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]
        assert ".env" in lines
        assert "venv/" in lines
        assert "__pycache__/" in lines
        assert "*.db" in lines
        assert "*.log" in lines
