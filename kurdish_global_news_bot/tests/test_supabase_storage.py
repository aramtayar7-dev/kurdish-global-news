"""
Unit tests for the Supabase PostgREST persistent backend (`storage/supabase_db.py`),
automatic SQLite fallback (`storage/__init__.py`), SQL schema consistency,
and GitHub Actions secret/cache workflow configuration.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from config import ArticleStatus
from core.engine import NewsEngine
from storage import Database, SupabaseDatabase, create_database
from storage.models import Article


def _make_article(
    title: str = "UN Security Council convenes emergency session",
    url: str = "https://reuters.com/world/un-session-1?utm_source=x",
    category: str = "politics_world",
    importance_score: float = 0.80,
    hours_ago: float = 1.0,
    status: str = ArticleStatus.PUBLISHED,
) -> Article:
    pub_iso = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    art = Article(
        original_url=url,
        canonical_url=Article.canonicalize_url(url),
        original_title=title,
        normalized_title=Article.normalize_title(title),
        translated_title="کۆبوونەوەی بەپەلەی ئەنجومەنی ئاسایش",
        original_summary="Full international dispatch from UN headquarters.",
        translated_summary="ڕاپۆرتی کوردی دەربارەی کۆبوونەوەکە.",
        source_name="Reuters",
        category=category,
        importance_score=importance_score,
        status=status,
        published_at=pub_iso,
        discovered_at=pub_iso,
    )
    art.compute_content_hash()
    art.compute_event_fingerprint()
    return art


class TestBackendSelectionAndSchema:
    def test_create_database_falls_back_to_sqlite_when_unset(self, monkeypatch, tmp_path):
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_KEY", raising=False)
        db = create_database(db_path=str(tmp_path / "local.db"))
        assert isinstance(db, Database)

        # Partial credentials must still fall back to SQLite safely
        monkeypatch.setenv("SUPABASE_URL", "https://abc.supabase.co")
        monkeypatch.delenv("SUPABASE_KEY", raising=False)
        assert isinstance(create_database(db_path=str(tmp_path / "partial.db")), Database)

    def test_create_database_selects_supabase_when_configured(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_URL", "https://proj123.supabase.co/")
        monkeypatch.setenv("SUPABASE_KEY", "sb_secret_test_key")
        db = create_database()
        assert isinstance(db, SupabaseDatabase)
        assert db.rest_url == "https://proj123.supabase.co/rest/v1"
        engine = NewsEngine()
        assert isinstance(engine._db, SupabaseDatabase)

    def test_supabase_schema_sql_contains_all_tables_and_columns(self):
        schema_path = Path(__file__).resolve().parents[1] / "storage" / "supabase_schema.sql"
        assert schema_path.exists()
        sql = schema_path.read_text(encoding="utf-8")

        for table in ("articles", "skipped_urls", "daily_digests", "sources", "events", "system_logs"):
            assert f"CREATE TABLE IF NOT EXISTS {table}" in sql
        for col in ("canonical_url", "url_hash", "is_breaking", "digest_date", "skipped_at"):
            assert col in sql

    def test_workflows_include_supabase_secrets_and_nonblocking_cache(self):
        repo_root = Path(__file__).resolve().parents[2]
        for wf_name in ("news_bot.yml", "daily_digest.yml"):
            wf_path = repo_root / ".github" / "workflows" / wf_name
            content = wf_path.read_text(encoding="utf-8")
            assert "SUPABASE_URL: ${{ secrets.SUPABASE_URL }}" in content
            assert "SUPABASE_KEY: ${{ secrets.SUPABASE_KEY }}" in content
            assert "continue-on-error: true" in content
            assert "GROQ_API_KEY" not in content


class TestSupabasePostgRESTOperations:
    @pytest.mark.asyncio
    async def test_article_insert_duplicate_lookup_and_status_update(self):
        recorded: list[httpx.Request] = []
        insert_call = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal insert_call
            recorded.append(request)
            assert request.headers["apikey"] == "sb_test_key"
            assert request.headers["Authorization"] == "Bearer sb_test_key"

            if request.method == "POST" and request.url.path == "/rest/v1/articles":
                insert_call += 1
                assert "resolution=ignore-duplicates" in request.headers["Prefer"]
                body = json.loads(request.content.decode("utf-8"))
                assert body["url_hash"] == SupabaseDatabase.compute_url_hash(body["canonical_url"])
                if insert_call == 1:
                    return httpx.Response(201, json=[{"id": 42, **body}])
                # Second insert for same canonical_url -> ignored duplicate
                return httpx.Response(200, json=[])

            if request.method == "PATCH" and request.url.path == "/rest/v1/articles":
                assert request.url.params.get("id") == "eq.42"
                return httpx.Response(204)

            if request.method == "GET" and request.url.path == "/rest/v1/articles":
                return httpx.Response(200, json=[{
                    "id": 42,
                    "canonical_url": "https://reuters.com/world/un-session-1",
                    "original_url": "https://reuters.com/world/un-session-1?utm_source=x",
                    "original_title": "UN Security Council convenes emergency session",
                    "normalized_title": "un security council convenes emergency session",
                    "translated_title": "کۆبوونەوەی بەپەلە",
                    "original_summary": "Summary",
                    "translated_summary": "پوختە",
                    "source_name": "Reuters",
                    "category": "politics_world",
                    "priority": "CRITICAL",
                    "importance_score": 0.9,
                    "status": "PUBLISHED",
                    "is_breaking": True,
                }])
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            db = SupabaseDatabase(
                url="https://testproj.supabase.co",
                key="sb_test_key",
                client=client,
                base_backoff=0.0,
            )
            await db.connect()

            art = _make_article()
            new_id = await db.insert_article(art)
            assert new_id == 42
            assert art.id == 42

            dup_id = await db.insert_article(art)
            assert dup_id is None

            await db.update_article_status(42, ArticleStatus.PUBLISHED, telegram_message_id=9001)
            fetched = await db.get_article_by_canonical_url(art.canonical_url)
            assert fetched is not None
            assert fetched.id == 42
            assert fetched.is_breaking is True

    @pytest.mark.asyncio
    async def test_skipped_urls_upsert_and_lookup(self):
        stored_hashes: dict[str, dict] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST" and request.url.path == "/rest/v1/skipped_urls":
                assert "resolution=merge-duplicates" in request.headers["Prefer"]
                assert request.url.params.get("on_conflict") == "url_hash"
                payload = json.loads(request.content.decode("utf-8"))
                stored_hashes[payload["url_hash"]] = payload
                return httpx.Response(201)

            if request.method == "GET" and request.url.path == "/rest/v1/skipped_urls":
                url_hash_filter = request.url.params.get("url_hash", "")
                key = url_hash_filter.removeprefix("eq.")
                return httpx.Response(200, json=[stored_hashes[key]] if key in stored_hashes else [])

            if request.method == "GET" and request.url.path == "/rest/v1/articles":
                return httpx.Response(200, json=[])

            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            db = SupabaseDatabase(url="https://test.supabase.co", key="k", client=client, base_backoff=0.0)
            target_url = "https://example.com/local-fluff?utm_source=rss"
            assert await db.is_url_skipped(target_url) is False

            url_hash = await db.mark_url_skipped(target_url, title="Fluff", score=3.0, reason="Local only")
            assert len(url_hash) == 64
            assert await db.is_url_skipped("https://example.com/local-fluff") is True

    @pytest.mark.asyncio
    async def test_12h_dedup_fetch_and_24h_digest_prioritization(self):
        now = datetime.now(timezone.utc)
        rows_12h = [
            {
                "id": 10,
                "original_title": "Recent 2h Article",
                "translated_title": "بابەتی ٢ کاتژمێر پێش ئێستا",
                "category": "military_conflict",
                "published_at": (now - timedelta(hours=2)).isoformat(),
                "discovered_at": (now - timedelta(hours=2)).isoformat(),
            },
            {
                "id": 11,
                "original_title": "Outside Window 14h Article",
                "translated_title": "کۆن",
                "category": "politics_world",
                "published_at": (now - timedelta(hours=14)).isoformat(),
                "discovered_at": (now - timedelta(hours=14)).isoformat(),
            },
        ]
        rows_24h = [
            {
                "id": 1,
                "canonical_url": "https://example.com/global",
                "original_url": "https://example.com/global",
                "original_title": "Global News",
                "normalized_title": "global news",
                "category": "politics_world",
                "importance_score": 0.85,
                "status": "PUBLISHED",
                "published_at": (now - timedelta(hours=2)).isoformat(),
                "discovered_at": (now - timedelta(hours=2)).isoformat(),
            },
            {
                "id": 2,
                "canonical_url": "https://example.com/erbil",
                "original_url": "https://example.com/erbil",
                "original_title": "Erbil Summit",
                "normalized_title": "erbil summit",
                "category": "kurdistan",
                "importance_score": 0.78,
                "status": "PUBLISHED",
                "published_at": (now - timedelta(hours=3)).isoformat(),
                "discovered_at": (now - timedelta(hours=3)).isoformat(),
            },
            {
                "id": 3,
                "canonical_url": "https://example.com/stale",
                "original_url": "https://example.com/stale",
                "original_title": "Stale 30h",
                "normalized_title": "stale 30h",
                "category": "kurdistan",
                "importance_score": 0.99,
                "status": "PUBLISHED",
                "published_at": (now - timedelta(hours=30)).isoformat(),
                "discovered_at": (now - timedelta(hours=30)).isoformat(),
            },
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            select = request.url.params.get("select", "")
            if "original_title" in select:
                assert request.url.params.get("status") == "eq.PUBLISHED"
                assert (request.url.params.get("published_at") or "").startswith("gte.")
                return httpx.Response(200, json=rows_12h)
            return httpx.Response(200, json=rows_24h)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            db = SupabaseDatabase(url="https://test.supabase.co", key="k", client=client, base_backoff=0.0)
            recent = await db.get_recent_published_headlines(hours=12)
            assert [r["id"] for r in recent] == [10]

            digest_candidates = await db.get_published_articles_last_24h(limit=10)
            # Kurdistan story #2 gets +0.20 priority boost ahead of #1; stale #3 dropped
            assert [a.id for a in digest_candidates] == [2, 1]

    @pytest.mark.asyncio
    async def test_daily_digests_upsert_and_transient_retry(self):
        digests: dict[str, dict] = {}
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.ConnectTimeout("Transient network hiccup")
            if attempts == 2:
                return httpx.Response(503, text="Supabase temporarily unavailable")

            if request.method == "POST" and request.url.path == "/rest/v1/daily_digests":
                assert "resolution=merge-duplicates" in request.headers["Prefer"]
                payload = json.loads(request.content.decode("utf-8"))
                digests[payload["digest_date"]] = payload
                return httpx.Response(201)

            if request.method == "GET" and request.url.path == "/rest/v1/daily_digests":
                date_key = (request.url.params.get("digest_date") or "").removeprefix("eq.")
                return httpx.Response(200, json=[digests[date_key]] if date_key in digests else [])

            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            db = SupabaseDatabase(
                url="https://test.supabase.co",
                key="k",
                client=client,
                max_retries=3,
                base_backoff=0.0,
            )
            # Attempts 1 & 2 fail transiently, attempt 3 succeeds cleanly
            await db.mark_digest_published("2026-11-24", 5, "🌙 Digest text", telegram_message_id=888)
            assert attempts == 3
            assert await db.is_digest_published_for_date("2026-11-24") is True
            assert await db.is_digest_published_for_date("2026-11-25") is False

    @pytest.mark.asyncio
    async def test_engine_run_daily_digest_skips_sqlite_migrations_for_supabase(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co")
        monkeypatch.setenv("SUPABASE_KEY", "sb_secret")
        engine = NewsEngine()
        assert isinstance(engine._db, SupabaseDatabase)

        engine._db.connect = AsyncMock()
        engine._db.is_digest_published_for_date = AsyncMock(return_value=True)
        engine._db.close = AsyncMock()
        engine._publisher.cleanup = AsyncMock()

        res = await engine.run_daily_digest(force=False)
        assert res is False
        engine._db.connect.assert_awaited_once()
        engine._db.is_digest_published_for_date.assert_awaited_once()
        engine._db.close.assert_awaited_once()


class TestSupabaseAuthoritativeGlobalRPC:
    """Verify PostgREST RPC client behavior, fail-closed handling, and SQL artifact hygiene.

    Note: Uses httpx.MockTransport to test client RPC wiring and fail-closed logic;
    does NOT substitute for live Supabase PostgreSQL concurrency staging verification.
    """

    @pytest.mark.asyncio
    async def test_rpc_wrapper_endpoints_and_payloads(self):
        import json
        recorded_rpcs: list[tuple[str, dict]] = []
        unexpected_rest_calls: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if "/rest/v1/rpc/" in path:
                rpc_name = path.rsplit("/rest/v1/rpc/", 1)[-1]
                payload = json.loads(request.content.decode("utf-8") or "{}")
                recorded_rpcs.append((rpc_name, payload))
                responses = {
                    "get_global_quota_status": {
                        "ok": True,
                        "published_global_24h": 0,
                        "published_total_24h": 9,
                        "active_reservations": 0,
                        "ambiguous_reservations": 0,
                        "active_urls": [],
                    },
                    "reserve_global_slot": {
                        "admitted": True,
                        "state": "RESERVED",
                        "reservation_id": payload.get("p_reservation_id") or "res-rpc-1",
                        "owner_token": payload.get("p_owner_token"),
                        "expires_at": "2026-11-26T12:03:00+00:00",
                        "published_global_24h": 0,
                        "published_total_24h": 9,
                        "active_reservations": 0,
                        "reason": "admitted",
                    },
                    "renew_global_reservation": {
                        "ok": True,
                        "state": "RENEWED",
                        "reservation_id": payload.get("p_reservation_id"),
                        "expires_at": "2026-11-26T12:06:00+00:00",
                        "reason": "renewed",
                    },
                    "validate_global_reservation": {
                        "valid": True,
                        "state": "RENEWED",
                        "published_global_24h": 0,
                        "published_total_24h": 9,
                        "active_reservations_other": 0,
                        "reason": "valid",
                    },
                    "mark_global_publishing": {
                        "ok": True,
                        "admitted": True,
                        "state": "PUBLISHING",
                        "reservation_id": payload.get("p_reservation_id"),
                        "expires_at": "2026-11-26T12:09:00+00:00",
                        "reason": "publishing",
                    },
                    "finalize_global_publication": {
                        "ok": True,
                        "state": "COMMITTED",
                        "reservation_id": payload.get("p_reservation_id"),
                        "article_id": payload.get("p_article_id"),
                        "telegram_message_id": payload.get("p_telegram_message_id"),
                        "published_at": "2026-11-26T12:01:00+00:00",
                        "reason": "committed",
                    },
                    "release_global_reservation": {
                        "ok": True,
                        "state": "RELEASED",
                        "reservation_id": payload.get("p_reservation_id"),
                        "reason": "released",
                    },
                    "mark_global_ambiguous": {
                        "ok": True,
                        "state": "AMBIGUOUS",
                        "reservation_id": payload.get("p_reservation_id"),
                        "reason": "marked_ambiguous",
                    },
                    "reconcile_ambiguous_publication": {
                        "ok": True,
                        "reconciled": [{"reservation_id": "stale-1", "state": "AMBIGUOUS"}],
                        "ambiguous_count": 1,
                        "reason": "reconciled_auto",
                    },
                }
                return httpx.Response(200, json=responses[rpc_name])
            unexpected_rest_calls.append((request.method, path))
            return httpx.Response(500, json={"error": "unexpected_rest_fallback"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            db = SupabaseDatabase(
                url="https://proj.supabase.co",
                key="sb_service_key",
                client=client,
                base_backoff=0.0,
            )
            quota = await db.get_global_quota_status()
            assert quota["ok"] is True
            assert quota["published_total_24h"] == 9

            res = await db.reserve_global_slot(
                canonical_url="https://reuters.com/world/item-1",
                owner_token="worker:abc",
                article_id=101,
                batch_id="batch:1",
                lease_seconds=180,
            )
            assert res["admitted"] is True
            rid = res["reservation_id"]

            ren = await db.renew_global_reservation(rid, "worker:abc", lease_seconds=180)
            assert ren["state"] == "RENEWED"

            val = await db.validate_global_reservation(rid, "worker:abc")
            assert val["valid"] is True

            pub = await db.mark_global_publishing(rid, "worker:abc", lease_seconds=180)
            assert pub["admitted"] is True
            assert pub["state"] == "PUBLISHING"

            fin = await db.finalize_global_publication(
                reservation_id=rid,
                owner_token="worker:abc",
                article_id=101,
                telegram_message_id=5555,
            )
            assert fin["ok"] is True
            assert fin["state"] == "COMMITTED"

            rel = await db.release_global_reservation(rid, "worker:abc", reason="test")
            assert rel["state"] == "RELEASED"

            amb = await db.mark_global_ambiguous(rid, "worker:abc", reason="timeout", article_id=101)
            assert amb["state"] == "AMBIGUOUS"

            rec = await db.reconcile_ambiguous_publication(outcome="auto")
            assert rec["ok"] is True
            assert rec["ambiguous_count"] == 1

        assert len(recorded_rpcs) == 9
        assert [name for name, _ in recorded_rpcs] == [
            "get_global_quota_status",
            "reserve_global_slot",
            "renew_global_reservation",
            "validate_global_reservation",
            "mark_global_publishing",
            "finalize_global_publication",
            "release_global_reservation",
            "mark_global_ambiguous",
            "reconcile_ambiguous_publication",
        ]
        assert unexpected_rest_calls == []

    @pytest.mark.asyncio
    async def test_missing_rpc_fails_closed_for_global_while_regional_publishes(self):
        import config
        from core.pipeline import ArticlePipeline
        from monitoring.health import HealthMonitor
        from monitoring.metrics import Metrics
        from storage.models import Article

        rest_system_logs_calls: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if "/rest/v1/rpc/" in path:
                return httpx.Response(
                    404,
                    json={"code": "PGRST202", "message": "Could not find function in schema cache"},
                )
            if "/rest/v1/system_logs" in path:
                rest_system_logs_calls.append((request.method, path))
                return httpx.Response(500)
            if "/rest/v1/articles" in path and request.method in ("PATCH", "POST"):
                return httpx.Response(200, json=[])
            return httpx.Response(200, json=[])

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            db = SupabaseDatabase(
                url="https://proj.supabase.co",
                key="sb_service_key",
                client=client,
                max_retries=1,
                base_backoff=0.0,
            )
            # Fail-closed reservation helpers never touch /rest/v1/system_logs
            r_slot = await db.reserve_global_slot("https://bbc.com/news/global-1", "worker:failclosed")
            assert r_slot["admitted"] is False
            assert await db.get_active_global_reservations_count() == config.MAX_GLOBAL_POSTS_PER_DAY
            assert rest_system_logs_calls == []

            publisher = AsyncMock()
            publisher.publish_article = AsyncMock(return_value=777)
            pipeline = ArticlePipeline(db=db, publisher=publisher, health=HealthMonitor(), metrics=Metrics())

            # Global article MUST be blocked without calling Telegram
            global_art = Article(
                id=201,
                canonical_url="https://bbc.com/news/global-crisis",
                original_url="https://bbc.com/news/global-crisis",
                original_title="US Congress votes on emergency NATO defense treaty",
                translated_title="کۆنگرێسی ئەمریکا دەنگ لەسەر پەیماننامەی بەرگری ناتۆ دەدات",
                translated_summary="کۆنگرێسی ئەمریکا لە واشنتن کۆبووەوە بۆ دەنگدان لەسەر پرسی بەرگری.",
                source_name="BBC",
                category="جیهان",
                region="global",
                importance_score=9.2,
                priority="HIGH",
            )
            ok_global = await pipeline._publish_immediately(global_art)
            assert ok_global is False
            publisher.publish_article.assert_not_awaited()

            # Middle East / Kurdistan regional article MUST still publish cleanly
            me_art = Article(
                id=202,
                canonical_url="https://rudaw.net/sorani/kurdistan/erbil-security-1",
                original_url="https://rudaw.net/sorani/kurdistan/erbil-security-1",
                original_title="Peshmerga defense ministry announces Erbil border deployment",
                translated_title="وەزارەتی پێشمەرگە بڵاوکردنەوەی هێزی لە سنووری هەولێر ڕاگەیاند",
                translated_summary="وەزارەتی پێشمەرگە ئامادەکارییە ئەمنییەکانی سنووری هەولێری ڕاگەیاند.",
                source_name="Rudaw",
                category="کوردستان",
                region="kurdistan_bashur",
                importance_score=9.4,
                priority="HIGH",
            )
            ok_me = await pipeline._publish_immediately(me_art)
            assert ok_me is True
            publisher.publish_article.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_paginated_published_counts_deduplicates_canonical_urls(self):
        from datetime import datetime, timezone

        now_iso = datetime.now(timezone.utc).isoformat()
        rows = [
            {"id": 1, "region": "global", "canonical_url": "https://reuters.com/world/ukraine?utm_source=x", "published_at": now_iso},
            {"id": 2, "region": "global", "canonical_url": "https://reuters.com/world/ukraine/", "published_at": now_iso},
            {"id": 3, "region": "middle_east", "canonical_url": "", "published_at": now_iso},
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            if "/rest/v1/articles" in request.url.path:
                offset = int(request.url.params.get("offset", "0"))
                return httpx.Response(200, json=rows if offset == 0 else [])
            return httpx.Response(200, json=[])

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            db = SupabaseDatabase(url="https://proj.supabase.co", key="sb_key", client=client, base_backoff=0.0)
            g_cnt, t_cnt = await db.get_published_counts_last_24h()
            assert (g_cnt, t_cnt) == (1, 2)

    def test_sql_rpc_artifact_security_and_structure(self):
        from pathlib import Path

        sql_path = Path(__file__).resolve().parent.parent / "storage" / "sql" / "global_admission_rpc.sql"
        assert sql_path.exists()
        sql_text = sql_path.read_text(encoding="utf-8")

        for required_rpc in (
            "public.get_global_quota_status",
            "public.reserve_global_slot",
            "public.renew_global_reservation",
            "public.validate_global_reservation",
            "public.mark_global_publishing",
            "public.finalize_global_publication",
            "public.release_global_reservation",
            "public.mark_global_ambiguous",
            "public.reconcile_ambiguous_publication",
        ):
            assert required_rpc in sql_text
            assert f"GRANT EXECUTE ON FUNCTION {required_rpc}" in sql_text

        assert "pg_advisory_xact_lock(8472910348291042)" in sql_text
        assert "SECURITY DEFINER" in sql_text
        assert "SET search_path = public, pg_temp" in sql_text
        assert "FROM PUBLIC;" in sql_text
        assert "FROM anon;" in sql_text
        assert "FROM authenticated;" in sql_text
        assert "TO service_role;" in sql_text
        assert "10 * (v_global_24h + v_active_other + 1) > (v_total_24h + v_active_other + 1)" in sql_text
        for state in ("RESERVED", "RENEWED", "PUBLISHING", "COMMITTED", "RELEASED", "EXPIRED", "AMBIGUOUS"):
            assert state in sql_text
        assert "DROP TABLE" not in sql_text.upper()
        assert "TRUNCATE" not in sql_text.upper()
