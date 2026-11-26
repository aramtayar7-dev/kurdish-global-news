-- ============================================================================
-- Kurdish Global News Bot — Staging Schema & RPC Verification Queries (03)
-- Target Supabase Project: kurdish-global-news-staging
-- Status: READ-ONLY VERIFICATION QUERIES (DO NOT RUN AGAINST PRODUCTION)
--
-- Run after `staging_01_schema.sql` and `staging_02_global_admission_rpc.sql`
-- inside the Supabase Staging SQL Editor.
-- ============================================================================

-- 1. Verify all 6 required tables exist in `public`, RLS is enabled,
--    and `anon` / `authenticated` have zero privileges while `service_role` has RW.
SELECT
    c.relname AS table_name,
    c.relrowsecurity AS rls_enabled,
    has_table_privilege('anon', c.oid, 'SELECT') AS anon_select,
    has_table_privilege('authenticated', c.oid, 'SELECT') AS auth_select,
    has_table_privilege('service_role', c.oid, 'SELECT,INSERT,UPDATE,DELETE') AS service_rw
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public'
  AND c.relkind = 'r'
  AND c.relname IN (
      'articles',
      'skipped_urls',
      'daily_digests',
      'sources',
      'events',
      'system_logs'
  )
ORDER BY c.relname;
-- Expected: 6 rows; rls_enabled=true, anon_select=false, auth_select=false, service_rw=true.


-- 2. Verify every required column on `public.articles` (all 32 columns)
WITH expected_articles_cols(column_name) AS (
    VALUES
        ('id'), ('canonical_url'), ('url_hash'), ('original_url'),
        ('original_title'), ('normalized_title'), ('translated_title'),
        ('original_summary'), ('translated_summary'), ('source_name'),
        ('category'), ('priority'), ('importance_score'), ('publication_date'),
        ('discovered_at'), ('processed_at'), ('published_at'), ('status'),
        ('content_hash'), ('event_fingerprint'), ('retry_count'), ('error_message'),
        ('telegram_message_id'), ('thread_root_message_id'), ('image_url'),
        ('video_url'), ('country'), ('region'), ('named_entities'),
        ('event_key'), ('event_entities'), ('is_breaking')
)
SELECT
    e.column_name AS expected_column,
    c.data_type,
    c.is_nullable,
    (c.column_name IS NOT NULL) AS exists_in_table
FROM expected_articles_cols e
LEFT JOIN information_schema.columns c
  ON c.table_schema = 'public'
 AND c.table_name = 'articles'
 AND c.column_name = e.column_name
ORDER BY e.column_name;
-- Expected: 32 rows, every row has exists_in_table = true.


-- 3. Verify all required columns on the remaining 5 tables
WITH expected_other_cols(table_name, column_name) AS (
    VALUES
        ('skipped_urls', 'url_hash'), ('skipped_urls', 'canonical_url'),
        ('skipped_urls', 'title'), ('skipped_urls', 'score'),
        ('skipped_urls', 'reason'), ('skipped_urls', 'skipped_at'),
        ('daily_digests', 'digest_date'), ('daily_digests', 'story_count'),
        ('daily_digests', 'digest_text'), ('daily_digests', 'telegram_message_id'),
        ('daily_digests', 'published_at'),
        ('sources', 'id'), ('sources', 'name'), ('sources', 'source_type'),
        ('sources', 'url'), ('sources', 'category'), ('sources', 'enabled'),
        ('sources', 'last_checked_at'), ('sources', 'last_success_at'),
        ('sources', 'last_error'), ('sources', 'consecutive_failures'),
        ('events', 'id'), ('events', 'event_fingerprint'), ('events', 'event_title'),
        ('events', 'first_seen_at'), ('events', 'last_updated_at'), ('events', 'article_count'),
        ('system_logs', 'id'), ('system_logs', 'timestamp'), ('system_logs', 'level'),
        ('system_logs', 'component'), ('system_logs', 'message')
)
SELECT
    e.table_name,
    e.column_name AS expected_column,
    c.data_type,
    (c.column_name IS NOT NULL) AS exists_in_table
FROM expected_other_cols e
LEFT JOIN information_schema.columns c
  ON c.table_schema = 'public'
 AND c.table_name = e.table_name
 AND c.column_name = e.column_name
ORDER BY e.table_name, e.column_name;
-- Expected: 32 rows, every row has exists_in_table = true.


-- 4. Verify all PRIMARY KEY and UNIQUE constraints required by PostgREST `on_conflict`
SELECT
    tc.table_name,
    tc.constraint_type,
    tc.constraint_name,
    string_agg(kcu.column_name, ', ' ORDER BY kcu.ordinal_position) AS columns
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
  ON tc.constraint_name = kcu.constraint_name
 AND tc.table_schema = kcu.table_schema
WHERE tc.table_schema = 'public'
  AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE')
  AND tc.table_name IN (
      'articles',
      'skipped_urls',
      'daily_digests',
      'sources',
      'events',
      'system_logs'
  )
GROUP BY tc.table_name, tc.constraint_type, tc.constraint_name
ORDER BY tc.table_name, tc.constraint_type;
-- Expected on_conflict targets present:
--   articles(canonical_url), skipped_urls(url_hash), daily_digests(digest_date),
--   sources(name), events(event_fingerprint).


-- 5. Verify all performance, threading, and ledger replay indexes
SELECT
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
  AND tablename IN (
      'articles',
      'skipped_urls',
      'daily_digests',
      'sources',
      'events',
      'system_logs'
  )
ORDER BY tablename, indexname;


-- 6. Verify schema `public` CREATE privilege is revoked from untrusted roles
SELECT
    has_schema_privilege('anon', 'public', 'CREATE') AS anon_create_public,
    has_schema_privilege('authenticated', 'public', 'CREATE') AS auth_create_public;
-- Expected: both false.


-- 7. Verify all 9 public Global RPC functions + 4 internal `_kgn_*` helpers,
--    their SECURITY DEFINER / search_path configuration, and least-privilege grants
WITH expected_funcs(function_name, expected_secdef) AS (
    VALUES
        ('get_global_quota_status', TRUE),
        ('reserve_global_slot', TRUE),
        ('renew_global_reservation', TRUE),
        ('validate_global_reservation', TRUE),
        ('mark_global_publishing', TRUE),
        ('finalize_global_publication', TRUE),
        ('release_global_reservation', TRUE),
        ('mark_global_ambiguous', TRUE),
        ('reconcile_ambiguous_publication', TRUE),
        ('_kgn_reconstruct_global_reservations', TRUE),
        ('_kgn_try_parse_jsonb', FALSE),
        ('_kgn_try_parse_timestamptz', FALSE),
        ('_kgn_try_parse_bigint', FALSE),
        ('_kgn_try_parse_integer', FALSE)
)
SELECT
    e.function_name,
    (p.oid IS NOT NULL) AS exists_in_public,
    p.prosecdef AS security_definer,
    p.proconfig AS function_config,
    pg_get_function_identity_arguments(p.oid) AS args,
    has_function_privilege('anon', p.oid, 'EXECUTE') AS anon_exec,
    has_function_privilege('authenticated', p.oid, 'EXECUTE') AS auth_exec,
    has_function_privilege('service_role', p.oid, 'EXECUTE') AS service_exec
FROM expected_funcs e
LEFT JOIN pg_proc p
  ON p.proname = e.function_name
 AND p.pronamespace = 'public'::regnamespace
ORDER BY e.function_name;
-- Expected: all 14 rows have exists_in_public=true, function_config={'search_path=pg_catalog, public'},
-- anon_exec=false, auth_exec=false, service_exec=true.


-- 8. Read-only smoke test of Global quota RPC on empty staging project
SELECT public.get_global_quota_status(4);
-- Expected JSONB: ok=true, published_global_24h=0, published_total_24h=0,
-- active_reservations=0, ambiguous_reservations=0, max_daily_global=4.
