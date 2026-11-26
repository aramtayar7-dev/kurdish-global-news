-- ============================================================================
-- Kurdish Global News Bot — Authoritative PostgreSQL RPC Admission & Lifecycle
-- Target Environment: Supabase PostgreSQL (15+) + PostgREST RPC
-- Status: REVIEWABLE DEPLOYMENT ARTIFACT (DO NOT AUTO-EXECUTE)
-- ============================================================================
--
-- ARCHITECTURAL PURPOSE
--   Makes PostgreSQL the single authoritative decision-maker for Global article
--   admission, reservation ownership, lease renewal, pre-publish revalidation,
--   atomic publication finalization, and ambiguous-outcome protection across
--   overlapping GitHub Actions runners and cloud Python workers.
--
-- LOCKING MODEL
--   Every state-mutating or capacity-evaluating RPC immediately acquires:
--     PERFORM pg_advisory_xact_lock(8472910348291042);
--   BEFORE reading `public.articles` or `public.system_logs`.
--   Advisory locks are strictly transaction-scoped (released on COMMIT/ABORT)
--   and are NEVER held across external Telegram HTTP requests.
--
-- SECURITY MODEL (Least Privilege)
--   Inspected `storage/supabase_schema.sql` & `.env.example`:
--     - The bot executes with Supabase `service_role` / secret API credentials
--       for server-to-server execution, not end-user `anon` or `authenticated`
--       browser clients.
--     - Functions are declared `SECURITY DEFINER SET search_path = public, pg_temp`
--       and ALL execution rights are explicitly revoked from `PUBLIC`, `anon`,
--       and `authenticated`, then granted ONLY to `service_role`.
--     - All table and function references are schema-qualified (`public.*`).
--
-- EXISTING SCHEMA REUSE (`public.system_logs` + `public.articles`)
--   Uses existing `public.system_logs` (`component = 'GLOBAL_RESERVATION'`)
--   without schema alteration. Versioned JSON v1 payloads in `message`:
--     {
--       "v": 1,
--       "reservation_id": "<uuid>",
--       "owner_token": "<worker_token>",
--       "canonical_url": "<normalized_url>",
--       "url_hash": "<sha256>",
--       "article_id": 123,
--       "batch_id": "<batch_id>",
--       "state": "RESERVED|RENEWED|PUBLISHING|COMMITTED|RELEASED|AMBIGUOUS|EXPIRED",
--       "lease_seconds": 180,
--       "expires_at": "<iso8601_utc>",
--       "telegram_message_id": null,
--       "reason": ""
--     }
--   State reconstruction orders deterministically by `system_logs.id ASC`
--   (monotonic identity PK, immune to timestamp collisions) and safely handles
--   legacy non-JSON raw-URL messages if present. Watchdog queries filter on
--   `component IN ('WATCHDOG_ALERT', 'WATCHDOG_REPORT', 'WATCHDOG_TELEMETRY')`
--   and never inspect `'GLOBAL_RESERVATION'`.
--
-- ============================================================================


-- ----------------------------------------------------------------------------
-- Internal Helper: Safe JSONB parser (never throws on legacy raw-URL text)
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public._kgn_try_parse_jsonb(p_raw TEXT)
RETURNS JSONB
LANGUAGE plpgsql
IMMUTABLE
SET search_path = public, pg_temp
AS $$
DECLARE
    v_trimmed TEXT;
BEGIN
    v_trimmed := BTRIM(COALESCE(p_raw, ''));
    IF LEFT(v_trimmed, 1) <> '{' THEN
        RETURN NULL;
    END IF;
    RETURN v_trimmed::JSONB;
EXCEPTION WHEN OTHERS THEN
    RETURN NULL;
END;
$$;

REVOKE ALL ON FUNCTION public._kgn_try_parse_jsonb(TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION public._kgn_try_parse_jsonb(TEXT) FROM anon;
REVOKE ALL ON FUNCTION public._kgn_try_parse_jsonb(TEXT) FROM authenticated;
GRANT EXECUTE ON FUNCTION public._kgn_try_parse_jsonb(TEXT) TO service_role;


-- ----------------------------------------------------------------------------
-- Internal Helper: Deterministic reservation ledger reconstruction (`id ASC`)
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public._kgn_reconstruct_global_reservations(
    p_now TIMESTAMPTZ DEFAULT timezone('utc', now())
)
RETURNS TABLE (
    reservation_id TEXT,
    owner_token TEXT,
    canonical_url TEXT,
    url_hash TEXT,
    article_id BIGINT,
    batch_id TEXT,
    state TEXT,
    is_legacy BOOLEAN,
    reserved_at TIMESTAMPTZ,
    state_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    lease_seconds INTEGER,
    telegram_message_id BIGINT,
    reason TEXT,
    counts_for_quota BOOLEAN,
    counts_for_batch BOOLEAN
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
WITH raw_events AS (
    SELECT
        sl.id,
        sl.timestamp AS row_ts,
        UPPER(BTRIM(COALESCE(sl.level, ''))) AS row_level,
        COALESCE(sl.message, '') AS raw_msg,
        public._kgn_try_parse_jsonb(sl.message) AS payload
    FROM public.system_logs sl
    WHERE sl.component = 'GLOBAL_RESERVATION'
      AND sl.timestamp >= (p_now - INTERVAL '48 hours')
),
parsed AS (
    SELECT
        e.id,
        e.row_ts,
        (e.payload IS NULL) AS is_legacy,
        CASE
            WHEN e.payload IS NOT NULL
                 AND NULLIF(BTRIM(COALESCE(e.payload->>'reservation_id', '')), '') IS NOT NULL
                THEN BTRIM(e.payload->>'reservation_id')
            WHEN NULLIF(BTRIM(e.raw_msg), '') IS NOT NULL
                THEN 'legacy:' || BTRIM(e.raw_msg)
            ELSE 'legacy_row:' || e.id::TEXT
        END AS res_id,
        CASE
            WHEN e.payload IS NOT NULL
                THEN NULLIF(BTRIM(COALESCE(e.payload->>'owner_token', '')), '')
            ELSE NULL
        END AS owner_tok,
        CASE
            WHEN e.payload IS NOT NULL
                THEN BTRIM(COALESCE(e.payload->>'canonical_url', ''))
            ELSE BTRIM(e.raw_msg)
        END AS canon_url,
        CASE
            WHEN e.payload IS NOT NULL
                THEN BTRIM(COALESCE(e.payload->>'url_hash', ''))
            ELSE ''
        END AS u_hash,
        CASE
            WHEN e.payload IS NOT NULL
                 AND (e.payload->>'article_id') ~ '^\d+$'
                THEN (e.payload->>'article_id')::BIGINT
            ELSE NULL
        END AS art_id,
        CASE
            WHEN e.payload IS NOT NULL
                THEN BTRIM(COALESCE(e.payload->>'batch_id', ''))
            ELSE ''
        END AS b_id,
        CASE
            WHEN e.payload IS NOT NULL
                 AND NULLIF(BTRIM(COALESCE(e.payload->>'state', '')), '') IS NOT NULL
                THEN UPPER(BTRIM(e.payload->>'state'))
            ELSE e.row_level
        END AS state_name,
        CASE
            WHEN e.payload IS NOT NULL
                 AND NULLIF(BTRIM(COALESCE(e.payload->>'expires_at', '')), '') IS NOT NULL
                THEN (e.payload->>'expires_at')::TIMESTAMPTZ
            ELSE (e.row_ts + INTERVAL '15 minutes')
        END AS exp_ts,
        CASE
            WHEN e.payload IS NOT NULL
                 AND (e.payload->>'lease_seconds') ~ '^\d+$'
                THEN GREATEST(1, (e.payload->>'lease_seconds')::INTEGER)
            ELSE 900
        END AS lease_sec,
        CASE
            WHEN e.payload IS NOT NULL
                 AND (e.payload->>'telegram_message_id') ~ '^\d+$'
                THEN (e.payload->>'telegram_message_id')::BIGINT
            ELSE NULL
        END AS tg_msg_id,
        CASE
            WHEN e.payload IS NOT NULL
                THEN COALESCE(e.payload->>'reason', '')
            ELSE ''
        END AS res_reason
    FROM raw_events e
),
with_reserved_ts AS (
    SELECT
        p.*,
        COALESCE(
            MIN(p.row_ts) FILTER (WHERE p.state_name = 'RESERVED') OVER (PARTITION BY p.res_id),
            MIN(p.row_ts) OVER (PARTITION BY p.res_id)
        ) AS first_reserved_ts
    FROM parsed p
),
latest AS (
    SELECT DISTINCT ON (w.res_id)
        w.res_id AS reservation_id,
        w.owner_tok AS owner_token,
        w.canon_url AS canonical_url,
        w.u_hash AS url_hash,
        w.art_id AS article_id,
        w.b_id AS batch_id,
        w.state_name AS state,
        w.is_legacy,
        w.first_reserved_ts AS reserved_at,
        w.row_ts AS state_at,
        w.exp_ts AS expires_at,
        w.lease_sec AS lease_seconds,
        w.tg_msg_id AS telegram_message_id,
        w.res_reason AS reason
    FROM with_reserved_ts w
    ORDER BY w.res_id, w.id DESC
)
SELECT
    l.reservation_id,
    l.owner_token,
    l.canonical_url,
    l.url_hash,
    l.article_id,
    l.batch_id,
    l.state,
    l.is_legacy,
    l.reserved_at,
    l.state_at,
    l.expires_at,
    l.lease_seconds,
    l.telegram_message_id,
    l.reason,
    (
        (
            l.state IN ('RESERVED', 'RENEWED')
            AND l.expires_at > p_now
            AND l.reserved_at >= (p_now - INTERVAL '24 hours')
        )
        OR (
            l.state IN ('PUBLISHING', 'AMBIGUOUS')
            AND l.state_at >= (p_now - INTERVAL '24 hours')
        )
    ) AS counts_for_quota,
    (
        l.batch_id <> ''
        AND (
            (l.state IN ('RESERVED', 'RENEWED') AND l.expires_at > p_now AND l.reserved_at >= (p_now - INTERVAL '24 hours'))
            OR (l.state IN ('PUBLISHING', 'AMBIGUOUS', 'COMMITTED') AND l.state_at >= (p_now - INTERVAL '24 hours'))
        )
    ) AS counts_for_batch
FROM latest l;
$$;

REVOKE ALL ON FUNCTION public._kgn_reconstruct_global_reservations(TIMESTAMPTZ) FROM PUBLIC;
REVOKE ALL ON FUNCTION public._kgn_reconstruct_global_reservations(TIMESTAMPTZ) FROM anon;
REVOKE ALL ON FUNCTION public._kgn_reconstruct_global_reservations(TIMESTAMPTZ) FROM authenticated;
GRANT EXECUTE ON FUNCTION public._kgn_reconstruct_global_reservations(TIMESTAMPTZ) TO service_role;


-- ----------------------------------------------------------------------------
-- RPC H: Read-Only Quota Status Snapshot (under advisory lock for consistency)
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.get_global_quota_status(
    p_max_daily_global INTEGER DEFAULT 4
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
    v_now TIMESTAMPTZ := timezone('utc', now());
    v_cutoff TIMESTAMPTZ := v_now - INTERVAL '24 hours';
    v_max_daily INTEGER := GREATEST(0, LEAST(4, COALESCE(p_max_daily_global, 4)));
    v_global_24h INTEGER := 0;
    v_total_24h INTEGER := 0;
    v_active_res INTEGER := 0;
    v_ambiguous_res INTEGER := 0;
    v_active_urls JSONB := '[]'::JSONB;
BEGIN
    PERFORM pg_advisory_xact_lock(8472910348291042);

    SELECT
        COALESCE(COUNT(DISTINCT COALESCE(NULLIF(BTRIM(a.canonical_url), ''), 'id:' || a.id::TEXT))
            FILTER (WHERE LOWER(BTRIM(COALESCE(a.region, ''))) = 'global'), 0)::INTEGER,
        COALESCE(COUNT(DISTINCT COALESCE(NULLIF(BTRIM(a.canonical_url), ''), 'id:' || a.id::TEXT)), 0)::INTEGER
    INTO v_global_24h, v_total_24h
    FROM public.articles a
    WHERE a.status = 'PUBLISHED'
      AND a.published_at IS NOT NULL
      AND a.published_at >= v_cutoff;

    SELECT
        COALESCE(COUNT(*) FILTER (WHERE r.counts_for_quota), 0)::INTEGER,
        COALESCE(COUNT(*) FILTER (WHERE r.counts_for_quota AND r.state = 'AMBIGUOUS'), 0)::INTEGER,
        COALESCE(
            jsonb_agg(r.canonical_url) FILTER (WHERE r.counts_for_quota AND r.canonical_url <> ''),
            '[]'::JSONB
        )
    INTO v_active_res, v_ambiguous_res, v_active_urls
    FROM public._kgn_reconstruct_global_reservations(v_now) r;

    RETURN jsonb_build_object(
        'ok', TRUE,
        'now_utc', to_char(v_now AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
        'published_global_24h', v_global_24h,
        'published_total_24h', v_total_24h,
        'active_reservations', v_active_res,
        'ambiguous_reservations', v_ambiguous_res,
        'active_urls', v_active_urls,
        'max_daily_global', v_max_daily
    );
END;
$$;

REVOKE ALL ON FUNCTION public.get_global_quota_status(INTEGER) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.get_global_quota_status(INTEGER) FROM anon;
REVOKE ALL ON FUNCTION public.get_global_quota_status(INTEGER) FROM authenticated;
GRANT EXECUTE ON FUNCTION public.get_global_quota_status(INTEGER) TO service_role;


-- ----------------------------------------------------------------------------
-- RPC A: reserve_global_slot()
-- Atomically checks rolling 24h Global/Total counts, active reservations,
-- duplicates, batch cap, daily cap, and 90/10 ratio under advisory lock.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.reserve_global_slot(
    p_canonical_url TEXT,
    p_owner_token TEXT,
    p_reservation_id TEXT DEFAULT NULL,
    p_article_id BIGINT DEFAULT NULL,
    p_batch_id TEXT DEFAULT '',
    p_lease_seconds INTEGER DEFAULT 180,
    p_url_hash TEXT DEFAULT '',
    p_max_daily_global INTEGER DEFAULT 4,
    p_max_batch_global INTEGER DEFAULT 1
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
    v_now TIMESTAMPTZ := timezone('utc', now());
    v_cutoff TIMESTAMPTZ := v_now - INTERVAL '24 hours';
    v_url TEXT := BTRIM(COALESCE(p_canonical_url, ''));
    v_owner TEXT := BTRIM(COALESCE(p_owner_token, ''));
    v_res_id TEXT := NULLIF(BTRIM(COALESCE(p_reservation_id, '')), '');
    v_batch TEXT := BTRIM(COALESCE(p_batch_id, ''));
    v_hash TEXT := BTRIM(COALESCE(p_url_hash, ''));
    v_lease INTEGER := GREATEST(30, LEAST(900, COALESCE(p_lease_seconds, 180)));
    v_max_daily INTEGER := GREATEST(0, LEAST(4, COALESCE(p_max_daily_global, 4)));
    v_max_batch INTEGER := GREATEST(0, LEAST(1, COALESCE(p_max_batch_global, 1)));
    v_expires TIMESTAMPTZ := v_now + make_interval(secs => v_lease);
    v_global_24h INTEGER := 0;
    v_total_24h INTEGER := 0;
    v_active_other INTEGER := 0;
    v_batch_used INTEGER := 0;
    v_already_pub BOOLEAN := FALSE;
    v_existing RECORD;
    v_dup RECORD;
    v_payload JSONB;
BEGIN
    PERFORM pg_advisory_xact_lock(8472910348291042);

    IF v_owner = '' THEN
        RETURN jsonb_build_object(
            'admitted', FALSE,
            'state', 'REJECTED',
            'reason', 'missing_owner_token'
        );
    END IF;

    IF v_url = '' AND p_article_id IS NULL THEN
        RETURN jsonb_build_object(
            'admitted', FALSE,
            'state', 'REJECTED',
            'reason', 'missing_article_identity'
        );
    END IF;

    IF v_res_id IS NULL THEN
        v_res_id := md5(random()::TEXT || clock_timestamp()::TEXT || v_url || v_owner);
    END IF;

    -- 1. Idempotent retry by reservation_id
    SELECT * INTO v_existing
    FROM public._kgn_reconstruct_global_reservations(v_now) r
    WHERE r.reservation_id = v_res_id
    LIMIT 1;

    IF FOUND THEN
        IF v_existing.owner_token IS DISTINCT FROM v_owner THEN
            RETURN jsonb_build_object(
                'admitted', FALSE,
                'state', v_existing.state,
                'reservation_id', v_res_id,
                'reason', 'reservation_id_owner_mismatch'
            );
        END IF;

        IF v_existing.counts_for_quota THEN
            RETURN jsonb_build_object(
                'admitted', TRUE,
                'state', v_existing.state,
                'reservation_id', v_existing.reservation_id,
                'owner_token', v_existing.owner_token,
                'expires_at', to_char(v_existing.expires_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
                'reason', 'idempotent_existing_reservation'
            );
        ELSIF NOT (v_res_id LIKE 'legacy_url:%' AND v_existing.state IN ('RELEASED', 'EXPIRED')) THEN
            RETURN jsonb_build_object(
                'admitted', FALSE,
                'state', v_existing.state,
                'reservation_id', v_existing.reservation_id,
                'reason', 'reservation_not_active_' || LOWER(v_existing.state)
            );
        END IF;
    END IF;

    -- 2. Check if article already PUBLISHED in articles table
    SELECT EXISTS (
        SELECT 1
        FROM public.articles a
        WHERE a.status = 'PUBLISHED'
          AND (
              (p_article_id IS NOT NULL AND a.id = p_article_id)
              OR (v_url <> '' AND a.canonical_url = v_url)
          )
    ) INTO v_already_pub;

    IF v_already_pub THEN
        RETURN jsonb_build_object(
            'admitted', FALSE,
            'state', 'REJECTED',
            'reason', 'article_already_published'
        );
    END IF;

    -- 3. Check duplicate active/publishing/ambiguous reservation for same article identity
    SELECT * INTO v_dup
    FROM public._kgn_reconstruct_global_reservations(v_now) r
    WHERE r.counts_for_quota
      AND (
          (v_url <> '' AND r.canonical_url = v_url)
          OR (v_hash <> '' AND r.url_hash = v_hash)
          OR (p_article_id IS NOT NULL AND r.article_id = p_article_id)
      )
    LIMIT 1;

    IF FOUND THEN
        -- Idempotent match ONLY if same worker owns an active RESERVED/RENEWED lease
        IF v_dup.owner_token = v_owner AND v_dup.state IN ('RESERVED', 'RENEWED') THEN
            RETURN jsonb_build_object(
                'admitted', TRUE,
                'state', v_dup.state,
                'reservation_id', v_dup.reservation_id,
                'owner_token', v_dup.owner_token,
                'expires_at', to_char(v_dup.expires_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
                'reason', 'idempotent_existing_article_reservation'
            );
        END IF;

        RETURN jsonb_build_object(
            'admitted', FALSE,
            'state', v_dup.state,
            'reservation_id', v_dup.reservation_id,
            'reason', 'duplicate_active_reservation'
        );
    END IF;

    -- 4. Count rolling 24h published Global & Total articles
    SELECT
        COALESCE(COUNT(DISTINCT COALESCE(NULLIF(BTRIM(a.canonical_url), ''), 'id:' || a.id::TEXT))
            FILTER (WHERE LOWER(BTRIM(COALESCE(a.region, ''))) = 'global'), 0)::INTEGER,
        COALESCE(COUNT(DISTINCT COALESCE(NULLIF(BTRIM(a.canonical_url), ''), 'id:' || a.id::TEXT)), 0)::INTEGER
    INTO v_global_24h, v_total_24h
    FROM public.articles a
    WHERE a.status = 'PUBLISHED'
      AND a.published_at IS NOT NULL
      AND a.published_at >= v_cutoff;

    -- 5. Count other active/publishing/ambiguous reservations & batch slots
    SELECT
        COALESCE(COUNT(*) FILTER (WHERE r.counts_for_quota), 0)::INTEGER,
        COALESCE(COUNT(*) FILTER (WHERE r.counts_for_batch AND v_batch <> '' AND r.batch_id = v_batch), 0)::INTEGER
    INTO v_active_other, v_batch_used
    FROM public._kgn_reconstruct_global_reservations(v_now) r;

    -- 6. Enforce per-batch cap (MAX_GLOBAL_POSTS_PER_BATCH = 1)
    IF v_batch <> '' AND v_batch_used >= v_max_batch THEN
        RETURN jsonb_build_object(
            'admitted', FALSE,
            'state', 'REJECTED',
            'published_global_24h', v_global_24h,
            'published_total_24h', v_total_24h,
            'active_reservations', v_active_other,
            'reason', 'batch_global_cap_reached'
        );
    END IF;

    -- 7. Enforce rolling 24h daily cap (MAX_GLOBAL_POSTS_PER_DAY = 4)
    IF (v_global_24h + v_active_other + 1) > v_max_daily THEN
        RETURN jsonb_build_object(
            'admitted', FALSE,
            'state', 'REJECTED',
            'published_global_24h', v_global_24h,
            'published_total_24h', v_total_24h,
            'active_reservations', v_active_other,
            'reason', 'daily_global_cap_reached'
        );
    END IF;

    -- 8. Enforce strict 90/10 ratio: 10 * (G + R + 1) <= (T + R + 1)
    IF 10 * (v_global_24h + v_active_other + 1) > (v_total_24h + v_active_other + 1) THEN
        RETURN jsonb_build_object(
            'admitted', FALSE,
            'state', 'REJECTED',
            'published_global_24h', v_global_24h,
            'published_total_24h', v_total_24h,
            'active_reservations', v_active_other,
            'reason', 'rolling_90_10_ratio_not_satisfied'
        );
    END IF;

    -- 9. Persist RESERVED state in system_logs inside the lock
    v_payload := jsonb_build_object(
        'v', 1,
        'reservation_id', v_res_id,
        'owner_token', v_owner,
        'canonical_url', v_url,
        'url_hash', v_hash,
        'article_id', p_article_id,
        'batch_id', v_batch,
        'state', 'RESERVED',
        'lease_seconds', v_lease,
        'expires_at', to_char(v_expires AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
        'telegram_message_id', NULL,
        'reason', 'admitted'
    );

    INSERT INTO public.system_logs (timestamp, level, component, message)
    VALUES (v_now, 'RESERVED', 'GLOBAL_RESERVATION', v_payload::TEXT);

    RETURN jsonb_build_object(
        'admitted', TRUE,
        'state', 'RESERVED',
        'reservation_id', v_res_id,
        'owner_token', v_owner,
        'expires_at', to_char(v_expires AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
        'published_global_24h', v_global_24h,
        'published_total_24h', v_total_24h,
        'active_reservations', v_active_other,
        'reason', 'admitted'
    );
END;
$$;

REVOKE ALL ON FUNCTION public.reserve_global_slot(TEXT, TEXT, TEXT, BIGINT, TEXT, INTEGER, TEXT, INTEGER, INTEGER) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.reserve_global_slot(TEXT, TEXT, TEXT, BIGINT, TEXT, INTEGER, TEXT, INTEGER, INTEGER) FROM anon;
REVOKE ALL ON FUNCTION public.reserve_global_slot(TEXT, TEXT, TEXT, BIGINT, TEXT, INTEGER, TEXT, INTEGER, INTEGER) FROM authenticated;
GRANT EXECUTE ON FUNCTION public.reserve_global_slot(TEXT, TEXT, TEXT, BIGINT, TEXT, INTEGER, TEXT, INTEGER, INTEGER) TO service_role;


-- ----------------------------------------------------------------------------
-- RPC B: renew_global_reservation()
-- Owner-authorized lease extension; preserves PUBLISHING state if active.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.renew_global_reservation(
    p_reservation_id TEXT,
    p_owner_token TEXT,
    p_lease_seconds INTEGER DEFAULT 180
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
    v_now TIMESTAMPTZ := timezone('utc', now());
    v_res_id TEXT := BTRIM(COALESCE(p_reservation_id, ''));
    v_owner TEXT := BTRIM(COALESCE(p_owner_token, ''));
    v_lease INTEGER := GREATEST(30, LEAST(900, COALESCE(p_lease_seconds, 180)));
    v_expires TIMESTAMPTZ := v_now + make_interval(secs => v_lease);
    v_cur RECORD;
    v_next_state TEXT;
    v_payload JSONB;
BEGIN
    PERFORM pg_advisory_xact_lock(8472910348291042);

    SELECT * INTO v_cur
    FROM public._kgn_reconstruct_global_reservations(v_now) r
    WHERE r.reservation_id = v_res_id
    LIMIT 1;

    IF NOT FOUND THEN
        RETURN jsonb_build_object('ok', FALSE, 'state', 'NOT_FOUND', 'reason', 'reservation_not_found');
    END IF;

    IF v_cur.owner_token IS DISTINCT FROM v_owner THEN
        RETURN jsonb_build_object('ok', FALSE, 'state', v_cur.state, 'reason', 'owner_mismatch');
    END IF;

    IF v_cur.state IN ('RESERVED', 'RENEWED') AND v_cur.expires_at <= v_now THEN
        v_payload := jsonb_build_object(
            'v', 1, 'reservation_id', v_cur.reservation_id, 'owner_token', v_cur.owner_token,
            'canonical_url', v_cur.canonical_url, 'url_hash', v_cur.url_hash,
            'article_id', v_cur.article_id, 'batch_id', v_cur.batch_id,
            'state', 'EXPIRED', 'lease_seconds', v_cur.lease_seconds,
            'expires_at', to_char(v_cur.expires_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
            'telegram_message_id', NULL, 'reason', 'lease_expired_before_renewal'
        );
        INSERT INTO public.system_logs (timestamp, level, component, message)
        VALUES (v_now, 'EXPIRED', 'GLOBAL_RESERVATION', v_payload::TEXT);
        RETURN jsonb_build_object('ok', FALSE, 'state', 'EXPIRED', 'reason', 'reservation_expired');
    END IF;

    IF v_cur.state NOT IN ('RESERVED', 'RENEWED', 'PUBLISHING') THEN
        RETURN jsonb_build_object('ok', FALSE, 'state', v_cur.state, 'reason', 'invalid_state_' || LOWER(v_cur.state));
    END IF;

    -- Keep PUBLISHING as PUBLISHING when renewing lease during long Telegram upload
    v_next_state := CASE WHEN v_cur.state = 'PUBLISHING' THEN 'PUBLISHING' ELSE 'RENEWED' END;

    v_payload := jsonb_build_object(
        'v', 1,
        'reservation_id', v_cur.reservation_id,
        'owner_token', v_cur.owner_token,
        'canonical_url', v_cur.canonical_url,
        'url_hash', v_cur.url_hash,
        'article_id', v_cur.article_id,
        'batch_id', v_cur.batch_id,
        'state', v_next_state,
        'lease_seconds', v_lease,
        'expires_at', to_char(v_expires AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
        'telegram_message_id', NULL,
        'reason', 'lease_renewed'
    );

    INSERT INTO public.system_logs (timestamp, level, component, message)
    VALUES (v_now, v_next_state, 'GLOBAL_RESERVATION', v_payload::TEXT);

    RETURN jsonb_build_object(
        'ok', TRUE,
        'state', v_next_state,
        'reservation_id', v_cur.reservation_id,
        'expires_at', to_char(v_expires AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
        'reason', 'renewed'
    );
END;
$$;

REVOKE ALL ON FUNCTION public.renew_global_reservation(TEXT, TEXT, INTEGER) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.renew_global_reservation(TEXT, TEXT, INTEGER) FROM anon;
REVOKE ALL ON FUNCTION public.renew_global_reservation(TEXT, TEXT, INTEGER) FROM authenticated;
GRANT EXECUTE ON FUNCTION public.renew_global_reservation(TEXT, TEXT, INTEGER) TO service_role;


-- ----------------------------------------------------------------------------
-- RPC C: validate_global_reservation()
-- Validates ownership, active state, lease expiry, and optional live 90/10 ratio.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.validate_global_reservation(
    p_reservation_id TEXT,
    p_owner_token TEXT,
    p_check_ratio BOOLEAN DEFAULT TRUE,
    p_max_daily_global INTEGER DEFAULT 4
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
    v_now TIMESTAMPTZ := timezone('utc', now());
    v_cutoff TIMESTAMPTZ := v_now - INTERVAL '24 hours';
    v_res_id TEXT := BTRIM(COALESCE(p_reservation_id, ''));
    v_owner TEXT := BTRIM(COALESCE(p_owner_token, ''));
    v_max_daily INTEGER := GREATEST(0, LEAST(4, COALESCE(p_max_daily_global, 4)));
    v_cur RECORD;
    v_global_24h INTEGER := 0;
    v_total_24h INTEGER := 0;
    v_active_other INTEGER := 0;
BEGIN
    PERFORM pg_advisory_xact_lock(8472910348291042);

    SELECT * INTO v_cur
    FROM public._kgn_reconstruct_global_reservations(v_now) r
    WHERE r.reservation_id = v_res_id
    LIMIT 1;

    IF NOT FOUND THEN
        RETURN jsonb_build_object('valid', FALSE, 'state', 'NOT_FOUND', 'reason', 'reservation_not_found');
    END IF;

    IF v_cur.owner_token IS DISTINCT FROM v_owner THEN
        RETURN jsonb_build_object('valid', FALSE, 'state', v_cur.state, 'reason', 'owner_mismatch');
    END IF;

    IF NOT v_cur.counts_for_quota THEN
        RETURN jsonb_build_object('valid', FALSE, 'state', v_cur.state, 'reason', 'reservation_inactive_or_expired');
    END IF;

    SELECT
        COALESCE(COUNT(DISTINCT COALESCE(NULLIF(BTRIM(a.canonical_url), ''), 'id:' || a.id::TEXT))
            FILTER (WHERE LOWER(BTRIM(COALESCE(a.region, ''))) = 'global'), 0)::INTEGER,
        COALESCE(COUNT(DISTINCT COALESCE(NULLIF(BTRIM(a.canonical_url), ''), 'id:' || a.id::TEXT)), 0)::INTEGER
    INTO v_global_24h, v_total_24h
    FROM public.articles a
    WHERE a.status = 'PUBLISHED'
      AND a.published_at IS NOT NULL
      AND a.published_at >= v_cutoff;

    SELECT COALESCE(COUNT(*) FILTER (WHERE r.counts_for_quota AND r.reservation_id <> v_res_id), 0)::INTEGER
    INTO v_active_other
    FROM public._kgn_reconstruct_global_reservations(v_now) r;

    IF COALESCE(p_check_ratio, TRUE) THEN
        IF (v_global_24h + v_active_other + 1) > v_max_daily THEN
            RETURN jsonb_build_object(
                'valid', FALSE, 'state', v_cur.state,
                'published_global_24h', v_global_24h, 'published_total_24h', v_total_24h,
                'active_reservations_other', v_active_other,
                'reason', 'daily_global_cap_reached'
            );
        END IF;
        IF 10 * (v_global_24h + v_active_other + 1) > (v_total_24h + v_active_other + 1) THEN
            RETURN jsonb_build_object(
                'valid', FALSE, 'state', v_cur.state,
                'published_global_24h', v_global_24h, 'published_total_24h', v_total_24h,
                'active_reservations_other', v_active_other,
                'reason', 'rolling_90_10_ratio_not_satisfied'
            );
        END IF;
    END IF;

    RETURN jsonb_build_object(
        'valid', TRUE,
        'state', v_cur.state,
        'published_global_24h', v_global_24h,
        'published_total_24h', v_total_24h,
        'active_reservations_other', v_active_other,
        'reason', 'valid'
    );
END;
$$;

REVOKE ALL ON FUNCTION public.validate_global_reservation(TEXT, TEXT, BOOLEAN, INTEGER) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.validate_global_reservation(TEXT, TEXT, BOOLEAN, INTEGER) FROM anon;
REVOKE ALL ON FUNCTION public.validate_global_reservation(TEXT, TEXT, BOOLEAN, INTEGER) FROM authenticated;
GRANT EXECUTE ON FUNCTION public.validate_global_reservation(TEXT, TEXT, BOOLEAN, INTEGER) TO service_role;


-- ----------------------------------------------------------------------------
-- RPC D: mark_global_publishing()
-- Final pre-publish authorization immediately before calling Telegram.
-- Revalidates live 24h 90/10 ratio & daily cap; auto-releases if invalid.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.mark_global_publishing(
    p_reservation_id TEXT,
    p_owner_token TEXT,
    p_lease_seconds INTEGER DEFAULT 180,
    p_max_daily_global INTEGER DEFAULT 4
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
    v_now TIMESTAMPTZ := timezone('utc', now());
    v_cutoff TIMESTAMPTZ := v_now - INTERVAL '24 hours';
    v_res_id TEXT := BTRIM(COALESCE(p_reservation_id, ''));
    v_owner TEXT := BTRIM(COALESCE(p_owner_token, ''));
    v_lease INTEGER := GREATEST(30, LEAST(900, COALESCE(p_lease_seconds, 180)));
    v_max_daily INTEGER := GREATEST(0, LEAST(4, COALESCE(p_max_daily_global, 4)));
    v_expires TIMESTAMPTZ := v_now + make_interval(secs => v_lease);
    v_cur RECORD;
    v_already_pub BOOLEAN := FALSE;
    v_global_24h INTEGER := 0;
    v_total_24h INTEGER := 0;
    v_active_other INTEGER := 0;
    v_payload JSONB;
BEGIN
    PERFORM pg_advisory_xact_lock(8472910348291042);

    SELECT * INTO v_cur
    FROM public._kgn_reconstruct_global_reservations(v_now) r
    WHERE r.reservation_id = v_res_id
    LIMIT 1;

    IF NOT FOUND THEN
        RETURN jsonb_build_object('ok', FALSE, 'admitted', FALSE, 'state', 'NOT_FOUND', 'reason', 'reservation_not_found');
    END IF;

    IF v_cur.owner_token IS DISTINCT FROM v_owner THEN
        RETURN jsonb_build_object('ok', FALSE, 'admitted', FALSE, 'state', v_cur.state, 'reason', 'owner_mismatch');
    END IF;

    IF v_cur.state = 'PUBLISHING' THEN
        RETURN jsonb_build_object(
            'ok', TRUE, 'admitted', TRUE, 'state', 'PUBLISHING',
            'reservation_id', v_cur.reservation_id, 'reason', 'already_publishing'
        );
    END IF;

    IF v_cur.state IN ('RESERVED', 'RENEWED') AND v_cur.expires_at <= v_now THEN
        v_payload := jsonb_build_object(
            'v', 1, 'reservation_id', v_cur.reservation_id, 'owner_token', v_cur.owner_token,
            'canonical_url', v_cur.canonical_url, 'url_hash', v_cur.url_hash,
            'article_id', v_cur.article_id, 'batch_id', v_cur.batch_id,
            'state', 'EXPIRED', 'lease_seconds', v_cur.lease_seconds,
            'expires_at', to_char(v_cur.expires_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
            'telegram_message_id', NULL, 'reason', 'lease_expired_before_publish'
        );
        INSERT INTO public.system_logs (timestamp, level, component, message)
        VALUES (v_now, 'EXPIRED', 'GLOBAL_RESERVATION', v_payload::TEXT);
        RETURN jsonb_build_object('ok', FALSE, 'admitted', FALSE, 'state', 'EXPIRED', 'reason', 'reservation_expired');
    END IF;

    IF v_cur.state NOT IN ('RESERVED', 'RENEWED') THEN
        RETURN jsonb_build_object(
            'ok', FALSE, 'admitted', FALSE, 'state', v_cur.state,
            'reason', 'invalid_state_' || LOWER(v_cur.state)
        );
    END IF;

    -- Verify article was not already published elsewhere
    SELECT EXISTS (
        SELECT 1 FROM public.articles a
        WHERE a.status = 'PUBLISHED'
          AND (
              (v_cur.article_id IS NOT NULL AND a.id = v_cur.article_id)
              OR (v_cur.canonical_url <> '' AND a.canonical_url = v_cur.canonical_url)
          )
    ) INTO v_already_pub;

    IF v_already_pub THEN
        RETURN jsonb_build_object('ok', FALSE, 'admitted', FALSE, 'state', v_cur.state, 'reason', 'article_already_published');
    END IF;

    -- Re-check LIVE 24h counts & 90/10 ratio right before Telegram send
    SELECT
        COALESCE(COUNT(DISTINCT COALESCE(NULLIF(BTRIM(a.canonical_url), ''), 'id:' || a.id::TEXT))
            FILTER (WHERE LOWER(BTRIM(COALESCE(a.region, ''))) = 'global'), 0)::INTEGER,
        COALESCE(COUNT(DISTINCT COALESCE(NULLIF(BTRIM(a.canonical_url), ''), 'id:' || a.id::TEXT)), 0)::INTEGER
    INTO v_global_24h, v_total_24h
    FROM public.articles a
    WHERE a.status = 'PUBLISHED'
      AND a.published_at IS NOT NULL
      AND a.published_at >= v_cutoff;

    SELECT COALESCE(COUNT(*) FILTER (WHERE r.counts_for_quota AND r.reservation_id <> v_res_id), 0)::INTEGER
    INTO v_active_other
    FROM public._kgn_reconstruct_global_reservations(v_now) r;

    IF (v_global_24h + v_active_other + 1) > v_max_daily
       OR 10 * (v_global_24h + v_active_other + 1) > (v_total_24h + v_active_other + 1) THEN
        v_payload := jsonb_build_object(
            'v', 1, 'reservation_id', v_cur.reservation_id, 'owner_token', v_cur.owner_token,
            'canonical_url', v_cur.canonical_url, 'url_hash', v_cur.url_hash,
            'article_id', v_cur.article_id, 'batch_id', v_cur.batch_id,
            'state', 'RELEASED', 'lease_seconds', v_cur.lease_seconds,
            'expires_at', to_char(v_cur.expires_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
            'telegram_message_id', NULL, 'reason', 'ratio_or_cap_failed_at_publish'
        );
        INSERT INTO public.system_logs (timestamp, level, component, message)
        VALUES (v_now, 'RELEASED', 'GLOBAL_RESERVATION', v_payload::TEXT);
        RETURN jsonb_build_object(
            'ok', FALSE, 'admitted', FALSE, 'state', 'RELEASED',
            'published_global_24h', v_global_24h, 'published_total_24h', v_total_24h,
            'active_reservations_other', v_active_other,
            'reason', 'ratio_or_cap_failed_at_publish'
        );
    END IF;

    v_payload := jsonb_build_object(
        'v', 1,
        'reservation_id', v_cur.reservation_id,
        'owner_token', v_cur.owner_token,
        'canonical_url', v_cur.canonical_url,
        'url_hash', v_cur.url_hash,
        'article_id', v_cur.article_id,
        'batch_id', v_cur.batch_id,
        'state', 'PUBLISHING',
        'lease_seconds', v_lease,
        'expires_at', to_char(v_expires AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
        'telegram_message_id', NULL,
        'reason', 'publishing_started'
    );
    INSERT INTO public.system_logs (timestamp, level, component, message)
    VALUES (v_now, 'PUBLISHING', 'GLOBAL_RESERVATION', v_payload::TEXT);

    RETURN jsonb_build_object(
        'ok', TRUE,
        'admitted', TRUE,
        'state', 'PUBLISHING',
        'reservation_id', v_cur.reservation_id,
        'expires_at', to_char(v_expires AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
        'published_global_24h', v_global_24h,
        'published_total_24h', v_total_24h,
        'reason', 'publishing'
    );
END;
$$;

REVOKE ALL ON FUNCTION public.mark_global_publishing(TEXT, TEXT, INTEGER, INTEGER) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.mark_global_publishing(TEXT, TEXT, INTEGER, INTEGER) FROM anon;
REVOKE ALL ON FUNCTION public.mark_global_publishing(TEXT, TEXT, INTEGER, INTEGER) FROM authenticated;
GRANT EXECUTE ON FUNCTION public.mark_global_publishing(TEXT, TEXT, INTEGER, INTEGER) TO service_role;


-- ----------------------------------------------------------------------------
-- RPC E: finalize_global_publication()
-- Atomically marks article PUBLISHED in `public.articles` AND logs COMMITTED
-- in `public.system_logs` inside ONE transaction under the advisory lock.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.finalize_global_publication(
    p_reservation_id TEXT,
    p_owner_token TEXT,
    p_article_id BIGINT,
    p_telegram_message_id BIGINT,
    p_thread_root_message_id BIGINT DEFAULT NULL,
    p_is_breaking BOOLEAN DEFAULT FALSE,
    p_published_at TIMESTAMPTZ DEFAULT NULL
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
    v_now TIMESTAMPTZ := timezone('utc', now());
    v_pub_ts TIMESTAMPTZ := COALESCE(p_published_at, v_now);
    v_res_id TEXT := BTRIM(COALESCE(p_reservation_id, ''));
    v_owner TEXT := NULLIF(BTRIM(COALESCE(p_owner_token, '')), '');
    v_cur RECORD;
    v_target_art_id BIGINT;
    v_has_tg_cols BOOLEAN := FALSE;
    v_payload JSONB;
BEGIN
    PERFORM pg_advisory_xact_lock(8472910348291042);

    SELECT * INTO v_cur
    FROM public._kgn_reconstruct_global_reservations(v_now) r
    WHERE r.reservation_id = v_res_id
    LIMIT 1;

    IF NOT FOUND THEN
        RETURN jsonb_build_object('ok', FALSE, 'state', 'NOT_FOUND', 'reason', 'reservation_not_found');
    END IF;

    IF (NOT v_cur.is_legacy AND (v_owner IS NULL OR v_cur.owner_token IS DISTINCT FROM v_owner))
       OR (v_cur.is_legacy AND v_cur.owner_token IS DISTINCT FROM v_owner) THEN
        RETURN jsonb_build_object('ok', FALSE, 'state', v_cur.state, 'reason', 'owner_mismatch');
    END IF;

    IF v_cur.state = 'COMMITTED' THEN
        RETURN jsonb_build_object(
            'ok', TRUE, 'state', 'COMMITTED',
            'reservation_id', v_cur.reservation_id,
            'article_id', COALESCE(p_article_id, v_cur.article_id),
            'telegram_message_id', COALESCE(p_telegram_message_id, v_cur.telegram_message_id),
            'reason', 'already_committed'
        );
    END IF;

    IF v_cur.state NOT IN ('PUBLISHING', 'AMBIGUOUS', 'RESERVED', 'RENEWED') THEN
        RETURN jsonb_build_object('ok', FALSE, 'state', v_cur.state, 'reason', 'invalid_state_' || LOWER(v_cur.state));
    END IF;

    v_target_art_id := COALESCE(p_article_id, v_cur.article_id);
    IF v_target_art_id IS NULL AND v_cur.canonical_url <> '' THEN
        SELECT a.id INTO v_target_art_id
        FROM public.articles a
        WHERE a.canonical_url = v_cur.canonical_url
        LIMIT 1;
    END IF;

    IF v_target_art_id IS NOT NULL THEN
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'articles'
              AND column_name = 'telegram_message_id'
        ) INTO v_has_tg_cols;

        IF v_has_tg_cols THEN
            UPDATE public.articles
            SET status = 'PUBLISHED',
                published_at = v_pub_ts,
                telegram_message_id = p_telegram_message_id,
                thread_root_message_id = p_thread_root_message_id,
                is_breaking = COALESCE(p_is_breaking, FALSE),
                error_message = ''
            WHERE id = v_target_art_id;
        ELSE
            UPDATE public.articles
            SET status = 'PUBLISHED',
                published_at = v_pub_ts,
                is_breaking = COALESCE(p_is_breaking, FALSE),
                error_message = ''
            WHERE id = v_target_art_id;
        END IF;
    END IF;

    v_payload := jsonb_build_object(
        'v', 1,
        'reservation_id', v_cur.reservation_id,
        'owner_token', v_cur.owner_token,
        'canonical_url', v_cur.canonical_url,
        'url_hash', v_cur.url_hash,
        'article_id', v_target_art_id,
        'batch_id', v_cur.batch_id,
        'state', 'COMMITTED',
        'lease_seconds', v_cur.lease_seconds,
        'expires_at', to_char(v_now AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
        'telegram_message_id', p_telegram_message_id,
        'reason', 'published_and_committed'
    );
    INSERT INTO public.system_logs (timestamp, level, component, message)
    VALUES (v_now, 'COMMITTED', 'GLOBAL_RESERVATION', v_payload::TEXT);

    RETURN jsonb_build_object(
        'ok', TRUE,
        'state', 'COMMITTED',
        'reservation_id', v_cur.reservation_id,
        'article_id', v_target_art_id,
        'telegram_message_id', p_telegram_message_id,
        'published_at', to_char(v_pub_ts AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
        'reason', 'committed'
    );
END;
$$;

REVOKE ALL ON FUNCTION public.finalize_global_publication(TEXT, TEXT, BIGINT, BIGINT, BIGINT, BOOLEAN, TIMESTAMPTZ) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.finalize_global_publication(TEXT, TEXT, BIGINT, BIGINT, BIGINT, BOOLEAN, TIMESTAMPTZ) FROM anon;
REVOKE ALL ON FUNCTION public.finalize_global_publication(TEXT, TEXT, BIGINT, BIGINT, BIGINT, BOOLEAN, TIMESTAMPTZ) FROM authenticated;
GRANT EXECUTE ON FUNCTION public.finalize_global_publication(TEXT, TEXT, BIGINT, BIGINT, BIGINT, BOOLEAN, TIMESTAMPTZ) TO service_role;


-- ----------------------------------------------------------------------------
-- RPC F: release_global_reservation()
-- Releases unused RESERVED/RENEWED or definitively-rejected PUBLISHING slot.
-- Refuses to release AMBIGUOUS or COMMITTED states, or non-owner calls.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.release_global_reservation(
    p_reservation_id TEXT,
    p_owner_token TEXT,
    p_reason TEXT DEFAULT 'definitive_failure'
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
    v_now TIMESTAMPTZ := timezone('utc', now());
    v_res_id TEXT := BTRIM(COALESCE(p_reservation_id, ''));
    v_owner TEXT := BTRIM(COALESCE(p_owner_token, ''));
    v_cur RECORD;
    v_payload JSONB;
BEGIN
    PERFORM pg_advisory_xact_lock(8472910348291042);

    SELECT * INTO v_cur
    FROM public._kgn_reconstruct_global_reservations(v_now) r
    WHERE r.reservation_id = v_res_id
    LIMIT 1;

    IF NOT FOUND THEN
        RETURN jsonb_build_object('ok', FALSE, 'state', 'NOT_FOUND', 'reason', 'reservation_not_found');
    END IF;

    IF v_cur.owner_token IS DISTINCT FROM v_owner THEN
        RETURN jsonb_build_object('ok', FALSE, 'state', v_cur.state, 'reason', 'owner_mismatch');
    END IF;

    IF v_cur.state IN ('RELEASED', 'EXPIRED') THEN
        RETURN jsonb_build_object('ok', TRUE, 'state', v_cur.state, 'reason', 'already_released');
    END IF;

    IF v_cur.state IN ('COMMITTED', 'AMBIGUOUS') THEN
        RETURN jsonb_build_object(
            'ok', FALSE, 'state', v_cur.state,
            'reason', 'cannot_release_' || LOWER(v_cur.state)
        );
    END IF;

    v_payload := jsonb_build_object(
        'v', 1,
        'reservation_id', v_cur.reservation_id,
        'owner_token', v_cur.owner_token,
        'canonical_url', v_cur.canonical_url,
        'url_hash', v_cur.url_hash,
        'article_id', v_cur.article_id,
        'batch_id', v_cur.batch_id,
        'state', 'RELEASED',
        'lease_seconds', v_cur.lease_seconds,
        'expires_at', to_char(v_now AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
        'telegram_message_id', NULL,
        'reason', LEFT(COALESCE(p_reason, 'released'), 250)
    );
    INSERT INTO public.system_logs (timestamp, level, component, message)
    VALUES (v_now, 'RELEASED', 'GLOBAL_RESERVATION', v_payload::TEXT);

    RETURN jsonb_build_object('ok', TRUE, 'state', 'RELEASED', 'reservation_id', v_cur.reservation_id, 'reason', 'released');
END;
$$;

REVOKE ALL ON FUNCTION public.release_global_reservation(TEXT, TEXT, TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.release_global_reservation(TEXT, TEXT, TEXT) FROM anon;
REVOKE ALL ON FUNCTION public.release_global_reservation(TEXT, TEXT, TEXT) FROM authenticated;
GRANT EXECUTE ON FUNCTION public.release_global_reservation(TEXT, TEXT, TEXT) TO service_role;


-- ----------------------------------------------------------------------------
-- RPC F2: mark_global_ambiguous()
-- Keeps reservation counted in Global quota after uncertain Telegram outcome.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.mark_global_ambiguous(
    p_reservation_id TEXT,
    p_owner_token TEXT,
    p_reason TEXT DEFAULT 'uncertain_telegram_outcome',
    p_article_id BIGINT DEFAULT NULL
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
    v_now TIMESTAMPTZ := timezone('utc', now());
    v_res_id TEXT := BTRIM(COALESCE(p_reservation_id, ''));
    v_owner TEXT := BTRIM(COALESCE(p_owner_token, ''));
    v_cur RECORD;
    v_target_art_id BIGINT;
    v_payload JSONB;
BEGIN
    PERFORM pg_advisory_xact_lock(8472910348291042);

    SELECT * INTO v_cur
    FROM public._kgn_reconstruct_global_reservations(v_now) r
    WHERE r.reservation_id = v_res_id
    LIMIT 1;

    IF NOT FOUND THEN
        RETURN jsonb_build_object('ok', FALSE, 'state', 'NOT_FOUND', 'reason', 'reservation_not_found');
    END IF;

    IF v_cur.owner_token IS DISTINCT FROM v_owner THEN
        RETURN jsonb_build_object('ok', FALSE, 'state', v_cur.state, 'reason', 'owner_mismatch');
    END IF;

    IF v_cur.state = 'AMBIGUOUS' THEN
        RETURN jsonb_build_object('ok', TRUE, 'state', 'AMBIGUOUS', 'reason', 'already_ambiguous');
    END IF;

    IF v_cur.state = 'COMMITTED' THEN
        RETURN jsonb_build_object('ok', FALSE, 'state', 'COMMITTED', 'reason', 'already_committed');
    END IF;

    v_target_art_id := COALESCE(p_article_id, v_cur.article_id);
    IF v_target_art_id IS NOT NULL THEN
        UPDATE public.articles
        SET error_message = LEFT('AMBIGUOUS [Global Delivery Uncertain]: ' || COALESCE(p_reason, ''), 500)
        WHERE id = v_target_art_id
          AND status <> 'PUBLISHED';
    END IF;

    v_payload := jsonb_build_object(
        'v', 1,
        'reservation_id', v_cur.reservation_id,
        'owner_token', v_cur.owner_token,
        'canonical_url', v_cur.canonical_url,
        'url_hash', v_cur.url_hash,
        'article_id', v_target_art_id,
        'batch_id', v_cur.batch_id,
        'state', 'AMBIGUOUS',
        'lease_seconds', v_cur.lease_seconds,
        'expires_at', to_char((v_now + INTERVAL '24 hours') AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
        'telegram_message_id', NULL,
        'reason', LEFT(COALESCE(p_reason, 'ambiguous'), 250)
    );
    INSERT INTO public.system_logs (timestamp, level, component, message)
    VALUES (v_now, 'AMBIGUOUS', 'GLOBAL_RESERVATION', v_payload::TEXT);

    RETURN jsonb_build_object(
        'ok', TRUE, 'state', 'AMBIGUOUS',
        'reservation_id', v_cur.reservation_id, 'reason', 'marked_ambiguous'
    );
END;
$$;

REVOKE ALL ON FUNCTION public.mark_global_ambiguous(TEXT, TEXT, TEXT, BIGINT) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.mark_global_ambiguous(TEXT, TEXT, TEXT, BIGINT) FROM anon;
REVOKE ALL ON FUNCTION public.mark_global_ambiguous(TEXT, TEXT, TEXT, BIGINT) FROM authenticated;
GRANT EXECUTE ON FUNCTION public.mark_global_ambiguous(TEXT, TEXT, TEXT, BIGINT) TO service_role;


-- ----------------------------------------------------------------------------
-- RPC G: reconcile_ambiguous_publication()
-- 1. p_outcome = 'auto': transitions stale expired PUBLISHING -> AMBIGUOUS
--    (fail-closed, stays counted in quota) OR promotes already-PUBLISHED
--    articles -> COMMITTED. Never auto-releases capacity.
-- 2. p_outcome = 'confirmed_published': atomically finalizes with tg_msg_id.
-- 3. p_outcome = 'confirmed_not_published': releases slot after verification.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.reconcile_ambiguous_publication(
    p_reservation_id TEXT DEFAULT NULL,
    p_owner_token TEXT DEFAULT NULL,
    p_outcome TEXT DEFAULT 'auto',
    p_telegram_message_id BIGINT DEFAULT NULL,
    p_reason TEXT DEFAULT 'reconciled'
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
    v_now TIMESTAMPTZ := timezone('utc', now());
    v_res_id TEXT := NULLIF(BTRIM(COALESCE(p_reservation_id, '')), '');
    v_owner TEXT := NULLIF(BTRIM(COALESCE(p_owner_token, '')), '');
    v_outcome TEXT := LOWER(BTRIM(COALESCE(p_outcome, 'auto')));
    v_cur RECORD;
    v_art RECORD;
    v_reconciled JSONB := '[]'::JSONB;
    v_ambig_count INTEGER := 0;
    v_payload JSONB;
BEGIN
    PERFORM pg_advisory_xact_lock(8472910348291042);

    IF v_outcome = 'auto' THEN
        FOR v_cur IN
            SELECT * FROM public._kgn_reconstruct_global_reservations(v_now) r
            WHERE (v_res_id IS NULL OR r.reservation_id = v_res_id)
              AND r.state IN ('PUBLISHING', 'AMBIGUOUS')
              AND r.state_at >= (v_now - INTERVAL '24 hours')
        LOOP
            SELECT a.id, a.status, a.published_at
            INTO v_art
            FROM public.articles a
            WHERE (v_cur.article_id IS NOT NULL AND a.id = v_cur.article_id)
               OR (v_cur.canonical_url <> '' AND a.canonical_url = v_cur.canonical_url)
            LIMIT 1;

            IF FOUND AND v_art.status = 'PUBLISHED' THEN
                v_payload := jsonb_build_object(
                    'v', 1, 'reservation_id', v_cur.reservation_id, 'owner_token', v_cur.owner_token,
                    'canonical_url', v_cur.canonical_url, 'url_hash', v_cur.url_hash,
                    'article_id', v_art.id, 'batch_id', v_cur.batch_id,
                    'state', 'COMMITTED', 'lease_seconds', v_cur.lease_seconds,
                    'expires_at', to_char(v_now AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
                    'telegram_message_id', v_cur.telegram_message_id, 'reason', 'auto_reconciled_already_published'
                );
                INSERT INTO public.system_logs (timestamp, level, component, message)
                VALUES (v_now, 'COMMITTED', 'GLOBAL_RESERVATION', v_payload::TEXT);
                v_reconciled := v_reconciled || jsonb_build_array(jsonb_build_object('reservation_id', v_cur.reservation_id, 'action', 'COMMITTED'));
            ELSIF v_cur.state = 'PUBLISHING' AND v_cur.expires_at <= v_now THEN
                v_payload := jsonb_build_object(
                    'v', 1, 'reservation_id', v_cur.reservation_id, 'owner_token', v_cur.owner_token,
                    'canonical_url', v_cur.canonical_url, 'url_hash', v_cur.url_hash,
                    'article_id', v_cur.article_id, 'batch_id', v_cur.batch_id,
                    'state', 'AMBIGUOUS', 'lease_seconds', v_cur.lease_seconds,
                    'expires_at', to_char((v_now + INTERVAL '24 hours') AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
                    'telegram_message_id', NULL, 'reason', 'stale_publishing_lease_fail_closed'
                );
                INSERT INTO public.system_logs (timestamp, level, component, message)
                VALUES (v_now, 'AMBIGUOUS', 'GLOBAL_RESERVATION', v_payload::TEXT);
                v_reconciled := v_reconciled || jsonb_build_array(jsonb_build_object('reservation_id', v_cur.reservation_id, 'action', 'AMBIGUOUS'));
                v_ambig_count := v_ambig_count + 1;
            ELSIF v_cur.state = 'AMBIGUOUS' THEN
                v_ambig_count := v_ambig_count + 1;
            END IF;
        END LOOP;

        RETURN jsonb_build_object(
            'ok', TRUE,
            'reconciled', v_reconciled,
            'ambiguous_count', v_ambig_count,
            'reason', 'auto_reconciled'
        );
    END IF;

    -- Explicit resolution requires reservation_id and matching owner_token (when set)
    IF v_res_id IS NULL THEN
        RETURN jsonb_build_object('ok', FALSE, 'reason', 'reservation_id_required');
    END IF;

    SELECT * INTO v_cur
    FROM public._kgn_reconstruct_global_reservations(v_now) r
    WHERE r.reservation_id = v_res_id
    LIMIT 1;

    IF NOT FOUND THEN
        RETURN jsonb_build_object('ok', FALSE, 'reason', 'reservation_not_found');
    END IF;

    IF v_cur.owner_token IS NOT NULL AND v_owner IS DISTINCT FROM v_cur.owner_token THEN
        RETURN jsonb_build_object('ok', FALSE, 'state', v_cur.state, 'reason', 'owner_mismatch');
    END IF;

    IF v_outcome = 'confirmed_published' THEN
        IF p_telegram_message_id IS NULL THEN
            RETURN jsonb_build_object('ok', FALSE, 'reason', 'telegram_message_id_required');
        END IF;
        RETURN public.finalize_global_publication(
            v_cur.reservation_id,
            COALESCE(v_cur.owner_token, ''),
            v_cur.article_id,
            p_telegram_message_id,
            NULL,
            FALSE,
            v_now
        );
    ELSIF v_outcome = 'confirmed_not_published' THEN
        v_payload := jsonb_build_object(
            'v', 1, 'reservation_id', v_cur.reservation_id, 'owner_token', v_cur.owner_token,
            'canonical_url', v_cur.canonical_url, 'url_hash', v_cur.url_hash,
            'article_id', v_cur.article_id, 'batch_id', v_cur.batch_id,
            'state', 'RELEASED', 'lease_seconds', v_cur.lease_seconds,
            'expires_at', to_char(v_now AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
            'telegram_message_id', NULL, 'reason', LEFT(COALESCE(p_reason, 'confirmed_not_published'), 250)
        );
        INSERT INTO public.system_logs (timestamp, level, component, message)
        VALUES (v_now, 'RELEASED', 'GLOBAL_RESERVATION', v_payload::TEXT);
        RETURN jsonb_build_object('ok', TRUE, 'state', 'RELEASED', 'reservation_id', v_cur.reservation_id, 'reason', 'confirmed_not_published');
    END IF;

    RETURN jsonb_build_object('ok', FALSE, 'reason', 'invalid_outcome');
END;
$$;

REVOKE ALL ON FUNCTION public.reconcile_ambiguous_publication(TEXT, TEXT, TEXT, BIGINT, TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.reconcile_ambiguous_publication(TEXT, TEXT, TEXT, BIGINT, TEXT) FROM anon;
REVOKE ALL ON FUNCTION public.reconcile_ambiguous_publication(TEXT, TEXT, TEXT, BIGINT, TEXT) FROM authenticated;
GRANT EXECUTE ON FUNCTION public.reconcile_ambiguous_publication(TEXT, TEXT, TEXT, BIGINT, TEXT) TO service_role;


-- ============================================================================
-- MANUAL DEPLOYMENT & VERIFICATION INSTRUCTIONS (FOR ADMINISTRATOR)
-- ============================================================================
-- 1. Open Supabase Dashboard -> SQL Editor -> New Query.
-- 2. Paste this entire file (`storage/sql/global_admission_rpc.sql`) and Run.
-- 3. Verify functions and privilege grants by running:
--
--    SELECT p.proname,
--           pg_get_function_identity_arguments(p.oid) AS args,
--           has_function_privilege('anon', p.oid, 'EXECUTE') AS anon_exec,
--           has_function_privilege('authenticated', p.oid, 'EXECUTE') AS auth_exec,
--           has_function_privilege('service_role', p.oid, 'EXECUTE') AS service_exec
--    FROM pg_proc p
--    JOIN pg_namespace n ON n.oid = p.pronamespace
--    WHERE n.nspname = 'public'
--      AND (p.proname LIKE '%global%' OR p.proname LIKE '\_kgn\_%')
--    ORDER BY p.proname;
--
--    Expected output: anon_exec = false, auth_exec = false, service_exec = true!
--
-- 4. Smoke-test read-only status RPC:
--    SELECT public.get_global_quota_status(4);
--
-- ============================================================================
-- ROLLBACK / REMOVAL INSTRUCTIONS
-- ============================================================================
-- If you ever need to roll back these RPCs cleanly:
--   DROP FUNCTION IF EXISTS public.reconcile_ambiguous_publication(TEXT, TEXT, TEXT, BIGINT, TEXT);
--   DROP FUNCTION IF EXISTS public.mark_global_ambiguous(TEXT, TEXT, TEXT, BIGINT);
--   DROP FUNCTION IF EXISTS public.release_global_reservation(TEXT, TEXT, TEXT);
--   DROP FUNCTION IF EXISTS public.finalize_global_publication(TEXT, TEXT, BIGINT, BIGINT, BIGINT, BOOLEAN, TIMESTAMPTZ);
--   DROP FUNCTION IF EXISTS public.mark_global_publishing(TEXT, TEXT, INTEGER, INTEGER);
--   DROP FUNCTION IF EXISTS public.validate_global_reservation(TEXT, TEXT, BOOLEAN, INTEGER);
--   DROP FUNCTION IF EXISTS public.renew_global_reservation(TEXT, TEXT, INTEGER);
--   DROP FUNCTION IF EXISTS public.reserve_global_slot(TEXT, TEXT, TEXT, BIGINT, TEXT, INTEGER, TEXT, INTEGER, INTEGER);
--   DROP FUNCTION IF EXISTS public.get_global_quota_status(INTEGER);
--   DROP FUNCTION IF EXISTS public._kgn_reconstruct_global_reservations(TIMESTAMPTZ);
--   DROP FUNCTION IF EXISTS public._kgn_try_parse_jsonb(TEXT);
--
-- ============================================================================
-- OPTIONAL FUTURE SCHEMA PROPOSAL (NOT PART OF CURRENT SCHEMA — DO NOT APPLY)
-- ============================================================================
-- Reusing `system_logs` avoids altering schema while staying 100% atomic under
-- `pg_advisory_xact_lock`. For higher write throughput or dedicated operational
-- auditing later, an explicit schema migration could introduce a dedicated
-- `global_reservations` table:
--
--   CREATE TABLE IF NOT EXISTS public.global_reservations (
--       reservation_id UUID PRIMARY KEY,
--       owner_token TEXT NOT NULL,
--       canonical_url TEXT NOT NULL,
--       url_hash TEXT NOT NULL,
--       article_id BIGINT REFERENCES public.articles(id),
--       batch_id TEXT NOT NULL DEFAULT '',
--       state TEXT NOT NULL CHECK (state IN (
--           'RESERVED','RENEWED','PUBLISHING','COMMITTED','RELEASED','AMBIGUOUS','EXPIRED'
--       )),
--       reserved_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
--       expires_at TIMESTAMPTZ NOT NULL,
--       updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
--       telegram_message_id BIGINT,
--       reason TEXT NOT NULL DEFAULT ''
--   );
--   CREATE UNIQUE INDEX uq_active_global_url ON public.global_reservations (canonical_url)
--       WHERE state IN ('RESERVED', 'RENEWED', 'PUBLISHING', 'AMBIGUOUS');
-- ============================================================================
