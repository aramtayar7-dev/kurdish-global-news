-- ============================================================================
-- Kurdish Global News Bot — Staging Global Admission RPC Deployment (02)
-- Target Supabase Project: kurdish-global-news-staging
-- Status: REVIEWABLE STAGING ARTIFACT (DO NOT AUTO-EXECUTE)
--
-- Prerequisites:
--   Run `staging_01_schema.sql` FIRST. These RPCs depend on:
--     - `public.articles` (32 columns including id, canonical_url, region,
--       status, published_at, telegram_message_id, thread_root_message_id,
--       is_breaking, error_message)
--     - `public.system_logs` (id IDENTITY PK, timestamp, level, component, message)
--     - `REVOKE CREATE ON SCHEMA public FROM PUBLIC, anon, authenticated`
--     - `service_role` privileges on `public.*`
--
-- Deployment Order:
--   1. staging_01_schema.sql
--   2. staging_02_global_admission_rpc.sql  (THIS FILE)
--   3. staging_03_verify.sql
-- ============================================================================


-- ----------------------------------------------------------------------------
-- Internal Helper 1: Safe JSONB object parser (never throws on malformed text)
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public._kgn_try_parse_jsonb(p_raw TEXT)
RETURNS JSONB
LANGUAGE plpgsql
IMMUTABLE
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_trimmed TEXT;
    v_parsed JSONB;
BEGIN
    v_trimmed := pg_catalog.btrim(COALESCE(p_raw, ''));
    IF pg_catalog.left(v_trimmed, 1) <> '{' THEN
        RETURN NULL;
    END IF;
    v_parsed := v_trimmed::JSONB;
    IF pg_catalog.jsonb_typeof(v_parsed) <> 'object' THEN
        RETURN NULL;
    END IF;
    RETURN v_parsed;
EXCEPTION WHEN OTHERS THEN
    RETURN NULL;
END;
$$;

REVOKE ALL ON FUNCTION public._kgn_try_parse_jsonb(TEXT) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public._kgn_try_parse_jsonb(TEXT) TO service_role;


-- ----------------------------------------------------------------------------
-- Internal Helper 2: Safe scalar parsers (never throw on malformed JSON fields)
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public._kgn_try_parse_timestamptz(p_raw TEXT)
RETURNS TIMESTAMPTZ
LANGUAGE plpgsql
IMMUTABLE
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_trimmed TEXT;
BEGIN
    v_trimmed := pg_catalog.btrim(COALESCE(p_raw, ''));
    IF v_trimmed = '' THEN
        RETURN NULL;
    END IF;
    RETURN v_trimmed::TIMESTAMPTZ;
EXCEPTION WHEN OTHERS THEN
    RETURN NULL;
END;
$$;

REVOKE ALL ON FUNCTION public._kgn_try_parse_timestamptz(TEXT) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public._kgn_try_parse_timestamptz(TEXT) TO service_role;


CREATE OR REPLACE FUNCTION public._kgn_try_parse_bigint(p_raw TEXT)
RETURNS BIGINT
LANGUAGE plpgsql
IMMUTABLE
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_trimmed TEXT;
BEGIN
    v_trimmed := pg_catalog.btrim(COALESCE(p_raw, ''));
    IF v_trimmed !~ '^\d{1,18}$' THEN
        RETURN NULL;
    END IF;
    RETURN v_trimmed::BIGINT;
EXCEPTION WHEN OTHERS THEN
    RETURN NULL;
END;
$$;

REVOKE ALL ON FUNCTION public._kgn_try_parse_bigint(TEXT) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public._kgn_try_parse_bigint(TEXT) TO service_role;


CREATE OR REPLACE FUNCTION public._kgn_try_parse_integer(p_raw TEXT)
RETURNS INTEGER
LANGUAGE plpgsql
IMMUTABLE
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_trimmed TEXT;
BEGIN
    v_trimmed := pg_catalog.btrim(COALESCE(p_raw, ''));
    IF v_trimmed !~ '^\d{1,9}$' THEN
        RETURN NULL;
    END IF;
    RETURN v_trimmed::INTEGER;
EXCEPTION WHEN OTHERS THEN
    RETURN NULL;
END;
$$;

REVOKE ALL ON FUNCTION public._kgn_try_parse_integer(TEXT) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public._kgn_try_parse_integer(TEXT) TO service_role;


-- ----------------------------------------------------------------------------
-- Internal Helper 3: Deterministic reservation ledger reconstruction (`id ASC`)
-- Note: Malformed JSON or legacy rows fail CLOSED (counted toward quota within
--       24h unless explicitly terminated by RELEASED/EXPIRED/COMMITTED).
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public._kgn_reconstruct_global_reservations(
    p_now TIMESTAMPTZ DEFAULT pg_catalog.now()
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
SET search_path = pg_catalog, public
AS $$
WITH raw_events AS (
    SELECT
        sl.id,
        sl.timestamp AS row_ts,
        pg_catalog.upper(pg_catalog.btrim(COALESCE(sl.level, ''))) AS row_level,
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
        CASE
            WHEN e.payload IS NOT NULL AND pg_catalog.btrim(COALESCE(e.payload->>'reservation_id', '')) <> ''
                THEN pg_catalog.btrim(e.payload->>'reservation_id')
            WHEN e.payload IS NOT NULL
                THEN 'malformed:' || e.id::TEXT
            ELSE 'legacy:' || e.raw_msg
        END AS reservation_id,
        CASE
            WHEN e.payload IS NOT NULL THEN NULLIF(pg_catalog.btrim(COALESCE(e.payload->>'owner_token', '')), '')
            ELSE NULL
        END AS owner_token,
        CASE
            WHEN e.payload IS NOT NULL THEN pg_catalog.btrim(COALESCE(e.payload->>'canonical_url', ''))
            ELSE pg_catalog.btrim(e.raw_msg)
        END AS canonical_url,
        CASE
            WHEN e.payload IS NOT NULL THEN pg_catalog.btrim(COALESCE(e.payload->>'url_hash', ''))
            ELSE ''
        END AS url_hash,
        CASE
            WHEN e.payload IS NOT NULL THEN public._kgn_try_parse_bigint(e.payload->>'article_id')
            ELSE NULL
        END AS article_id,
        CASE
            WHEN e.payload IS NOT NULL THEN pg_catalog.btrim(COALESCE(e.payload->>'batch_id', ''))
            ELSE ''
        END AS batch_id,
        pg_catalog.upper(pg_catalog.btrim(COALESCE(
            CASE WHEN e.payload IS NOT NULL THEN NULLIF(pg_catalog.btrim(COALESCE(e.payload->>'state', '')), '') ELSE NULL END,
            NULLIF(e.row_level, ''),
            'RESERVED'
        ))) AS state,
        (e.payload IS NULL) AS is_legacy,
        pg_catalog.greatest(
            30,
            pg_catalog.least(
                900,
                COALESCE(
                    CASE WHEN e.payload IS NOT NULL THEN public._kgn_try_parse_integer(e.payload->>'lease_seconds') ELSE NULL END,
                    900
                )
            )
        ) AS lease_seconds,
        CASE
            WHEN e.payload IS NOT NULL AND public._kgn_try_parse_timestamptz(e.payload->>'expires_at') IS NOT NULL
                THEN public._kgn_try_parse_timestamptz(e.payload->>'expires_at')
            WHEN e.payload IS NOT NULL
                THEN e.row_ts + pg_catalog.make_interval(secs => pg_catalog.greatest(30, pg_catalog.least(900, COALESCE(public._kgn_try_parse_integer(e.payload->>'lease_seconds'), 180))))
            ELSE e.row_ts + INTERVAL '15 minutes'
        END AS expires_at,
        CASE
            WHEN e.payload IS NOT NULL THEN public._kgn_try_parse_bigint(e.payload->>'telegram_message_id')
            ELSE NULL
        END AS telegram_message_id,
        CASE
            WHEN e.payload IS NOT NULL THEN COALESCE(e.payload->>'reason', '')
            ELSE ''
        END AS reason
    FROM raw_events e
),
grouped AS (
    SELECT
        p.*,
        MIN(p.row_ts) FILTER (WHERE p.state = 'RESERVED') OVER (PARTITION BY p.reservation_id) AS first_reserved_at,
        ROW_NUMBER() OVER (PARTITION BY p.reservation_id ORDER BY p.id DESC) AS rn_desc
    FROM parsed p
),
latest AS (
    SELECT
        g.reservation_id,
        g.owner_token,
        g.canonical_url,
        g.url_hash,
        g.article_id,
        g.batch_id,
        g.state,
        g.is_legacy,
        COALESCE(g.first_reserved_at, g.row_ts) AS reserved_at,
        g.row_ts AS state_at,
        g.expires_at,
        g.lease_seconds,
        g.telegram_message_id,
        g.reason
    FROM grouped g
    WHERE g.rn_desc = 1
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
        OR (
            -- Fail closed on unrecognized/malformed states within 24h window
            l.state NOT IN ('RESERVED', 'RENEWED', 'PUBLISHING', 'AMBIGUOUS', 'COMMITTED', 'RELEASED', 'EXPIRED')
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

REVOKE ALL ON FUNCTION public._kgn_reconstruct_global_reservations(TIMESTAMPTZ) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public._kgn_reconstruct_global_reservations(TIMESTAMPTZ) TO service_role;


-- ----------------------------------------------------------------------------
-- RPC 1: get_global_quota_status()
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.get_global_quota_status(
    p_max_daily_global INTEGER DEFAULT 4
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_now TIMESTAMPTZ := pg_catalog.now();
    v_cutoff TIMESTAMPTZ := v_now - INTERVAL '24 hours';
    v_max_daily INTEGER := pg_catalog.greatest(0, pg_catalog.least(4, COALESCE(p_max_daily_global, 4)));
    v_global_24h INTEGER := 0;
    v_total_24h INTEGER := 0;
    v_active_res INTEGER := 0;
    v_ambiguous_res INTEGER := 0;
    v_active_urls JSONB := '[]'::JSONB;
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(8472910348291042);

    SELECT
        COALESCE(COUNT(DISTINCT COALESCE(NULLIF(pg_catalog.btrim(a.canonical_url), ''), 'id:' || a.id::TEXT))
            FILTER (WHERE pg_catalog.lower(pg_catalog.btrim(COALESCE(a.region, ''))) = 'global'), 0)::INTEGER,
        COALESCE(COUNT(DISTINCT COALESCE(NULLIF(pg_catalog.btrim(a.canonical_url), ''), 'id:' || a.id::TEXT)), 0)::INTEGER
    INTO v_global_24h, v_total_24h
    FROM public.articles a
    WHERE a.status = 'PUBLISHED'
      AND a.published_at IS NOT NULL
      AND a.published_at >= v_cutoff;

    SELECT
        COALESCE(COUNT(*) FILTER (WHERE r.counts_for_quota), 0)::INTEGER,
        COALESCE(COUNT(*) FILTER (WHERE r.counts_for_quota AND r.state = 'AMBIGUOUS'), 0)::INTEGER,
        COALESCE(
            pg_catalog.jsonb_agg(r.canonical_url) FILTER (WHERE r.counts_for_quota AND r.canonical_url <> ''),
            '[]'::JSONB
        )
    INTO v_active_res, v_ambiguous_res, v_active_urls
    FROM public._kgn_reconstruct_global_reservations(v_now) r;

    RETURN pg_catalog.jsonb_build_object(
        'ok', TRUE,
        'now_utc', pg_catalog.to_char(v_now AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
        'published_global_24h', v_global_24h,
        'published_total_24h', v_total_24h,
        'active_reservations', v_active_res,
        'ambiguous_reservations', v_ambiguous_res,
        'active_urls', v_active_urls,
        'max_daily_global', v_max_daily
    );
END;
$$;

REVOKE ALL ON FUNCTION public.get_global_quota_status(INTEGER) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.get_global_quota_status(INTEGER) TO service_role;


-- ----------------------------------------------------------------------------
-- RPC 2: reserve_global_slot()
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
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_now TIMESTAMPTZ := pg_catalog.now();
    v_cutoff TIMESTAMPTZ := v_now - INTERVAL '24 hours';
    v_url TEXT := pg_catalog.btrim(COALESCE(p_canonical_url, ''));
    v_owner TEXT := pg_catalog.btrim(COALESCE(p_owner_token, ''));
    v_res_id TEXT := NULLIF(pg_catalog.btrim(COALESCE(p_reservation_id, '')), '');
    v_batch TEXT := pg_catalog.btrim(COALESCE(p_batch_id, ''));
    v_hash TEXT := pg_catalog.btrim(COALESCE(p_url_hash, ''));
    v_lease INTEGER := pg_catalog.greatest(30, pg_catalog.least(900, COALESCE(p_lease_seconds, 180)));
    v_max_daily INTEGER := pg_catalog.greatest(0, pg_catalog.least(4, COALESCE(p_max_daily_global, 4)));
    v_max_batch INTEGER := pg_catalog.greatest(0, pg_catalog.least(1, COALESCE(p_max_batch_global, 1)));
    v_expires TIMESTAMPTZ := v_now + pg_catalog.make_interval(secs => v_lease);
    v_global_24h INTEGER := 0;
    v_total_24h INTEGER := 0;
    v_active_other INTEGER := 0;
    v_batch_used INTEGER := 0;
    v_already_pub BOOLEAN := FALSE;
    v_existing RECORD;
    v_dup RECORD;
    v_payload JSONB;
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(8472910348291042);

    IF v_owner = '' THEN
        RETURN pg_catalog.jsonb_build_object('admitted', FALSE, 'state', 'REJECTED', 'reason', 'missing_owner_token');
    END IF;

    IF v_url = '' AND p_article_id IS NULL THEN
        RETURN pg_catalog.jsonb_build_object('admitted', FALSE, 'state', 'REJECTED', 'reason', 'missing_article_identity');
    END IF;

    IF v_res_id IS NULL THEN
        v_res_id := pg_catalog.md5(pg_catalog.random()::TEXT || pg_catalog.clock_timestamp()::TEXT || v_url || v_owner);
    END IF;

    SELECT * INTO v_existing
    FROM public._kgn_reconstruct_global_reservations(v_now) r
    WHERE r.reservation_id = v_res_id
    LIMIT 1;

    IF FOUND THEN
        IF v_existing.owner_token IS DISTINCT FROM v_owner THEN
            RETURN pg_catalog.jsonb_build_object(
                'admitted', FALSE, 'state', v_existing.state,
                'reservation_id', v_res_id, 'reason', 'reservation_id_owner_mismatch'
            );
        END IF;

        IF v_existing.counts_for_quota AND v_existing.state IN ('RESERVED', 'RENEWED') THEN
            RETURN pg_catalog.jsonb_build_object(
                'admitted', TRUE, 'state', v_existing.state,
                'reservation_id', v_existing.reservation_id, 'owner_token', v_existing.owner_token,
                'expires_at', pg_catalog.to_char(v_existing.expires_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
                'reason', 'idempotent_existing_reservation'
            );
        ELSIF NOT (v_res_id LIKE 'legacy_url:%' AND v_existing.state IN ('RELEASED', 'EXPIRED')) THEN
            RETURN pg_catalog.jsonb_build_object(
                'admitted', FALSE, 'state', v_existing.state,
                'reservation_id', v_existing.reservation_id,
                'reason', 'reservation_not_active_' || pg_catalog.lower(v_existing.state)
            );
        END IF;
    END IF;

    SELECT EXISTS (
        SELECT 1 FROM public.articles a
        WHERE a.status = 'PUBLISHED'
          AND ((p_article_id IS NOT NULL AND a.id = p_article_id) OR (v_url <> '' AND a.canonical_url = v_url))
    ) INTO v_already_pub;

    IF v_already_pub THEN
        RETURN pg_catalog.jsonb_build_object('admitted', FALSE, 'state', 'REJECTED', 'reason', 'article_already_published');
    END IF;

    SELECT * INTO v_dup
    FROM public._kgn_reconstruct_global_reservations(v_now) r
    WHERE r.counts_for_quota
      AND ((v_url <> '' AND r.canonical_url = v_url) OR (v_hash <> '' AND r.url_hash = v_hash) OR (p_article_id IS NOT NULL AND r.article_id = p_article_id))
    LIMIT 1;

    IF FOUND THEN
        IF v_dup.owner_token = v_owner AND v_dup.state IN ('RESERVED', 'RENEWED') THEN
            RETURN pg_catalog.jsonb_build_object(
                'admitted', TRUE, 'state', v_dup.state,
                'reservation_id', v_dup.reservation_id, 'owner_token', v_dup.owner_token,
                'expires_at', pg_catalog.to_char(v_dup.expires_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
                'reason', 'idempotent_existing_article_reservation'
            );
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'admitted', FALSE, 'state', v_dup.state,
            'reservation_id', v_dup.reservation_id, 'reason', 'duplicate_active_reservation'
        );
    END IF;

    SELECT
        COALESCE(COUNT(DISTINCT COALESCE(NULLIF(pg_catalog.btrim(a.canonical_url), ''), 'id:' || a.id::TEXT))
            FILTER (WHERE pg_catalog.lower(pg_catalog.btrim(COALESCE(a.region, ''))) = 'global'), 0)::INTEGER,
        COALESCE(COUNT(DISTINCT COALESCE(NULLIF(pg_catalog.btrim(a.canonical_url), ''), 'id:' || a.id::TEXT)), 0)::INTEGER
    INTO v_global_24h, v_total_24h
    FROM public.articles a
    WHERE a.status = 'PUBLISHED'
      AND a.published_at IS NOT NULL
      AND a.published_at >= v_cutoff;

    SELECT
        COALESCE(COUNT(*) FILTER (WHERE r.counts_for_quota), 0)::INTEGER,
        COALESCE(COUNT(*) FILTER (WHERE r.counts_for_batch AND v_batch <> '' AND r.batch_id = v_batch), 0)::INTEGER
    INTO v_active_other, v_batch_used
    FROM public._kgn_reconstruct_global_reservations(v_now) r;

    IF v_batch <> '' AND v_batch_used >= v_max_batch THEN
        RETURN pg_catalog.jsonb_build_object(
            'admitted', FALSE, 'state', 'REJECTED',
            'published_global_24h', v_global_24h, 'published_total_24h', v_total_24h,
            'active_reservations', v_active_other, 'reason', 'batch_global_cap_reached'
        );
    END IF;

    IF (v_global_24h + v_active_other + 1) > v_max_daily THEN
        RETURN pg_catalog.jsonb_build_object(
            'admitted', FALSE, 'state', 'REJECTED',
            'published_global_24h', v_global_24h, 'published_total_24h', v_total_24h,
            'active_reservations', v_active_other, 'reason', 'daily_global_cap_reached'
        );
    END IF;

    IF 10 * (v_global_24h + v_active_other + 1) > (v_total_24h + v_active_other + 1) THEN
        RETURN pg_catalog.jsonb_build_object(
            'admitted', FALSE, 'state', 'REJECTED',
            'published_global_24h', v_global_24h, 'published_total_24h', v_total_24h,
            'active_reservations', v_active_other, 'reason', 'rolling_90_10_ratio_not_satisfied'
        );
    END IF;

    v_payload := pg_catalog.jsonb_build_object(
        'v', 1, 'reservation_id', v_res_id, 'owner_token', v_owner,
        'canonical_url', v_url, 'url_hash', v_hash, 'article_id', p_article_id,
        'batch_id', v_batch, 'state', 'RESERVED', 'lease_seconds', v_lease,
        'expires_at', pg_catalog.to_char(v_expires AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
        'telegram_message_id', NULL, 'reason', 'admitted'
    );
    INSERT INTO public.system_logs (timestamp, level, component, message)
    VALUES (v_now, 'RESERVED', 'GLOBAL_RESERVATION', v_payload::TEXT);

    RETURN pg_catalog.jsonb_build_object(
        'admitted', TRUE, 'state', 'RESERVED', 'reservation_id', v_res_id, 'owner_token', v_owner,
        'expires_at', pg_catalog.to_char(v_expires AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
        'published_global_24h', v_global_24h, 'published_total_24h', v_total_24h,
        'active_reservations', v_active_other, 'reason', 'admitted'
    );
END;
$$;

REVOKE ALL ON FUNCTION public.reserve_global_slot(TEXT, TEXT, TEXT, BIGINT, TEXT, INTEGER, TEXT, INTEGER, INTEGER) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.reserve_global_slot(TEXT, TEXT, TEXT, BIGINT, TEXT, INTEGER, TEXT, INTEGER, INTEGER) TO service_role;


-- ----------------------------------------------------------------------------
-- RPC 3: renew_global_reservation()
-- Valid source states: RESERVED (unexpired), RENEWED (unexpired), PUBLISHING.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.renew_global_reservation(
    p_reservation_id TEXT,
    p_owner_token TEXT,
    p_lease_seconds INTEGER DEFAULT 180
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_now TIMESTAMPTZ := pg_catalog.now();
    v_res_id TEXT := pg_catalog.btrim(COALESCE(p_reservation_id, ''));
    v_owner TEXT := pg_catalog.btrim(COALESCE(p_owner_token, ''));
    v_lease INTEGER := pg_catalog.greatest(30, pg_catalog.least(900, COALESCE(p_lease_seconds, 180)));
    v_expires TIMESTAMPTZ := v_now + pg_catalog.make_interval(secs => v_lease);
    v_cur RECORD;
    v_next_state TEXT;
    v_payload JSONB;
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(8472910348291042);

    SELECT * INTO v_cur
    FROM public._kgn_reconstruct_global_reservations(v_now) r
    WHERE r.reservation_id = v_res_id
    LIMIT 1;

    IF NOT FOUND THEN
        RETURN pg_catalog.jsonb_build_object('ok', FALSE, 'state', 'NOT_FOUND', 'reason', 'reservation_not_found');
    END IF;
    IF v_cur.owner_token IS DISTINCT FROM v_owner THEN
        RETURN pg_catalog.jsonb_build_object('ok', FALSE, 'state', v_cur.state, 'reason', 'owner_mismatch');
    END IF;

    IF v_cur.state IN ('RESERVED', 'RENEWED') AND v_cur.expires_at <= v_now THEN
        v_payload := pg_catalog.jsonb_build_object(
            'v', 1, 'reservation_id', v_cur.reservation_id, 'owner_token', v_cur.owner_token,
            'canonical_url', v_cur.canonical_url, 'url_hash', v_cur.url_hash,
            'article_id', v_cur.article_id, 'batch_id', v_cur.batch_id,
            'state', 'EXPIRED', 'lease_seconds', v_cur.lease_seconds,
            'expires_at', pg_catalog.to_char(v_cur.expires_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
            'telegram_message_id', NULL, 'reason', 'lease_expired_before_renewal'
        );
        INSERT INTO public.system_logs (timestamp, level, component, message)
        VALUES (v_now, 'EXPIRED', 'GLOBAL_RESERVATION', v_payload::TEXT);
        RETURN pg_catalog.jsonb_build_object('ok', FALSE, 'state', 'EXPIRED', 'reason', 'reservation_expired');
    END IF;

    IF v_cur.state NOT IN ('RESERVED', 'RENEWED', 'PUBLISHING') THEN
        RETURN pg_catalog.jsonb_build_object('ok', FALSE, 'state', v_cur.state, 'reason', 'invalid_state_' || pg_catalog.lower(v_cur.state));
    END IF;

    v_next_state := CASE WHEN v_cur.state = 'PUBLISHING' THEN 'PUBLISHING' ELSE 'RENEWED' END;
    v_payload := pg_catalog.jsonb_build_object(
        'v', 1, 'reservation_id', v_cur.reservation_id, 'owner_token', v_cur.owner_token,
        'canonical_url', v_cur.canonical_url, 'url_hash', v_cur.url_hash,
        'article_id', v_cur.article_id, 'batch_id', v_cur.batch_id,
        'state', v_next_state, 'lease_seconds', v_lease,
        'expires_at', pg_catalog.to_char(v_expires AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
        'telegram_message_id', NULL, 'reason', 'lease_renewed'
    );
    INSERT INTO public.system_logs (timestamp, level, component, message)
    VALUES (v_now, v_next_state, 'GLOBAL_RESERVATION', v_payload::TEXT);

    RETURN pg_catalog.jsonb_build_object(
        'ok', TRUE, 'state', v_next_state, 'reservation_id', v_cur.reservation_id,
        'expires_at', pg_catalog.to_char(v_expires AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
        'reason', 'renewed'
    );
END;
$$;

REVOKE ALL ON FUNCTION public.renew_global_reservation(TEXT, TEXT, INTEGER) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.renew_global_reservation(TEXT, TEXT, INTEGER) TO service_role;


-- ----------------------------------------------------------------------------
-- RPC 4: validate_global_reservation()
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
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_now TIMESTAMPTZ := pg_catalog.now();
    v_cutoff TIMESTAMPTZ := v_now - INTERVAL '24 hours';
    v_res_id TEXT := pg_catalog.btrim(COALESCE(p_reservation_id, ''));
    v_owner TEXT := pg_catalog.btrim(COALESCE(p_owner_token, ''));
    v_max_daily INTEGER := pg_catalog.greatest(0, pg_catalog.least(4, COALESCE(p_max_daily_global, 4)));
    v_cur RECORD;
    v_global_24h INTEGER := 0;
    v_total_24h INTEGER := 0;
    v_active_other INTEGER := 0;
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(8472910348291042);

    SELECT * INTO v_cur
    FROM public._kgn_reconstruct_global_reservations(v_now) r
    WHERE r.reservation_id = v_res_id
    LIMIT 1;

    IF NOT FOUND THEN
        RETURN pg_catalog.jsonb_build_object('valid', FALSE, 'state', 'NOT_FOUND', 'reason', 'reservation_not_found');
    END IF;
    IF v_cur.owner_token IS DISTINCT FROM v_owner THEN
        RETURN pg_catalog.jsonb_build_object('valid', FALSE, 'state', v_cur.state, 'reason', 'owner_mismatch');
    END IF;
    IF v_cur.state NOT IN ('RESERVED', 'RENEWED', 'PUBLISHING') OR NOT v_cur.counts_for_quota THEN
        RETURN pg_catalog.jsonb_build_object('valid', FALSE, 'state', v_cur.state, 'reason', 'reservation_inactive_or_expired');
    END IF;

    SELECT
        COALESCE(COUNT(DISTINCT COALESCE(NULLIF(pg_catalog.btrim(a.canonical_url), ''), 'id:' || a.id::TEXT))
            FILTER (WHERE pg_catalog.lower(pg_catalog.btrim(COALESCE(a.region, ''))) = 'global'), 0)::INTEGER,
        COALESCE(COUNT(DISTINCT COALESCE(NULLIF(pg_catalog.btrim(a.canonical_url), ''), 'id:' || a.id::TEXT)), 0)::INTEGER
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
            RETURN pg_catalog.jsonb_build_object(
                'valid', FALSE, 'state', v_cur.state,
                'published_global_24h', v_global_24h, 'published_total_24h', v_total_24h,
                'active_reservations_other', v_active_other, 'reason', 'daily_global_cap_reached'
            );
        END IF;
        IF 10 * (v_global_24h + v_active_other + 1) > (v_total_24h + v_active_other + 1) THEN
            RETURN pg_catalog.jsonb_build_object(
                'valid', FALSE, 'state', v_cur.state,
                'published_global_24h', v_global_24h, 'published_total_24h', v_total_24h,
                'active_reservations_other', v_active_other, 'reason', 'rolling_90_10_ratio_not_satisfied'
            );
        END IF;
    END IF;

    RETURN pg_catalog.jsonb_build_object(
        'valid', TRUE, 'state', v_cur.state,
        'published_global_24h', v_global_24h, 'published_total_24h', v_total_24h,
        'active_reservations_other', v_active_other, 'reason', 'valid'
    );
END;
$$;

REVOKE ALL ON FUNCTION public.validate_global_reservation(TEXT, TEXT, BOOLEAN, INTEGER) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.validate_global_reservation(TEXT, TEXT, BOOLEAN, INTEGER) TO service_role;


-- ----------------------------------------------------------------------------
-- RPC 5: mark_global_publishing()
-- Valid source states: unexpired RESERVED or RENEWED -> PUBLISHING (or idempotent PUBLISHING).
-- Explicitly blocks EXPIRED -> PUBLISHING, RELEASED -> PUBLISHING, AMBIGUOUS -> PUBLISHING.
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
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_now TIMESTAMPTZ := pg_catalog.now();
    v_cutoff TIMESTAMPTZ := v_now - INTERVAL '24 hours';
    v_res_id TEXT := pg_catalog.btrim(COALESCE(p_reservation_id, ''));
    v_owner TEXT := pg_catalog.btrim(COALESCE(p_owner_token, ''));
    v_lease INTEGER := pg_catalog.greatest(30, pg_catalog.least(900, COALESCE(p_lease_seconds, 180)));
    v_max_daily INTEGER := pg_catalog.greatest(0, pg_catalog.least(4, COALESCE(p_max_daily_global, 4)));
    v_expires TIMESTAMPTZ := v_now + pg_catalog.make_interval(secs => v_lease);
    v_cur RECORD;
    v_already_pub BOOLEAN := FALSE;
    v_global_24h INTEGER := 0;
    v_total_24h INTEGER := 0;
    v_active_other INTEGER := 0;
    v_payload JSONB;
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(8472910348291042);

    SELECT * INTO v_cur
    FROM public._kgn_reconstruct_global_reservations(v_now) r
    WHERE r.reservation_id = v_res_id
    LIMIT 1;

    IF NOT FOUND THEN
        RETURN pg_catalog.jsonb_build_object('ok', FALSE, 'admitted', FALSE, 'state', 'NOT_FOUND', 'reason', 'reservation_not_found');
    END IF;
    IF v_cur.owner_token IS DISTINCT FROM v_owner THEN
        RETURN pg_catalog.jsonb_build_object('ok', FALSE, 'admitted', FALSE, 'state', v_cur.state, 'reason', 'owner_mismatch');
    END IF;
    IF v_cur.state = 'PUBLISHING' THEN
        RETURN pg_catalog.jsonb_build_object(
            'ok', TRUE, 'admitted', TRUE, 'state', 'PUBLISHING',
            'reservation_id', v_cur.reservation_id, 'reason', 'already_publishing'
        );
    END IF;

    IF v_cur.state IN ('RESERVED', 'RENEWED') AND v_cur.expires_at <= v_now THEN
        v_payload := pg_catalog.jsonb_build_object(
            'v', 1, 'reservation_id', v_cur.reservation_id, 'owner_token', v_cur.owner_token,
            'canonical_url', v_cur.canonical_url, 'url_hash', v_cur.url_hash,
            'article_id', v_cur.article_id, 'batch_id', v_cur.batch_id,
            'state', 'EXPIRED', 'lease_seconds', v_cur.lease_seconds,
            'expires_at', pg_catalog.to_char(v_cur.expires_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
            'telegram_message_id', NULL, 'reason', 'lease_expired_before_publish'
        );
        INSERT INTO public.system_logs (timestamp, level, component, message)
        VALUES (v_now, 'EXPIRED', 'GLOBAL_RESERVATION', v_payload::TEXT);
        RETURN pg_catalog.jsonb_build_object('ok', FALSE, 'admitted', FALSE, 'state', 'EXPIRED', 'reason', 'reservation_expired');
    END IF;

    IF v_cur.state NOT IN ('RESERVED', 'RENEWED') THEN
        RETURN pg_catalog.jsonb_build_object(
            'ok', FALSE, 'admitted', FALSE, 'state', v_cur.state,
            'reason', 'invalid_state_' || pg_catalog.lower(v_cur.state)
        );
    END IF;

    SELECT EXISTS (
        SELECT 1 FROM public.articles a
        WHERE a.status = 'PUBLISHED'
          AND ((v_cur.article_id IS NOT NULL AND a.id = v_cur.article_id) OR (v_cur.canonical_url <> '' AND a.canonical_url = v_cur.canonical_url))
    ) INTO v_already_pub;

    IF v_already_pub THEN
        RETURN pg_catalog.jsonb_build_object('ok', FALSE, 'admitted', FALSE, 'state', v_cur.state, 'reason', 'article_already_published');
    END IF;

    SELECT
        COALESCE(COUNT(DISTINCT COALESCE(NULLIF(pg_catalog.btrim(a.canonical_url), ''), 'id:' || a.id::TEXT))
            FILTER (WHERE pg_catalog.lower(pg_catalog.btrim(COALESCE(a.region, ''))) = 'global'), 0)::INTEGER,
        COALESCE(COUNT(DISTINCT COALESCE(NULLIF(pg_catalog.btrim(a.canonical_url), ''), 'id:' || a.id::TEXT)), 0)::INTEGER
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
        v_payload := pg_catalog.jsonb_build_object(
            'v', 1, 'reservation_id', v_cur.reservation_id, 'owner_token', v_cur.owner_token,
            'canonical_url', v_cur.canonical_url, 'url_hash', v_cur.url_hash,
            'article_id', v_cur.article_id, 'batch_id', v_cur.batch_id,
            'state', 'RELEASED', 'lease_seconds', v_cur.lease_seconds,
            'expires_at', pg_catalog.to_char(v_cur.expires_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
            'telegram_message_id', NULL, 'reason', 'ratio_or_cap_failed_at_publish'
        );
        INSERT INTO public.system_logs (timestamp, level, component, message)
        VALUES (v_now, 'RELEASED', 'GLOBAL_RESERVATION', v_payload::TEXT);
        RETURN pg_catalog.jsonb_build_object(
            'ok', FALSE, 'admitted', FALSE, 'state', 'RELEASED',
            'published_global_24h', v_global_24h, 'published_total_24h', v_total_24h,
            'active_reservations_other', v_active_other, 'reason', 'ratio_or_cap_failed_at_publish'
        );
    END IF;

    v_payload := pg_catalog.jsonb_build_object(
        'v', 1, 'reservation_id', v_cur.reservation_id, 'owner_token', v_cur.owner_token,
        'canonical_url', v_cur.canonical_url, 'url_hash', v_cur.url_hash,
        'article_id', v_cur.article_id, 'batch_id', v_cur.batch_id,
        'state', 'PUBLISHING', 'lease_seconds', v_lease,
        'expires_at', pg_catalog.to_char(v_expires AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
        'telegram_message_id', NULL, 'reason', 'publishing_started'
    );
    INSERT INTO public.system_logs (timestamp, level, component, message)
    VALUES (v_now, 'PUBLISHING', 'GLOBAL_RESERVATION', v_payload::TEXT);

    RETURN pg_catalog.jsonb_build_object(
        'ok', TRUE, 'admitted', TRUE, 'state', 'PUBLISHING', 'reservation_id', v_cur.reservation_id,
        'expires_at', pg_catalog.to_char(v_expires AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
        'published_global_24h', v_global_24h, 'published_total_24h', v_total_24h, 'reason', 'publishing'
    );
END;
$$;

REVOKE ALL ON FUNCTION public.mark_global_publishing(TEXT, TEXT, INTEGER, INTEGER) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.mark_global_publishing(TEXT, TEXT, INTEGER, INTEGER) TO service_role;


-- ----------------------------------------------------------------------------
-- RPC 6: finalize_global_publication()
-- Hardened atomic finalization:
--   - Requires valid reservation, matching owner_token, and positive telegram_message_id.
--   - Rejects expired RESERVED/RENEWED, RELEASED, EXPIRED, or unknown states.
--   - Verifies target article exists (FOR UPDATE), matches reservation identity,
--     has region = 'global', and UPDATE affects exactly 1 row.
--   - Never logs COMMITTED if article verification or UPDATE fails.
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
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_now TIMESTAMPTZ := pg_catalog.now();
    v_pub_ts TIMESTAMPTZ := COALESCE(p_published_at, v_now);
    v_res_id TEXT := pg_catalog.btrim(COALESCE(p_reservation_id, ''));
    v_owner TEXT := NULLIF(pg_catalog.btrim(COALESCE(p_owner_token, '')), '');
    v_cur RECORD;
    v_art RECORD;
    v_target_art_id BIGINT;
    v_updated_rows INTEGER := 0;
    v_payload JSONB;
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(8472910348291042);

    IF v_res_id = '' THEN
        RETURN pg_catalog.jsonb_build_object('ok', FALSE, 'state', 'REJECTED', 'reason', 'missing_reservation_id');
    END IF;
    IF p_telegram_message_id IS NULL OR p_telegram_message_id <= 0 THEN
        RETURN pg_catalog.jsonb_build_object('ok', FALSE, 'state', 'REJECTED', 'reason', 'invalid_telegram_message_id');
    END IF;

    SELECT * INTO v_cur
    FROM public._kgn_reconstruct_global_reservations(v_now) r
    WHERE r.reservation_id = v_res_id
    LIMIT 1;

    IF NOT FOUND THEN
        RETURN pg_catalog.jsonb_build_object('ok', FALSE, 'state', 'NOT_FOUND', 'reason', 'reservation_not_found');
    END IF;
    IF v_owner IS NULL OR v_cur.owner_token IS DISTINCT FROM v_owner THEN
        RETURN pg_catalog.jsonb_build_object('ok', FALSE, 'state', v_cur.state, 'reason', 'owner_mismatch');
    END IF;

    IF v_cur.state = 'COMMITTED' THEN
        IF v_cur.telegram_message_id IS NOT NULL AND v_cur.telegram_message_id <> p_telegram_message_id THEN
            RETURN pg_catalog.jsonb_build_object(
                'ok', FALSE, 'state', 'COMMITTED', 'reservation_id', v_cur.reservation_id,
                'reason', 'committed_message_id_mismatch'
            );
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'ok', TRUE, 'state', 'COMMITTED', 'reservation_id', v_cur.reservation_id,
            'article_id', COALESCE(p_article_id, v_cur.article_id),
            'telegram_message_id', COALESCE(v_cur.telegram_message_id, p_telegram_message_id),
            'reason', 'already_committed'
        );
    END IF;

    IF v_cur.state IN ('RESERVED', 'RENEWED') AND v_cur.expires_at <= v_now THEN
        RETURN pg_catalog.jsonb_build_object('ok', FALSE, 'state', 'EXPIRED', 'reason', 'reservation_expired');
    END IF;

    IF v_cur.state NOT IN ('PUBLISHING', 'AMBIGUOUS', 'RESERVED', 'RENEWED') THEN
        RETURN pg_catalog.jsonb_build_object('ok', FALSE, 'state', v_cur.state, 'reason', 'invalid_state_' || pg_catalog.lower(v_cur.state));
    END IF;

    v_target_art_id := COALESCE(p_article_id, v_cur.article_id);
    IF v_target_art_id IS NULL AND v_cur.canonical_url <> '' THEN
        SELECT a.id INTO v_target_art_id
        FROM public.articles a
        WHERE a.canonical_url = v_cur.canonical_url
        LIMIT 1;
    END IF;

    IF v_target_art_id IS NULL THEN
        RETURN pg_catalog.jsonb_build_object('ok', FALSE, 'state', v_cur.state, 'reason', 'target_article_not_found');
    END IF;

    SELECT a.id, a.canonical_url, a.region, a.status
    INTO v_art
    FROM public.articles a
    WHERE a.id = v_target_art_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RETURN pg_catalog.jsonb_build_object('ok', FALSE, 'state', v_cur.state, 'reason', 'target_article_not_found');
    END IF;

    IF v_cur.article_id IS NOT NULL AND v_cur.article_id <> v_art.id THEN
        RETURN pg_catalog.jsonb_build_object('ok', FALSE, 'state', v_cur.state, 'reason', 'article_id_mismatch');
    END IF;

    IF v_cur.canonical_url <> '' AND COALESCE(v_art.canonical_url, '') <> '' AND v_art.canonical_url <> v_cur.canonical_url THEN
        RETURN pg_catalog.jsonb_build_object('ok', FALSE, 'state', v_cur.state, 'reason', 'article_url_mismatch');
    END IF;

    IF pg_catalog.lower(pg_catalog.btrim(COALESCE(v_art.region, ''))) <> 'global' THEN
        RETURN pg_catalog.jsonb_build_object('ok', FALSE, 'state', v_cur.state, 'reason', 'article_not_global');
    END IF;

    UPDATE public.articles
    SET status = 'PUBLISHED',
        published_at = v_pub_ts,
        telegram_message_id = p_telegram_message_id,
        thread_root_message_id = p_thread_root_message_id,
        is_breaking = COALESCE(p_is_breaking, FALSE),
        error_message = ''
    WHERE id = v_art.id;

    GET DIAGNOSTICS v_updated_rows = ROW_COUNT;
    IF v_updated_rows <> 1 THEN
        RAISE EXCEPTION 'finalize_global_publication expected 1 updated row, got %', v_updated_rows;
    END IF;

    v_payload := pg_catalog.jsonb_build_object(
        'v', 1, 'reservation_id', v_cur.reservation_id, 'owner_token', v_cur.owner_token,
        'canonical_url', v_cur.canonical_url, 'url_hash', v_cur.url_hash,
        'article_id', v_art.id, 'batch_id', v_cur.batch_id,
        'state', 'COMMITTED', 'lease_seconds', v_cur.lease_seconds,
        'expires_at', pg_catalog.to_char(v_now AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
        'telegram_message_id', p_telegram_message_id, 'reason', 'published_and_committed'
    );
    INSERT INTO public.system_logs (timestamp, level, component, message)
    VALUES (v_now, 'COMMITTED', 'GLOBAL_RESERVATION', v_payload::TEXT);

    RETURN pg_catalog.jsonb_build_object(
        'ok', TRUE, 'state', 'COMMITTED', 'reservation_id', v_cur.reservation_id,
        'article_id', v_art.id, 'telegram_message_id', p_telegram_message_id,
        'published_at', pg_catalog.to_char(v_pub_ts AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
        'reason', 'committed'
    );
END;
$$;

REVOKE ALL ON FUNCTION public.finalize_global_publication(TEXT, TEXT, BIGINT, BIGINT, BIGINT, BOOLEAN, TIMESTAMPTZ) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.finalize_global_publication(TEXT, TEXT, BIGINT, BIGINT, BIGINT, BOOLEAN, TIMESTAMPTZ) TO service_role;


-- ----------------------------------------------------------------------------
-- RPC 7: release_global_reservation()
-- Valid source states: RESERVED, RENEWED, PUBLISHING -> RELEASED.
-- Explicitly blocks COMMITTED -> RELEASED and AMBIGUOUS -> RELEASED.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.release_global_reservation(
    p_reservation_id TEXT,
    p_owner_token TEXT,
    p_reason TEXT DEFAULT 'definitive_failure'
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_now TIMESTAMPTZ := pg_catalog.now();
    v_res_id TEXT := pg_catalog.btrim(COALESCE(p_reservation_id, ''));
    v_owner TEXT := pg_catalog.btrim(COALESCE(p_owner_token, ''));
    v_cur RECORD;
    v_payload JSONB;
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(8472910348291042);

    SELECT * INTO v_cur
    FROM public._kgn_reconstruct_global_reservations(v_now) r
    WHERE r.reservation_id = v_res_id
    LIMIT 1;

    IF NOT FOUND THEN
        RETURN pg_catalog.jsonb_build_object('ok', FALSE, 'state', 'NOT_FOUND', 'reason', 'reservation_not_found');
    END IF;
    IF v_cur.owner_token IS DISTINCT FROM v_owner THEN
        RETURN pg_catalog.jsonb_build_object('ok', FALSE, 'state', v_cur.state, 'reason', 'owner_mismatch');
    END IF;
    IF v_cur.state IN ('RELEASED', 'EXPIRED') THEN
        RETURN pg_catalog.jsonb_build_object('ok', TRUE, 'state', v_cur.state, 'reason', 'already_released');
    END IF;
    IF v_cur.state NOT IN ('RESERVED', 'RENEWED', 'PUBLISHING') THEN
        RETURN pg_catalog.jsonb_build_object('ok', FALSE, 'state', v_cur.state, 'reason', 'cannot_release_' || pg_catalog.lower(v_cur.state));
    END IF;

    v_payload := pg_catalog.jsonb_build_object(
        'v', 1, 'reservation_id', v_cur.reservation_id, 'owner_token', v_cur.owner_token,
        'canonical_url', v_cur.canonical_url, 'url_hash', v_cur.url_hash,
        'article_id', v_cur.article_id, 'batch_id', v_cur.batch_id,
        'state', 'RELEASED', 'lease_seconds', v_cur.lease_seconds,
        'expires_at', pg_catalog.to_char(v_now AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
        'telegram_message_id', NULL, 'reason', pg_catalog.left(COALESCE(p_reason, 'released'), 250)
    );
    INSERT INTO public.system_logs (timestamp, level, component, message)
    VALUES (v_now, 'RELEASED', 'GLOBAL_RESERVATION', v_payload::TEXT);

    RETURN pg_catalog.jsonb_build_object('ok', TRUE, 'state', 'RELEASED', 'reservation_id', v_cur.reservation_id, 'reason', 'released');
END;
$$;

REVOKE ALL ON FUNCTION public.release_global_reservation(TEXT, TEXT, TEXT) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.release_global_reservation(TEXT, TEXT, TEXT) TO service_role;


-- ----------------------------------------------------------------------------
-- RPC 8: mark_global_ambiguous()
-- Valid source states: PUBLISHING (and unexpired RESERVED/RENEWED).
-- Explicitly blocks RELEASED -> AMBIGUOUS, EXPIRED -> AMBIGUOUS, COMMITTED -> AMBIGUOUS.
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
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_now TIMESTAMPTZ := pg_catalog.now();
    v_res_id TEXT := pg_catalog.btrim(COALESCE(p_reservation_id, ''));
    v_owner TEXT := pg_catalog.btrim(COALESCE(p_owner_token, ''));
    v_cur RECORD;
    v_target_art_id BIGINT;
    v_payload JSONB;
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(8472910348291042);

    SELECT * INTO v_cur
    FROM public._kgn_reconstruct_global_reservations(v_now) r
    WHERE r.reservation_id = v_res_id
    LIMIT 1;

    IF NOT FOUND THEN
        RETURN pg_catalog.jsonb_build_object('ok', FALSE, 'state', 'NOT_FOUND', 'reason', 'reservation_not_found');
    END IF;
    IF v_cur.owner_token IS DISTINCT FROM v_owner THEN
        RETURN pg_catalog.jsonb_build_object('ok', FALSE, 'state', v_cur.state, 'reason', 'owner_mismatch');
    END IF;
    IF v_cur.state = 'AMBIGUOUS' THEN
        RETURN pg_catalog.jsonb_build_object('ok', TRUE, 'state', 'AMBIGUOUS', 'reason', 'already_ambiguous');
    END IF;
    IF v_cur.state IN ('RESERVED', 'RENEWED') AND v_cur.expires_at <= v_now THEN
        RETURN pg_catalog.jsonb_build_object('ok', FALSE, 'state', 'EXPIRED', 'reason', 'invalid_transition_expired_to_ambiguous');
    END IF;
    IF v_cur.state NOT IN ('PUBLISHING', 'RESERVED', 'RENEWED') THEN
        RETURN pg_catalog.jsonb_build_object(
            'ok', FALSE, 'state', v_cur.state,
            'reason', 'invalid_transition_' || pg_catalog.lower(v_cur.state) || '_to_ambiguous'
        );
    END IF;

    v_target_art_id := COALESCE(p_article_id, v_cur.article_id);
    IF v_target_art_id IS NOT NULL THEN
        UPDATE public.articles
        SET error_message = pg_catalog.left('AMBIGUOUS [Global Delivery Uncertain]: ' || COALESCE(p_reason, ''), 500)
        WHERE id = v_target_art_id
          AND status <> 'PUBLISHED';
    END IF;

    v_payload := pg_catalog.jsonb_build_object(
        'v', 1, 'reservation_id', v_cur.reservation_id, 'owner_token', v_cur.owner_token,
        'canonical_url', v_cur.canonical_url, 'url_hash', v_cur.url_hash,
        'article_id', v_target_art_id, 'batch_id', v_cur.batch_id,
        'state', 'AMBIGUOUS', 'lease_seconds', v_cur.lease_seconds,
        'expires_at', pg_catalog.to_char((v_now + INTERVAL '24 hours') AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
        'telegram_message_id', NULL, 'reason', pg_catalog.left(COALESCE(p_reason, 'ambiguous'), 250)
    );
    INSERT INTO public.system_logs (timestamp, level, component, message)
    VALUES (v_now, 'AMBIGUOUS', 'GLOBAL_RESERVATION', v_payload::TEXT);

    RETURN pg_catalog.jsonb_build_object('ok', TRUE, 'state', 'AMBIGUOUS', 'reservation_id', v_cur.reservation_id, 'reason', 'marked_ambiguous');
END;
$$;

REVOKE ALL ON FUNCTION public.mark_global_ambiguous(TEXT, TEXT, TEXT, BIGINT) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.mark_global_ambiguous(TEXT, TEXT, TEXT, BIGINT) TO service_role;


-- ----------------------------------------------------------------------------
-- RPC 9: reconcile_ambiguous_publication()
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
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_now TIMESTAMPTZ := pg_catalog.now();
    v_res_id TEXT := NULLIF(pg_catalog.btrim(COALESCE(p_reservation_id, '')), '');
    v_owner TEXT := NULLIF(pg_catalog.btrim(COALESCE(p_owner_token, '')), '');
    v_outcome TEXT := pg_catalog.lower(pg_catalog.btrim(COALESCE(p_outcome, 'auto')));
    v_cur RECORD;
    v_art RECORD;
    v_reconciled JSONB := '[]'::JSONB;
    v_ambig_count INTEGER := 0;
    v_payload JSONB;
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(8472910348291042);

    IF v_outcome = 'auto' THEN
        FOR v_cur IN
            SELECT * FROM public._kgn_reconstruct_global_reservations(v_now) r
            WHERE (v_res_id IS NULL OR r.reservation_id = v_res_id)
              AND r.state IN ('PUBLISHING', 'AMBIGUOUS')
              AND r.state_at >= (v_now - INTERVAL '24 hours')
        LOOP
            SELECT a.id, a.status, a.published_at INTO v_art
            FROM public.articles a
            WHERE (v_cur.article_id IS NOT NULL AND a.id = v_cur.article_id)
               OR (v_cur.canonical_url <> '' AND a.canonical_url = v_cur.canonical_url)
            LIMIT 1;

            IF FOUND AND v_art.status = 'PUBLISHED' THEN
                v_payload := pg_catalog.jsonb_build_object(
                    'v', 1, 'reservation_id', v_cur.reservation_id, 'owner_token', v_cur.owner_token,
                    'canonical_url', v_cur.canonical_url, 'url_hash', v_cur.url_hash,
                    'article_id', v_art.id, 'batch_id', v_cur.batch_id,
                    'state', 'COMMITTED', 'lease_seconds', v_cur.lease_seconds,
                    'expires_at', pg_catalog.to_char(v_now AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
                    'telegram_message_id', v_cur.telegram_message_id, 'reason', 'auto_reconciled_already_published'
                );
                INSERT INTO public.system_logs (timestamp, level, component, message)
                VALUES (v_now, 'COMMITTED', 'GLOBAL_RESERVATION', v_payload::TEXT);
                v_reconciled := v_reconciled || pg_catalog.jsonb_build_array(
                    pg_catalog.jsonb_build_object('reservation_id', v_cur.reservation_id, 'action', 'COMMITTED')
                );
            ELSIF v_cur.state = 'PUBLISHING' AND v_cur.expires_at <= v_now THEN
                v_payload := pg_catalog.jsonb_build_object(
                    'v', 1, 'reservation_id', v_cur.reservation_id, 'owner_token', v_cur.owner_token,
                    'canonical_url', v_cur.canonical_url, 'url_hash', v_cur.url_hash,
                    'article_id', v_cur.article_id, 'batch_id', v_cur.batch_id,
                    'state', 'AMBIGUOUS', 'lease_seconds', v_cur.lease_seconds,
                    'expires_at', pg_catalog.to_char((v_now + INTERVAL '24 hours') AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
                    'telegram_message_id', NULL, 'reason', 'stale_publishing_lease_fail_closed'
                );
                INSERT INTO public.system_logs (timestamp, level, component, message)
                VALUES (v_now, 'AMBIGUOUS', 'GLOBAL_RESERVATION', v_payload::TEXT);
                v_reconciled := v_reconciled || pg_catalog.jsonb_build_array(
                    pg_catalog.jsonb_build_object('reservation_id', v_cur.reservation_id, 'action', 'AMBIGUOUS')
                );
                v_ambig_count := v_ambig_count + 1;
            ELSIF v_cur.state = 'AMBIGUOUS' THEN
                v_ambig_count := v_ambig_count + 1;
            END IF;
        END LOOP;

        RETURN pg_catalog.jsonb_build_object('ok', TRUE, 'reconciled', v_reconciled, 'ambiguous_count', v_ambig_count, 'reason', 'auto_reconciled');
    END IF;

    IF v_res_id IS NULL THEN
        RETURN pg_catalog.jsonb_build_object('ok', FALSE, 'reason', 'reservation_id_required');
    END IF;

    SELECT * INTO v_cur
    FROM public._kgn_reconstruct_global_reservations(v_now) r
    WHERE r.reservation_id = v_res_id
    LIMIT 1;

    IF NOT FOUND THEN
        RETURN pg_catalog.jsonb_build_object('ok', FALSE, 'reason', 'reservation_not_found');
    END IF;
    IF v_cur.owner_token IS NOT NULL AND v_owner IS DISTINCT FROM v_cur.owner_token THEN
        RETURN pg_catalog.jsonb_build_object('ok', FALSE, 'state', v_cur.state, 'reason', 'owner_mismatch');
    END IF;
    IF v_cur.state NOT IN ('PUBLISHING', 'AMBIGUOUS') THEN
        RETURN pg_catalog.jsonb_build_object(
            'ok', FALSE, 'state', v_cur.state,
            'reason', 'cannot_reconcile_state_' || pg_catalog.lower(v_cur.state)
        );
    END IF;

    IF v_outcome = 'confirmed_published' THEN
        IF p_telegram_message_id IS NULL OR p_telegram_message_id <= 0 THEN
            RETURN pg_catalog.jsonb_build_object('ok', FALSE, 'reason', 'telegram_message_id_required');
        END IF;
        RETURN public.finalize_global_publication(
            v_cur.reservation_id, COALESCE(v_cur.owner_token, ''),
            v_cur.article_id, p_telegram_message_id, NULL, FALSE, v_now
        );
    ELSIF v_outcome = 'confirmed_not_published' THEN
        v_payload := pg_catalog.jsonb_build_object(
            'v', 1, 'reservation_id', v_cur.reservation_id, 'owner_token', v_cur.owner_token,
            'canonical_url', v_cur.canonical_url, 'url_hash', v_cur.url_hash,
            'article_id', v_cur.article_id, 'batch_id', v_cur.batch_id,
            'state', 'RELEASED', 'lease_seconds', v_cur.lease_seconds,
            'expires_at', pg_catalog.to_char(v_now AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
            'telegram_message_id', NULL, 'reason', pg_catalog.left(COALESCE(p_reason, 'confirmed_not_published'), 250)
        );
        INSERT INTO public.system_logs (timestamp, level, component, message)
        VALUES (v_now, 'RELEASED', 'GLOBAL_RESERVATION', v_payload::TEXT);
        RETURN pg_catalog.jsonb_build_object('ok', TRUE, 'state', 'RELEASED', 'reservation_id', v_cur.reservation_id, 'reason', 'confirmed_not_published');
    END IF;

    RETURN pg_catalog.jsonb_build_object('ok', FALSE, 'reason', 'invalid_outcome');
END;
$$;

REVOKE ALL ON FUNCTION public.reconcile_ambiguous_publication(TEXT, TEXT, TEXT, BIGINT, TEXT) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.reconcile_ambiguous_publication(TEXT, TEXT, TEXT, BIGINT, TEXT) TO service_role;
