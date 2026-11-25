-- ============================================================================
-- Kurdish Global News Bot — Live Event Threading Migration (Supabase)
-- Run cleanly and idempotently in the Supabase SQL Editor.
-- Adds Telegram threading columns if they do not already exist on `articles`.
-- ============================================================================

ALTER TABLE articles
    ADD COLUMN IF NOT EXISTS telegram_message_id BIGINT;

ALTER TABLE articles
    ADD COLUMN IF NOT EXISTS thread_root_message_id BIGINT;

ALTER TABLE articles
    ADD COLUMN IF NOT EXISTS event_key TEXT NOT NULL DEFAULT '';

ALTER TABLE articles
    ADD COLUMN IF NOT EXISTS event_entities TEXT NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_articles_telegram_message_id
    ON articles (telegram_message_id);

CREATE INDEX IF NOT EXISTS idx_articles_thread_root_message_id
    ON articles (thread_root_message_id);
