"""Kurdish Global News Bot — Storage Package & Backend Factory."""

from __future__ import annotations

import logging
import os
from typing import Optional, Union

import config
from storage.database import Database, get_kurdistan_date_str
from storage.supabase_db import SupabaseAPIError, SupabaseDatabase

logger = logging.getLogger("storage")

DatabaseBackend = Union[Database, SupabaseDatabase]


def create_database(
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None,
    db_path: Optional[str] = None,
) -> DatabaseBackend:
    """
    Return the active persistent database backend:
    - Supabase PostgREST (`SupabaseDatabase`) when both SUPABASE_URL and SUPABASE_KEY are set.
    - Local SQLite (`Database`) fallback otherwise.
    """
    resolved_url = (
        supabase_url
        if supabase_url is not None
        else (os.getenv("SUPABASE_URL") or config.SUPABASE_URL)
    ).strip().rstrip("/")
    resolved_key = (
        supabase_key
        if supabase_key is not None
        else (os.getenv("SUPABASE_KEY") or config.SUPABASE_KEY)
    ).strip()

    if resolved_url and resolved_key:
        logger.info("Using Supabase PostgREST persistent backend (%s).", resolved_url)
        return SupabaseDatabase(url=resolved_url, key=resolved_key)

    logger.info("Using local SQLite database backend.")
    return Database(db_path=db_path or config.DATABASE_PATH)


__all__ = [
    "Database",
    "DatabaseBackend",
    "SupabaseAPIError",
    "SupabaseDatabase",
    "create_database",
    "get_kurdistan_date_str",
]
