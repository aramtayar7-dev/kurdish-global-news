"""
Kurdish Global News Bot — Logger Module

Structured logging setup with file and console handlers.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

from config import LOG_LEVEL


_INITIALIZED = False


def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Configure structured logging for the application."""
    global _INITIALIZED
    if _INITIALIZED:
        return logging.getLogger("newsbot")

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    console_format = logging.Formatter(
        "%(asctime)s │ %(levelname)-8s │ %(name)-20s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)

    # File handler — main log
    file_handler = RotatingFileHandler(
        log_path / "newsbot.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s │ %(levelname)-8s │ %(name)-20s │ %(funcName)s:%(lineno)d │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)

    # Error file handler
    error_handler = RotatingFileHandler(
        log_path / "errors.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_format)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)

    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)

    _INITIALIZED = True

    bot_logger = logging.getLogger("newsbot")
    bot_logger.info("Logging initialized. Level: %s, Log dir: %s", LOG_LEVEL, log_path.resolve())
    return bot_logger
