"""
Kurdish Global News Bot — Main Entry Point

Autonomous 24/7 international news monitoring and Telegram publishing platform.
Discovers news every 60 seconds and publishes eligible stories IMMEDIATELY.
Runs an embedded lightweight async health-check web server for Render / cloud deployment.

Usage:
    python main.py          # Continuous monitoring loop + health-check web server
    python main.py --once   # Execute a single discovery & publishing cycle, then exit cleanly
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from aiohttp import web
from config import validate_config

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Kurdish Global News Bot")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Execute a single discovery and publishing cycle, then exit cleanly.",
    )
    return parser.parse_args(args)


async def handle_health_check(request: web.Request) -> web.Response:
    """Return 200 OK JSON for Render / uptime monitoring keep-alive."""
    return web.json_response({
        "status": "ok",
        "service": "kurdish-global-news",
    })


def create_health_app() -> web.Application:
    """Create lightweight aiohttp web application for health checks."""
    app = web.Application()
    app.router.add_get("/", handle_health_check)
    app.router.add_get("/health", handle_health_check)
    return app


async def run_bot_with_web_server() -> None:
    """Run news engine alongside embedded health check web server."""
    logger = logging.getLogger("main")
    port = int(os.getenv("PORT", "10000"))

    # 1. Start embedded web server on 0.0.0.0:PORT
    app = create_health_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info("Embedded health server started on http://0.0.0.0:%d (/ and /health)", port)

    # 2. Run news bot engine
    from core.engine import NewsEngine
    engine = NewsEngine()

    try:
        await engine.run_forever()
    finally:
        logger.info("Shutting down embedded health server...")
        await runner.cleanup()


async def run_single_cycle() -> None:
    """Run a single news discovery & publishing cycle."""
    from core.engine import NewsEngine
    engine = NewsEngine()
    await engine.run_once()


def main(argv: list[str] | None = None) -> int:
    """Application entry point."""
    args = parse_args(argv)

    # Initialize logging first
    from monitoring.logger import setup_logging
    logger = setup_logging()

    mode_label = "Single-Run Mode (--once)" if args.once else "Continuous Mode"
    logger.info("=" * 60)
    logger.info("  Kurdish Global News Bot — Starting [%s]", mode_label)
    logger.info("  Autonomous International News Platform")
    logger.info("=" * 60)

    # Validate configuration
    errors = validate_config()
    if errors:
        for error in errors:
            logger.error("Configuration error: %s", error)
        logger.error(
            "Please configure your .env file. See .env.example for details."
        )
        sys.exit(1)

    # Run with Windows-compatible event loop if on Windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        if args.once:
            logger.info("Running single discovery and publishing cycle...")
            asyncio.run(run_single_cycle())
            logger.info("Single cycle completed successfully. Exiting with code 0.")
            return 0
        else:
            asyncio.run(run_bot_with_web_server())
            return 0
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user.")
        return 0
    except Exception as e:
        logger.critical("Fatal error: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
