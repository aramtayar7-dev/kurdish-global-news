"""
Diagnostic CLI for live Telegram (`t.me/s/`) and X (`Nitter` / `syndication.twitter.com` / `fxtwitter`)
ingestion, media extraction, age filtering, and pre-LLM editorial validation.

Usage:
    python scripts/diagnose_x_telegram.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import aiohttp
from bs4 import BeautifulSoup

import config
from core.importance_ranker import check_pre_filter_off_topic
from processing.extractor import ArticleExtractor
from processing.validator import ArticleValidator
from sources.telegram_source import (
    TELEGRAM_WEB_BASE_URL,
    TelegramSource,
    clean_telegram_message_text,
    extract_telegram_photo_url,
    extract_telegram_video_url,
    is_rejected_telegram_content,
)
from sources.x_video_source import XVideoSource


async def diagnose_telegram(channels: list[dict[str, str]]) -> dict[str, int]:
    print("\n=== TELEGRAM DIAGNOSTIC (t.me/s/) ===")
    src = TelegramSource(channels=channels)
    validator = ArticleValidator()
    extractor = ArticleExtractor()
    totals = {"fetched_dom": 0, "recent": 0, "photos": 0, "videos": 0, "validated": 0, "editorial_passed": 0}

    await src.initialize()
    try:
        for ch in channels:
            username = ch["username"]
            url = TELEGRAM_WEB_BASE_URL.format(username=username)
            assert src._session is not None
            try:
                async with src._session.get(url) as resp:
                    status = resp.status
                    html_text = await resp.text(errors="replace") if status == 200 else ""
            except Exception as exc:
                print(f"[TG] @{username}: HTTP ERROR -> {exc}")
                continue

            soup = BeautifulSoup(html_text, "html.parser") if html_text else None
            dom_nodes = soup.select(".tgme_widget_message") if soup else []
            totals["fetched_dom"] += len(dom_nodes)

            drop_reasons: dict[str, int] = {}
            ch_photos = 0
            ch_videos = 0
            for node in dom_nodes:
                if extract_telegram_photo_url(node):
                    ch_photos += 1
                if extract_telegram_video_url(node):
                    ch_videos += 1

            raw_articles = src.parse_channel_html(html_text, ch)
            totals["recent"] += len(raw_articles)
            totals["photos"] += sum(1 for a in raw_articles if a.image_url)
            totals["videos"] += sum(1 for a in raw_articles if a.video_url)

            val_ok = 0
            ed_ok = 0
            for raw in raw_articles:
                art = extractor.extract(raw)
                ok, reason = validator.validate(art, is_cold_start=False)
                if not ok:
                    drop_reasons[f"validator:{reason}"] = drop_reasons.get(f"validator:{reason}", 0) + 1
                    continue
                val_ok += 1
                ed_reason = check_pre_filter_off_topic(art)
                if ed_reason:
                    drop_reasons[f"editorial:{ed_reason}"] = drop_reasons.get(f"editorial:{ed_reason}", 0) + 1
                    continue
                ed_ok += 1

            totals["validated"] += val_ok
            totals["editorial_passed"] += ed_ok
            print(
                f"[TG] @{username}: http={status}, dom_msgs={len(dom_nodes)}, "
                f"dom_photos={ch_photos}, dom_videos={ch_videos}, parsed_recent={len(raw_articles)}, "
                f"raw_img={sum(1 for a in raw_articles if a.image_url)}, raw_vid={sum(1 for a in raw_articles if a.video_url)}, "
                f"validator_ok={val_ok}, editorial_ok={ed_ok}"
            )
            if drop_reasons:
                print(f"     drops: {drop_reasons}")
    finally:
        await src.cleanup()

    return totals


async def diagnose_x(accounts: list[dict[str, object]]) -> dict[str, int]:
    print("\n=== X (TWITTER) DIAGNOSTIC ===")
    src = XVideoSource(accounts=accounts)
    validator = ArticleValidator()
    extractor = ArticleExtractor()
    totals = {"discovered": 0, "photos": 0, "videos": 0, "validated": 0, "editorial_passed": 0}

    await src.initialize()
    try:
        # Check mirror health once
        for mirror in src._mirrors:
            handle = str(accounts[0]["handle"])
            rss_url = f"{mirror}/{handle}/rss"
            body = await src._http_get_text(rss_url)
            ok = bool(body and ("<rss" in body[:200].lower() or "<feed" in body[:200].lower() or "<?xml" in body[:200].lower()))
            print(f"[X] Mirror {mirror}: {'OK' if ok else 'FAIL'} (len={len(body or '')})")

        if hasattr(src, "fetch_channel_syndication"):
            for acc in accounts:
                handle = str(acc["handle"])
                syn_items, syn_ok = await src.fetch_channel_syndication(acc)
                print(f"[X] Syndication @{handle}: ok={syn_ok}, tweets={len(syn_items)}")

        raw_articles = await src.fetch_articles()
        totals["discovered"] = len(raw_articles)
        totals["photos"] = sum(1 for a in raw_articles if a.image_url)
        totals["videos"] = sum(1 for a in raw_articles if a.video_url)

        drop_reasons: dict[str, int] = {}
        for raw in raw_articles:
            art = extractor.extract(raw)
            ok, reason = validator.validate(art, is_cold_start=False)
            if not ok:
                drop_reasons[f"validator:{reason}"] = drop_reasons.get(f"validator:{reason}", 0) + 1
                continue
            totals["validated"] += 1
            ed_reason = check_pre_filter_off_topic(art)
            if ed_reason:
                drop_reasons[f"editorial:{ed_reason}"] = drop_reasons.get(f"editorial:{ed_reason}", 0) + 1
                continue
            totals["editorial_passed"] += 1

        print(
            f"[X] Total fetch_articles={totals['discovered']}, photos={totals['photos']}, "
            f"videos={totals['videos']}, validator_ok={totals['validated']}, editorial_ok={totals['editorial_passed']}"
        )
        for raw in raw_articles[:5]:
            print(f"     sample: {raw.source_name} | vid={bool(raw.video_url)} img={bool(raw.image_url)} | {raw.title[:70]}")
        if drop_reasons:
            print(f"     drops: {drop_reasons}")
    finally:
        await src.cleanup()

    return totals


async def main() -> None:
    tg_channels = [
        {"username": "rudawdigital", "name": "Rudaw", "region_hint": "kurdistan_bashur", "lang": "ckb"},
        {"username": "Kurdistan24news", "name": "Kurdistan24", "region_hint": "kurdistan_bashur", "lang": "ckb"},
        {"username": "ClashReport", "name": "Clash Report", "region_hint": "global", "lang": "en"},
    ]
    x_accounts = [
        {"handle": "CENTCOM", "display_name": "CENTCOM", "region": "middle_east", "lang": "en", "credibility_tier": 1, "source_group": "centcom", "is_official_media": True},
        {"handle": "ClashReport", "display_name": "Clash Report", "region": "middle_east", "lang": "en", "credibility_tier": 2, "source_group": "clashreport", "is_official_media": False},
        {"handle": "Faytuks", "display_name": "Faytuks News", "region": "global", "lang": "en", "credibility_tier": 2, "source_group": "faytuks", "is_official_media": False},
    ]
    tg_totals = await diagnose_telegram(tg_channels)
    x_totals = await diagnose_x(x_accounts)
    print("\n=== SUMMARY ===")
    print("Telegram:", tg_totals)
    print("X:", x_totals)


if __name__ == "__main__":
    asyncio.run(main())
