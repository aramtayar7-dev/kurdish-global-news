"""
Kurdish Global News Bot — Private Admin Watchdog & Daily Health Telemetry

Provides:
  - Hard public-channel isolation (`is_valid_admin_target`) so admin messages NEVER hit public channel
  - Deterministic secret sanitization (`redact_secrets`) scrubbing tokens, API keys, and JWTs
  - Stateless alert cooldown & daily report tracking via existing `system_logs` table (no schema migration)
  - Real-time deduplicated critical alerts (`GEMINI_EXHAUSTED`, `SUPABASE_FALLBACK`,
    `PUBLISH_DROUGHT`, `SOURCE_OUTAGE`, `PIPELINE_CRASH`)
  - Daily Sorani Kurdish operational health summary sent once per Kurdistan UTC+3 day
    with `disable_notification=True` and ZERO extra Gemini API calls.
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import config
from core.importance_ranker import classify_topic_bucket, is_bashur_article, is_strict_breaking
from storage.utils import parse_utc_timestamp

logger = logging.getLogger("monitoring.watchdog")

KURDISTAN_TZ = timezone(timedelta(hours=3))

ALERT_GEMINI_EXHAUSTED = "GEMINI_EXHAUSTED"
ALERT_SUPABASE_FALLBACK = "SUPABASE_FALLBACK"
ALERT_PUBLISH_DROUGHT = "PUBLISH_DROUGHT"
ALERT_SOURCE_OUTAGE = "SOURCE_OUTAGE"
ALERT_PIPELINE_CRASH = "PIPELINE_CRASH"

COMPONENT_ALERT = "watchdog_alert"
COMPONENT_DAILY = "watchdog_daily"
COMPONENT_TELEMETRY = "watchdog_telemetry"

# Regex scrubbers for tokens/keys that might leak into HTTP errors or stack traces
_SECRET_REGEXES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"bot[0-9]+:[A-Za-z0-9_-]+"), "[REDACTED]"),
    (re.compile(r"(?<![A-Za-z0-9_-])[0-9]{6,15}:[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])"), "[REDACTED]"),
    (re.compile(r"AIza[0-9A-Za-z_-]+"), "[REDACTED]"),
    (re.compile(r"AQ\.[A-Za-z0-9._-]+"), "[REDACTED]"),
    (re.compile(r"sk-or-[A-Za-z0-9._-]+"), "[REDACTED]"),
    (re.compile(r"gsk_[A-Za-z0-9._-]+"), "[REDACTED]"),
    (re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"), "[REDACTED]"),
    (re.compile(r"((?:apikey|api_key|x-goog-api-key|token)=)[^\s&\"']+"), r"\1[REDACTED]"),
)

_NUMERIC_CHAT_ID_RE = re.compile(r"^-?\d{4,20}$")
_USERNAME_HANDLE_RE = re.compile(r"^@[A-Za-z0-9_]{4,}$")


def is_valid_admin_target(admin_chat_id: Optional[str], public_channel_id: Optional[str]) -> bool:
    """
    Strictly validate that `admin_chat_id` is a valid private admin target and NEVER
    equals or targets `public_channel_id`.
    """
    admin_clean = (admin_chat_id or "").strip()
    if not admin_clean:
        return False

    pub_clean = (public_channel_id or "").strip()
    if pub_clean:
        if admin_clean.lower() == pub_clean.lower():
            return False
        if admin_clean.lstrip("@").lower() == pub_clean.lstrip("@").lower():
            return False

    if _NUMERIC_CHAT_ID_RE.fullmatch(admin_clean):
        return True
    if _USERNAME_HANDLE_RE.fullmatch(admin_clean):
        return True
    return False


def _gather_active_secrets(extra_secrets: Optional[list[str]] = None) -> list[str]:
    """Collect configured secret values dynamically (longest-first for partial safety)."""
    raw_candidates: list[str] = [
        str(getattr(config, "TELEGRAM_BOT_TOKEN", "") or ""),
        os.getenv("TELEGRAM_BOT_TOKEN", ""),
        str(getattr(config, "SUPABASE_KEY", "") or ""),
        os.getenv("SUPABASE_KEY", ""),
        str(getattr(config, "GEMINI_API_KEY", "") or ""),
        os.getenv("GEMINI_API_KEY", ""),
        os.getenv("GEMINI_API_KEYS", ""),
        str(getattr(config, "OPENROUTER_API_KEY", "") or ""),
        os.getenv("OPENROUTER_API_KEY", ""),
        os.getenv("OPENROUTER_API_KEYS", ""),
        str(getattr(config, "GROQ_API_KEY", "") or ""),
        os.getenv("GROQ_API_KEY", ""),
    ]
    raw_candidates.extend(getattr(config, "GEMINI_API_KEYS", []) or [])
    raw_candidates.extend(getattr(config, "OPENROUTER_API_KEYS", []) or [])
    if extra_secrets:
        raw_candidates.extend(extra_secrets)

    secrets: set[str] = set()
    for raw in raw_candidates:
        if not raw:
            continue
        for part in str(raw).split(","):
            tok = part.strip()
            if len(tok) >= 6 and tok not in {
                "your_bot_token_here",
                "your_gemini_api_key_here",
                "your_supabase_service_role_or_api_key",
            }:
                secrets.add(tok)

    return sorted(secrets, key=len, reverse=True)


def redact_secrets(text: str, extra_secrets: Optional[list[str]] = None) -> str:
    """Scrub API keys, Telegram bot tokens, Supabase JWTs, and query credentials."""
    if not text:
        return ""
    scrubbed = str(text)
    for secret in _gather_active_secrets(extra_secrets=extra_secrets):
        if secret in scrubbed:
            scrubbed = scrubbed.replace(secret, "[REDACTED]")
    for pat, repl in _SECRET_REGEXES:
        scrubbed = pat.sub(repl, scrubbed)
    return scrubbed


@dataclass
class WatchdogCycleMetrics:
    """In-memory per-cycle operational counters passed into AdminWatchdog."""
    discovered: int = 0
    published: int = 0
    duplicates_blocked: int = 0
    gemini_429_hits: int = 0
    gemini_exhausted: int = 0
    openrouter_rescued: int = 0
    supabase_fallback: bool = False
    supabase_error: str = ""
    storage_backend: str = "SQLite"
    discovery_ran: bool = True
    in_cooldown: bool = False
    extra_notes: list[str] = field(default_factory=list)


class AdminWatchdog:
    """
    Private SRE Watchdog & Daily Health Telemetry Reporter.
    Strictly isolated from public Telegram channels and completely non-blocking.
    """

    def __init__(
        self,
        db: Optional[Any],
        publisher: Optional[Any],
        admin_chat_id: Optional[str] = None,
        public_channel_id: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self._db = db
        self._publisher = publisher
        self._admin_override = admin_chat_id
        self._public_override = public_channel_id
        self._enabled_override = enabled
        self._memory_cooldowns: dict[str, datetime] = {}

    def _resolved_enabled(self) -> bool:
        if self._enabled_override is not None:
            return bool(self._enabled_override)
        return bool(getattr(config, "ENABLE_ADMIN_WATCHDOG", True))

    def _resolved_admin_id(self) -> str:
        if self._admin_override is not None:
            return str(self._admin_override).strip()
        return str(getattr(config, "ADMIN_CHAT_ID", "") or os.getenv("ADMIN_CHAT_ID", "")).strip()

    def _resolved_public_id(self) -> str:
        if self._public_override is not None:
            return str(self._public_override).strip()
        pub = ""
        if self._publisher is not None:
            pub = str(getattr(self._publisher, "_channel_id", "") or "").strip()
        return pub or str(getattr(config, "TELEGRAM_CHANNEL_ID", "") or os.getenv("TELEGRAM_CHANNEL_ID", "")).strip()

    def _can_send(self) -> Optional[str]:
        if not self._resolved_enabled():
            return None
        admin_id = self._resolved_admin_id()
        public_id = self._resolved_public_id()
        if not admin_id:
            logger.debug("Admin watchdog disabled: ADMIN_CHAT_ID is not configured.")
            return None
        if not is_valid_admin_target(admin_id, public_id):
            logger.warning(
                "Admin watchdog REFUSED send: ADMIN_CHAT_ID is invalid or matches public channel (%s).",
                public_id or "unset",
            )
            return None
        return admin_id

    async def _send_private_message(self, text: str, disable_notification: bool = False) -> bool:
        """Send sanitized private admin message; strictly non-blocking and public-channel isolated."""
        admin_id = self._can_send()
        if not admin_id:
            return False

        safe_text = redact_secrets(text)
        if len(safe_text) > 4000:
            safe_text = safe_text[:3990].rstrip() + "\n..."

        if self._publisher is None:
            logger.warning("Admin watchdog cannot send alert: Telegram publisher is unavailable.")
            return False

        try:
            if hasattr(self._publisher, "send_admin_message"):
                return bool(
                    await self._publisher.send_admin_message(
                        chat_id=admin_id,
                        text=safe_text,
                        disable_notification=disable_notification,
                    )
                )
            bot = getattr(self._publisher, "_bot", None)
            if bot is None:
                logger.warning("Admin watchdog cannot send alert: Telegram bot client not initialized.")
                return False
            from telegram.constants import ParseMode
            await bot.send_message(
                chat_id=admin_id,
                text=safe_text,
                parse_mode=ParseMode.HTML,
                disable_notification=disable_notification,
            )
            return True
        except Exception as exc:
            logger.warning(
                "Non-blocking admin watchdog Telegram send failure: %s",
                redact_secrets(str(exc))[:200],
            )
            return False

    async def _is_alert_in_cooldown(
        self,
        alert_type: str,
        cooldown_minutes: int,
        now_utc: datetime,
    ) -> bool:
        """Check memory + persisted system_logs cooldown to deduplicate identical alerts."""
        cutoff = now_utc - timedelta(minutes=max(1, cooldown_minutes))
        mem_ts = self._memory_cooldowns.get(alert_type)
        if mem_ts is not None and mem_ts > cutoff:
            return True

        if self._db is not None and hasattr(self._db, "get_recent_system_logs"):
            try:
                logs = await self._db.get_recent_system_logs(
                    component=COMPONENT_ALERT,
                    limit=50,
                    now=now_utc,
                )
                prefix = f"alert:{alert_type}|"
                for row in logs:
                    msg = str(row.get("message") or "")
                    if not (msg == f"alert:{alert_type}" or msg.startswith(prefix)):
                        continue
                    ts = parse_utc_timestamp(row.get("timestamp"))
                    if ts is not None and ts > cutoff:
                        self._memory_cooldowns[alert_type] = ts
                        return True
            except Exception as exc:
                logger.debug("Watchdog cooldown DB lookup error (%s): %s", alert_type, str(exc)[:120])
        return False

    async def _record_alert_sent(
        self,
        alert_type: str,
        summary: str,
        now_utc: datetime,
    ) -> None:
        self._memory_cooldowns[alert_type] = now_utc
        if self._db is not None and hasattr(self._db, "log_system_event"):
            try:
                safe_summary = redact_secrets(summary)[:400]
                await self._db.log_system_event(
                    level="WARNING",
                    component=COMPONENT_ALERT,
                    message=f"alert:{alert_type}|{safe_summary}",
                    timestamp=now_utc.isoformat(),
                )
            except Exception as exc:
                logger.debug("Watchdog failed to persist alert log (%s): %s", alert_type, str(exc)[:120])

    async def dispatch_alert(
        self,
        alert_type: str,
        title: str,
        details: str,
        now: Optional[datetime] = None,
    ) -> bool:
        """Send a deduplicated critical admin alert respecting cooldown and secret redaction."""
        if not self._can_send():
            return False

        now_utc = now or datetime.now(timezone.utc)
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)

        cooldown_min = int(getattr(config, "WATCHDOG_ALERT_COOLDOWN_MINUTES", 60))
        if await self._is_alert_in_cooldown(alert_type, cooldown_min, now_utc):
            logger.debug("Watchdog alert suppressed by %dm cooldown: %s", cooldown_min, alert_type)
            return False

        erbil_time = now_utc.astimezone(KURDISTAN_TZ).strftime("%Y-%m-%d %H:%M UTC+3")
        safe_details = html.escape(redact_secrets(details), quote=False)
        safe_title = html.escape(redact_secrets(title), quote=False)
        message = (
            f"🚨 <b>ئاگادارکردنەوەی سیستەم [{html.escape(alert_type, quote=False)}]</b>\n"
            f"<b>{safe_title}</b>\n\n"
            f"{safe_details}\n\n"
            f"🕒 {erbil_time}"
        )
        sent = await self._send_private_message(message, disable_notification=False)
        if sent:
            await self._record_alert_sent(alert_type, title, now_utc)
        return sent

    async def record_cycle_telemetry(
        self,
        metrics: WatchdogCycleMetrics,
        now: Optional[datetime] = None,
    ) -> None:
        """Persist non-zero Gemini 429 or fallback events into system_logs for 24h rollups."""
        if self._db is None or not hasattr(self._db, "log_system_event"):
            return
        if (
            metrics.gemini_429_hits <= 0
            and metrics.openrouter_rescued <= 0
            and not metrics.supabase_fallback
        ):
            return
        now_utc = now or datetime.now(timezone.utc)
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)
        try:
            payload = {
                "gemini_429_hits": int(metrics.gemini_429_hits),
                "gemini_exhausted": int(metrics.gemini_exhausted),
                "openrouter_rescued": int(metrics.openrouter_rescued),
                "supabase_fallback": bool(metrics.supabase_fallback),
            }
            await self._db.log_system_event(
                level="INFO",
                component=COMPONENT_TELEMETRY,
                message=json.dumps(payload, ensure_ascii=True),
                timestamp=now_utc.isoformat(),
            )
        except Exception as exc:
            logger.debug("Watchdog telemetry write skipped: %s", str(exc)[:120])

    async def notify_pipeline_crash(
        self,
        exc: BaseException,
        now: Optional[datetime] = None,
    ) -> bool:
        """Dispatch a redacted PIPELINE_CRASH alert cleanly when an unhandled cycle error occurs."""
        err_type = type(exc).__name__
        err_msg = redact_secrets(str(exc) or "Unhandled exception")[:500]
        return await self.dispatch_alert(
            ALERT_PIPELINE_CRASH,
            title=f"پەککەوتنی خولی بۆت ({err_type})",
            details=f"هەڵەی چاوەڕواننەکراو: <code>{html.escape(err_type, quote=False)}: {html.escape(err_msg, quote=False)}</code>",
            now=now,
        )

    async def _daily_report_already_sent(self, today_str: str, now_utc: datetime) -> bool:
        if self._db is None or not hasattr(self._db, "get_recent_system_logs"):
            return False
        try:
            logs = await self._db.get_recent_system_logs(
                component=COMPONENT_DAILY,
                limit=20,
                now=now_utc,
            )
            marker = f"daily_report:{today_str}"
            return any(str(row.get("message") or "").strip() == marker for row in logs)
        except Exception as exc:
            logger.debug("Watchdog daily-report lookup failed: %s", str(exc)[:120])
            return False

    async def maybe_send_daily_report(
        self,
        metrics: Optional[WatchdogCycleMetrics] = None,
        now: Optional[datetime] = None,
    ) -> bool:
        """
        Send the daily Sorani Kurdish health report once per Kurdistan (UTC+3) date
        once local hour >= WATCHDOG_DAILY_REPORT_HOUR_UTC3.
        """
        if not self._can_send() or self._db is None:
            return False

        now_utc = now or datetime.now(timezone.utc)
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)
        local_now = now_utc.astimezone(KURDISTAN_TZ)
        target_hour = int(getattr(config, "WATCHDOG_DAILY_REPORT_HOUR_UTC3", 22))
        if local_now.hour < target_hour:
            return False

        today_str = local_now.strftime("%Y-%m-%d")
        if await self._daily_report_already_sent(today_str, now_utc):
            return False

        try:
            art_metrics = await self._db.get_watchdog_article_metrics(hours=24, now=now_utc)
            sources = await self._db.get_sources_status() if hasattr(self._db, "get_sources_status") else []
            tel_logs = (
                await self._db.get_recent_system_logs(
                    component=COMPONENT_TELEMETRY,
                    limit=200,
                    hours=24,
                    now=now_utc,
                )
                if hasattr(self._db, "get_recent_system_logs")
                else []
            )
        except Exception as exc:
            logger.warning("Watchdog failed gathering 24h daily report metrics: %s", redact_secrets(str(exc))[:160])
            return False

        published_articles = list(art_metrics.get("published_articles") or [])
        total_pub = len(published_articles)
        bashur_count = sum(1 for a in published_articles if is_bashur_article(a))
        global_count = max(0, total_pub - bashur_count)
        pw_count = sum(1 for a in published_articles if classify_topic_bucket(a) == "politics_war")
        econ_count = sum(1 for a in published_articles if classify_topic_bucket(a) == "economy")
        cyber_count = sum(1 for a in published_articles if classify_topic_bucket(a) == "cybersecurity")
        breaking_count = sum(1 for a in published_articles if is_strict_breaking(a))
        threaded_count = sum(
            1 for a in published_articles
            if getattr(a, "thread_root_message_id", None) is not None
        )

        dup_24h = int(art_metrics.get("duplicates_count") or 0)
        skip_24h = int(art_metrics.get("skipped_count") or 0)
        queued_cnt = int(art_metrics.get("queued_count") or 0)

        gemini_429_24h = 0
        openrouter_rescued_24h = 0
        saw_sb_fallback = bool(metrics and metrics.supabase_fallback)
        for row in tel_logs:
            try:
                data = json.loads(str(row.get("message") or "{}"))
                gemini_429_24h += int(data.get("gemini_429_hits", 0) or 0)
                openrouter_rescued_24h += int(data.get("openrouter_rescued", 0) or 0)
                if data.get("supabase_fallback"):
                    saw_sb_fallback = True
            except Exception:
                continue

        # Add current cycle hits only if not yet recorded to telemetry table
        if metrics and not tel_logs:
            if metrics.gemini_429_hits > 0:
                gemini_429_24h += int(metrics.gemini_429_hits)
            if metrics.openrouter_rescued > 0:
                openrouter_rescued_24h += int(metrics.openrouter_rescued)

        active_sources = len(sources)
        failing_sources = [s for s in sources if int(s.get("consecutive_failures") or 0) >= 3]
        backend_label = (
            "SQLite fallback"
            if saw_sb_fallback
            else (metrics.storage_backend if metrics else type(self._db).__name__)
        )

        failing_line = (
            ", ".join(
                f"{html.escape(str(s.get('name', 'unknown')), quote=False)} ({int(s.get('consecutive_failures') or 0)})"
                for s in failing_sources[:5]
            )
            if failing_sources
            else "هیچ"
        )

        or_line = (
            f"\n• فریاکەوتنی OpenRouter (٢٤ کاتژمێر): {openrouter_rescued_24h}"
            if openrouter_rescued_24h > 0
            else ""
        )

        report_text = (
            f"📊 <b>ڕاپۆرتی ڕۆژانەی تەندروستیی بۆت — {today_str}</b>\n\n"
            f"📰 <b>بڵاوکراوەکانی ٢٤ کاتژمێر:</b> {total_pub}\n"
            f"• هەرێم: باشوور {bashur_count} | جیهان {global_count}\n"
            f"• بابەت: سیاسەت و جەنگ {pw_count} | ئابووری {econ_count} | سایبەری {cyber_count}\n"
            f"• تایبەت: 🔴 بەپەلە {breaking_count} | 🔄 نوێکردنەوەی زنجیرە {threaded_count}\n\n"
            f"⚙️ <b>کارایی بۆت:</b>\n"
            f"• دووبارە ڕاگیراو: {dup_24h} | پشتگوێخراو: {skip_24h}\n"
            f"• چاوەڕوان لە نۆرە (QUEUED): {queued_cnt}\n\n"
            f"🔌 <b>API و کۆگا:</b>\n"
            f"• هەڵەی Gemini 429 (٢٤ کاتژمێر): {gemini_429_24h}{or_line}\n"
            f"• بنکەدراوی چالاک: {html.escape(str(backend_label), quote=False)}\n\n"
            f"📡 <b>سەرچاوەکان ({active_sources} چالاک):</b>\n"
            f"• سەرچاوە پەککەوتووەکان (≥٣ هەڵە): {len(failing_sources)} ({failing_line})"
        )

        sent = await self._send_private_message(report_text, disable_notification=True)
        if sent and hasattr(self._db, "log_system_event"):
            try:
                await self._db.log_system_event(
                    level="INFO",
                    component=COMPONENT_DAILY,
                    message=f"daily_report:{today_str}",
                    timestamp=now_utc.isoformat(),
                )
            except Exception as exc:
                logger.debug("Watchdog failed saving daily report marker: %s", str(exc)[:120])
        return sent

    async def run_cycle_checks(
        self,
        metrics: Optional[WatchdogCycleMetrics] = None,
        now: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """
        Execute all real-time anomaly checks + optional daily health summary.
        Never raises exceptions into the news pipeline.
        """
        summary: dict[str, Any] = {"alerts_sent": [], "daily_report_sent": False}
        if not self._resolved_enabled():
            return summary

        now_utc = now or datetime.now(timezone.utc)
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)

        cycle_metrics = metrics or WatchdogCycleMetrics()

        try:
            await self.record_cycle_telemetry(cycle_metrics, now=now_utc)

            # 1. Supabase fallback alert
            if cycle_metrics.supabase_fallback:
                details = (
                    "بنکەدراوی Supabase وەڵامی نەدایەوە و سیستەم بە شێوەی کاتی گەڕایەوە سەر SQLite.\n"
                    f"وردەکاری: {cycle_metrics.supabase_error or 'Connection/query error'}"
                )
                if await self.dispatch_alert(
                    ALERT_SUPABASE_FALLBACK,
                    title="پەککەوتنی Supabase و گەڕانەوە بۆ SQLite",
                    details=details,
                    now=now_utc,
                ):
                    summary["alerts_sent"].append(ALERT_SUPABASE_FALLBACK)

            # 2. Gemini all-keys exhausted alert (suppressed if OpenRouter fallback rescued the requests)
            if cycle_metrics.gemini_exhausted > 0 and cycle_metrics.openrouter_rescued <= 0:
                details = (
                    f"هەموو کلیلەکانی Gemini تووشی سنوورداریی HTTP 429 بوون.\n"
                    f"ژمارەی 429 لەم خولەدا: {cycle_metrics.gemini_429_hits}\n"
                    "دۆخ: ڕێزبەندکردن گەڕاوەتەوە سەر فلتەری ناوخۆیی و وەرگێڕان هەڵگیراوە بۆ دووبارە تاقیکردنەوە."
                )
                if await self.dispatch_alert(
                    ALERT_GEMINI_EXHAUSTED,
                    title="تەواوبوونی پشکی هەموو کلیلەکانی Gemini (HTTP 429)",
                    details=details,
                    now=now_utc,
                ):
                    summary["alerts_sent"].append(ALERT_GEMINI_EXHAUSTED)

            # Remaining DB-backed checks
            if self._db is not None:
                # 3. Publish drought check (never trigger false alerts while intentionally resting)
                if (
                    not getattr(cycle_metrics, "in_cooldown", False)
                    and cycle_metrics.discovery_ran
                    and hasattr(self._db, "get_watchdog_article_metrics")
                ):
                    try:
                        art_metrics = await self._db.get_watchdog_article_metrics(hours=24, now=now_utc)
                        total_articles = int(art_metrics.get("total_articles") or 0)
                        last_pub_dt = art_metrics.get("last_published_at")
                        silence_hours = int(getattr(config, "WATCHDOG_SILENCE_ALERT_HOURS", 6))

                        if total_articles > 0:
                            drought = False
                            drought_desc = ""
                            if last_pub_dt is None:
                                drought = True
                                drought_desc = (
                                    f"هیچ بابەتێک بڵاونەکراوەتەوە سەرەڕای هەبوونی {total_articles} بابەت لە بنکەدراودا."
                                )
                            else:
                                elapsed_h = (now_utc - last_pub_dt).total_seconds() / 3600.0
                                if elapsed_h >= silence_hours:
                                    drought = True
                                    drought_desc = (
                                        f"لە ماوەی {elapsed_h:.1f} کاتژمێری ڕابردوودا هیچ هەواڵێک بڵاونەکراوەتەوە "
                                        f"(سنوور: {silence_hours} کاتژمێر)."
                                    )
                            if drought:
                                if await self.dispatch_alert(
                                    ALERT_PUBLISH_DROUGHT,
                                    title="وەستانی بڵاوکردنەوەی هەواڵ (Publish Drought)",
                                    details=drought_desc,
                                    now=now_utc,
                                ):
                                    summary["alerts_sent"].append(ALERT_PUBLISH_DROUGHT)
                    except Exception as exc:
                        logger.debug("Watchdog drought check error: %s", redact_secrets(str(exc))[:120])

                # 4. Source outage check
                if hasattr(self._db, "get_sources_status"):
                    try:
                        threshold = int(getattr(config, "WATCHDOG_SOURCE_FAILURE_THRESHOLD", 5))
                        sources = await self._db.get_sources_status()
                        outages = [
                            s for s in sources
                            if int(s.get("consecutive_failures") or 0) >= threshold
                        ]
                        if outages:
                            lines = [
                                f"• {s.get('name', 'unknown')}: {int(s.get('consecutive_failures') or 0)} هەڵەی لەسەریەک "
                                f"({redact_secrets(str(s.get('last_error') or 'unknown error'))[:100]})"
                                for s in outages[:8]
                            ]
                            if await self.dispatch_alert(
                                ALERT_SOURCE_OUTAGE,
                                title=f"پەککەوتنی {len(outages)} سەرچاوەی هەواڵ",
                                details="\n".join(lines),
                                now=now_utc,
                            ):
                                summary["alerts_sent"].append(ALERT_SOURCE_OUTAGE)
                    except Exception as exc:
                        logger.debug("Watchdog source outage check error: %s", redact_secrets(str(exc))[:120])

                # 5. Daily Health Report (once per Kurdistan UTC+3 day once hour >= 22)
                summary["daily_report_sent"] = await self.maybe_send_daily_report(
                    metrics=cycle_metrics,
                    now=now_utc,
                )
        except Exception as exc:
            logger.warning(
                "Admin watchdog cycle check failed non-fatally: %s",
                redact_secrets(str(exc))[:200],
            )

        return summary
