"""
Kurdish Global News Bot — Metrics Module

Track and report operational metrics for discovery, processing, and publishing.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("metrics")


class Metrics:
    """Operational metrics tracker."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = defaultdict(int)
        self._timings: dict[str, list[float]] = defaultdict(list)
        self._start_time = time.time()

    def increment(self, name: str, amount: int = 1) -> None:
        """Increment a counter."""
        self._counters[name] += amount

    def record_timing(self, name: str, duration_seconds: float) -> None:
        """Record a timing measurement."""
        self._timings[name].append(duration_seconds)
        # Keep only last 1000 measurements
        if len(self._timings[name]) > 1000:
            self._timings[name] = self._timings[name][-500:]

    def get_average_timing(self, name: str) -> float:
        """Get average timing for a metric."""
        vals = self._timings.get(name, [])
        return sum(vals) / len(vals) if vals else 0.0

    def get_report(self) -> dict[str, object]:
        """Generate a metrics report."""
        report: dict[str, object] = {
            "uptime_seconds": round(time.time() - self._start_time, 1),
            "counters": dict(self._counters),
            "timings": {},
        }
        for name, vals in self._timings.items():
            if vals:
                report["timings"][name] = {  # type: ignore
                    "count": len(vals),
                    "avg_seconds": round(sum(vals) / len(vals), 3),
                    "min_seconds": round(min(vals), 3),
                    "max_seconds": round(max(vals), 3),
                }
        return report

    def log_report(self) -> None:
        """Log the current metrics report."""
        report = self.get_report()
        logger.info("── Metrics Report ──")
        logger.info("Uptime: %s seconds", report["uptime_seconds"])
        for name, value in report.get("counters", {}).items():  # type: ignore
            logger.info("  %s: %s", name, value)
        for name, stats in report.get("timings", {}).items():  # type: ignore
            logger.info("  %s: avg=%.3fs, min=%.3fs, max=%.3fs, count=%d",
                        name, stats["avg_seconds"], stats["min_seconds"],
                        stats["max_seconds"], stats["count"])


class Timer:
    """Context manager for timing operations."""

    def __init__(self, metrics: Metrics, name: str) -> None:
        self._metrics = metrics
        self._name = name
        self._start: float = 0

    def __enter__(self) -> "Timer":
        self._start = time.time()
        return self

    def __exit__(self, *args: object) -> None:
        duration = time.time() - self._start
        self._metrics.record_timing(self._name, duration)
