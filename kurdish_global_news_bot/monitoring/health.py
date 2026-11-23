"""
Kurdish Global News Bot — Health Monitoring

Tracks system health, component status, and uptime.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("health")


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    healthy: bool = True
    last_success: Optional[float] = None
    last_error: Optional[str] = None
    error_count: int = 0
    total_runs: int = 0

    def record_success(self) -> None:
        self.healthy = True
        self.last_success = time.time()
        self.total_runs += 1

    def record_failure(self, error: str) -> None:
        self.last_error = error
        self.error_count += 1
        self.total_runs += 1
        if self.error_count > 5:
            self.healthy = False


class HealthMonitor:
    """Monitors the health of all system components."""

    def __init__(self) -> None:
        self._components: dict[str, ComponentHealth] = {}
        self._start_time = time.time()

    def register_component(self, name: str) -> None:
        """Register a component for health tracking."""
        if name not in self._components:
            self._components[name] = ComponentHealth(name=name)

    def record_success(self, component: str) -> None:
        """Record a successful operation for a component."""
        if component not in self._components:
            self.register_component(component)
        self._components[component].record_success()

    def record_failure(self, component: str, error: str) -> None:
        """Record a failed operation for a component."""
        if component not in self._components:
            self.register_component(component)
        self._components[component].record_failure(error)
        logger.warning("Component '%s' failure: %s", component, error[:200])

    def is_healthy(self) -> bool:
        """Check if all components are healthy."""
        return all(c.healthy for c in self._components.values())

    def get_status(self) -> dict[str, object]:
        """Get a full health status report."""
        uptime = time.time() - self._start_time
        return {
            "healthy": self.is_healthy(),
            "uptime_seconds": round(uptime, 1),
            "uptime_human": self._format_uptime(uptime),
            "components": {
                name: {
                    "healthy": c.healthy,
                    "total_runs": c.total_runs,
                    "error_count": c.error_count,
                    "last_error": c.last_error,
                }
                for name, c in self._components.items()
            },
        }

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours}h {minutes}m {secs}s"
