"""Tests for the scheduler."""

import asyncio
import pytest
from unittest.mock import AsyncMock
from core.scheduler import DiscoveryScheduler
from config import NEWS_CHECK_INTERVAL_SECONDS


class TestDiscoveryScheduler:
    """Test the discovery scheduler."""

    def test_default_interval(self):
        """TEST 1: Verify default 60-second discovery interval."""
        scheduler = DiscoveryScheduler()
        assert scheduler.interval_seconds == 60
        assert NEWS_CHECK_INTERVAL_SECONDS == 60

    def test_scheduler_not_running_initially(self):
        scheduler = DiscoveryScheduler()
        assert scheduler.is_running is False
        assert scheduler.cycle_count == 0

    @pytest.mark.asyncio
    async def test_scheduler_starts_and_calls_callback(self):
        """TEST 1: The discovery engine starts automatically and calls the callback."""
        callback = AsyncMock()
        scheduler = DiscoveryScheduler()
        scheduler._interval = 1  # 1 second for testing
        scheduler.set_discovery_callback(callback)

        await scheduler.start()
        assert scheduler.is_running is True

        # Wait for at least one cycle
        await asyncio.sleep(2.5)

        await scheduler.stop()
        assert scheduler.is_running is False
        assert callback.call_count >= 1

    @pytest.mark.asyncio
    async def test_scheduler_no_callback_raises(self):
        """Test that starting without a callback raises an error."""
        scheduler = DiscoveryScheduler()
        with pytest.raises(RuntimeError):
            await scheduler.start()

    @pytest.mark.asyncio
    async def test_scheduler_survives_callback_error(self):
        """Test that the scheduler continues after callback errors."""
        call_count = 0

        async def failing_callback():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First cycle fails")
            # Second cycle succeeds

        scheduler = DiscoveryScheduler()
        scheduler._interval = 1
        scheduler.set_discovery_callback(failing_callback)

        await scheduler.start()
        await asyncio.sleep(3.5)
        await scheduler.stop()

        # Should have continued past the error
        assert call_count >= 2

    def test_no_hourly_publishing_scheduler(self):
        """TEST 4: Verify no hourly publishing scheduler exists."""
        import inspect
        source = inspect.getsource(DiscoveryScheduler)
        # Should not contain hourly publishing logic
        assert "3600" not in source or "backup" in source.lower()
        # The word "hourly" should only appear in negation context
        lines_with_hourly = [
            line for line in source.split("\n")
            if "hourly" in line.lower() and "no hourly" not in line.lower()
        ]
        assert len(lines_with_hourly) == 0, f"Found hourly references: {lines_with_hourly}"

    def test_no_publishing_cooldown(self):
        """TEST 2 & TEST 3: No artificial cooldown or delay."""
        # The scheduler only handles discovery, not publishing
        scheduler = DiscoveryScheduler()
        # No cooldown attribute should exist
        assert not hasattr(scheduler, '_publish_cooldown')
        assert not hasattr(scheduler, '_batch_size')
        assert not hasattr(scheduler, '_publish_interval')
