#!/usr/bin/env python3
"""
Safety Monitor for Agricultural Spray System

Implements safety interlocks to prevent spray system misuse:
- Maximum spray duration enforcement
- Cooldown period between sprays
- Hourly spray count limit
- Heartbeat monitoring
- Emergency stop functionality

SAFETY CRITICAL: This module is the final safeguard before spray commands
are sent to the ESP32. All safety checks must pass before spraying.

Author: Agricultural Robotics Team
License: MIT
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class SprayRecord:
    """Record of a spray event for safety tracking."""
    timestamp: datetime
    duration_ms: int


class SafetyMonitor:
    """
    Safety interlock system for spray control.

    Enforces safety limits:
    - Maximum single spray duration
    - Minimum cooldown between sprays
    - Maximum sprays per hour
    - Emergency stop capability

    All spray commands must pass through this monitor before execution.

    Attributes:
        max_spray_duration: Maximum allowed spray duration in ms
        cooldown_seconds: Minimum time between sprays
        max_sprays_per_hour: Maximum spray count per hour
        emergency_stopped: Whether emergency stop is active
    """

    def __init__(
        self,
        max_spray_duration: int = 10000,
        cooldown_seconds: float = 2.0,
        max_sprays_per_hour: int = 20,
        heartbeat_interval: float = 5.0
    ) -> None:
        """
        Initialize the safety monitor.

        Args:
            max_spray_duration: Max spray time in milliseconds
            cooldown_seconds: Minimum seconds between sprays
            max_sprays_per_hour: Maximum spray operations per hour
            heartbeat_interval: Expected heartbeat interval
        """
        self.max_spray_duration = max_spray_duration
        self.cooldown_seconds = cooldown_seconds
        self.max_sprays_per_hour = max_sprays_per_hour
        self.heartbeat_interval = heartbeat_interval

        # State tracking
        self.spray_history: deque = deque(maxlen=1000)
        self.last_spray_time: Optional[datetime] = None
        self.emergency_stopped = False
        self.emergency_reason: Optional[str] = None

        # Heartbeat tracking
        self.last_heartbeat: Optional[datetime] = None
        self.heartbeat_failures = 0
        self.max_heartbeat_failures = 3

        # Statistics
        self.total_sprays = 0
        self.total_duration_ms = 0
        self.blocked_attempts = 0

        logger.info(
            f"Safety Monitor initialized: max_duration={max_spray_duration}ms, "
            f"cooldown={cooldown_seconds}s, max_per_hour={max_sprays_per_hour}"
        )

    def can_spray(self, requested_duration: int) -> Tuple[bool, str]:
        """
        Check if spray operation is allowed.

        Performs all safety checks:
        1. Emergency stop not active
        2. Duration within limits
        3. Cooldown period elapsed
        4. Hourly limit not exceeded

        Args:
            requested_duration: Requested spray duration in ms

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        # Check emergency stop
        if self.emergency_stopped:
            self.blocked_attempts += 1
            return False, f"Emergency stop active: {self.emergency_reason}"

        # Check duration limit
        if requested_duration > self.max_spray_duration:
            self.blocked_attempts += 1
            return (
                False,
                f"Duration {requested_duration}ms exceeds max {self.max_spray_duration}ms"
            )

        # Check duration is positive
        if requested_duration <= 0:
            self.blocked_attempts += 1
            return False, "Duration must be positive"

        # Check cooldown
        if self.last_spray_time is not None:
            elapsed = (datetime.now() - self.last_spray_time).total_seconds()
            if elapsed < self.cooldown_seconds:
                remaining = self.cooldown_seconds - elapsed
                self.blocked_attempts += 1
                return False, f"Cooldown active: {remaining:.1f}s remaining"

        # Check hourly limit
        hourly_count = self._get_hourly_spray_count()
        if hourly_count >= self.max_sprays_per_hour:
            self.blocked_attempts += 1
            return (
                False,
                f"Hourly limit reached: {hourly_count}/{self.max_sprays_per_hour}"
            )

        return True, "All safety checks passed"

    def record_spray(self, duration_ms: int) -> None:
        """
        Record a completed spray operation.

        Call this after successful spray execution.

        Args:
            duration_ms: Actual spray duration in ms
        """
        now = datetime.now()
        record = SprayRecord(timestamp=now, duration_ms=duration_ms)
        self.spray_history.append(record)

        self.last_spray_time = now
        self.total_sprays += 1
        self.total_duration_ms += duration_ms

        logger.debug(f"Spray recorded: {duration_ms}ms at {now}")

    def _get_hourly_spray_count(self) -> int:
        """Get number of sprays in the last hour."""
        one_hour_ago = datetime.now() - timedelta(hours=1)
        return sum(
            1 for record in self.spray_history
            if record.timestamp > one_hour_ago
        )

    def get_hourly_duration(self) -> int:
        """Get total spray duration in the last hour (ms)."""
        one_hour_ago = datetime.now() - timedelta(hours=1)
        return sum(
            record.duration_ms for record in self.spray_history
            if record.timestamp > one_hour_ago
        )

    def emergency_stop(self, reason: str = "Manual stop") -> None:
        """
        Activate emergency stop.

        Args:
            reason: Reason for emergency stop
        """
        self.emergency_stopped = True
        self.emergency_reason = reason
        logger.warning(f"EMERGENCY STOP activated: {reason}")

    def clear_emergency(self) -> bool:
        """
        Clear emergency stop state.

        Returns:
            True if emergency was cleared, False if not in emergency state
        """
        if not self.emergency_stopped:
            return False

        self.emergency_stopped = False
        previous_reason = self.emergency_reason
        self.emergency_reason = None
        logger.info(f"Emergency stop cleared (was: {previous_reason})")
        return True

    def record_heartbeat(self, success: bool = True) -> None:
        """
        Record heartbeat status from ESP32.

        Args:
            success: Whether heartbeat was successful
        """
        if success:
            self.last_heartbeat = datetime.now()
            self.heartbeat_failures = 0
        else:
            self.heartbeat_failures += 1
            if self.heartbeat_failures >= self.max_heartbeat_failures:
                self.emergency_stop(
                    f"Heartbeat failed {self.heartbeat_failures} times"
                )

    def check_heartbeat_timeout(self, timeout_seconds: float = 15.0) -> bool:
        """
        Check if heartbeat has timed out.

        Args:
            timeout_seconds: Maximum time since last heartbeat

        Returns:
            True if heartbeat is OK, False if timed out
        """
        if self.last_heartbeat is None:
            return True  # No heartbeat expected yet

        elapsed = (datetime.now() - self.last_heartbeat).total_seconds()
        if elapsed > timeout_seconds:
            self.emergency_stop(f"Heartbeat timeout: {elapsed:.1f}s")
            return False

        return True

    def clamp_duration(self, requested_duration: int) -> int:
        """
        Clamp duration to safe limits.

        Args:
            requested_duration: Requested spray duration

        Returns:
            Duration clamped to max_spray_duration
        """
        return min(max(0, requested_duration), self.max_spray_duration)

    def get_status(self) -> Dict[str, Any]:
        """
        Get current safety monitor status.

        Returns:
            Dictionary with safety status information
        """
        now = datetime.now()

        # Calculate time until cooldown expires
        cooldown_remaining = 0.0
        if self.last_spray_time is not None:
            elapsed = (now - self.last_spray_time).total_seconds()
            cooldown_remaining = max(0, self.cooldown_seconds - elapsed)

        # Calculate sprays remaining this hour
        hourly_count = self._get_hourly_spray_count()
        sprays_remaining = max(0, self.max_sprays_per_hour - hourly_count)

        return {
            'emergency_stopped': self.emergency_stopped,
            'emergency_reason': self.emergency_reason,
            'cooldown_remaining': round(cooldown_remaining, 1),
            'sprays_this_hour': hourly_count,
            'sprays_remaining': sprays_remaining,
            'max_sprays_per_hour': self.max_sprays_per_hour,
            'hourly_duration_ms': self.get_hourly_duration(),
            'total_sprays': self.total_sprays,
            'total_duration_ms': self.total_duration_ms,
            'blocked_attempts': self.blocked_attempts,
            'last_spray': (
                self.last_spray_time.isoformat()
                if self.last_spray_time else None
            ),
            'last_heartbeat': (
                self.last_heartbeat.isoformat()
                if self.last_heartbeat else None
            ),
            'heartbeat_failures': self.heartbeat_failures
        }

    def get_recent_sprays(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent spray records.

        Args:
            count: Number of records to return

        Returns:
            List of spray record dictionaries
        """
        recent = list(self.spray_history)[-count:]
        return [
            {
                'timestamp': record.timestamp.isoformat(),
                'duration_ms': record.duration_ms
            }
            for record in recent
        ]

    def reset_statistics(self) -> None:
        """Reset accumulated statistics (not safety limits)."""
        self.total_sprays = 0
        self.total_duration_ms = 0
        self.blocked_attempts = 0
        logger.info("Safety monitor statistics reset")


class TankMonitor:
    """
    Optional tank level monitoring.

    Tracks spray fluid tank level and provides warnings.
    """

    def __init__(
        self,
        warning_level: float = 20.0,
        critical_level: float = 5.0
    ) -> None:
        """
        Initialize tank monitor.

        Args:
            warning_level: Percentage level for warning
            critical_level: Percentage level to block spraying
        """
        self.warning_level = warning_level
        self.critical_level = critical_level
        self.current_level: Optional[float] = None
        self.last_update: Optional[datetime] = None

    def update_level(self, level: float) -> None:
        """
        Update tank level reading.

        Args:
            level: Tank level percentage (0-100)
        """
        self.current_level = max(0.0, min(100.0, level))
        self.last_update = datetime.now()

    def can_spray(self) -> Tuple[bool, str]:
        """
        Check if tank level allows spraying.

        Returns:
            Tuple of (allowed, reason)
        """
        if self.current_level is None:
            return True, "Tank level unknown"

        if self.current_level <= self.critical_level:
            return False, f"Tank critical: {self.current_level:.1f}%"

        if self.current_level <= self.warning_level:
            logger.warning(f"Tank low: {self.current_level:.1f}%")

        return True, f"Tank level: {self.current_level:.1f}%"

    def get_status(self) -> Dict[str, Any]:
        """Get tank status."""
        return {
            'level': self.current_level,
            'warning_level': self.warning_level,
            'critical_level': self.critical_level,
            'last_update': (
                self.last_update.isoformat()
                if self.last_update else None
            ),
            'status': self._get_status_string()
        }

    def _get_status_string(self) -> str:
        """Get human-readable status string."""
        if self.current_level is None:
            return "unknown"
        elif self.current_level <= self.critical_level:
            return "critical"
        elif self.current_level <= self.warning_level:
            return "low"
        else:
            return "ok"


if __name__ == '__main__':
    # Quick test
    logging.basicConfig(level=logging.DEBUG)

    monitor = SafetyMonitor(
        max_spray_duration=10000,
        cooldown_seconds=2.0,
        max_sprays_per_hour=5
    )

    print(f"Initial status: {monitor.get_status()}")

    # Test spray checks
    test_durations = [5000, 15000, 5000, 5000, 5000, 5000, 5000]

    for duration in test_durations:
        allowed, reason = monitor.can_spray(duration)
        print(f"\nSpray {duration}ms: {allowed} - {reason}")

        if allowed:
            monitor.record_spray(duration)
            time.sleep(0.5)  # Shorter than cooldown

    print(f"\nFinal status: {monitor.get_status()}")
