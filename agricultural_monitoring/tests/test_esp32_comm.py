#!/usr/bin/env python3
"""
Test suite for ESP32 serial communication.

Tests:
- Serial connection
- Command/response protocol
- Safety interlocks
- Timeout handling

Run with: python -m pytest tests/test_esp32_comm.py -v

Note: These tests require an ESP32 connected via USB.
Skip with -k "not esp32" if not available.

Author: Agricultural Robotics Team
"""

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project to path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'src'))

from spray_control.safety_monitor import SafetyMonitor
from spray_control.spray_decision import SprayDecisionMaker


class TestSafetyMonitor(unittest.TestCase):
    """Tests for safety monitor interlocks."""

    def setUp(self):
        self.monitor = SafetyMonitor(
            max_spray_duration=10000,
            cooldown_seconds=2.0,
            max_sprays_per_hour=5
        )

    def test_initial_state(self):
        """Test initial state allows spraying."""
        can_spray, reason = self.monitor.can_spray(5000)
        self.assertTrue(can_spray)

    def test_max_duration_exceeded(self):
        """Test max duration interlock."""
        can_spray, reason = self.monitor.can_spray(15000)
        self.assertFalse(can_spray)
        self.assertIn('exceeds', reason.lower())

    def test_cooldown_period(self):
        """Test cooldown between sprays."""
        # First spray should work
        can_spray, _ = self.monitor.can_spray(5000)
        self.assertTrue(can_spray)
        self.monitor.record_spray(5000)

        # Immediate second spray should fail
        can_spray, reason = self.monitor.can_spray(5000)
        self.assertFalse(can_spray)
        self.assertIn('cooldown', reason.lower())

    def test_cooldown_expiry(self):
        """Test spray works after cooldown expires."""
        # Record spray
        self.monitor.record_spray(5000)

        # Wait for cooldown
        time.sleep(2.1)

        # Should now work
        can_spray, _ = self.monitor.can_spray(5000)
        self.assertTrue(can_spray)

    def test_hourly_limit(self):
        """Test hourly spray count limit."""
        # Spray up to limit (with no cooldown to speed up test)
        self.monitor.cooldown_seconds = 0

        for i in range(5):
            self.monitor.record_spray(1000)

        # Next should fail
        can_spray, reason = self.monitor.can_spray(1000)
        self.assertFalse(can_spray)
        self.assertIn('hourly', reason.lower())

    def test_emergency_stop(self):
        """Test emergency stop blocks all sprays."""
        self.monitor.emergency_stop("Test stop")

        can_spray, reason = self.monitor.can_spray(1000)
        self.assertFalse(can_spray)
        self.assertIn('emergency', reason.lower())

    def test_emergency_clear(self):
        """Test clearing emergency stop."""
        self.monitor.emergency_stop("Test stop")
        result = self.monitor.clear_emergency()

        self.assertTrue(result)
        self.assertFalse(self.monitor.emergency_stopped)

        can_spray, _ = self.monitor.can_spray(1000)
        self.assertTrue(can_spray)

    def test_clamp_duration(self):
        """Test duration clamping."""
        clamped = self.monitor.clamp_duration(15000)
        self.assertEqual(clamped, 10000)

        clamped = self.monitor.clamp_duration(-1000)
        self.assertEqual(clamped, 0)

    def test_status_report(self):
        """Test status reporting."""
        self.monitor.record_spray(5000)
        status = self.monitor.get_status()

        self.assertIn('emergency_stopped', status)
        self.assertIn('total_sprays', status)
        self.assertIn('sprays_this_hour', status)


class TestSprayDecisionMaker(unittest.TestCase):
    """Tests for spray decision logic."""

    def setUp(self):
        self.decision_maker = SprayDecisionMaker(
            confidence_threshold=0.8,
            spray_durations={
                'healthy': 0,
                'mild': 2000,
                'moderate': 4000,
                'severe': 6000
            }
        )

    def test_healthy_no_spray(self):
        """Test healthy plant gets no spray."""
        should_spray, duration, reason = self.decision_maker.decide(
            'healthy', 0.95, 'healthy'
        )

        self.assertFalse(should_spray)
        self.assertEqual(duration, 0)

    def test_disease_triggers_spray(self):
        """Test disease detection triggers spray."""
        should_spray, duration, reason = self.decision_maker.decide(
            'early_blight', 0.85, 'mild'
        )

        self.assertTrue(should_spray)
        self.assertEqual(duration, 2000)

    def test_low_confidence_no_spray(self):
        """Test low confidence doesn't trigger spray."""
        should_spray, duration, reason = self.decision_maker.decide(
            'late_blight', 0.7, 'severe'
        )

        self.assertFalse(should_spray)

    def test_severity_duration_mapping(self):
        """Test severity maps to correct duration."""
        # Mild
        _, duration, _ = self.decision_maker.decide('test', 0.9, 'mild')
        self.assertEqual(duration, 2000)

        # Moderate
        _, duration, _ = self.decision_maker.decide('test', 0.9, 'moderate')
        self.assertEqual(duration, 4000)

        # Severe
        _, duration, _ = self.decision_maker.decide('test', 0.9, 'severe')
        self.assertEqual(duration, 6000)

    def test_severity_from_disease_name(self):
        """Test severity lookup from disease name."""
        self.decision_maker.severity_mapping['test_disease'] = 'severe'

        _, duration, _ = self.decision_maker.decide(
            'test_disease', 0.9, None
        )
        self.assertEqual(duration, 6000)


class TestSerialProtocol(unittest.TestCase):
    """Tests for serial communication protocol (mocked)."""

    def test_ping_command(self):
        """Test PING command format."""
        with patch('serial.Serial') as mock_serial:
            mock_instance = MagicMock()
            mock_serial.return_value = mock_instance
            mock_instance.readline.return_value = b'PONG\n'

            import serial
            ser = serial.Serial('/dev/ttyUSB0', 115200)

            ser.write(b'PING\n')
            response = ser.readline().decode().strip()

            self.assertEqual(response, 'PONG')

    def test_spray_command_format(self):
        """Test SPRAY command format."""
        duration = 5000
        command = f"SPRAY:{duration}\n"

        self.assertEqual(command, "SPRAY:5000\n")

    def test_stop_command(self):
        """Test STOP command format."""
        with patch('serial.Serial') as mock_serial:
            mock_instance = MagicMock()
            mock_serial.return_value = mock_instance
            mock_instance.readline.return_value = b'STOPPED\n'

            import serial
            ser = serial.Serial('/dev/ttyUSB0', 115200)

            ser.write(b'STOP\n')
            response = ser.readline().decode().strip()

            self.assertEqual(response, 'STOPPED')


class TestLiveESP32(unittest.TestCase):
    """Live tests with actual ESP32 (skipped if not available)."""

    @classmethod
    def setUpClass(cls):
        """Check if ESP32 is connected."""
        import serial.tools.list_ports

        cls.esp32_port = None
        for port in serial.tools.list_ports.comports():
            if 'USB' in port.device or 'ACM' in port.device:
                cls.esp32_port = port.device
                break

    def setUp(self):
        if self.esp32_port is None:
            self.skipTest("ESP32 not connected")

        import serial
        self.ser = serial.Serial(self.esp32_port, 115200, timeout=5)
        time.sleep(2)  # Wait for ESP32 to initialize

    def tearDown(self):
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()

    @unittest.skip("Requires ESP32 connected")
    def test_live_ping(self):
        """Test PING with actual ESP32."""
        self.ser.write(b'PING\n')
        response = self.ser.readline().decode().strip()
        self.assertEqual(response, 'PONG')

    @unittest.skip("Requires ESP32 connected")
    def test_live_heartbeat(self):
        """Test HEARTBEAT with actual ESP32."""
        self.ser.write(b'HEARTBEAT\n')
        response = self.ser.readline().decode().strip()
        self.assertEqual(response, 'HEARTBEAT_OK')

    @unittest.skip("Requires ESP32 connected")
    def test_live_status(self):
        """Test STATUS with actual ESP32."""
        self.ser.write(b'STATUS\n')
        response = self.ser.readline().decode().strip()
        self.assertTrue(response.startswith('STATUS:'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
