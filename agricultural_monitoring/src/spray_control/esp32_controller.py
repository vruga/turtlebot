#!/usr/bin/env python3
"""
ESP32 Spray Controller for Agricultural Disease Detection

Handles serial communication with ESP32 microcontroller that
controls the spray nozzle system. Includes safety interlocks
and heartbeat monitoring.

SAFETY CRITICAL: This module controls agricultural spray equipment.
All commands are validated and logged.

Author: Agricultural Robotics Team
License: MIT
"""

import os
import sys
import time
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import json
import csv
import yaml

import serial
import serial.tools.list_ports

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.spray_control.safety_monitor import SafetyMonitor
from src.spray_control.spray_decision import SprayDecisionMaker


logger = logging.getLogger(__name__)


@dataclass
class SprayEvent:
    """Record of a spray event for logging."""
    timestamp: datetime
    duration_ms: int
    disease: str
    confidence: float
    trigger: str  # 'auto' or 'manual'
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'duration_ms': self.duration_ms,
            'disease': self.disease,
            'confidence': self.confidence,
            'trigger': self.trigger,
            'success': self.success,
            'error': self.error
        }


class ESP32Controller(Node):
    """
    ROS2 node for ESP32 spray controller communication.

    Handles:
    - Serial communication with ESP32
    - Safety interlocks (max duration, cooldown, hourly limit)
    - Heartbeat monitoring
    - Spray event logging

    Topics:
        Subscribed:
            - /agricultural/disease_detection (String): Detection results
            - /agricultural/spray_command (String): Manual spray commands
        Published:
            - /agricultural/spray_status (String): Spray status updates
    """

    def __init__(self) -> None:
        super().__init__('esp32_controller')

        # Load configuration
        self.config = self._load_config()

        # Serial connection
        self.serial_port: Optional[serial.Serial] = None
        self.serial_lock = threading.Lock()

        # Safety monitor
        self.safety_monitor = SafetyMonitor(
            max_spray_duration=self.config.get('safety', {}).get('max_duration', 10000),
            cooldown_seconds=self.config.get('safety', {}).get('cooldown_seconds', 2.0),
            max_sprays_per_hour=self.config.get('safety', {}).get('max_sprays_per_hour', 20),
            heartbeat_interval=self.config.get('safety', {}).get('heartbeat_interval', 5.0)
        )

        # Spray decision maker
        self.decision_maker = SprayDecisionMaker(
            config_path=Path(__file__).parent.parent.parent / 'config'
        )

        # State
        self.connected = False
        self.last_spray_time: Optional[datetime] = None
        self.spray_count = 0
        self.spray_history: list[SprayEvent] = []
        self.emergency_stopped = False
        self.pending_confirmation: Optional[Dict] = None

        # Heartbeat thread
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.heartbeat_running = False

        # Setup logging
        self._setup_spray_log()

        # QoS profiles
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )

        # Subscribe to disease detection
        detection_topic = self.config.get('topics', {}).get(
            'disease_detection', '/agricultural/disease_detection'
        )
        self.detection_sub = self.create_subscription(
            String,
            detection_topic,
            self._detection_callback,
            reliable_qos
        )

        # Subscribe to manual spray commands
        self.command_sub = self.create_subscription(
            String,
            '/agricultural/spray_command',
            self._command_callback,
            reliable_qos
        )

        # Subscribe to confirmation responses
        self.confirm_sub = self.create_subscription(
            String,
            '/agricultural/spray_confirm',
            self._confirm_callback,
            reliable_qos
        )

        # Publisher for spray status
        self.status_pub = self.create_publisher(
            String,
            '/agricultural/spray_status',
            reliable_qos
        )

        # Try to connect to ESP32
        self._connect_esp32()

        # Run startup self-test if configured
        if self.config.get('safety', {}).get('startup_self_test', True):
            self._run_self_test()

        # Start heartbeat monitoring
        self._start_heartbeat()

        self.get_logger().info(
            f"ESP32 Controller initialized. "
            f"Connected: {self.connected}"
        )

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML files."""
        config = {}
        config_dir = Path(__file__).parent.parent.parent / 'config'

        # Load system config
        system_config = config_dir / 'system_config.yaml'
        if system_config.exists():
            try:
                with open(system_config, 'r') as f:
                    loaded = yaml.safe_load(f)
                    if loaded:
                        config.update(loaded)
            except Exception as e:
                logger.warning(f"Failed to load system config: {e}")

        # Load spray config
        spray_config = config_dir / 'spray_config.yaml'
        if spray_config.exists():
            try:
                with open(spray_config, 'r') as f:
                    loaded = yaml.safe_load(f)
                    if loaded:
                        config.update(loaded)
            except Exception as e:
                logger.warning(f"Failed to load spray config: {e}")

        return config

    def _setup_spray_log(self) -> None:
        """Setup CSV log file for spray events."""
        log_dir = Path(__file__).parent.parent.parent / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = log_dir / 'spray_log.csv'

        # Create header if file doesn't exist
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'duration_ms', 'disease',
                    'confidence', 'trigger', 'success', 'error'
                ])

    def _connect_esp32(self) -> bool:
        """
        Connect to ESP32 via serial port.

        Returns:
            True if connection successful, False otherwise
        """
        serial_config = self.config.get('serial', {})
        port = serial_config.get('port', '/dev/ttyUSB0')
        baud_rate = serial_config.get('baud_rate', 115200)
        timeout = serial_config.get('timeout', 5.0)
        retry_attempts = serial_config.get('retry_attempts', 3)
        retry_delay = serial_config.get('retry_delay', 2.0)

        for attempt in range(retry_attempts):
            try:
                self.get_logger().info(
                    f"Connecting to ESP32 on {port} (attempt {attempt + 1})"
                )

                # Check if port exists
                if not Path(port).exists():
                    # Try to find ESP32
                    available_ports = list(serial.tools.list_ports.comports())
                    esp_ports = [
                        p.device for p in available_ports
                        if 'USB' in p.device or 'ACM' in p.device
                    ]

                    if esp_ports:
                        port = esp_ports[0]
                        self.get_logger().info(f"Found serial port: {port}")
                    else:
                        raise serial.SerialException(f"Port {port} not found")

                self.serial_port = serial.Serial(
                    port=port,
                    baudrate=baud_rate,
                    timeout=timeout,
                    write_timeout=timeout
                )

                # Wait for ESP32 to initialize
                time.sleep(2.0)

                # Send ping to verify connection
                if self._send_command("PING"):
                    self.connected = True
                    self.get_logger().info(f"Connected to ESP32 on {port}")
                    return True

            except serial.SerialException as e:
                self.get_logger().warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < retry_attempts - 1:
                    time.sleep(retry_delay)

        self.get_logger().error("Failed to connect to ESP32")
        self.connected = False
        return False

    def _run_self_test(self) -> bool:
        """
        Run startup self-test to verify ESP32 functionality.

        Tests:
        1. Ping response
        2. Short relay cycle

        Returns:
            True if self-test passed, False otherwise
        """
        if not self.connected:
            self.get_logger().warning("Cannot run self-test: not connected")
            return False

        self.get_logger().info("Running ESP32 self-test...")

        # Test 1: Ping
        if not self._send_command("PING"):
            self.get_logger().error("Self-test FAILED: Ping failed")
            return False

        # Test 2: Short spray (100ms) to verify relay
        self.get_logger().info("Testing relay (100ms pulse)...")
        if not self._send_command("TEST:100"):
            self.get_logger().error("Self-test FAILED: Relay test failed")
            return False

        self.get_logger().info("Self-test PASSED")
        self._publish_status("self_test", {"passed": True})
        return True

    def _start_heartbeat(self) -> None:
        """Start heartbeat monitoring thread."""
        if self.heartbeat_thread is not None:
            return

        self.heartbeat_running = True
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True
        )
        self.heartbeat_thread.start()
        self.get_logger().debug("Heartbeat monitoring started")

    def _heartbeat_loop(self) -> None:
        """Heartbeat monitoring loop."""
        interval = self.config.get('safety', {}).get('heartbeat_interval', 5.0)
        timeout = self.config.get('safety', {}).get('heartbeat_timeout', 15.0)
        last_successful = time.time()

        while self.heartbeat_running:
            time.sleep(interval)

            if not self.connected:
                continue

            # Send heartbeat
            if self._send_command("HEARTBEAT", expect_response=True):
                last_successful = time.time()
            else:
                # Check if we've exceeded timeout
                elapsed = time.time() - last_successful
                if elapsed > timeout:
                    self.get_logger().error(
                        f"Heartbeat timeout ({elapsed:.1f}s) - "
                        "triggering emergency stop"
                    )
                    self._emergency_stop("Heartbeat timeout")

    def _detection_callback(self, msg: String) -> None:
        """
        Handle disease detection results.

        Args:
            msg: JSON string with detection result
        """
        if self.emergency_stopped:
            self.get_logger().warning("System in emergency stop, ignoring detection")
            return

        try:
            detection = json.loads(msg.data)

            # Check for error
            if detection.get('error'):
                self.get_logger().warning(f"Detection error: {detection.get('message')}")
                return

            # Make spray decision
            should_spray, duration, reason = self.decision_maker.decide(
                disease_name=detection.get('disease_name', 'unknown'),
                confidence=detection.get('confidence', 0),
                severity=detection.get('severity', 'unknown')
            )

            if not should_spray:
                self.get_logger().info(f"No spray: {reason}")
                self._publish_status("no_spray", {
                    "disease": detection.get('disease_name'),
                    "reason": reason
                })
                return

            # Check safety
            can_spray, safety_reason = self.safety_monitor.can_spray(duration)
            if not can_spray:
                self.get_logger().warning(f"Safety interlock: {safety_reason}")
                self._publish_status("safety_block", {
                    "disease": detection.get('disease_name'),
                    "reason": safety_reason
                })
                return

            # Check if confirmation required
            if self.config.get('safety', {}).get('confirmation_required', True):
                self._request_confirmation(detection, duration)
            else:
                self._execute_spray(
                    duration=duration,
                    disease=detection.get('disease_name', 'unknown'),
                    confidence=detection.get('confidence', 0),
                    trigger='auto'
                )

        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse detection: {e}")

    def _command_callback(self, msg: String) -> None:
        """
        Handle manual spray commands.

        Commands:
            - SPRAY:<duration_ms> - Manual spray
            - STOP - Emergency stop
            - RESUME - Resume from emergency stop
            - STATUS - Get current status

        Args:
            msg: Command string
        """
        command = msg.data.strip().upper()

        if command == "STOP":
            self._emergency_stop("Manual stop requested")
        elif command == "RESUME":
            self._resume_from_emergency()
        elif command == "STATUS":
            self._publish_full_status()
        elif command.startswith("SPRAY:"):
            try:
                duration = int(command.split(':')[1])
                # Manual spray bypasses confirmation but respects safety
                can_spray, reason = self.safety_monitor.can_spray(duration)
                if can_spray:
                    self._execute_spray(
                        duration=duration,
                        disease='manual',
                        confidence=1.0,
                        trigger='manual'
                    )
                else:
                    self.get_logger().warning(f"Manual spray blocked: {reason}")
            except (ValueError, IndexError):
                self.get_logger().error(f"Invalid spray command: {command}")

    def _confirm_callback(self, msg: String) -> None:
        """
        Handle confirmation response from dashboard.

        Args:
            msg: 'CONFIRM' or 'CANCEL'
        """
        if self.pending_confirmation is None:
            return

        if msg.data.strip().upper() == "CONFIRM":
            self._execute_spray(
                duration=self.pending_confirmation['duration'],
                disease=self.pending_confirmation['disease'],
                confidence=self.pending_confirmation['confidence'],
                trigger='auto_confirmed'
            )
        else:
            self.get_logger().info("Spray cancelled by user")
            self._publish_status("spray_cancelled", self.pending_confirmation)

        self.pending_confirmation = None

    def _request_confirmation(
        self,
        detection: Dict,
        duration: int
    ) -> None:
        """Request user confirmation before spraying."""
        self.pending_confirmation = {
            'duration': duration,
            'disease': detection.get('disease_name', 'unknown'),
            'confidence': detection.get('confidence', 0),
            'timestamp': datetime.now().isoformat()
        }

        self._publish_status("awaiting_confirmation", self.pending_confirmation)
        self.get_logger().info(
            f"Awaiting confirmation: spray {duration}ms for "
            f"{detection.get('disease_name')}"
        )

    def _execute_spray(
        self,
        duration: int,
        disease: str,
        confidence: float,
        trigger: str
    ) -> bool:
        """
        Execute spray command with full safety checks.

        Args:
            duration: Spray duration in milliseconds
            disease: Disease being treated
            confidence: Detection confidence
            trigger: Trigger source ('auto', 'manual', etc.)

        Returns:
            True if spray successful, False otherwise
        """
        if self.emergency_stopped:
            self.get_logger().error("Cannot spray: emergency stop active")
            return False

        if not self.connected:
            self.get_logger().error("Cannot spray: not connected to ESP32")
            self._log_spray_event(SprayEvent(
                timestamp=datetime.now(),
                duration_ms=duration,
                disease=disease,
                confidence=confidence,
                trigger=trigger,
                success=False,
                error="Not connected"
            ))
            return False

        # Final safety check
        max_duration = self.config.get('safety', {}).get('max_duration', 10000)
        duration = min(duration, max_duration)

        # Execute spray
        self.get_logger().info(f"Spraying for {duration}ms (disease: {disease})")

        success = self._send_spray_command(duration)

        # Record event
        event = SprayEvent(
            timestamp=datetime.now(),
            duration_ms=duration,
            disease=disease,
            confidence=confidence,
            trigger=trigger,
            success=success,
            error=None if success else "Command failed"
        )

        self._log_spray_event(event)
        self.safety_monitor.record_spray(duration)

        if success:
            self.spray_count += 1
            self.last_spray_time = datetime.now()
            self._publish_status("spray_complete", event.to_dict())
        else:
            self._publish_status("spray_failed", event.to_dict())

        return success

    def _send_spray_command(self, duration: int) -> bool:
        """
        Send spray command to ESP32.

        Args:
            duration: Spray duration in milliseconds

        Returns:
            True if command acknowledged, False otherwise
        """
        command = f"SPRAY:{duration}"
        return self._send_command(command, expect_response=True)

    def _send_command(
        self,
        command: str,
        expect_response: bool = False
    ) -> bool:
        """
        Send command to ESP32 and optionally wait for response.

        Args:
            command: Command string to send
            expect_response: Whether to wait for acknowledgment

        Returns:
            True if command sent (and ack received if expected)
        """
        if self.serial_port is None:
            return False

        with self.serial_lock:
            try:
                # Send command with newline
                self.serial_port.write(f"{command}\n".encode())
                self.serial_port.flush()

                if expect_response:
                    # Wait for response
                    response = self.serial_port.readline().decode().strip()

                    if response in ["OK", "PONG", "SPRAY_COMPLETE", "HEARTBEAT_OK"]:
                        return True
                    elif response == "ERROR":
                        self.get_logger().error(f"ESP32 error for command: {command}")
                        return False
                    else:
                        self.get_logger().warning(f"Unexpected response: {response}")
                        return False

                return True

            except serial.SerialException as e:
                self.get_logger().error(f"Serial error: {e}")
                self.connected = False
                return False

    def _emergency_stop(self, reason: str) -> None:
        """
        Trigger emergency stop.

        Args:
            reason: Reason for emergency stop
        """
        self.get_logger().error(f"EMERGENCY STOP: {reason}")
        self.emergency_stopped = True

        # Send stop command to ESP32
        if self.connected:
            self._send_command("STOP")

        self._publish_status("emergency_stop", {"reason": reason})

    def _resume_from_emergency(self) -> None:
        """Resume operation after emergency stop."""
        if not self.emergency_stopped:
            return

        require_confirm = self.config.get('emergency', {}).get(
            'require_resume_confirm', True
        )

        if require_confirm:
            self.get_logger().info("Resume requested - awaiting confirmation")
            self._publish_status("resume_pending", {})
        else:
            self.emergency_stopped = False
            self.get_logger().info("Resumed from emergency stop")
            self._publish_status("resumed", {})

    def _log_spray_event(self, event: SprayEvent) -> None:
        """Log spray event to CSV file."""
        self.spray_history.append(event)

        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    event.timestamp.isoformat(),
                    event.duration_ms,
                    event.disease,
                    event.confidence,
                    event.trigger,
                    event.success,
                    event.error
                ])
        except Exception as e:
            self.get_logger().error(f"Failed to log spray event: {e}")

    def _publish_status(self, event_type: str, data: Dict) -> None:
        """Publish status update."""
        status = {
            'event': event_type,
            'timestamp': datetime.now().isoformat(),
            'connected': self.connected,
            'emergency_stopped': self.emergency_stopped,
            'spray_count': self.spray_count,
            **data
        }

        msg = String()
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)

    def _publish_full_status(self) -> None:
        """Publish complete status information."""
        status = {
            'connected': self.connected,
            'emergency_stopped': self.emergency_stopped,
            'spray_count': self.spray_count,
            'last_spray': (
                self.last_spray_time.isoformat()
                if self.last_spray_time else None
            ),
            'pending_confirmation': self.pending_confirmation is not None,
            'safety_status': self.safety_monitor.get_status()
        }
        self._publish_status("status", status)

    def destroy_node(self) -> None:
        """Clean shutdown."""
        self.get_logger().info("Shutting down ESP32 Controller")

        # Stop heartbeat
        self.heartbeat_running = False
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=2.0)

        # Close serial connection
        if self.serial_port and self.serial_port.is_open:
            self._send_command("STOP")  # Safety stop
            self.serial_port.close()

        super().destroy_node()


def main(args=None) -> None:
    """Main entry point for ESP32 controller."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    rclpy.init(args=args)

    node = ESP32Controller()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
