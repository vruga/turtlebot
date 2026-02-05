#!/usr/bin/env python3
"""
Mock ESP32 Serial Simulator

Simulates ESP32 spray controller responses for testing without hardware.
Creates a pseudo-terminal that responds like the real ESP32.

Usage:
    python simulation/mock_esp32.py

    Then connect your code to the PTY path printed at startup.

Author: Agricultural Robotics Team
"""

import os
import sys
import pty
import time
import threading
import select
from datetime import datetime
from typing import Optional


class MockESP32:
    """
    Simulates ESP32 spray controller serial interface.

    Creates a pseudo-terminal (PTY) that behaves like a serial port.
    Responds to all standard commands.
    """

    def __init__(self):
        self.master_fd: Optional[int] = None
        self.slave_fd: Optional[int] = None
        self.slave_path: Optional[str] = None

        self.running = False
        self.state = "IDLE"
        self.spray_start_time: Optional[float] = None
        self.spray_duration: int = 0
        self.emergency_stopped = False

        # Statistics
        self.spray_count = 0
        self.total_spray_time = 0

        self._create_pty()

    def _create_pty(self):
        """Create pseudo-terminal for serial simulation."""
        self.master_fd, self.slave_fd = pty.openpty()
        self.slave_path = os.ttyname(self.slave_fd)
        print(f"Mock ESP32 ready at: {self.slave_path}")
        print(f"Use this path in your code instead of /dev/ttyUSB0")

    def start(self):
        """Start the mock ESP32."""
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        print("Mock ESP32 started. Waiting for commands...")
        print("Commands: PING, HEARTBEAT, STATUS, SPRAY:<ms>, TEST:<ms>, STOP, RESUME")

    def stop(self):
        """Stop the mock ESP32."""
        self.running = False
        if self.master_fd:
            os.close(self.master_fd)
        if self.slave_fd:
            os.close(self.slave_fd)
        print("Mock ESP32 stopped")

    def _run_loop(self):
        """Main loop reading commands and sending responses."""
        buffer = ""

        while self.running:
            # Check for spray completion
            if self.state == "SPRAYING":
                elapsed = (time.time() - self.spray_start_time) * 1000
                if elapsed >= self.spray_duration:
                    self._complete_spray()

            # Check for incoming data
            readable, _, _ = select.select([self.master_fd], [], [], 0.1)

            if readable:
                try:
                    data = os.read(self.master_fd, 1024).decode('utf-8')
                    buffer += data

                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if line:
                            self._process_command(line)

                except OSError:
                    break

    def _process_command(self, command: str):
        """Process incoming command and send response."""
        command = command.upper().strip()
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] RX: {command}")

        if self.emergency_stopped and command != "RESUME":
            self._send("ERROR:EMERGENCY_STOPPED")
            return

        if command == "PING":
            self._send("PONG")

        elif command == "HEARTBEAT":
            self._send("HEARTBEAT_OK")

        elif command == "STATUS":
            self._send_status()

        elif command == "STOP":
            self._emergency_stop()
            self._send("STOPPED")

        elif command == "RESUME":
            self.emergency_stopped = False
            self.state = "IDLE"
            self._send("RESUMED")

        elif command.startswith("SPRAY:"):
            try:
                duration = int(command.split(':')[1])
                self._start_spray(duration)
            except (ValueError, IndexError):
                self._send("ERROR:INVALID_DURATION")

        elif command.startswith("TEST:"):
            try:
                duration = int(command.split(':')[1])
                if 50 <= duration <= 500:
                    print(f"  [TEST] Relay pulse for {duration}ms")
                    time.sleep(duration / 1000)
                    self._send("TEST_COMPLETE")
                else:
                    self._send("ERROR:INVALID_TEST_DURATION")
            except (ValueError, IndexError):
                self._send("ERROR:INVALID_DURATION")

        else:
            self._send("ERROR:UNKNOWN_COMMAND")

    def _send(self, response: str):
        """Send response to the PTY."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] TX: {response}")
        os.write(self.master_fd, (response + '\n').encode('utf-8'))

    def _send_status(self):
        """Send current status."""
        if self.state == "IDLE":
            self._send("STATUS:IDLE")
        elif self.state == "SPRAYING":
            elapsed = int((time.time() - self.spray_start_time) * 1000)
            self._send(f"STATUS:SPRAYING:{elapsed}/{self.spray_duration}")
        elif self.emergency_stopped:
            self._send("STATUS:EMERGENCY_STOPPED")

    def _start_spray(self, duration: int):
        """Start spray operation."""
        if duration < 100 or duration > 10000:
            self._send("ERROR:INVALID_DURATION")
            return

        if self.state == "SPRAYING":
            self._send("ERROR:ALREADY_SPRAYING")
            return

        self.state = "SPRAYING"
        self.spray_duration = duration
        self.spray_start_time = time.time()
        self.spray_count += 1

        print(f"  [SPRAY] Starting {duration}ms spray (#{self.spray_count})")
        self._send(f"SPRAY_STARTED:{duration}")

    def _complete_spray(self):
        """Complete spray operation."""
        actual_duration = int((time.time() - self.spray_start_time) * 1000)
        self.total_spray_time += actual_duration
        self.state = "IDLE"

        print(f"  [SPRAY] Complete ({actual_duration}ms actual)")
        self._send("SPRAY_COMPLETE")

    def _emergency_stop(self):
        """Trigger emergency stop."""
        self.emergency_stopped = True
        self.state = "STOPPED"

        if self.spray_start_time:
            # Was spraying, stop immediately
            print("  [EMERGENCY] Spray aborted!")

        print("  [EMERGENCY] System stopped")


def main():
    print("=" * 50)
    print("Mock ESP32 Spray Controller")
    print("=" * 50)

    esp32 = MockESP32()
    esp32.start()

    print("\nPress Ctrl+C to stop\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        esp32.stop()

        print(f"\nSession stats:")
        print(f"  Total sprays: {esp32.spray_count}")
        print(f"  Total spray time: {esp32.total_spray_time}ms")


if __name__ == '__main__':
    main()
