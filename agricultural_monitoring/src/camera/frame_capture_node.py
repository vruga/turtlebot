#!/usr/bin/env python3
"""
Frame Capture Node for Agricultural Disease Detection System

This ROS2 node subscribes to the camera image topic and captures frames
on demand when triggered by keyboard input (SPACEBAR) or dashboard button.

CRITICAL: This node must be lightweight (<5% CPU) to avoid interfering
with teleop control. Frame capture is non-blocking and runs in a
separate callback from the main ROS2 spin.

Author: Agricultural Robotics Team
License: MIT
"""

import os
import sys
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String, Header
from cv_bridge import CvBridge
import cv2
import yaml

# Keyboard input handling
try:
    import pynput
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("Warning: pynput not available, keyboard capture disabled")


class FrameCaptureNode(Node):
    """
    ROS2 node for capturing camera frames on keyboard trigger.

    This node:
    1. Subscribes to /camera/image_raw (existing camera topic)
    2. Maintains a buffer of recent frames
    3. On SPACEBAR press, saves latest frame and publishes capture event
    4. Designed to be lightweight and non-blocking

    Topics:
        Subscribed:
            - /camera/image_raw (sensor_msgs/Image)
        Published:
            - /agricultural/capture_event (std_msgs/String)
    """

    def __init__(self) -> None:
        super().__init__('frame_capture_node')

        # Load configuration
        self.config = self._load_config()

        # Initialize CV bridge for ROS image conversion
        self.bridge = CvBridge()

        # Frame buffer (stores most recent frame)
        self.latest_frame: Optional[cv2.typing.MatLike] = None
        self.frame_timestamp: Optional[datetime] = None
        self.frame_lock = threading.Lock()

        # Capture state
        self.capture_in_progress = False
        self.capture_count = 0

        # Setup output directory
        self.output_dir = Path(self.config.get('capture', {}).get(
            'output_dir', '/tmp/agricultural_captures'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # QoS profile for camera subscription (best effort for performance)
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1  # Only keep latest frame
        )

        # QoS profile for capture event (reliable delivery)
        event_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )

        # Subscribe to camera topic
        camera_topic = self.config.get('topics', {}).get(
            'camera_raw', '/camera/image_raw')
        self.camera_sub = self.create_subscription(
            Image,
            camera_topic,
            self._camera_callback,
            camera_qos
        )

        # Publisher for capture events
        capture_topic = self.config.get('topics', {}).get(
            'capture_event', '/agricultural/capture_event')
        self.capture_pub = self.create_publisher(
            String,
            capture_topic,
            event_qos
        )

        # Subscribe to dashboard trigger topic
        self.dashboard_trigger_sub = self.create_subscription(
            String,
            '/agricultural/capture_trigger',
            self._dashboard_trigger_callback,
            event_qos
        )

        # Setup keyboard listener (non-blocking)
        self.keyboard_listener: Optional[keyboard.Listener] = None
        if PYNPUT_AVAILABLE:
            self._setup_keyboard_listener()

        self.get_logger().info(
            f"Frame Capture Node initialized. "
            f"Listening on {camera_topic}, "
            f"Publishing to {capture_topic}"
        )
        self.get_logger().info(f"Output directory: {self.output_dir}")
        self.get_logger().info("Press SPACEBAR to capture frame")

    def _load_config(self) -> dict:
        """Load configuration from YAML files."""
        config = {}

        # Try to load system config
        config_paths = [
            Path(__file__).parent.parent.parent / 'config' / 'system_config.yaml',
            Path(__file__).parent.parent.parent / 'config' / 'model_config.yaml',
        ]

        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        loaded = yaml.safe_load(f)
                        if loaded:
                            config.update(loaded)
                except Exception as e:
                    self.get_logger().warning(f"Failed to load {config_path}: {e}")

        return config

    def _setup_keyboard_listener(self) -> None:
        """
        Setup non-blocking keyboard listener for SPACEBAR capture.

        Uses pynput for cross-platform keyboard monitoring.
        The listener runs in a separate thread and doesn't block ROS2.
        """
        capture_key = self.config.get('keybinds', {}).get('capture_frame', 'space')

        def on_press(key):
            try:
                # Check for spacebar
                if key == keyboard.Key.space:
                    self._trigger_capture("keyboard")
                # Check for escape (emergency stop)
                elif key == keyboard.Key.esc:
                    self.get_logger().warning("Escape pressed - emergency stop")
                    self._publish_emergency_stop()
            except AttributeError:
                pass

        self.keyboard_listener = keyboard.Listener(on_press=on_press)
        self.keyboard_listener.daemon = True  # Thread dies with main program
        self.keyboard_listener.start()

        self.get_logger().info(f"Keyboard listener started (capture key: {capture_key})")

    def _camera_callback(self, msg: Image) -> None:
        """
        Callback for camera image messages.

        This callback is kept minimal to avoid blocking:
        - Convert ROS image to OpenCV format
        - Store in buffer (thread-safe)
        - Exit immediately

        Args:
            msg: ROS2 Image message from camera
        """
        try:
            # Convert ROS image to OpenCV (BGR format)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Thread-safe buffer update
            with self.frame_lock:
                self.latest_frame = cv_image
                self.frame_timestamp = datetime.now()

        except Exception as e:
            self.get_logger().error(f"Failed to process camera frame: {e}")

    def _dashboard_trigger_callback(self, msg: String) -> None:
        """
        Callback for dashboard capture trigger.

        Allows the web dashboard to trigger frame capture via ROS2 topic.

        Args:
            msg: String message (content not used, trigger on receive)
        """
        self._trigger_capture("dashboard")

    def _trigger_capture(self, source: str) -> None:
        """
        Trigger frame capture from any source.

        Thread-safe capture that:
        1. Checks if capture is already in progress
        2. Copies latest frame from buffer
        3. Saves to disk in background thread
        4. Publishes capture event

        Args:
            source: Trigger source ("keyboard" or "dashboard")
        """
        # Prevent concurrent captures
        if self.capture_in_progress:
            self.get_logger().debug("Capture already in progress, ignoring trigger")
            return

        # Check if we have a frame
        with self.frame_lock:
            if self.latest_frame is None:
                self.get_logger().warning("No frame available to capture")
                return

            # Copy frame to avoid holding lock during save
            frame_copy = self.latest_frame.copy()
            timestamp = self.frame_timestamp

        # Mark capture in progress
        self.capture_in_progress = True
        self.capture_count += 1

        # Generate filename with timestamp
        filename = f"capture_{timestamp.strftime('%Y%m%d_%H%M%S')}_{self.capture_count:04d}.jpg"
        filepath = self.output_dir / filename

        # Save in background thread to avoid blocking
        save_thread = threading.Thread(
            target=self._save_frame,
            args=(frame_copy, filepath, source),
            daemon=True
        )
        save_thread.start()

        self.get_logger().info(f"Capture triggered ({source}): {filename}")

    def _save_frame(
        self,
        frame: cv2.typing.MatLike,
        filepath: Path,
        source: str
    ) -> None:
        """
        Save frame to disk and publish capture event.

        Runs in background thread to avoid blocking ROS2 callbacks.

        Args:
            frame: OpenCV image to save
            filepath: Destination path
            source: Trigger source for event metadata
        """
        try:
            # Get quality setting
            quality = self.config.get('camera', {}).get('capture_quality', 95)

            # Save frame as JPEG
            cv2.imwrite(
                str(filepath),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, quality]
            )

            # Also save to standard location for inference worker
            standard_path = Path('/tmp/captured_frame.jpg')
            cv2.imwrite(
                str(standard_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, quality]
            )

            # Publish capture event
            event_msg = String()
            event_msg.data = f"{filepath}|{source}|{datetime.now().isoformat()}"
            self.capture_pub.publish(event_msg)

            self.get_logger().debug(f"Frame saved: {filepath}")

            # Cleanup old frames if needed
            self._cleanup_old_frames()

        except Exception as e:
            self.get_logger().error(f"Failed to save frame: {e}")
        finally:
            self.capture_in_progress = False

    def _cleanup_old_frames(self) -> None:
        """Remove old captured frames if over limit."""
        max_frames = self.config.get('capture', {}).get('max_frames', 100)
        keep_frames = self.config.get('capture', {}).get('keep_frames', True)

        if not keep_frames:
            return

        try:
            # Get all capture files sorted by modification time
            captures = sorted(
                self.output_dir.glob('capture_*.jpg'),
                key=lambda p: p.stat().st_mtime
            )

            # Remove oldest if over limit
            while len(captures) > max_frames:
                oldest = captures.pop(0)
                oldest.unlink()
                self.get_logger().debug(f"Removed old capture: {oldest.name}")

        except Exception as e:
            self.get_logger().warning(f"Cleanup failed: {e}")

    def _publish_emergency_stop(self) -> None:
        """Publish emergency stop event."""
        stop_msg = String()
        stop_msg.data = "EMERGENCY_STOP"
        self.capture_pub.publish(stop_msg)

    def get_latest_frame(self) -> Optional[tuple]:
        """
        Get the latest captured frame (for testing/debugging).

        Returns:
            Tuple of (frame, timestamp) or None if no frame available
        """
        with self.frame_lock:
            if self.latest_frame is not None:
                return (self.latest_frame.copy(), self.frame_timestamp)
        return None

    def destroy_node(self) -> None:
        """Clean shutdown of the node."""
        self.get_logger().info("Shutting down Frame Capture Node")

        # Stop keyboard listener
        if self.keyboard_listener is not None:
            self.keyboard_listener.stop()

        super().destroy_node()


def main(args=None) -> None:
    """Main entry point for the frame capture node."""
    rclpy.init(args=args)

    node = FrameCaptureNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
