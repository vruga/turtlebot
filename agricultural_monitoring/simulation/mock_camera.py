#!/usr/bin/env python3
"""
Mock Camera for Testing on macOS

Simulates camera input by:
1. Using laptop webcam if available
2. Generating synthetic plant images
3. Loading test images from disk

Usage:
    python simulation/mock_camera.py [--webcam] [--synthetic] [--images-dir DIR]

Author: Agricultural Robotics Team
"""

import os
import sys
import time
import threading
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available. Install with: pip install opencv-python")

# Add project to path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'src'))


class MockCamera:
    """
    Mock camera for testing without ROS2 or real hardware.

    Can use:
    - macOS webcam (if available)
    - Synthetic test images
    - Images from a directory
    """

    def __init__(
        self,
        use_webcam: bool = False,
        images_dir: Optional[Path] = None,
        frame_rate: int = 30
    ):
        self.use_webcam = use_webcam
        self.images_dir = images_dir
        self.frame_rate = frame_rate

        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()

        self.test_images: list = []
        self.image_index = 0

        # Callbacks
        self.on_frame: Optional[Callable] = None

        self._setup()

    def _setup(self):
        """Initialize camera or load test images."""
        if self.use_webcam and CV2_AVAILABLE:
            # Try to open webcam
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                print(f"Webcam opened successfully")
                return
            else:
                print("Webcam not available, using synthetic images")
                self.cap = None

        # Load test images if directory provided
        if self.images_dir and self.images_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                self.test_images.extend(list(self.images_dir.glob(ext)))
            if self.test_images:
                print(f"Loaded {len(self.test_images)} test images")
                return

        # Fall back to synthetic images
        print("Using synthetic test images")

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generate a synthetic plant-like test image."""
        # Create base green background (plant-like)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Green background with variation
        frame[:, :, 1] = np.random.randint(80, 150)  # Green channel
        frame[:, :, 0] = np.random.randint(20, 60)   # Blue
        frame[:, :, 2] = np.random.randint(20, 80)   # Red

        # Add some leaf-like shapes
        for _ in range(5):
            center = (np.random.randint(100, 540), np.random.randint(100, 380))
            axes = (np.random.randint(30, 80), np.random.randint(20, 50))
            angle = np.random.randint(0, 180)

            # Leaf color (green with slight variation)
            color = (
                np.random.randint(30, 70),
                np.random.randint(100, 180),
                np.random.randint(30, 80)
            )

            cv2.ellipse(frame, center, axes, angle, 0, 360, color, -1)

        # Randomly add "disease spots" (brown/yellow patches)
        if np.random.random() > 0.3:  # 70% chance of disease
            num_spots = np.random.randint(3, 15)
            for _ in range(num_spots):
                center = (np.random.randint(50, 590), np.random.randint(50, 430))
                radius = np.random.randint(5, 25)

                # Disease color (brown/yellow)
                color = (
                    np.random.randint(20, 60),
                    np.random.randint(80, 150),
                    np.random.randint(100, 200)
                )

                cv2.circle(frame, center, radius, color, -1)

        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"MOCK: {timestamp}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the current frame."""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame

        if self.test_images:
            img_path = self.test_images[self.image_index % len(self.test_images)]
            self.image_index += 1
            frame = cv2.imread(str(img_path))
            if frame is not None:
                return frame

        return self._generate_synthetic_frame()

    def start(self):
        """Start continuous frame capture in background."""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("Mock camera started")

    def stop(self):
        """Stop frame capture."""
        self.running = False
        if self.cap:
            self.cap.release()
        print("Mock camera stopped")

    def _capture_loop(self):
        """Background frame capture loop."""
        interval = 1.0 / self.frame_rate

        while self.running:
            frame = self.get_frame()

            if frame is not None:
                with self.frame_lock:
                    self.latest_frame = frame

                if self.on_frame:
                    self.on_frame(frame)

            time.sleep(interval)

    def capture_to_file(self, filepath: str) -> bool:
        """Capture current frame to file."""
        frame = self.get_frame()
        if frame is not None:
            cv2.imwrite(filepath, frame)
            print(f"Frame saved to {filepath}")
            return True
        return False

    def show_preview(self):
        """Show live preview window (for testing)."""
        print("Press 'q' to quit, 'c' to capture")

        while True:
            frame = self.get_frame()
            if frame is not None:
                cv2.imshow('Mock Camera Preview', frame)

            key = cv2.waitKey(33) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.capture_to_file('/tmp/captured_frame.jpg')

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Mock Camera for Testing')
    parser.add_argument('--webcam', action='store_true', help='Use laptop webcam')
    parser.add_argument('--images-dir', type=str, help='Directory with test images')
    parser.add_argument('--preview', action='store_true', help='Show preview window')
    parser.add_argument('--capture', type=str, help='Capture single frame to file')

    args = parser.parse_args()

    images_dir = Path(args.images_dir) if args.images_dir else None

    camera = MockCamera(
        use_webcam=args.webcam,
        images_dir=images_dir
    )

    if args.capture:
        camera.capture_to_file(args.capture)
    elif args.preview:
        camera.show_preview()
    else:
        # Just capture one frame to /tmp
        camera.capture_to_file('/tmp/captured_frame.jpg')


if __name__ == '__main__':
    main()
