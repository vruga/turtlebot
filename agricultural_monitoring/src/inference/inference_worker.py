#!/usr/bin/env python3
"""
Inference Worker for Agricultural Disease Detection

This ROS2 node runs plant disease inference in a separate process
with LOW PRIORITY (nice +15) to ensure teleop responsiveness.

CRITICAL: This worker must NEVER block the main ROS2 system.
- Runs with nice +15 priority
- Uses separate thread for inference
- Publishes results asynchronously

Author: Agricultural Robotics Team
License: MIT
"""

import os
import sys
import time
import threading
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.camera.image_preprocessor import ImagePreprocessor
from src.inference.model_loader import ModelLoader
from src.inference.disease_classifier import DiseaseClassifier, DetectionResult, DetectionHistory


logger = logging.getLogger(__name__)


class InferenceWorker(Node):
    """
    ROS2 node for plant disease model inference.

    Runs as a separate low-priority process to avoid blocking teleop.

    Topics:
        Subscribed:
            - /agricultural/capture_event (String): Triggers inference
        Published:
            - /agricultural/disease_detection (String): Detection results as JSON
    """

    def __init__(self) -> None:
        super().__init__('inference_worker')

        # Set process priority (nice +15 for low priority)
        self._set_low_priority()

        # Load configuration
        self.config = self._load_config()

        # Initialize components
        config_dir = Path(__file__).parent.parent.parent / 'config'

        self.preprocessor = ImagePreprocessor(
            config_path=config_dir / 'model_config.yaml'
        )

        self.model_loader = ModelLoader(
            config_path=config_dir / 'model_config.yaml'
        )

        self.classifier = DiseaseClassifier(
            config_path=config_dir
        )

        # Detection history
        self.history = DetectionHistory(max_entries=100)

        # Inference state
        self.inference_in_progress = False
        self.inference_lock = threading.Lock()
        self.last_inference_time: Optional[datetime] = None
        self.inference_count = 0
        self.total_inference_time = 0.0

        # QoS profiles
        event_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )

        result_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )

        # Subscriber for capture events
        capture_topic = self.config.get('topics', {}).get(
            'capture_event', '/agricultural/capture_event'
        )
        self.capture_sub = self.create_subscription(
            String,
            capture_topic,
            self._capture_event_callback,
            event_qos
        )

        # Publisher for detection results
        detection_topic = self.config.get('topics', {}).get(
            'disease_detection', '/agricultural/disease_detection'
        )
        self.detection_pub = self.create_publisher(
            String,
            detection_topic,
            result_qos
        )

        # Status publisher
        self.status_pub = self.create_publisher(
            String,
            '/agricultural/inference_status',
            result_qos
        )

        # Log startup status
        model_loaded = self.model_loader.is_loaded()
        status = "ready" if model_loaded else "model not loaded"

        self.get_logger().info(f"Inference Worker initialized ({status})")
        self.get_logger().info(f"Listening on: {capture_topic}")
        self.get_logger().info(f"Publishing to: {detection_topic}")

        if model_loaded:
            self.get_logger().info(f"Model info: {self.model_loader.get_model_info()}")
        else:
            self.get_logger().warning(
                "Model not loaded! Ensure plant_disease_model.tflite exists"
            )

    def _set_low_priority(self) -> None:
        """Set process to low priority (nice +15) to not interfere with teleop."""
        try:
            # Get configured priority
            config_dir = Path(__file__).parent.parent.parent / 'config'
            priority = 15  # default

            system_config = config_dir / 'system_config.yaml'
            if system_config.exists():
                with open(system_config, 'r') as f:
                    config = yaml.safe_load(f)
                    priority = config.get('priority', {}).get('inference', 15)

            # Set nice value (higher = lower priority)
            os.nice(priority)
            self.get_logger().info(f"Process priority set to nice +{priority}")

        except PermissionError:
            self.get_logger().warning(
                "Could not set process priority (not running as root)"
            )
        except Exception as e:
            self.get_logger().warning(f"Failed to set priority: {e}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML files."""
        config = {}
        config_dir = Path(__file__).parent.parent.parent / 'config'

        for config_file in ['system_config.yaml', 'model_config.yaml']:
            config_path = config_dir / config_file
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        loaded = yaml.safe_load(f)
                        if loaded:
                            config.update(loaded)
                except Exception as e:
                    self.get_logger().warning(f"Failed to load {config_file}: {e}")

        return config

    def _capture_event_callback(self, msg: String) -> None:
        """
        Handle capture event and trigger inference.

        Runs inference in a separate thread to avoid blocking ROS2.

        Args:
            msg: Capture event message containing "filepath|source|timestamp"
        """
        # Check for emergency stop
        if msg.data == "EMERGENCY_STOP":
            self.get_logger().warning("Emergency stop received")
            return

        # Parse message
        try:
            parts = msg.data.split('|')
            filepath = parts[0]
            source = parts[1] if len(parts) > 1 else 'unknown'
            timestamp = parts[2] if len(parts) > 2 else None
        except Exception as e:
            self.get_logger().error(f"Failed to parse capture event: {e}")
            return

        # Check if inference already in progress
        with self.inference_lock:
            if self.inference_in_progress:
                self.get_logger().warning("Inference already in progress, queuing...")
                return
            self.inference_in_progress = True

        # Run inference in thread
        inference_thread = threading.Thread(
            target=self._run_inference,
            args=(filepath, source),
            daemon=True
        )
        inference_thread.start()

    def _run_inference(self, filepath: str, source: str) -> None:
        """
        Run model inference on captured frame.

        This runs in a separate thread with low priority.

        Args:
            filepath: Path to captured image
            source: Source of capture trigger
        """
        start_time = time.time()

        try:
            # Check if model is loaded
            if not self.model_loader.is_loaded():
                self.get_logger().error("Model not loaded, cannot run inference")
                self._publish_error("Model not loaded")
                return

            # Load and preprocess image
            self.get_logger().debug(f"Processing image: {filepath}")

            # Use standard capture path if file doesn't exist
            if not Path(filepath).exists():
                filepath = '/tmp/captured_frame.jpg'

            if not Path(filepath).exists():
                self.get_logger().error(f"Image not found: {filepath}")
                self._publish_error("Image not found")
                return

            # Preprocess
            preprocessed = self.preprocessor.preprocess(filepath)

            # Run inference
            predictions = self.model_loader.predict(preprocessed)

            # Classify result
            result = self.classifier.classify(predictions, image_path=filepath)

            # Calculate inference time
            inference_time = time.time() - start_time
            self.total_inference_time += inference_time
            self.inference_count += 1
            self.last_inference_time = datetime.now()

            # Add to history
            self.history.add(result)

            # Log result
            self.get_logger().info(
                f"Detection: {result.disease_name} "
                f"({result.confidence:.1%}) - {inference_time:.2f}s"
            )

            # Check target time
            target_time = self.config.get('inference', {}).get('target_time', 2.0)
            if inference_time > target_time:
                self.get_logger().warning(
                    f"Inference time {inference_time:.2f}s exceeds "
                    f"target {target_time}s"
                )

            # Publish result
            self._publish_result(result, inference_time)

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            self._publish_error(str(e))

        finally:
            with self.inference_lock:
                self.inference_in_progress = False

    def _publish_result(
        self,
        result: DetectionResult,
        inference_time: float
    ) -> None:
        """
        Publish detection result to ROS2 topic.

        Args:
            result: Detection result to publish
            inference_time: Time taken for inference
        """
        # Convert to JSON
        result_dict = result.to_dict()
        result_dict['inference_time'] = round(inference_time, 3)
        result_dict['detection_count'] = self.inference_count

        msg = String()
        msg.data = json.dumps(result_dict)

        self.detection_pub.publish(msg)
        self.get_logger().debug(f"Published detection result")

    def _publish_error(self, error_message: str) -> None:
        """Publish error message."""
        error_dict = {
            'error': True,
            'message': error_message,
            'timestamp': datetime.now().isoformat()
        }

        msg = String()
        msg.data = json.dumps(error_dict)

        self.detection_pub.publish(msg)

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        avg_time = (
            self.total_inference_time / self.inference_count
            if self.inference_count > 0 else 0
        )

        return {
            'total_inferences': self.inference_count,
            'total_time': round(self.total_inference_time, 2),
            'average_time': round(avg_time, 3),
            'last_inference': (
                self.last_inference_time.isoformat()
                if self.last_inference_time else None
            ),
            'model_loaded': self.model_loader.is_loaded(),
            'history_summary': self.history.get_summary()
        }

    def destroy_node(self) -> None:
        """Clean shutdown."""
        self.get_logger().info("Shutting down Inference Worker")
        self.get_logger().info(f"Final stats: {self.get_stats()}")
        super().destroy_node()


def main(args=None) -> None:
    """Main entry point for inference worker."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    rclpy.init(args=args)

    node = InferenceWorker()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
