#!/usr/bin/env python3
"""
Farmer Dashboard Web Application

Flask-based web dashboard for agricultural disease detection system.
Provides real-time monitoring and control interface optimized for
tablet use in field conditions.

Features:
- Live camera feed
- Detection history
- LLM recommendations display
- Manual spray control
- Emergency stop
- System status indicators

Author: Agricultural Robotics Team
License: MIT
"""

import os
import sys
import json
import logging
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Generator
import yaml

from flask import Flask, render_template, jsonify, request, Response, send_file

# ROS2 imports (conditional)
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from std_msgs.msg import String
    from sensor_msgs.msg import CompressedImage
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    logging.warning("ROS2 not available - running in standalone mode")


logger = logging.getLogger(__name__)


class DashboardState:
    """Shared state for dashboard data."""

    def __init__(self):
        self.latest_detection: Optional[Dict] = None
        self.detection_history: list = []
        self.max_history = 10

        self.latest_recommendation: Optional[Dict] = None
        self.spray_status: Optional[Dict] = None

        self.system_status = {
            'camera': 'unknown',
            'model': 'unknown',
            'esp32': 'unknown',
            'llm': 'unknown'
        }

        self.latest_frame: Optional[bytes] = None
        self.frame_timestamp: Optional[datetime] = None

        self.lock = threading.Lock()

        # SSE event queue
        self.event_queue: queue.Queue = queue.Queue(maxsize=100)

    def update_detection(self, detection: Dict) -> None:
        """Update with new detection result."""
        with self.lock:
            self.latest_detection = detection
            self.detection_history.insert(0, detection)
            if len(self.detection_history) > self.max_history:
                self.detection_history = self.detection_history[:self.max_history]

        self._push_event('detection', detection)

    def update_recommendation(self, recommendation: Dict) -> None:
        """Update with new LLM recommendation."""
        with self.lock:
            self.latest_recommendation = recommendation
        self._push_event('recommendation', recommendation)

    def update_spray_status(self, status: Dict) -> None:
        """Update spray system status."""
        with self.lock:
            self.spray_status = status
        self._push_event('spray_status', status)

    def update_frame(self, frame_data: bytes) -> None:
        """Update latest camera frame."""
        with self.lock:
            self.latest_frame = frame_data
            self.frame_timestamp = datetime.now()

    def update_system_status(self, component: str, status: str) -> None:
        """Update system component status."""
        with self.lock:
            self.system_status[component] = status
        self._push_event('system_status', self.system_status.copy())

    def _push_event(self, event_type: str, data: Any) -> None:
        """Push event to SSE queue."""
        try:
            self.event_queue.put_nowait({
                'type': event_type,
                'data': data,
                'timestamp': datetime.now().isoformat()
            })
        except queue.Full:
            pass  # Drop event if queue full

    def get_state(self) -> Dict[str, Any]:
        """Get current dashboard state."""
        with self.lock:
            return {
                'latest_detection': self.latest_detection,
                'detection_history': self.detection_history,
                'latest_recommendation': self.latest_recommendation,
                'spray_status': self.spray_status,
                'system_status': self.system_status.copy()
            }


# Global state instance
dashboard_state = DashboardState()


class ROS2Bridge(Node):
    """ROS2 node for bridging dashboard with ROS2 topics."""

    def __init__(self, state: DashboardState):
        super().__init__('dashboard_bridge')
        self.state = state

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )

        # Subscribe to detection results
        self.detection_sub = self.create_subscription(
            String,
            '/agricultural/disease_detection',
            self._detection_callback,
            qos
        )

        # Subscribe to recommendations
        self.recommendation_sub = self.create_subscription(
            String,
            '/agricultural/recommendation',
            self._recommendation_callback,
            qos
        )

        # Subscribe to spray status
        self.spray_sub = self.create_subscription(
            String,
            '/agricultural/spray_status',
            self._spray_callback,
            qos
        )

        # Subscribe to compressed image for MJPEG stream
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/image_compressed',
            self._image_callback,
            QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
                depth=1
            )
        )

        # Publisher for capture trigger
        self.capture_pub = self.create_publisher(
            String,
            '/agricultural/capture_trigger',
            qos
        )

        # Publisher for spray commands
        self.spray_cmd_pub = self.create_publisher(
            String,
            '/agricultural/spray_command',
            qos
        )

        # Publisher for spray confirmation
        self.confirm_pub = self.create_publisher(
            String,
            '/agricultural/spray_confirm',
            qos
        )

        self.get_logger().info("Dashboard ROS2 bridge initialized")

    def _detection_callback(self, msg: String) -> None:
        try:
            detection = json.loads(msg.data)
            self.state.update_detection(detection)
            self.state.update_system_status('model', 'ok')
        except json.JSONDecodeError:
            pass

    def _recommendation_callback(self, msg: String) -> None:
        try:
            recommendation = json.loads(msg.data)
            self.state.update_recommendation(recommendation)
            self.state.update_system_status('llm', 'ok')
        except json.JSONDecodeError:
            pass

    def _spray_callback(self, msg: String) -> None:
        try:
            status = json.loads(msg.data)
            self.state.update_spray_status(status)
            self.state.update_system_status('esp32', 'ok' if status.get('connected') else 'error')
        except json.JSONDecodeError:
            pass

    def _image_callback(self, msg: CompressedImage) -> None:
        self.state.update_frame(bytes(msg.data))
        self.state.update_system_status('camera', 'ok')

    def trigger_capture(self) -> None:
        """Trigger frame capture."""
        msg = String()
        msg.data = "dashboard_trigger"
        self.capture_pub.publish(msg)

    def send_spray_command(self, command: str) -> None:
        """Send spray command."""
        msg = String()
        msg.data = command
        self.spray_cmd_pub.publish(msg)

    def send_confirmation(self, confirm: bool) -> None:
        """Send spray confirmation."""
        msg = String()
        msg.data = "CONFIRM" if confirm else "CANCEL"
        self.confirm_pub.publish(msg)


# Global ROS2 bridge instance
ros2_bridge: Optional[ROS2Bridge] = None


def create_app(config_path: Optional[Path] = None) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        config_path: Path to system_config.yaml

    Returns:
        Configured Flask application
    """
    # Get template and static directories
    template_dir = Path(__file__).parent / 'templates'
    static_dir = Path(__file__).parent / 'static'

    app = Flask(
        __name__,
        template_folder=str(template_dir),
        static_folder=str(static_dir)
    )

    # Load configuration
    config = {}
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'config' / 'system_config.yaml'

    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")

    dashboard_config = config.get('dashboard', {})
    app.config['SECRET_KEY'] = os.environ.get(
        dashboard_config.get('secret_key_env', 'DASHBOARD_SECRET_KEY'),
        'dev-secret-key-change-in-production'
    )
    app.config['DEBUG'] = dashboard_config.get('debug', False)

    # Routes
    @app.route('/')
    def index():
        """Main dashboard page."""
        return render_template('index.html')

    @app.route('/api/state')
    def get_state():
        """Get current dashboard state."""
        return jsonify(dashboard_state.get_state())

    @app.route('/api/capture', methods=['POST'])
    def trigger_capture():
        """Trigger frame capture."""
        if ros2_bridge:
            ros2_bridge.trigger_capture()
            return jsonify({'success': True, 'message': 'Capture triggered'})
        return jsonify({'success': False, 'message': 'ROS2 not available'}), 503

    @app.route('/api/spray', methods=['POST'])
    def spray_command():
        """Send spray command."""
        data = request.get_json() or {}
        command = data.get('command', '')

        if ros2_bridge:
            ros2_bridge.send_spray_command(command)
            return jsonify({'success': True, 'message': f'Command sent: {command}'})
        return jsonify({'success': False, 'message': 'ROS2 not available'}), 503

    @app.route('/api/spray/confirm', methods=['POST'])
    def spray_confirm():
        """Confirm or cancel pending spray."""
        data = request.get_json() or {}
        confirm = data.get('confirm', False)

        if ros2_bridge:
            ros2_bridge.send_confirmation(confirm)
            return jsonify({'success': True, 'confirmed': confirm})
        return jsonify({'success': False, 'message': 'ROS2 not available'}), 503

    @app.route('/api/emergency-stop', methods=['POST'])
    def emergency_stop():
        """Trigger emergency stop."""
        if ros2_bridge:
            ros2_bridge.send_spray_command('STOP')
            return jsonify({'success': True, 'message': 'Emergency stop triggered'})
        return jsonify({'success': False, 'message': 'ROS2 not available'}), 503

    @app.route('/api/resume', methods=['POST'])
    def resume():
        """Resume from emergency stop."""
        if ros2_bridge:
            ros2_bridge.send_spray_command('RESUME')
            return jsonify({'success': True, 'message': 'Resume requested'})
        return jsonify({'success': False, 'message': 'ROS2 not available'}), 503

    @app.route('/video_feed')
    def video_feed():
        """MJPEG video stream."""
        def generate() -> Generator[bytes, None, None]:
            while True:
                with dashboard_state.lock:
                    frame = dashboard_state.latest_frame

                if frame:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                # Limit frame rate
                import time
                time.sleep(0.033)  # ~30fps

        return Response(
            generate(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    @app.route('/events')
    def events():
        """Server-Sent Events endpoint."""
        def generate() -> Generator[str, None, None]:
            while True:
                try:
                    event = dashboard_state.event_queue.get(timeout=1.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except queue.Empty:
                    # Send keepalive
                    yield ": keepalive\n\n"

        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive'
            }
        )

    @app.route('/api/latest-frame')
    def latest_frame():
        """Get latest captured frame as JPEG."""
        capture_path = Path('/tmp/captured_frame.jpg')
        if capture_path.exists():
            return send_file(capture_path, mimetype='image/jpeg')
        return jsonify({'error': 'No frame available'}), 404

    return app


def run_dashboard(host: str = '0.0.0.0', port: int = 8080) -> None:
    """
    Run the dashboard with ROS2 integration.

    Args:
        host: Host to bind to
        port: Port to listen on
    """
    global ros2_bridge

    # Initialize ROS2 if available
    if ROS2_AVAILABLE:
        rclpy.init()
        ros2_bridge = ROS2Bridge(dashboard_state)

        # Spin ROS2 in background thread
        ros2_thread = threading.Thread(
            target=lambda: rclpy.spin(ros2_bridge),
            daemon=True
        )
        ros2_thread.start()
        logger.info("ROS2 bridge started")

    # Create and run Flask app
    app = create_app()

    try:
        logger.info(f"Starting dashboard at http://{host}:{port}")
        app.run(host=host, port=port, threaded=True)
    finally:
        if ROS2_AVAILABLE and ros2_bridge:
            ros2_bridge.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    run_dashboard()
