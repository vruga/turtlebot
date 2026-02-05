#!/usr/bin/env python3
"""
Standalone Test Runner for macOS

Runs the complete disease detection pipeline WITHOUT ROS2.
Perfect for testing on macOS before deploying to Raspberry Pi.

Features:
- Mock camera (webcam or synthetic images)
- Mock ESP32 (simulated serial)
- Real TFLite inference (if model available)
- Real Claude API (if key provided)
- Web dashboard

Usage:
    python simulation/run_standalone.py [options]

    Options:
        --webcam        Use laptop webcam instead of synthetic images
        --no-dashboard  Don't start web dashboard
        --no-llm        Don't use Claude API
        --mock-model    Use random predictions instead of real model

Author: Agricultural Robotics Team
"""

import os
import sys
import time
import json
import threading
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import queue

# Add project to path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / 'src'))

import numpy as np

# Import project modules
from src.camera.image_preprocessor import ImagePreprocessor
from src.inference.disease_classifier import DiseaseClassifier, DetectionResult, DetectionHistory
from src.spray_control.safety_monitor import SafetyMonitor
from src.spray_control.spray_decision import SprayDecisionMaker
from src.llm.prompt_builder import PromptBuilder
from src.llm.recommendation_cache import RecommendationCache

# Try to import optional modules
try:
    from src.inference.model_loader import ModelLoader
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from flask import Flask, render_template, jsonify, request, Response
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


class StandaloneSystem:
    """
    Standalone agricultural detection system for testing.

    Runs without ROS2, using direct Python calls.
    """

    def __init__(
        self,
        use_webcam: bool = False,
        use_mock_model: bool = False,
        enable_llm: bool = True,
        enable_dashboard: bool = True
    ):
        self.use_webcam = use_webcam
        self.use_mock_model = use_mock_model
        self.enable_llm = enable_llm and ANTHROPIC_AVAILABLE
        self.enable_dashboard = enable_dashboard and FLASK_AVAILABLE

        # State
        self.running = False
        self.latest_detection: Optional[DetectionResult] = None
        self.latest_recommendation: Optional[str] = None
        self.detection_history = DetectionHistory(max_entries=50)

        # Event queue for dashboard
        self.event_queue: queue.Queue = queue.Queue(maxsize=100)

        # Initialize components
        self._init_components()

        print("\n" + "=" * 50)
        print("Standalone Agricultural Detection System")
        print("=" * 50)
        print(f"Camera: {'Webcam' if use_webcam else 'Synthetic'}")
        print(f"Model: {'Mock' if use_mock_model else 'TFLite'}")
        print(f"LLM: {'Enabled' if self.enable_llm else 'Disabled'}")
        print(f"Dashboard: {'Enabled' if self.enable_dashboard else 'Disabled'}")
        print("=" * 50 + "\n")

    def _init_components(self):
        """Initialize all components."""
        config_dir = PROJECT_DIR / 'config'

        # Image preprocessor
        self.preprocessor = ImagePreprocessor(
            config_path=config_dir / 'model_config.yaml'
        )

        # Model loader (or mock)
        self.model_loader = None
        model_path = PROJECT_DIR / 'models' / 'plant_disease_model.tflite'

        if not self.use_mock_model and MODEL_AVAILABLE and model_path.exists():
            self.model_loader = ModelLoader(model_path=model_path)
            if self.model_loader.is_loaded():
                print(f"Model loaded: {model_path.name}")
            else:
                print("Model failed to load, using mock predictions")
                self.model_loader = None

        if self.model_loader is None:
            print("Using mock model predictions")

        # Classifier
        self.classifier = DiseaseClassifier(config_path=config_dir)

        # Spray decision and safety
        self.decision_maker = SprayDecisionMaker(config_path=config_dir)
        self.safety_monitor = SafetyMonitor(
            max_spray_duration=10000,
            cooldown_seconds=2.0,
            max_sprays_per_hour=20
        )

        # LLM components
        self.prompt_builder = PromptBuilder(config_path=config_dir / 'llm_config.yaml')
        self.cache = RecommendationCache(enabled=True, ttl_hours=24)
        self.anthropic_client = None

        if self.enable_llm:
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if api_key:
                self.anthropic_client = anthropic.Anthropic(api_key=api_key)
                print("Claude API initialized")
            else:
                print("ANTHROPIC_API_KEY not set, LLM disabled")
                self.enable_llm = False

        # Camera
        self.cap = None
        if self.use_webcam and CV2_AVAILABLE:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Webcam not available")
                self.cap = None

    def capture_frame(self) -> np.ndarray:
        """Capture a frame from camera or generate synthetic."""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame

        # Generate synthetic frame
        return self._generate_synthetic_frame()

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generate synthetic plant image for testing."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Green background
        frame[:, :, 1] = np.random.randint(80, 150)
        frame[:, :, 0] = np.random.randint(20, 60)
        frame[:, :, 2] = np.random.randint(20, 80)

        # Add leaf shapes
        if CV2_AVAILABLE:
            for _ in range(5):
                center = (np.random.randint(100, 540), np.random.randint(100, 380))
                axes = (np.random.randint(30, 80), np.random.randint(20, 50))
                color = (np.random.randint(30, 70), np.random.randint(100, 180), np.random.randint(30, 80))
                cv2.ellipse(frame, center, axes, np.random.randint(0, 180), 0, 360, color, -1)

            # Disease spots
            if np.random.random() > 0.3:
                for _ in range(np.random.randint(3, 15)):
                    center = (np.random.randint(50, 590), np.random.randint(50, 430))
                    radius = np.random.randint(5, 25)
                    color = (np.random.randint(20, 60), np.random.randint(80, 150), np.random.randint(100, 200))
                    cv2.circle(frame, center, radius, color, -1)

        return frame

    def run_inference(self, frame: np.ndarray) -> DetectionResult:
        """Run inference on a frame."""
        # Preprocess
        processed = self.preprocessor.preprocess(frame)

        # Get predictions
        if self.model_loader and self.model_loader.is_loaded():
            predictions = self.model_loader.predict(processed)
        else:
            # Mock predictions
            num_classes = len(self.classifier.classes)
            predictions = np.random.random((1, num_classes))
            # Make one class dominant
            dominant = np.random.randint(0, num_classes)
            predictions[0, dominant] = np.random.uniform(0.7, 0.95)
            predictions = predictions / predictions.sum()

        # Classify
        result = self.classifier.classify(predictions)
        return result

    def make_spray_decision(self, detection: DetectionResult) -> Dict[str, Any]:
        """Make spray decision based on detection."""
        should_spray, duration, reason = self.decision_maker.decide(
            detection.disease_name,
            detection.confidence,
            detection.severity
        )

        spray_executed = False
        safety_reason = ""

        if should_spray:
            can_spray, safety_reason = self.safety_monitor.can_spray(duration)
            if can_spray:
                self.safety_monitor.record_spray(duration)
                spray_executed = True
                print(f"  [SPRAY] {duration}ms for {detection.disease_name}")
            else:
                print(f"  [BLOCKED] {safety_reason}")

        return {
            'should_spray': should_spray,
            'spray_executed': spray_executed,
            'duration': duration,
            'reason': reason,
            'safety_reason': safety_reason
        }

    def get_recommendation(self, detection: DetectionResult) -> str:
        """Get LLM recommendation for detection."""
        if not self.enable_llm or not self.anthropic_client:
            return self._get_fallback_recommendation(detection.disease_name)

        # Check cache
        cache_key = self.cache.build_key(detection.disease_name, detection.confidence)
        cached = self.cache.get(cache_key)
        if cached:
            print("  [LLM] Cache hit")
            return cached

        # Build prompt
        prompt = self.prompt_builder.build_prompt(
            detection=detection.to_dict(),
            history=[d.to_dict() for d in self.detection_history.get_recent(5)],
            context={'is_first_today': len(self.detection_history.get_today()) <= 1}
        )

        try:
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                system=self.prompt_builder.get_system_prompt(),
                messages=[{"role": "user", "content": prompt}],
                timeout=10
            )

            recommendation = response.content[0].text
            self.cache.set(cache_key, recommendation)
            print("  [LLM] API response received")
            return recommendation

        except Exception as e:
            print(f"  [LLM] Error: {e}")
            return self._get_fallback_recommendation(detection.disease_name)

    def _get_fallback_recommendation(self, disease_name: str) -> str:
        """Get fallback recommendation without API."""
        fallbacks = {
            'healthy': "Plant appears healthy. Continue regular monitoring.",
            'early_blight': "Early blight detected. Remove affected leaves, improve air circulation.",
            'late_blight': "Late blight is serious. Remove infected plants immediately.",
            'leaf_mold': "Leaf mold present. Reduce humidity, improve ventilation.",
        }
        return fallbacks.get(disease_name, f"Disease '{disease_name}' detected. Monitor closely.")

    def process_capture(self) -> Dict[str, Any]:
        """Process a single capture through the full pipeline."""
        print("\n" + "-" * 40)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing capture...")

        # 1. Capture frame
        frame = self.capture_frame()
        frame_path = '/tmp/captured_frame.jpg'
        if CV2_AVAILABLE:
            cv2.imwrite(frame_path, frame)

        # 2. Run inference
        start_time = time.time()
        detection = self.run_inference(frame)
        inference_time = time.time() - start_time

        print(f"  Detection: {detection.disease_name} ({detection.confidence:.1%})")
        print(f"  Severity: {detection.severity}")
        print(f"  Inference time: {inference_time*1000:.0f}ms")

        # 3. Add to history
        self.detection_history.add(detection)
        self.latest_detection = detection

        # 4. Make spray decision
        spray_result = self.make_spray_decision(detection)

        # 5. Get recommendation (async in real system)
        recommendation = self.get_recommendation(detection)
        self.latest_recommendation = recommendation

        print(f"  Recommendation: {recommendation[:100]}...")
        print("-" * 40)

        # Push to event queue for dashboard
        result = {
            'detection': detection.to_dict(),
            'spray': spray_result,
            'recommendation': recommendation,
            'inference_time': inference_time,
            'timestamp': datetime.now().isoformat()
        }

        try:
            self.event_queue.put_nowait(result)
        except queue.Full:
            pass

        return result

    def run_dashboard(self, port: int = 8080):
        """Run the web dashboard."""
        if not FLASK_AVAILABLE:
            print("Flask not available, dashboard disabled")
            return

        app = Flask(
            __name__,
            template_folder=str(PROJECT_DIR / 'src' / 'dashboard' / 'templates'),
            static_folder=str(PROJECT_DIR / 'src' / 'dashboard' / 'static')
        )

        system = self

        @app.route('/')
        def index():
            return render_template('index.html')

        @app.route('/api/state')
        def get_state():
            return jsonify({
                'latest_detection': system.latest_detection.to_dict() if system.latest_detection else None,
                'latest_recommendation': system.latest_recommendation,
                'detection_history': [d.to_dict() for d in system.detection_history.get_recent(10)],
                'spray_status': system.safety_monitor.get_status(),
                'system_status': {
                    'camera': 'ok',
                    'model': 'ok' if system.model_loader else 'mock',
                    'esp32': 'mock',
                    'llm': 'ok' if system.enable_llm else 'disabled'
                }
            })

        @app.route('/api/capture', methods=['POST'])
        def trigger_capture():
            result = system.process_capture()
            return jsonify({'success': True, 'result': result})

        @app.route('/api/latest-frame')
        def latest_frame():
            from flask import send_file
            frame_path = Path('/tmp/captured_frame.jpg')
            if frame_path.exists():
                return send_file(frame_path, mimetype='image/jpeg')
            return jsonify({'error': 'No frame'}), 404

        print(f"\nDashboard running at http://localhost:{port}")
        print("Press Ctrl+C to stop\n")

        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

    def run_interactive(self):
        """Run in interactive mode (CLI)."""
        print("\nInteractive Mode")
        print("Commands: [c]apture, [s]tatus, [h]istory, [q]uit")
        print("-" * 40)

        while True:
            try:
                cmd = input("\n> ").strip().lower()

                if cmd in ['c', 'capture']:
                    self.process_capture()

                elif cmd in ['s', 'status']:
                    print(f"\nSystem Status:")
                    print(f"  Model: {'Loaded' if self.model_loader else 'Mock'}")
                    print(f"  LLM: {'Enabled' if self.enable_llm else 'Disabled'}")
                    print(f"  Safety: {self.safety_monitor.get_status()}")

                elif cmd in ['h', 'history']:
                    print(f"\nDetection History ({len(self.detection_history.get_today())} today):")
                    for d in self.detection_history.get_recent(5):
                        print(f"  {d.timestamp.strftime('%H:%M:%S')}: {d.disease_name} ({d.confidence:.1%})")

                elif cmd in ['q', 'quit', 'exit']:
                    break

                else:
                    print("Unknown command. Use: capture, status, history, quit")

            except KeyboardInterrupt:
                break
            except EOFError:
                break

        print("\nGoodbye!")

    def cleanup(self):
        """Cleanup resources."""
        if self.cap:
            self.cap.release()
        self.cache.save()


def main():
    parser = argparse.ArgumentParser(description='Standalone Test System')
    parser.add_argument('--webcam', action='store_true', help='Use laptop webcam')
    parser.add_argument('--mock-model', action='store_true', help='Use mock predictions')
    parser.add_argument('--no-llm', action='store_true', help='Disable Claude API')
    parser.add_argument('--no-dashboard', action='store_true', help='Disable web dashboard')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive CLI mode')
    parser.add_argument('--port', type=int, default=8080, help='Dashboard port')

    args = parser.parse_args()

    system = StandaloneSystem(
        use_webcam=args.webcam,
        use_mock_model=args.mock_model,
        enable_llm=not args.no_llm,
        enable_dashboard=not args.no_dashboard
    )

    try:
        if args.interactive or args.no_dashboard:
            system.run_interactive()
        else:
            system.run_dashboard(port=args.port)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        system.cleanup()


if __name__ == '__main__':
    main()
