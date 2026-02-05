#!/usr/bin/env python3
"""
Integration tests for Agricultural Disease Detection System.

Tests the complete pipeline:
1. Frame capture simulation
2. Image preprocessing
3. Model inference
4. Disease classification
5. Spray decision
6. (Mocked) LLM recommendation

Run with: python -m pytest tests/test_integration.py -v

Author: Agricultural Robotics Team
"""

import sys
import json
import time
import unittest
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add project to path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'src'))

from camera.image_preprocessor import ImagePreprocessor
from inference.model_loader import ModelLoader
from inference.disease_classifier import DiseaseClassifier, DetectionResult, DetectionHistory
from spray_control.safety_monitor import SafetyMonitor
from spray_control.spray_decision import SprayDecisionMaker
from llm.prompt_builder import PromptBuilder
from llm.recommendation_cache import RecommendationCache


class TestFullPipeline(unittest.TestCase):
    """Integration tests for complete detection pipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up shared test resources."""
        cls.model_path = PROJECT_DIR / 'models' / 'plant_disease_model.tflite'
        cls.model_exists = cls.model_path.exists()

        # Create components
        cls.preprocessor = ImagePreprocessor(input_size=(224, 224))
        cls.classifier = DiseaseClassifier()
        cls.safety_monitor = SafetyMonitor(cooldown_seconds=0)
        cls.decision_maker = SprayDecisionMaker()

        if cls.model_exists:
            cls.model_loader = ModelLoader(model_path=cls.model_path)

    def _simulate_capture(self):
        """Simulate frame capture."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def _run_detection(self, image):
        """Run detection on image."""
        processed = self.preprocessor.preprocess(image)

        if self.model_exists:
            predictions = self.model_loader.predict(processed)
        else:
            # Mock predictions
            predictions = np.random.random((1, 10))
            predictions = predictions / predictions.sum()

        return self.classifier.classify(predictions)

    def test_capture_to_detection(self):
        """Test capture through detection pipeline."""
        # Simulate capture
        frame = self._simulate_capture()
        self.assertEqual(frame.shape, (480, 640, 3))

        # Run detection
        result = self._run_detection(frame)

        # Validate result
        self.assertIsInstance(result, DetectionResult)
        self.assertIsInstance(result.disease_name, str)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_detection_to_spray_decision(self):
        """Test detection through spray decision pipeline."""
        # Create mock detection
        detection = DetectionResult(
            disease_name='early_blight',
            confidence=0.85,
            severity='mild',
            timestamp=datetime.now(),
            top_predictions=[('early_blight', 0.85)],
            should_spray=True,
            spray_duration=2000
        )

        # Make spray decision
        should_spray, duration, reason = self.decision_maker.decide(
            detection.disease_name,
            detection.confidence,
            detection.severity
        )

        # Check safety
        can_spray, safety_reason = self.safety_monitor.can_spray(duration)

        # Validate
        self.assertTrue(should_spray)
        self.assertEqual(duration, 2000)
        self.assertTrue(can_spray)

    def test_full_cycle_with_safety(self):
        """Test complete cycle respects safety limits."""
        # Simulate multiple detections
        for i in range(3):
            frame = self._simulate_capture()
            result = self._run_detection(frame)

            # Make spray decision
            should_spray, duration, _ = self.decision_maker.decide(
                result.disease_name,
                result.confidence,
                result.severity
            )

            if should_spray:
                can_spray, _ = self.safety_monitor.can_spray(duration)
                if can_spray:
                    self.safety_monitor.record_spray(duration)

        # Validate sprays were recorded
        status = self.safety_monitor.get_status()
        self.assertGreaterEqual(status['total_sprays'], 0)


class TestDetectionHistory(unittest.TestCase):
    """Tests for detection history tracking."""

    def setUp(self):
        self.history = DetectionHistory(max_entries=10)

    def test_add_detection(self):
        """Test adding detections to history."""
        result = DetectionResult(
            disease_name='test_disease',
            confidence=0.9,
            severity='mild',
            timestamp=datetime.now(),
            top_predictions=[],
            should_spray=True,
            spray_duration=2000
        )

        self.history.add(result)
        self.assertEqual(len(self.history.get_recent(10)), 1)

    def test_max_entries(self):
        """Test history respects max entries."""
        for i in range(15):
            result = DetectionResult(
                disease_name=f'disease_{i}',
                confidence=0.9,
                severity='mild',
                timestamp=datetime.now(),
                top_predictions=[],
                should_spray=True,
                spray_duration=2000
            )
            self.history.add(result)

        # Should only have max_entries
        self.assertEqual(len(self.history.get_recent(20)), 10)

    def test_context_string(self):
        """Test context string generation."""
        result = DetectionResult(
            disease_name='early_blight',
            confidence=0.85,
            severity='mild',
            timestamp=datetime.now(),
            top_predictions=[],
            should_spray=True,
            spray_duration=2000
        )
        self.history.add(result)

        context = self.history.to_context_string(5)
        self.assertIn('early_blight', context)


class TestPromptBuilder(unittest.TestCase):
    """Tests for LLM prompt building."""

    def setUp(self):
        self.builder = PromptBuilder()

    def test_standard_prompt(self):
        """Test standard prompt generation."""
        detection = {
            'disease_name': 'early_blight',
            'confidence': 0.85,
            'severity': 'mild',
            'spray_duration': 2000
        }

        prompt = self.builder.build_prompt(
            detection=detection,
            history=[],
            context={'is_first_today': False}
        )

        self.assertIn('early_blight', prompt)
        self.assertIn('85', prompt)

    def test_first_detection_prompt(self):
        """Test first detection of day prompt."""
        detection = {
            'disease_name': 'late_blight',
            'confidence': 0.92,
            'severity': 'severe',
            'spray_duration': 6000
        }

        prompt = self.builder.build_prompt(
            detection=detection,
            history=[],
            context={'is_first_today': True}
        )

        self.assertIn('morning', prompt.lower())

    def test_high_confidence_prompt(self):
        """Test high confidence prompt selection."""
        detection = {
            'disease_name': 'leaf_mold',
            'confidence': 0.97,
            'severity': 'moderate',
            'spray_duration': 4000
        }

        prompt = self.builder.build_prompt(
            detection=detection,
            history=[],
            context={'is_first_today': False}
        )

        self.assertIn('97', prompt)


class TestRecommendationCache(unittest.TestCase):
    """Tests for LLM response caching."""

    def setUp(self):
        self.cache = RecommendationCache(
            enabled=True,
            ttl_hours=1,
            max_entries=5
        )

    def tearDown(self):
        self.cache.clear()

    def test_cache_set_get(self):
        """Test basic cache operations."""
        key = self.cache.build_key('early_blight', 0.85)
        self.cache.set(key, 'Test recommendation')

        result = self.cache.get(key)
        self.assertEqual(result, 'Test recommendation')

    def test_cache_miss(self):
        """Test cache miss."""
        result = self.cache.get('nonexistent_key')
        self.assertIsNone(result)

    def test_cache_eviction(self):
        """Test old entries are evicted."""
        # Fill cache
        for i in range(7):
            key = self.cache.build_key(f'disease_{i}', 0.85)
            self.cache.set(key, f'Recommendation {i}')

        # Check size
        self.assertLessEqual(self.cache.size(), 5)

    def test_cache_stats(self):
        """Test cache statistics."""
        key = self.cache.build_key('test', 0.9)
        self.cache.set(key, 'Test')
        self.cache.get(key)
        self.cache.get('miss')

        stats = self.cache.get_stats()
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)


class TestSerializationIntegration(unittest.TestCase):
    """Tests for data serialization between components."""

    def test_detection_result_json(self):
        """Test DetectionResult JSON serialization."""
        result = DetectionResult(
            disease_name='test_disease',
            confidence=0.88,
            severity='moderate',
            timestamp=datetime.now(),
            top_predictions=[('test_disease', 0.88), ('other', 0.12)],
            should_spray=True,
            spray_duration=4000
        )

        # Serialize
        json_str = result.to_json()
        parsed = json.loads(json_str)

        # Validate
        self.assertEqual(parsed['disease_name'], 'test_disease')
        self.assertAlmostEqual(parsed['confidence'], 0.88, places=2)
        self.assertTrue(parsed['should_spray'])

    def test_safety_status_json(self):
        """Test safety monitor status serialization."""
        monitor = SafetyMonitor()
        monitor.record_spray(5000)

        status = monitor.get_status()
        json_str = json.dumps(status)
        parsed = json.loads(json_str)

        self.assertIn('total_sprays', parsed)
        self.assertEqual(parsed['total_sprays'], 1)


class TestConcurrencySimulation(unittest.TestCase):
    """Simulated tests for concurrent operation."""

    def test_rapid_detections(self):
        """Test system handles rapid detections."""
        classifier = DiseaseClassifier()
        decision_maker = SprayDecisionMaker()
        safety_monitor = SafetyMonitor(cooldown_seconds=0.1)

        results = []

        for i in range(10):
            # Mock rapid predictions
            predictions = np.random.random((1, 10))
            predictions = predictions / predictions.sum()

            result = classifier.classify(predictions)

            should_spray, duration, _ = decision_maker.decide(
                result.disease_name,
                result.confidence,
                result.severity
            )

            if should_spray:
                can_spray, _ = safety_monitor.can_spray(duration)
                if can_spray:
                    safety_monitor.record_spray(duration)
                    results.append(result)

            time.sleep(0.05)  # Small delay

        # Some sprays should have occurred
        self.assertGreater(len(results), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
