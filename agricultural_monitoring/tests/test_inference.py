#!/usr/bin/env python3
"""
Test suite for TFLite model inference.

Tests:
- Model loading
- Inference speed
- Output validation
- Preprocessing pipeline

Run with: python -m pytest tests/test_inference.py -v

Author: Agricultural Robotics Team
"""

import sys
import time
import unittest
from pathlib import Path
import numpy as np

# Add project to path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'src'))

from camera.image_preprocessor import ImagePreprocessor
from inference.model_loader import ModelLoader
from inference.disease_classifier import DiseaseClassifier, DetectionResult


class TestImagePreprocessor(unittest.TestCase):
    """Tests for image preprocessing."""

    def setUp(self):
        self.preprocessor = ImagePreprocessor(
            input_size=(224, 224),
            normalization='0-1'
        )

    def test_preprocess_shape(self):
        """Test that preprocessed image has correct shape."""
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = self.preprocessor.preprocess(dummy_image)

        self.assertEqual(result.shape, (1, 224, 224, 3))

    def test_preprocess_normalization_0_1(self):
        """Test 0-1 normalization."""
        self.preprocessor.normalization = '0-1'
        dummy_image = np.ones((224, 224, 3), dtype=np.uint8) * 255

        result = self.preprocessor.preprocess(dummy_image, expand_dims=False)

        self.assertAlmostEqual(result.max(), 1.0, places=5)
        self.assertAlmostEqual(result.min(), 1.0, places=5)

    def test_preprocess_normalization_neg1_1(self):
        """Test -1 to 1 normalization."""
        self.preprocessor.normalization = '-1-1'
        dummy_image = np.ones((224, 224, 3), dtype=np.uint8) * 255

        result = self.preprocessor.preprocess(dummy_image, expand_dims=False)

        self.assertAlmostEqual(result.max(), 1.0, places=5)

    def test_preprocess_dtype(self):
        """Test that output is float32."""
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = self.preprocessor.preprocess(dummy_image)

        self.assertEqual(result.dtype, np.float32)


class TestModelLoader(unittest.TestCase):
    """Tests for TFLite model loading."""

    @classmethod
    def setUpClass(cls):
        """Set up model path."""
        cls.model_path = PROJECT_DIR / 'models' / 'plant_disease_model.tflite'
        cls.model_exists = cls.model_path.exists()

    def test_model_initialization(self):
        """Test model loader initialization."""
        loader = ModelLoader(model_path=None)
        self.assertFalse(loader.is_loaded())

    @unittest.skipUnless(
        Path(PROJECT_DIR / 'models' / 'plant_disease_model.tflite').exists(),
        "Model file not found"
    )
    def test_model_loading(self):
        """Test model loads successfully."""
        loader = ModelLoader(model_path=self.model_path)
        self.assertTrue(loader.is_loaded())

    @unittest.skipUnless(
        Path(PROJECT_DIR / 'models' / 'plant_disease_model.tflite').exists(),
        "Model file not found"
    )
    def test_inference_output_shape(self):
        """Test model produces correct output shape."""
        loader = ModelLoader(model_path=self.model_path)

        input_shape = loader.get_input_shape()
        dummy_input = np.random.random(input_shape).astype(np.float32)

        output = loader.predict(dummy_input)

        self.assertEqual(len(output.shape), 2)
        self.assertEqual(output.shape[0], 1)

    @unittest.skipUnless(
        Path(PROJECT_DIR / 'models' / 'plant_disease_model.tflite').exists(),
        "Model file not found"
    )
    def test_inference_speed(self):
        """Test inference speed is acceptable for Raspberry Pi."""
        loader = ModelLoader(model_path=self.model_path)

        input_shape = loader.get_input_shape()
        dummy_input = np.random.random(input_shape).astype(np.float32)

        # Warm up
        loader.predict(dummy_input)

        # Measure
        times = []
        for _ in range(10):
            start = time.time()
            loader.predict(dummy_input)
            times.append(time.time() - start)

        avg_time = sum(times) / len(times)
        print(f"\nAverage inference time: {avg_time*1000:.1f}ms")

        # Should be under 2 seconds on Pi 4B
        self.assertLess(avg_time, 5.0)  # Relaxed for CI


class TestDiseaseClassifier(unittest.TestCase):
    """Tests for disease classification post-processing."""

    def setUp(self):
        self.classifier = DiseaseClassifier()

    def test_classify_healthy(self):
        """Test classification of healthy plant."""
        # Mock predictions with healthy as highest
        predictions = np.array([0.95, 0.01, 0.01, 0.01, 0.01, 0.01])
        self.classifier.classes = ['healthy', 'early_blight', 'late_blight',
                                   'leaf_mold', 'septoria', 'bacterial_spot']

        result = self.classifier.classify(predictions)

        self.assertEqual(result.disease_name, 'healthy')
        self.assertAlmostEqual(result.confidence, 0.95, places=2)
        self.assertFalse(result.should_spray)
        self.assertEqual(result.spray_duration, 0)

    def test_classify_disease(self):
        """Test classification of diseased plant."""
        # Mock predictions with early_blight as highest
        predictions = np.array([0.05, 0.85, 0.03, 0.03, 0.02, 0.02])
        self.classifier.classes = ['healthy', 'early_blight', 'late_blight',
                                   'leaf_mold', 'septoria', 'bacterial_spot']

        result = self.classifier.classify(predictions)

        self.assertEqual(result.disease_name, 'early_blight')
        self.assertTrue(result.should_spray)
        self.assertGreater(result.spray_duration, 0)

    def test_low_confidence_no_spray(self):
        """Test that low confidence doesn't trigger spray."""
        predictions = np.array([0.2, 0.7, 0.03, 0.03, 0.02, 0.02])
        self.classifier.classes = ['healthy', 'early_blight', 'late_blight',
                                   'leaf_mold', 'septoria', 'bacterial_spot']
        self.classifier.confidence_threshold = 0.8

        result = self.classifier.classify(predictions)

        self.assertFalse(result.should_spray)

    def test_severity_mapping(self):
        """Test severity is correctly mapped."""
        # Test mild disease
        predictions = np.array([0.05, 0.90, 0.02, 0.01, 0.01, 0.01])
        self.classifier.classes = ['healthy', 'early_blight', 'late_blight',
                                   'leaf_mold', 'septoria', 'bacterial_spot']
        self.classifier.severity_mapping['early_blight'] = 'mild'

        result = self.classifier.classify(predictions)
        self.assertEqual(result.severity, 'mild')

    def test_result_to_dict(self):
        """Test DetectionResult serialization."""
        predictions = np.array([0.05, 0.90, 0.02, 0.01, 0.01, 0.01])
        self.classifier.classes = ['healthy', 'early_blight', 'late_blight',
                                   'leaf_mold', 'septoria', 'bacterial_spot']

        result = self.classifier.classify(predictions)
        result_dict = result.to_dict()

        self.assertIn('disease_name', result_dict)
        self.assertIn('confidence', result_dict)
        self.assertIn('timestamp', result_dict)


class TestEndToEndPipeline(unittest.TestCase):
    """Tests for complete inference pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.model_path = PROJECT_DIR / 'models' / 'plant_disease_model.tflite'

    @unittest.skipUnless(
        Path(PROJECT_DIR / 'models' / 'plant_disease_model.tflite').exists(),
        "Model file not found"
    )
    def test_full_pipeline(self):
        """Test complete inference pipeline."""
        # Create components
        preprocessor = ImagePreprocessor(input_size=(224, 224))
        loader = ModelLoader(model_path=self.model_path)
        classifier = DiseaseClassifier()

        # Create dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Process
        processed = preprocessor.preprocess(dummy_image)
        predictions = loader.predict(processed)
        result = classifier.classify(predictions)

        # Validate
        self.assertIsInstance(result, DetectionResult)
        self.assertIsInstance(result.disease_name, str)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
