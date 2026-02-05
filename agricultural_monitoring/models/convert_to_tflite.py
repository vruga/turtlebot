#!/usr/bin/env python3
"""
TensorFlow Lite Conversion Script

Converts Keras/TensorFlow plant disease detection models to TFLite format
with post-training quantization for optimal Raspberry Pi 4B performance.

Features:
- Full integer quantization (int8)
- Accuracy validation
- Before/after comparison (size, inference time)
- Metadata generation

Usage:
    python convert_to_tflite.py --model plant_disease_model.h5 --output plant_disease_model.tflite

Author: Agricultural Robotics Team
License: MIT
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not installed. Install with: pip install tensorflow")

logger = logging.getLogger(__name__)


class TFLiteConverter:
    """
    Converts Keras models to TFLite with quantization.

    Supports:
    - Float32 (no quantization)
    - Float16 (size reduction, good accuracy)
    - Int8 (best for Raspberry Pi)
    """

    def __init__(
        self,
        model_path: Path,
        output_path: Optional[Path] = None,
        quantization: str = 'int8',
        representative_data: Optional[np.ndarray] = None
    ):
        """
        Initialize the converter.

        Args:
            model_path: Path to Keras .h5 or SavedModel
            output_path: Output path for .tflite (default: same name)
            quantization: 'none', 'float16', or 'int8'
            representative_data: Sample data for int8 calibration
        """
        self.model_path = Path(model_path)
        self.output_path = output_path or self.model_path.with_suffix('.tflite')
        self.quantization = quantization
        self.representative_data = representative_data

        self.model = None
        self.tflite_model = None

        self.original_size = 0
        self.converted_size = 0
        self.original_accuracy = 0.0
        self.converted_accuracy = 0.0

    def load_model(self) -> bool:
        """Load the Keras model."""
        if not TF_AVAILABLE:
            logger.error("TensorFlow not available")
            return False

        try:
            logger.info(f"Loading model from {self.model_path}")

            if self.model_path.suffix == '.h5':
                self.model = tf.keras.models.load_model(str(self.model_path))
            else:
                # Assume SavedModel directory
                self.model = tf.keras.models.load_model(str(self.model_path))

            self.original_size = self.model_path.stat().st_size

            # Log model summary
            logger.info(f"Model loaded successfully")
            logger.info(f"Input shape: {self.model.input_shape}")
            logger.info(f"Output shape: {self.model.output_shape}")
            logger.info(f"Original size: {self.original_size / 1024 / 1024:.2f} MB")

            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def convert(self) -> bool:
        """Convert the model to TFLite format."""
        if self.model is None:
            logger.error("Model not loaded")
            return False

        try:
            logger.info(f"Converting with {self.quantization} quantization...")

            # Create converter
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

            if self.quantization == 'int8':
                # Full integer quantization
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                ]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8

                # Set representative dataset
                if self.representative_data is not None:
                    converter.representative_dataset = self._representative_dataset_gen
                else:
                    logger.warning(
                        "No representative data provided for int8 quantization. "
                        "Using random data for calibration."
                    )
                    converter.representative_dataset = self._random_dataset_gen

            elif self.quantization == 'float16':
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]

            elif self.quantization == 'dynamic':
                # Dynamic range quantization
                converter.optimizations = [tf.lite.Optimize.DEFAULT]

            # Convert
            self.tflite_model = converter.convert()

            # Save model
            with open(self.output_path, 'wb') as f:
                f.write(self.tflite_model)

            self.converted_size = self.output_path.stat().st_size

            logger.info(f"Conversion successful!")
            logger.info(f"Output: {self.output_path}")
            logger.info(f"Converted size: {self.converted_size / 1024 / 1024:.2f} MB")
            logger.info(f"Size reduction: {(1 - self.converted_size/self.original_size)*100:.1f}%")

            return True

        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            return False

    def _representative_dataset_gen(self):
        """Generator for representative dataset."""
        for sample in self.representative_data[:100]:
            yield [np.expand_dims(sample, axis=0).astype(np.float32)]

    def _random_dataset_gen(self):
        """Generator using random data for calibration."""
        input_shape = self.model.input_shape[1:]
        for _ in range(100):
            yield [np.random.random((1, *input_shape)).astype(np.float32)]

    def validate_accuracy(
        self,
        test_data: np.ndarray,
        test_labels: np.ndarray
    ) -> Tuple[float, float]:
        """
        Validate accuracy of original and converted models.

        Args:
            test_data: Test images
            test_labels: True labels

        Returns:
            Tuple of (original_accuracy, converted_accuracy)
        """
        if self.model is None or self.tflite_model is None:
            logger.error("Models not loaded/converted")
            return 0.0, 0.0

        # Evaluate original model
        logger.info("Evaluating original model...")
        original_preds = self.model.predict(test_data, verbose=0)
        original_pred_classes = np.argmax(original_preds, axis=1)
        self.original_accuracy = np.mean(original_pred_classes == test_labels)

        # Evaluate TFLite model
        logger.info("Evaluating TFLite model...")
        interpreter = tf.lite.Interpreter(model_content=self.tflite_model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        tflite_preds = []
        for sample in test_data:
            input_data = np.expand_dims(sample, axis=0)

            # Handle quantization
            if input_details[0]['dtype'] == np.uint8:
                input_scale, input_zero = input_details[0]['quantization']
                input_data = (input_data / input_scale + input_zero).astype(np.uint8)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])

            # Dequantize if needed
            if output_details[0]['dtype'] == np.uint8:
                output_scale, output_zero = output_details[0]['quantization']
                output = (output.astype(np.float32) - output_zero) * output_scale

            tflite_preds.append(output[0])

        tflite_pred_classes = np.argmax(np.array(tflite_preds), axis=1)
        self.converted_accuracy = np.mean(tflite_pred_classes == test_labels)

        accuracy_diff = abs(self.original_accuracy - self.converted_accuracy) * 100

        logger.info(f"Original accuracy: {self.original_accuracy*100:.2f}%")
        logger.info(f"TFLite accuracy: {self.converted_accuracy*100:.2f}%")
        logger.info(f"Accuracy difference: {accuracy_diff:.2f}%")

        if accuracy_diff > 3.0:
            logger.warning(f"Accuracy loss ({accuracy_diff:.2f}%) exceeds 3% threshold!")

        return self.original_accuracy, self.converted_accuracy

    def measure_inference_time(self, num_runs: int = 50) -> Tuple[float, float]:
        """
        Measure inference time for both models.

        Args:
            num_runs: Number of inference runs for averaging

        Returns:
            Tuple of (original_time_ms, tflite_time_ms)
        """
        input_shape = self.model.input_shape[1:]
        test_input = np.random.random((1, *input_shape)).astype(np.float32)

        # Measure original model
        logger.info("Measuring original model inference time...")
        times = []
        for _ in range(num_runs):
            start = time.time()
            self.model.predict(test_input, verbose=0)
            times.append((time.time() - start) * 1000)
        original_time = np.mean(times)

        # Measure TFLite model
        logger.info("Measuring TFLite model inference time...")
        interpreter = tf.lite.Interpreter(model_content=self.tflite_model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        times = []
        for _ in range(num_runs):
            input_data = test_input.copy()
            if input_details[0]['dtype'] == np.uint8:
                input_scale, input_zero = input_details[0]['quantization']
                input_data = (input_data / input_scale + input_zero).astype(np.uint8)

            start = time.time()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]['index'])
            times.append((time.time() - start) * 1000)
        tflite_time = np.mean(times)

        logger.info(f"Original inference time: {original_time:.2f} ms")
        logger.info(f"TFLite inference time: {tflite_time:.2f} ms")
        logger.info(f"Speedup: {original_time/tflite_time:.2f}x")

        return original_time, tflite_time

    def save_metadata(self, classes: Optional[List[str]] = None) -> None:
        """
        Save model metadata as JSON sidecar file.

        Args:
            classes: List of class names
        """
        metadata_path = self.output_path.with_suffix('.json')

        input_shape = self.model.input_shape
        output_shape = self.model.output_shape

        metadata = {
            'model_name': self.model_path.stem,
            'conversion_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'quantization': self.quantization,
            'input_shape': list(input_shape),
            'output_shape': list(output_shape),
            'classes': classes or [f'class_{i}' for i in range(output_shape[-1])],
            'normalization': '0-1',
            'original_size_bytes': self.original_size,
            'converted_size_bytes': self.converted_size,
            'size_reduction_percent': round((1 - self.converted_size/self.original_size) * 100, 2),
            'original_accuracy': round(self.original_accuracy * 100, 2) if self.original_accuracy else None,
            'converted_accuracy': round(self.converted_accuracy * 100, 2) if self.converted_accuracy else None
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved to {metadata_path}")

    def get_summary(self) -> dict:
        """Get conversion summary."""
        return {
            'original_path': str(self.model_path),
            'output_path': str(self.output_path),
            'quantization': self.quantization,
            'original_size_mb': round(self.original_size / 1024 / 1024, 2),
            'converted_size_mb': round(self.converted_size / 1024 / 1024, 2),
            'size_reduction': f"{(1 - self.converted_size/self.original_size)*100:.1f}%",
            'original_accuracy': f"{self.original_accuracy*100:.2f}%" if self.original_accuracy else "N/A",
            'converted_accuracy': f"{self.converted_accuracy*100:.2f}%" if self.converted_accuracy else "N/A"
        }


def main():
    parser = argparse.ArgumentParser(
        description='Convert Keras model to TFLite for Raspberry Pi'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='plant_disease_model.h5',
        help='Path to Keras model (.h5 or SavedModel)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output path for TFLite model'
    )
    parser.add_argument(
        '--quantization', '-q',
        type=str,
        choices=['none', 'float16', 'dynamic', 'int8'],
        default='int8',
        help='Quantization type (default: int8)'
    )
    parser.add_argument(
        '--classes',
        type=str,
        nargs='+',
        default=None,
        help='Class names for metadata'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip accuracy validation'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if not TF_AVAILABLE:
        logger.error("TensorFlow is required. Install with: pip install tensorflow")
        return 1

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return 1

    output_path = Path(args.output) if args.output else None

    # Create converter
    converter = TFLiteConverter(
        model_path=model_path,
        output_path=output_path,
        quantization=args.quantization
    )

    # Load and convert
    if not converter.load_model():
        return 1

    if not converter.convert():
        return 1

    # Measure inference time
    converter.measure_inference_time()

    # Save metadata
    converter.save_metadata(classes=args.classes)

    # Print summary
    print("\n" + "="*50)
    print("CONVERSION SUMMARY")
    print("="*50)
    for key, value in converter.get_summary().items():
        print(f"  {key}: {value}")
    print("="*50)

    return 0


if __name__ == '__main__':
    exit(main())
