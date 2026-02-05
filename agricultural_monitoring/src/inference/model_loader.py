#!/usr/bin/env python3
"""
TensorFlow Lite Model Loader for Plant Disease Detection

Handles loading and management of the TFLite model optimized for
Raspberry Pi 4B. Supports XNNPACK delegate for ARM acceleration.

Author: Agricultural Robotics Team
License: MIT
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml

import numpy as np

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_RUNTIME = True
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
        TFLITE_RUNTIME = False
    except ImportError:
        raise ImportError(
            "Neither tflite_runtime nor tensorflow found. "
            "Install with: pip install tflite-runtime"
        )


logger = logging.getLogger(__name__)


class ModelLoader:
    """
    TensorFlow Lite model loader and manager.

    Loads the plant disease detection model and provides inference interface.
    Optimized for Raspberry Pi 4B with optional XNNPACK acceleration.

    Attributes:
        model_path: Path to the TFLite model file
        interpreter: TFLite interpreter instance
        input_details: Model input tensor details
        output_details: Model output tensor details
        metadata: Model metadata (classes, normalization, etc.)
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        num_threads: int = 4,
        use_xnnpack: bool = True,
        config_path: Optional[Path] = None
    ) -> None:
        """
        Initialize the model loader.

        Args:
            model_path: Path to .tflite model file
            num_threads: Number of CPU threads for inference
            use_xnnpack: Enable XNNPACK delegate for ARM acceleration
            config_path: Path to model_config.yaml for settings
        """
        self.model_path: Optional[Path] = None
        self.interpreter: Optional[Any] = None
        self.input_details: Optional[List[Dict]] = None
        self.output_details: Optional[List[Dict]] = None
        self.metadata: Dict[str, Any] = {}

        self.num_threads = num_threads
        self.use_xnnpack = use_xnnpack

        # Load config if provided
        self.config = self._load_config(config_path)

        # Determine model path
        if model_path:
            self.model_path = Path(model_path)
        elif self.config:
            config_model_path = self.config.get('model_path', '')
            if config_model_path:
                # Resolve relative to config directory
                if config_path:
                    base_dir = config_path.parent.parent
                else:
                    base_dir = Path(__file__).parent.parent.parent
                self.model_path = base_dir / config_model_path

        # Load model if path is valid
        if self.model_path and self.model_path.exists():
            self.load()
        else:
            logger.warning(f"Model not found at {self.model_path}")

    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / 'config' / 'model_config.yaml'

        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return {}

        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def load(self) -> bool:
        """
        Load the TFLite model into memory.

        Sets up the interpreter with optional XNNPACK delegate.

        Returns:
            True if model loaded successfully, False otherwise
        """
        if not self.model_path or not self.model_path.exists():
            logger.error(f"Model file not found: {self.model_path}")
            return False

        try:
            logger.info(f"Loading TFLite model from {self.model_path}")

            # Build interpreter options
            if TFLITE_RUNTIME:
                # Using tflite-runtime
                if self.use_xnnpack:
                    # Try to use XNNPACK delegate
                    try:
                        self.interpreter = tflite.Interpreter(
                            model_path=str(self.model_path),
                            num_threads=self.num_threads,
                            experimental_delegates=[
                                tflite.load_delegate('libXNNPACK.so')
                            ] if os.path.exists('/usr/lib/libXNNPACK.so') else None
                        )
                    except Exception:
                        # Fall back to basic interpreter
                        self.interpreter = tflite.Interpreter(
                            model_path=str(self.model_path),
                            num_threads=self.num_threads
                        )
                else:
                    self.interpreter = tflite.Interpreter(
                        model_path=str(self.model_path),
                        num_threads=self.num_threads
                    )
            else:
                # Using full TensorFlow
                self.interpreter = tflite.Interpreter(
                    model_path=str(self.model_path),
                    num_threads=self.num_threads
                )

            # Allocate tensors
            self.interpreter.allocate_tensors()

            # Get input/output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            # Load metadata
            self._load_metadata()

            logger.info(f"Model loaded successfully")
            logger.info(f"Input shape: {self.get_input_shape()}")
            logger.info(f"Output shape: {self.get_output_shape()}")
            logger.info(f"Classes: {len(self.metadata.get('classes', []))}")

            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _load_metadata(self) -> None:
        """Load model metadata from config or sidecar file."""
        # Try to load from sidecar JSON file
        metadata_path = self.model_path.with_suffix('.json')
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                return
            except Exception as e:
                logger.warning(f"Failed to load metadata file: {e}")

        # Fall back to config
        if self.config:
            self.metadata = {
                'classes': self.config.get('classes', []),
                'input_size': self.config.get('input_size', {}),
                'normalization': self.config.get('normalization', '0-1'),
                'severity_mapping': self.config.get('severity_mapping', {})
            }

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on preprocessed input.

        Args:
            input_data: Preprocessed image(s) as numpy array
                       Shape: (batch, height, width, channels)

        Returns:
            Model output as numpy array (class probabilities)

        Raises:
            RuntimeError: If model not loaded
            ValueError: If input shape doesn't match model
        """
        if self.interpreter is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Validate input shape
        expected_shape = self.get_input_shape()
        if input_data.shape[1:] != expected_shape[1:]:
            raise ValueError(
                f"Input shape mismatch. Expected {expected_shape}, got {input_data.shape}"
            )

        # Ensure correct dtype
        input_dtype = self.input_details[0]['dtype']
        if input_data.dtype != input_dtype:
            input_data = input_data.astype(input_dtype)

        # Handle quantized models
        if input_dtype == np.uint8:
            input_scale, input_zero_point = self.input_details[0].get('quantization', (1.0, 0))
            input_data = input_data / input_scale + input_zero_point
            input_data = input_data.astype(np.uint8)

        # Set input tensor
        self.interpreter.set_tensor(
            self.input_details[0]['index'],
            input_data
        )

        # Run inference
        self.interpreter.invoke()

        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Dequantize if needed
        output_dtype = self.output_details[0]['dtype']
        if output_dtype == np.uint8:
            output_scale, output_zero_point = self.output_details[0].get('quantization', (1.0, 0))
            output = (output.astype(np.float32) - output_zero_point) * output_scale

        return output

    def get_input_shape(self) -> tuple:
        """Get expected input tensor shape."""
        if self.input_details:
            return tuple(self.input_details[0]['shape'])
        return (1, 224, 224, 3)

    def get_output_shape(self) -> tuple:
        """Get output tensor shape."""
        if self.output_details:
            return tuple(self.output_details[0]['shape'])
        return (1, 10)

    def get_classes(self) -> List[str]:
        """Get list of class labels."""
        return self.metadata.get('classes', [])

    def get_num_classes(self) -> int:
        """Get number of output classes."""
        return len(self.get_classes()) or self.get_output_shape()[-1]

    def get_severity(self, class_name: str) -> str:
        """
        Get severity level for a disease class.

        Args:
            class_name: Name of the detected disease

        Returns:
            Severity level: "healthy", "mild", "moderate", or "severe"
        """
        mapping = self.metadata.get('severity_mapping', {})
        return mapping.get(class_name, 'moderate')

    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self.interpreter is not None

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information for debugging/logging.

        Returns:
            Dictionary with model details
        """
        return {
            'path': str(self.model_path) if self.model_path else None,
            'loaded': self.is_loaded(),
            'input_shape': self.get_input_shape() if self.is_loaded() else None,
            'output_shape': self.get_output_shape() if self.is_loaded() else None,
            'num_classes': self.get_num_classes(),
            'classes': self.get_classes(),
            'num_threads': self.num_threads,
            'xnnpack': self.use_xnnpack,
            'runtime': 'tflite-runtime' if TFLITE_RUNTIME else 'tensorflow'
        }

    def __repr__(self) -> str:
        """String representation of model loader."""
        status = "loaded" if self.is_loaded() else "not loaded"
        return f"ModelLoader({self.model_path}, {status})"


def create_model_loader(config_dir: Optional[Path] = None) -> ModelLoader:
    """
    Factory function to create model loader from config.

    Args:
        config_dir: Path to config directory

    Returns:
        Configured ModelLoader instance
    """
    if config_dir is None:
        config_dir = Path(__file__).parent.parent.parent / 'config'

    config_path = config_dir / 'model_config.yaml'

    return ModelLoader(config_path=config_path)


if __name__ == '__main__':
    # Quick test
    logging.basicConfig(level=logging.INFO)

    loader = ModelLoader()
    print(f"Model info: {loader.get_model_info()}")

    if loader.is_loaded():
        # Test inference with dummy data
        input_shape = loader.get_input_shape()
        dummy_input = np.random.rand(*input_shape).astype(np.float32)
        output = loader.predict(dummy_input)
        print(f"Test output shape: {output.shape}")
        print(f"Test output: {output}")
