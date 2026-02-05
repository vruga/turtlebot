#!/usr/bin/env python3
"""
Image Preprocessor for Agricultural Disease Detection

Handles image preprocessing for the TensorFlow Lite model:
- Resize to model input dimensions
- Normalize pixel values
- Convert color spaces as needed

This module is designed to be efficient for Raspberry Pi 4B.

Author: Agricultural Robotics Team
License: MIT
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, Union
import yaml


class ImagePreprocessor:
    """
    Preprocessor for plant disease detection model input.

    Handles all image transformations required before model inference:
    - Resize to target dimensions
    - Color space conversion
    - Normalization (0-1, -1-1, or ImageNet)
    - Data type conversion

    Attributes:
        input_size: Target image dimensions (width, height)
        normalization: Normalization method
        imagenet_mean: Mean values for ImageNet normalization
        imagenet_std: Std values for ImageNet normalization
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (224, 224),
        normalization: str = "0-1",
        imagenet_mean: Optional[Tuple[float, float, float]] = None,
        imagenet_std: Optional[Tuple[float, float, float]] = None,
        config_path: Optional[Path] = None
    ) -> None:
        """
        Initialize the image preprocessor.

        Args:
            input_size: Target (width, height) for model input
            normalization: Method - "0-1", "-1-1", or "imagenet"
            imagenet_mean: Mean values for ImageNet normalization
            imagenet_std: Std values for ImageNet normalization
            config_path: Optional path to model_config.yaml
        """
        # Load from config if provided
        if config_path and config_path.exists():
            self._load_from_config(config_path)
        else:
            self.input_size = input_size
            self.normalization = normalization
            self.imagenet_mean = imagenet_mean or (0.485, 0.456, 0.406)
            self.imagenet_std = imagenet_std or (0.229, 0.224, 0.225)

    def _load_from_config(self, config_path: Path) -> None:
        """Load preprocessing settings from config file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            size_config = config.get('input_size', {})
            self.input_size = (
                size_config.get('width', 224),
                size_config.get('height', 224)
            )
            self.normalization = config.get('normalization', '0-1')
            self.imagenet_mean = tuple(config.get('imagenet_mean', [0.485, 0.456, 0.406]))
            self.imagenet_std = tuple(config.get('imagenet_std', [0.229, 0.224, 0.225]))

        except Exception as e:
            print(f"Warning: Failed to load config, using defaults: {e}")
            self.input_size = (224, 224)
            self.normalization = "0-1"
            self.imagenet_mean = (0.485, 0.456, 0.406)
            self.imagenet_std = (0.229, 0.224, 0.225)

    def preprocess(
        self,
        image: Union[np.ndarray, str, Path],
        expand_dims: bool = True
    ) -> np.ndarray:
        """
        Preprocess an image for model inference.

        Full preprocessing pipeline:
        1. Load image if path provided
        2. Convert BGR to RGB
        3. Resize to model input size
        4. Normalize pixel values
        5. Convert to float32
        6. Optionally add batch dimension

        Args:
            image: Input image (numpy array, file path, or Path object)
            expand_dims: If True, add batch dimension at axis 0

        Returns:
            Preprocessed image as numpy array ready for model input

        Raises:
            ValueError: If image cannot be loaded or processed
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            img = self._load_image(image)
        else:
            img = image.copy()

        # Validate input
        if img is None or img.size == 0:
            raise ValueError("Invalid or empty image")

        # Convert BGR to RGB (OpenCV loads as BGR)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        img = self._resize(img)

        # Normalize
        img = self._normalize(img)

        # Convert to float32
        img = img.astype(np.float32)

        # Add batch dimension if requested
        if expand_dims:
            img = np.expand_dims(img, axis=0)

        return img

    def _load_image(self, path: Union[str, Path]) -> np.ndarray:
        """
        Load image from file path.

        Args:
            path: Path to image file

        Returns:
            Loaded image as numpy array (BGR format)

        Raises:
            ValueError: If image cannot be loaded
        """
        path = Path(path)

        if not path.exists():
            raise ValueError(f"Image file not found: {path}")

        img = cv2.imread(str(path))

        if img is None:
            raise ValueError(f"Failed to load image: {path}")

        return img

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to model input size.

        Uses INTER_LINEAR interpolation for quality balance.

        Args:
            image: Input image

        Returns:
            Resized image
        """
        current_size = (image.shape[1], image.shape[0])

        if current_size == self.input_size:
            return image

        return cv2.resize(
            image,
            self.input_size,
            interpolation=cv2.INTER_LINEAR
        )

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values based on configured method.

        Methods:
        - "0-1": Scale to [0, 1]
        - "-1-1": Scale to [-1, 1]
        - "imagenet": ImageNet mean/std normalization

        Args:
            image: Input image with values [0, 255]

        Returns:
            Normalized image
        """
        # Convert to float for normalization
        img = image.astype(np.float32)

        if self.normalization == "0-1":
            # Scale to [0, 1]
            return img / 255.0

        elif self.normalization == "-1-1":
            # Scale to [-1, 1]
            return (img / 127.5) - 1.0

        elif self.normalization == "imagenet":
            # ImageNet normalization
            img = img / 255.0
            mean = np.array(self.imagenet_mean)
            std = np.array(self.imagenet_std)
            return (img - mean) / std

        else:
            # Default to 0-1
            return img / 255.0

    def preprocess_batch(
        self,
        images: list,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Preprocess a batch of images.

        Args:
            images: List of images (arrays or paths)
            show_progress: Print progress (for large batches)

        Returns:
            Batch of preprocessed images as numpy array
        """
        batch = []
        total = len(images)

        for i, img in enumerate(images):
            if show_progress and i % 10 == 0:
                print(f"Preprocessing: {i}/{total}")

            processed = self.preprocess(img, expand_dims=False)
            batch.append(processed)

        return np.array(batch)

    def denormalize(self, image: np.ndarray) -> np.ndarray:
        """
        Reverse normalization for visualization.

        Args:
            image: Normalized image

        Returns:
            Image with values in [0, 255] range (uint8)
        """
        img = image.copy()

        # Remove batch dimension if present
        if len(img.shape) == 4:
            img = img[0]

        if self.normalization == "0-1":
            img = img * 255.0

        elif self.normalization == "-1-1":
            img = (img + 1.0) * 127.5

        elif self.normalization == "imagenet":
            mean = np.array(self.imagenet_mean)
            std = np.array(self.imagenet_std)
            img = (img * std + mean) * 255.0

        # Clip and convert to uint8
        img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    def get_input_details(self) -> dict:
        """
        Get preprocessing details for documentation/debugging.

        Returns:
            Dictionary with preprocessing configuration
        """
        return {
            'input_size': self.input_size,
            'input_shape': (1, self.input_size[1], self.input_size[0], 3),
            'normalization': self.normalization,
            'dtype': 'float32',
            'color_space': 'RGB'
        }


def create_preprocessor_from_config(config_dir: Optional[Path] = None) -> ImagePreprocessor:
    """
    Factory function to create preprocessor from config directory.

    Args:
        config_dir: Path to config directory containing model_config.yaml

    Returns:
        Configured ImagePreprocessor instance
    """
    if config_dir is None:
        config_dir = Path(__file__).parent.parent.parent / 'config'

    config_path = config_dir / 'model_config.yaml'

    return ImagePreprocessor(config_path=config_path)


if __name__ == '__main__':
    # Quick test
    preprocessor = ImagePreprocessor()
    print(f"Preprocessor config: {preprocessor.get_input_details()}")

    # Test with a sample image if available
    test_image = Path('/tmp/captured_frame.jpg')
    if test_image.exists():
        try:
            processed = preprocessor.preprocess(test_image)
            print(f"Processed shape: {processed.shape}")
            print(f"Value range: [{processed.min():.3f}, {processed.max():.3f}]")
        except Exception as e:
            print(f"Test failed: {e}")
    else:
        print("No test image available")
