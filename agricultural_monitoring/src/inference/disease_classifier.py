#!/usr/bin/env python3
"""
Disease Classifier for Agricultural Disease Detection

Post-processes model predictions to generate actionable disease
classifications with confidence scores and severity levels.

Author: Agricultural Robotics Team
License: MIT
"""

import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import yaml
import json

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """
    Data class for disease detection results.

    Attributes:
        disease_name: Predicted disease class name
        confidence: Confidence score (0.0 - 1.0)
        severity: Severity level (healthy, mild, moderate, severe)
        timestamp: Detection timestamp
        top_predictions: Top N predictions with scores
        should_spray: Whether spray is recommended
        spray_duration: Recommended spray duration in ms
        image_path: Path to the source image
    """
    disease_name: str
    confidence: float
    severity: str
    timestamp: datetime
    top_predictions: List[Tuple[str, float]]
    should_spray: bool
    spray_duration: int
    image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'disease_name': self.disease_name,
            'confidence': round(self.confidence, 4),
            'confidence_percent': round(self.confidence * 100, 2),
            'severity': self.severity,
            'timestamp': self.timestamp.isoformat(),
            'top_predictions': [
                {'class': name, 'confidence': round(conf, 4)}
                for name, conf in self.top_predictions
            ],
            'should_spray': self.should_spray,
            'spray_duration': self.spray_duration,
            'image_path': self.image_path
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class DiseaseClassifier:
    """
    Post-processor for plant disease model predictions.

    Takes raw model output and produces actionable results including:
    - Class prediction with confidence
    - Severity classification
    - Spray recommendations

    Attributes:
        classes: List of disease class names
        confidence_threshold: Minimum confidence for action
        severity_mapping: Maps disease names to severity levels
        spray_durations: Maps severity to spray duration
    """

    def __init__(
        self,
        classes: Optional[List[str]] = None,
        confidence_threshold: float = 0.80,
        severity_mapping: Optional[Dict[str, str]] = None,
        spray_durations: Optional[Dict[str, int]] = None,
        config_path: Optional[Path] = None
    ) -> None:
        """
        Initialize the disease classifier.

        Args:
            classes: List of class names (must match model output order)
            confidence_threshold: Minimum confidence to recommend action
            severity_mapping: Disease name -> severity level mapping
            spray_durations: Severity level -> spray duration (ms) mapping
            config_path: Path to config directory
        """
        # Load configs
        if config_path:
            self._load_from_config(config_path)
        else:
            self._load_from_config(
                Path(__file__).parent.parent.parent / 'config'
            )

        # Override with provided parameters
        if classes is not None:
            self.classes = classes
        if confidence_threshold != 0.80:
            self.confidence_threshold = confidence_threshold
        if severity_mapping is not None:
            self.severity_mapping = severity_mapping
        if spray_durations is not None:
            self.spray_durations = spray_durations

    def _load_from_config(self, config_dir: Path) -> None:
        """Load settings from configuration files."""
        # Default values
        self.classes = [
            'healthy', 'early_blight', 'late_blight', 'leaf_mold',
            'septoria_leaf_spot', 'bacterial_spot', 'target_spot',
            'yellow_leaf_curl_virus', 'mosaic_virus', 'powdery_mildew'
        ]
        self.confidence_threshold = 0.80
        self.severity_mapping = {
            'healthy': 'healthy',
            'early_blight': 'mild',
            'late_blight': 'severe',
            'leaf_mold': 'moderate',
            'septoria_leaf_spot': 'moderate',
            'bacterial_spot': 'moderate',
            'target_spot': 'mild',
            'yellow_leaf_curl_virus': 'severe',
            'mosaic_virus': 'severe',
            'powdery_mildew': 'mild'
        }
        self.spray_durations = {
            'healthy': 0,
            'mild': 2000,
            'moderate': 4000,
            'severe': 6000
        }

        # Load model config
        model_config_path = config_dir / 'model_config.yaml'
        if model_config_path.exists():
            try:
                with open(model_config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    if config:
                        self.classes = config.get('classes', self.classes)
                        self.confidence_threshold = config.get(
                            'confidence_threshold', self.confidence_threshold
                        )
                        self.severity_mapping = config.get(
                            'severity_mapping', self.severity_mapping
                        )
            except Exception as e:
                logger.warning(f"Failed to load model config: {e}")

        # Load spray config
        spray_config_path = config_dir / 'spray_config.yaml'
        if spray_config_path.exists():
            try:
                with open(spray_config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    if config:
                        durations = config.get('spray_durations', {})
                        if durations:
                            self.spray_durations = {
                                'healthy': 0,
                                'mild': durations.get('mild', 2000),
                                'moderate': durations.get('moderate', 4000),
                                'severe': durations.get('severe', 6000)
                            }
            except Exception as e:
                logger.warning(f"Failed to load spray config: {e}")

    def classify(
        self,
        predictions: np.ndarray,
        image_path: Optional[str] = None,
        top_k: int = 3
    ) -> DetectionResult:
        """
        Process model predictions into a detection result.

        Args:
            predictions: Model output array (probabilities per class)
            image_path: Optional path to source image
            top_k: Number of top predictions to include

        Returns:
            DetectionResult with full classification details
        """
        # Handle batch dimension
        if len(predictions.shape) > 1:
            predictions = predictions[0]

        # Apply softmax if not already probabilities
        if predictions.min() < 0 or predictions.max() > 1:
            predictions = self._softmax(predictions)

        # Get top-k predictions
        top_indices = np.argsort(predictions)[::-1][:top_k]
        top_predictions = [
            (self._get_class_name(idx), float(predictions[idx]))
            for idx in top_indices
        ]

        # Get primary prediction
        primary_idx = top_indices[0]
        disease_name = self._get_class_name(primary_idx)
        confidence = float(predictions[primary_idx])

        # Determine severity and spray
        severity = self._get_severity(disease_name)
        should_spray, spray_duration = self._get_spray_decision(
            disease_name, confidence, severity
        )

        return DetectionResult(
            disease_name=disease_name,
            confidence=confidence,
            severity=severity,
            timestamp=datetime.now(),
            top_predictions=top_predictions,
            should_spray=should_spray,
            spray_duration=spray_duration,
            image_path=image_path
        )

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax to convert logits to probabilities."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def _get_class_name(self, index: int) -> str:
        """Get class name from index."""
        if 0 <= index < len(self.classes):
            return self.classes[index]
        return f"unknown_class_{index}"

    def _get_severity(self, disease_name: str) -> str:
        """Get severity level for disease."""
        return self.severity_mapping.get(disease_name, 'moderate')

    def _get_spray_decision(
        self,
        disease_name: str,
        confidence: float,
        severity: str
    ) -> Tuple[bool, int]:
        """
        Determine spray recommendation.

        Args:
            disease_name: Detected disease
            confidence: Detection confidence
            severity: Severity level

        Returns:
            Tuple of (should_spray, spray_duration_ms)
        """
        # No spray for healthy plants
        if disease_name == 'healthy' or severity == 'healthy':
            return False, 0

        # No spray if confidence below threshold
        if confidence < self.confidence_threshold:
            logger.info(
                f"Confidence {confidence:.2%} below threshold "
                f"{self.confidence_threshold:.2%}, no spray recommended"
            )
            return False, 0

        # Get spray duration for severity
        spray_duration = self.spray_durations.get(severity, 4000)

        return True, spray_duration

    def get_confidence_level(self, confidence: float) -> str:
        """
        Categorize confidence for UI display.

        Args:
            confidence: Confidence score (0-1)

        Returns:
            Confidence level string
        """
        if confidence >= 0.95:
            return "very_high"
        elif confidence >= 0.90:
            return "high"
        elif confidence >= 0.80:
            return "moderate"
        elif confidence >= 0.60:
            return "low"
        else:
            return "very_low"

    def format_for_display(self, result: DetectionResult) -> Dict[str, str]:
        """
        Format result for farmer-friendly display.

        Args:
            result: DetectionResult to format

        Returns:
            Dictionary with display-ready strings
        """
        confidence_percent = result.confidence * 100
        confidence_level = self.get_confidence_level(result.confidence)

        # Human-readable disease name
        disease_display = result.disease_name.replace('_', ' ').title()

        # Status message
        if result.severity == 'healthy':
            status = "Plant appears healthy"
            action = "No treatment needed"
        elif result.should_spray:
            status = f"Disease detected: {disease_display}"
            action = f"Spray treatment applied ({result.spray_duration}ms)"
        else:
            status = f"Possible: {disease_display}"
            action = "Manual inspection recommended"

        return {
            'disease': disease_display,
            'confidence': f"{confidence_percent:.1f}%",
            'confidence_level': confidence_level,
            'severity': result.severity.title(),
            'status': status,
            'action': action,
            'timestamp': result.timestamp.strftime('%H:%M:%S')
        }

    def get_config(self) -> Dict[str, Any]:
        """Get current classifier configuration."""
        return {
            'classes': self.classes,
            'confidence_threshold': self.confidence_threshold,
            'severity_mapping': self.severity_mapping,
            'spray_durations': self.spray_durations
        }


# Detection history tracker
class DetectionHistory:
    """
    Tracks recent detection results for pattern analysis.

    Used by LLM integration to provide context-aware recommendations.
    """

    def __init__(self, max_entries: int = 100) -> None:
        """
        Initialize detection history.

        Args:
            max_entries: Maximum history entries to keep
        """
        self.max_entries = max_entries
        self.history: List[DetectionResult] = []

    def add(self, result: DetectionResult) -> None:
        """Add detection to history."""
        self.history.append(result)
        # Trim if over limit
        if len(self.history) > self.max_entries:
            self.history = self.history[-self.max_entries:]

    def get_recent(self, count: int = 10) -> List[DetectionResult]:
        """Get most recent detections."""
        return self.history[-count:]

    def get_today(self) -> List[DetectionResult]:
        """Get all detections from today."""
        today = datetime.now().date()
        return [
            r for r in self.history
            if r.timestamp.date() == today
        ]

    def get_disease_count(self, disease_name: str, hours: int = 24) -> int:
        """Count detections of specific disease in time window."""
        cutoff = datetime.now().timestamp() - (hours * 3600)
        return sum(
            1 for r in self.history
            if r.disease_name == disease_name
            and r.timestamp.timestamp() > cutoff
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        today_detections = self.get_today()
        disease_counts = {}
        for r in today_detections:
            disease_counts[r.disease_name] = disease_counts.get(r.disease_name, 0) + 1

        return {
            'total_today': len(today_detections),
            'disease_counts': disease_counts,
            'total_history': len(self.history)
        }

    def to_context_string(self, count: int = 5) -> str:
        """Format recent history for LLM context."""
        recent = self.get_recent(count)
        if not recent:
            return "No recent detections"

        lines = []
        for r in recent:
            lines.append(
                f"- {r.timestamp.strftime('%H:%M')}: {r.disease_name} "
                f"({r.confidence:.0%} confidence)"
            )
        return "\n".join(lines)


if __name__ == '__main__':
    # Quick test
    logging.basicConfig(level=logging.INFO)

    classifier = DiseaseClassifier()
    print(f"Classifier config: {classifier.get_config()}")

    # Test with dummy predictions
    dummy_predictions = np.array([0.1, 0.7, 0.05, 0.05, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005])
    result = classifier.classify(dummy_predictions)

    print(f"\nTest result:")
    print(result.to_json())

    print(f"\nFormatted for display:")
    print(classifier.format_for_display(result))
