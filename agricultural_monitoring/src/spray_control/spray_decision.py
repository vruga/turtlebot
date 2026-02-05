#!/usr/bin/env python3
"""
Spray Decision Maker for Agricultural Disease Detection

Implements decision logic for when and how much to spray based on
disease detection results. Maps disease severity to spray duration.

Author: Agricultural Robotics Team
License: MIT
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import yaml

logger = logging.getLogger(__name__)


class SprayDecisionMaker:
    """
    Makes spray decisions based on disease detection results.

    Decision factors:
    - Disease type and severity
    - Detection confidence
    - Configured thresholds and durations

    Attributes:
        confidence_threshold: Minimum confidence to trigger spray
        spray_durations: Mapping of severity to spray duration (ms)
        severity_mapping: Mapping of disease names to severity levels
    """

    def __init__(
        self,
        confidence_threshold: float = 0.80,
        spray_durations: Optional[Dict[str, int]] = None,
        severity_mapping: Optional[Dict[str, str]] = None,
        config_path: Optional[Path] = None
    ) -> None:
        """
        Initialize the spray decision maker.

        Args:
            confidence_threshold: Minimum confidence to recommend spray
            spray_durations: Dict mapping severity -> duration in ms
            severity_mapping: Dict mapping disease name -> severity
            config_path: Path to config directory
        """
        # Default values
        self.confidence_threshold = confidence_threshold
        self.spray_durations = spray_durations or {
            'healthy': 0,
            'mild': 2000,
            'moderate': 4000,
            'severe': 6000
        }
        self.severity_mapping = severity_mapping or {
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
        self.spray_on_healthy = False
        self.max_duration = 10000

        # Load from config if provided
        if config_path:
            self._load_from_config(config_path)

    def _load_from_config(self, config_path: Path) -> None:
        """Load decision parameters from config files."""
        # Load model config for severity mapping
        model_config = config_path / 'model_config.yaml'
        if model_config.exists():
            try:
                with open(model_config, 'r') as f:
                    config = yaml.safe_load(f)
                    if config:
                        self.confidence_threshold = config.get(
                            'confidence_threshold', self.confidence_threshold
                        )
                        if 'severity_mapping' in config:
                            self.severity_mapping = config['severity_mapping']
            except Exception as e:
                logger.warning(f"Failed to load model config: {e}")

        # Load spray config for durations
        spray_config = config_path / 'spray_config.yaml'
        if spray_config.exists():
            try:
                with open(spray_config, 'r') as f:
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
                        self.max_duration = config.get('safety', {}).get(
                            'max_duration', self.max_duration
                        )
                        decision_config = config.get('decision', {})
                        self.spray_on_healthy = decision_config.get(
                            'spray_on_healthy', False
                        )
                        self.confidence_threshold = decision_config.get(
                            'min_confidence', self.confidence_threshold
                        )
            except Exception as e:
                logger.warning(f"Failed to load spray config: {e}")

    def decide(
        self,
        disease_name: str,
        confidence: float,
        severity: Optional[str] = None
    ) -> Tuple[bool, int, str]:
        """
        Make spray decision based on detection result.

        Decision logic:
        1. No spray for healthy plants (unless configured)
        2. No spray if confidence below threshold
        3. Spray duration based on severity

        Args:
            disease_name: Detected disease class name
            confidence: Detection confidence (0-1)
            severity: Severity level (if not provided, derived from disease_name)

        Returns:
            Tuple of (should_spray: bool, duration_ms: int, reason: str)
        """
        # Get severity if not provided
        if severity is None:
            severity = self._get_severity(disease_name)

        # Check for healthy plant
        if disease_name.lower() == 'healthy' or severity == 'healthy':
            if self.spray_on_healthy:
                return True, self.spray_durations.get('mild', 2000), "Preventive spray"
            return False, 0, "Plant is healthy"

        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return (
                False,
                0,
                f"Confidence {confidence:.1%} below threshold {self.confidence_threshold:.1%}"
            )

        # Get spray duration for severity
        duration = self.spray_durations.get(severity, self.spray_durations['moderate'])

        # Apply max duration limit
        duration = min(duration, self.max_duration)

        reason = (
            f"{disease_name} ({severity} severity) detected with "
            f"{confidence:.1%} confidence"
        )

        return True, duration, reason

    def _get_severity(self, disease_name: str) -> str:
        """
        Get severity level for a disease.

        Args:
            disease_name: Name of the disease

        Returns:
            Severity level string
        """
        # Normalize disease name
        normalized = disease_name.lower().replace(' ', '_')

        return self.severity_mapping.get(normalized, 'moderate')

    def get_duration_for_severity(self, severity: str) -> int:
        """
        Get spray duration for a severity level.

        Args:
            severity: Severity level ('mild', 'moderate', 'severe')

        Returns:
            Spray duration in milliseconds
        """
        return self.spray_durations.get(severity, self.spray_durations['moderate'])

    def get_config(self) -> Dict[str, Any]:
        """Get current decision maker configuration."""
        return {
            'confidence_threshold': self.confidence_threshold,
            'spray_durations': self.spray_durations,
            'severity_mapping': self.severity_mapping,
            'spray_on_healthy': self.spray_on_healthy,
            'max_duration': self.max_duration
        }

    def update_thresholds(
        self,
        confidence_threshold: Optional[float] = None,
        spray_durations: Optional[Dict[str, int]] = None
    ) -> None:
        """
        Update decision thresholds dynamically.

        Args:
            confidence_threshold: New confidence threshold
            spray_durations: New spray duration mapping
        """
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
            logger.info(f"Confidence threshold updated to {confidence_threshold}")

        if spray_durations is not None:
            self.spray_durations.update(spray_durations)
            logger.info(f"Spray durations updated: {spray_durations}")


class AdaptiveSprayDecision(SprayDecisionMaker):
    """
    Adaptive spray decision maker that adjusts based on history.

    Extends the base decision maker with:
    - Increased spray for repeated detections
    - Reduced spray after successful treatment
    - Weather-based adjustments
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detection_history: list = []
        self.max_history = 50

    def decide_with_context(
        self,
        disease_name: str,
        confidence: float,
        severity: Optional[str] = None,
        repeated_count: int = 0,
        weather_factor: float = 1.0
    ) -> Tuple[bool, int, str]:
        """
        Make spray decision with additional context.

        Args:
            disease_name: Detected disease
            confidence: Detection confidence
            severity: Severity level
            repeated_count: Number of times this disease was detected recently
            weather_factor: Multiplier based on weather (humidity, etc.)

        Returns:
            Tuple of (should_spray, duration_ms, reason)
        """
        # Get base decision
        should_spray, base_duration, reason = self.decide(
            disease_name, confidence, severity
        )

        if not should_spray:
            return should_spray, base_duration, reason

        # Adjust duration based on repeated detections
        if repeated_count > 3:
            adjusted_duration = int(base_duration * 1.5)
            reason += f" (increased for {repeated_count} repeated detections)"
        elif repeated_count > 5:
            adjusted_duration = int(base_duration * 2.0)
            reason += f" (doubled for {repeated_count} repeated detections)"
        else:
            adjusted_duration = base_duration

        # Apply weather factor
        adjusted_duration = int(adjusted_duration * weather_factor)

        # Apply max limit
        adjusted_duration = min(adjusted_duration, self.max_duration)

        return True, adjusted_duration, reason


if __name__ == '__main__':
    # Quick test
    logging.basicConfig(level=logging.INFO)

    decision_maker = SprayDecisionMaker()
    print(f"Config: {decision_maker.get_config()}")

    # Test cases
    test_cases = [
        ('healthy', 0.95, None),
        ('early_blight', 0.85, 'mild'),
        ('late_blight', 0.92, 'severe'),
        ('leaf_mold', 0.75, 'moderate'),  # Below threshold
        ('unknown_disease', 0.88, None),
    ]

    print("\nTest decisions:")
    for disease, confidence, severity in test_cases:
        should_spray, duration, reason = decision_maker.decide(
            disease, confidence, severity
        )
        print(f"  {disease} ({confidence:.0%}): spray={should_spray}, "
              f"duration={duration}ms - {reason}")
