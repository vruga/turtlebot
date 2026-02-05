#!/usr/bin/env python3
"""
Claude API Client for Agricultural Recommendations

Integrates with Anthropic's Claude API to provide farmer-friendly
recommendations based on disease detection results.

Features:
- Async API calls to avoid blocking
- Response caching to reduce API costs
- Fallback to local database if API fails
- Context-aware prompts based on detection history

Author: Agricultural Robotics Team
License: MIT
"""

import os
import sys
import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.llm.prompt_builder import PromptBuilder
from src.llm.recommendation_cache import RecommendationCache

# Try to import anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logging.warning("anthropic package not installed. Install with: pip install anthropic")


logger = logging.getLogger(__name__)


class ClaudeClient(Node):
    """
    ROS2 node for Claude API integration.

    Provides farmer recommendations based on disease detections.
    Uses async API calls to avoid blocking the ROS2 system.

    Topics:
        Subscribed:
            - /agricultural/disease_detection (String): Detection results
        Published:
            - /agricultural/recommendation (String): LLM recommendations
    """

    def __init__(self) -> None:
        super().__init__('claude_client')

        # Set lower priority
        self._set_priority()

        # Load configuration
        self.config = self._load_config()

        # Initialize API client
        self.api_client: Optional[anthropic.Anthropic] = None
        self._init_api_client()

        # Initialize prompt builder
        config_dir = Path(__file__).parent.parent.parent / 'config'
        self.prompt_builder = PromptBuilder(config_path=config_dir / 'llm_config.yaml')

        # Initialize cache
        cache_config = self.config.get('cache', {})
        self.cache = RecommendationCache(
            enabled=cache_config.get('enabled', True),
            ttl_hours=cache_config.get('ttl_hours', 24),
            max_entries=cache_config.get('max_entries', 100),
            cache_file=cache_config.get('cache_file')
        )

        # Load fallback responses
        self.fallback_responses = self._load_fallback_responses()

        # Detection history for context
        self.detection_history: List[Dict] = []
        self.max_history = 20

        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=2)

        # API settings
        api_config = self.config.get('api', {})
        self.model = api_config.get('model', 'claude-sonnet-4-20250514')
        self.timeout = api_config.get('timeout_seconds', 5)
        self.max_tokens = api_config.get('max_tokens', 500)
        self.retry_attempts = api_config.get('retry_attempts', 2)

        # Statistics
        self.total_requests = 0
        self.cache_hits = 0
        self.api_calls = 0
        self.fallback_uses = 0

        # QoS profiles
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )

        # Subscribe to disease detection
        self.detection_sub = self.create_subscription(
            String,
            '/agricultural/disease_detection',
            self._detection_callback,
            reliable_qos
        )

        # Publisher for recommendations
        self.recommendation_pub = self.create_publisher(
            String,
            '/agricultural/recommendation',
            reliable_qos
        )

        # Status publisher
        self.status_pub = self.create_publisher(
            String,
            '/agricultural/llm_status',
            reliable_qos
        )

        api_status = "available" if self.api_client else "not configured"
        self.get_logger().info(f"Claude Client initialized (API: {api_status})")

    def _set_priority(self) -> None:
        """Set process to lower priority."""
        try:
            os.nice(10)
            self.get_logger().info("Process priority set to nice +10")
        except Exception as e:
            self.get_logger().warning(f"Could not set priority: {e}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML files."""
        config = {}
        config_dir = Path(__file__).parent.parent.parent / 'config'

        llm_config = config_dir / 'llm_config.yaml'
        if llm_config.exists():
            try:
                with open(llm_config, 'r') as f:
                    config = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Failed to load LLM config: {e}")

        return config

    def _init_api_client(self) -> None:
        """Initialize Anthropic API client."""
        if not ANTHROPIC_AVAILABLE:
            self.get_logger().warning("Anthropic package not available")
            return

        # Get API key from environment
        api_key_env = self.config.get('api', {}).get('api_key_env', 'ANTHROPIC_API_KEY')
        api_key = os.environ.get(api_key_env)

        if not api_key:
            self.get_logger().warning(
                f"API key not found in environment variable {api_key_env}"
            )
            return

        try:
            self.api_client = anthropic.Anthropic(api_key=api_key)
            self.get_logger().info("Anthropic API client initialized")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize API client: {e}")

    def _load_fallback_responses(self) -> Dict[str, str]:
        """Load fallback responses for offline mode."""
        fallback_config = self.config.get('fallback', {})
        db_path = fallback_config.get('database_path')

        if db_path:
            full_path = Path(__file__).parent.parent.parent / db_path
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load fallback database: {e}")

        # Default fallback responses
        return {
            'early_blight': (
                "Early blight is a common fungal disease. Our automated spray "
                "treatment has been applied. Monitor the affected area and ensure "
                "good air circulation. Remove severely affected leaves. Consider "
                "copper-based fungicides for additional treatment."
            ),
            'late_blight': (
                "Late blight is a serious disease that spreads quickly. Our spray "
                "treatment has been applied, but this disease requires aggressive "
                "management. Remove all infected plant material immediately. "
                "Contact your local agricultural extension for guidance."
            ),
            'leaf_mold': (
                "Leaf mold thrives in humid conditions. Our treatment has been "
                "applied. Improve ventilation and reduce humidity if possible. "
                "Avoid overhead watering. Monitor closely for spread."
            ),
            'healthy': (
                "Your plant appears healthy. No treatment is needed at this time. "
                "Continue regular monitoring and maintain good growing conditions."
            ),
            'default': (
                "A plant disease has been detected and our automated system has "
                "applied spray treatment. Monitor the affected plants closely "
                "over the next 24-48 hours. If symptoms worsen or spread, "
                "contact your local agricultural extension service."
            )
        }

    def _detection_callback(self, msg: String) -> None:
        """
        Handle disease detection results.

        Generates recommendation asynchronously to avoid blocking.

        Args:
            msg: JSON string with detection result
        """
        try:
            detection = json.loads(msg.data)

            # Check for error
            if detection.get('error'):
                return

            # Add to history
            self.detection_history.append(detection)
            if len(self.detection_history) > self.max_history:
                self.detection_history = self.detection_history[-self.max_history:]

            # Generate recommendation asynchronously
            self.executor.submit(self._generate_recommendation, detection)

        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse detection: {e}")

    def _generate_recommendation(self, detection: Dict) -> None:
        """
        Generate recommendation for disease detection.

        Tries in order:
        1. Cache lookup
        2. Claude API call
        3. Fallback database

        Args:
            detection: Detection result dictionary
        """
        self.total_requests += 1
        disease_name = detection.get('disease_name', 'unknown')
        confidence = detection.get('confidence', 0)

        # Build cache key
        cache_key = self.cache.build_key(disease_name, confidence)

        # Check cache first
        cached = self.cache.get(cache_key)
        if cached:
            self.cache_hits += 1
            self.get_logger().debug(f"Cache hit for {disease_name}")
            self._publish_recommendation(cached, disease_name, "cache")
            return

        # Build prompt
        prompt = self.prompt_builder.build_prompt(
            detection=detection,
            history=self.detection_history[-5:],
            context=self._get_context()
        )

        # Try API call
        recommendation = None
        if self.api_client:
            recommendation = self._call_api(prompt)
            if recommendation:
                self.api_calls += 1
                # Cache the response
                self.cache.set(cache_key, recommendation)
                self._publish_recommendation(recommendation, disease_name, "api")
                return

        # Fall back to local database
        self.fallback_uses += 1
        recommendation = self._get_fallback(disease_name)
        self._publish_recommendation(recommendation, disease_name, "fallback")

    def _call_api(self, prompt: str) -> Optional[str]:
        """
        Call Claude API with retry logic.

        Args:
            prompt: Formatted prompt string

        Returns:
            API response text or None if failed
        """
        if not self.api_client:
            return None

        system_prompt = self.prompt_builder.get_system_prompt()

        for attempt in range(self.retry_attempts):
            try:
                response = self.api_client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    system=system_prompt,
                    timeout=self.timeout
                )

                if response.content:
                    return response.content[0].text

            except anthropic.APITimeoutError:
                self.get_logger().warning(
                    f"API timeout (attempt {attempt + 1}/{self.retry_attempts})"
                )
            except anthropic.APIError as e:
                self.get_logger().error(f"API error: {e}")
                break
            except Exception as e:
                self.get_logger().error(f"Unexpected error: {e}")
                break

        return None

    def _get_fallback(self, disease_name: str) -> str:
        """Get fallback response for disease."""
        normalized = disease_name.lower().replace(' ', '_')

        if normalized in self.fallback_responses:
            return self.fallback_responses[normalized]

        return self.fallback_responses.get('default', "Please consult an agricultural expert.")

    def _get_context(self) -> Dict[str, Any]:
        """Get context information for prompt building."""
        today_detections = [
            d for d in self.detection_history
            if datetime.fromisoformat(d.get('timestamp', '')).date() == datetime.now().date()
        ]

        # Count diseases today
        disease_counts = {}
        for d in today_detections:
            name = d.get('disease_name', 'unknown')
            disease_counts[name] = disease_counts.get(name, 0) + 1

        return {
            'detections_today': len(today_detections),
            'disease_counts': disease_counts,
            'is_first_today': len(today_detections) <= 1,
            'time_of_day': datetime.now().strftime('%H:%M')
        }

    def _publish_recommendation(
        self,
        recommendation: str,
        disease_name: str,
        source: str
    ) -> None:
        """Publish recommendation to ROS2 topic."""
        result = {
            'recommendation': recommendation,
            'disease_name': disease_name,
            'source': source,
            'timestamp': datetime.now().isoformat()
        }

        msg = String()
        msg.data = json.dumps(result)
        self.recommendation_pub.publish(msg)

        self.get_logger().info(f"Recommendation published for {disease_name} ({source})")

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            'total_requests': self.total_requests,
            'cache_hits': self.cache_hits,
            'api_calls': self.api_calls,
            'fallback_uses': self.fallback_uses,
            'cache_hit_rate': (
                round(self.cache_hits / self.total_requests, 2)
                if self.total_requests > 0 else 0
            ),
            'api_available': self.api_client is not None,
            'cache_size': self.cache.size()
        }

    def destroy_node(self) -> None:
        """Clean shutdown."""
        self.get_logger().info("Shutting down Claude Client")
        self.get_logger().info(f"Final stats: {self.get_stats()}")

        # Save cache
        self.cache.save()

        # Shutdown executor
        self.executor.shutdown(wait=False)

        super().destroy_node()


def main(args=None) -> None:
    """Main entry point for Claude client."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    rclpy.init(args=args)

    node = ClaudeClient()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
