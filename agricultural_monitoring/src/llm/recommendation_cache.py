#!/usr/bin/env python3
"""
Recommendation Cache for Claude LLM Integration

Caches LLM responses to reduce API costs and improve response time.
Uses a simple file-based cache with TTL expiration.

Author: Agricultural Robotics Team
License: MIT
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import threading

logger = logging.getLogger(__name__)


class RecommendationCache:
    """
    File-based cache for LLM recommendations.

    Features:
    - Configurable TTL (time-to-live)
    - Maximum entry limit
    - Thread-safe operations
    - Persistence across restarts

    Attributes:
        enabled: Whether caching is enabled
        ttl_hours: Time-to-live in hours
        max_entries: Maximum cached entries
        cache_file: Path to cache file
    """

    def __init__(
        self,
        enabled: bool = True,
        ttl_hours: int = 24,
        max_entries: int = 100,
        cache_file: Optional[str] = None
    ) -> None:
        """
        Initialize the recommendation cache.

        Args:
            enabled: Enable/disable caching
            ttl_hours: Hours before cache entries expire
            max_entries: Maximum number of cached entries
            cache_file: Path to cache file (default: ./cache/llm_responses.json)
        """
        self.enabled = enabled
        self.ttl_hours = ttl_hours
        self.max_entries = max_entries

        # Set cache file path
        if cache_file:
            self.cache_file = Path(cache_file)
        else:
            self.cache_file = (
                Path(__file__).parent.parent.parent / 'cache' / 'llm_responses.json'
            )

        # Ensure cache directory exists
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self.cache: Dict[str, Dict[str, Any]] = {}

        # Thread lock
        self.lock = threading.Lock()

        # Load existing cache
        self._load()

        # Statistics
        self.hits = 0
        self.misses = 0

        logger.info(
            f"Cache initialized: enabled={enabled}, ttl={ttl_hours}h, "
            f"max={max_entries}, loaded={len(self.cache)} entries"
        )

    def _load(self) -> None:
        """Load cache from disk."""
        if not self.cache_file.exists():
            return

        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)

            # Validate and load entries
            now = datetime.now()
            valid_entries = {}

            for key, entry in data.items():
                if self._is_valid_entry(entry, now):
                    valid_entries[key] = entry

            self.cache = valid_entries
            logger.debug(f"Loaded {len(self.cache)} valid cache entries")

        except json.JSONDecodeError as e:
            logger.warning(f"Cache file corrupted, starting fresh: {e}")
            self.cache = {}
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            self.cache = {}

    def _is_valid_entry(self, entry: Dict, now: datetime) -> bool:
        """Check if a cache entry is still valid."""
        try:
            created = datetime.fromisoformat(entry.get('created', ''))
            expiry = created + timedelta(hours=self.ttl_hours)
            return now < expiry
        except (ValueError, TypeError):
            return False

    def save(self) -> None:
        """Save cache to disk."""
        if not self.enabled:
            return

        with self.lock:
            try:
                with open(self.cache_file, 'w') as f:
                    json.dump(self.cache, f, indent=2)
                logger.debug(f"Saved {len(self.cache)} cache entries")
            except Exception as e:
                logger.error(f"Failed to save cache: {e}")

    def build_key(self, disease_name: str, confidence: float) -> str:
        """
        Build cache key from detection parameters.

        Args:
            disease_name: Detected disease name
            confidence: Detection confidence

        Returns:
            Hash-based cache key
        """
        # Round confidence to reduce key variations
        confidence_bucket = round(confidence, 1)

        key_data = f"{disease_name.lower()}:{confidence_bucket}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]

    def get(self, key: str) -> Optional[str]:
        """
        Get cached recommendation.

        Args:
            key: Cache key

        Returns:
            Cached recommendation or None if not found/expired
        """
        if not self.enabled:
            self.misses += 1
            return None

        with self.lock:
            entry = self.cache.get(key)

            if entry is None:
                self.misses += 1
                return None

            # Check expiration
            if not self._is_valid_entry(entry, datetime.now()):
                del self.cache[key]
                self.misses += 1
                return None

            # Update access time
            entry['accessed'] = datetime.now().isoformat()
            entry['access_count'] = entry.get('access_count', 0) + 1

            self.hits += 1
            return entry.get('recommendation')

    def set(self, key: str, recommendation: str) -> None:
        """
        Cache a recommendation.

        Args:
            key: Cache key
            recommendation: Recommendation text to cache
        """
        if not self.enabled:
            return

        with self.lock:
            # Enforce max entries limit
            if len(self.cache) >= self.max_entries:
                self._evict_oldest()

            self.cache[key] = {
                'recommendation': recommendation,
                'created': datetime.now().isoformat(),
                'accessed': datetime.now().isoformat(),
                'access_count': 0
            }

            # Save periodically (every 10 new entries)
            if len(self.cache) % 10 == 0:
                self._save_async()

    def _evict_oldest(self) -> None:
        """Remove oldest entries to make room."""
        if not self.cache:
            return

        # Sort by last access time
        sorted_keys = sorted(
            self.cache.keys(),
            key=lambda k: self.cache[k].get('accessed', '')
        )

        # Remove oldest 10%
        remove_count = max(1, len(sorted_keys) // 10)
        for key in sorted_keys[:remove_count]:
            del self.cache[key]

        logger.debug(f"Evicted {remove_count} old cache entries")

    def _save_async(self) -> None:
        """Save cache in background thread."""
        thread = threading.Thread(target=self.save, daemon=True)
        thread.start()

    def invalidate(self, key: str) -> bool:
        """
        Remove a specific cache entry.

        Args:
            key: Cache key to invalidate

        Returns:
            True if entry was removed, False if not found
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        with self.lock:
            count = len(self.cache)
            self.cache = {}
            self.save()
            return count

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        with self.lock:
            now = datetime.now()
            expired_keys = [
                key for key, entry in self.cache.items()
                if not self._is_valid_entry(entry, now)
            ]

            for key in expired_keys:
                del self.cache[key]

            if expired_keys:
                self.save()

            return len(expired_keys)

    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0

        return {
            'enabled': self.enabled,
            'size': len(self.cache),
            'max_entries': self.max_entries,
            'ttl_hours': self.ttl_hours,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': round(hit_rate, 3)
        }

    def get_entry_info(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a cache entry.

        Args:
            key: Cache key

        Returns:
            Entry metadata or None if not found
        """
        with self.lock:
            entry = self.cache.get(key)
            if entry:
                return {
                    'created': entry.get('created'),
                    'accessed': entry.get('accessed'),
                    'access_count': entry.get('access_count', 0),
                    'recommendation_length': len(entry.get('recommendation', ''))
                }
        return None


class DiseaseSpecificCache(RecommendationCache):
    """
    Extended cache with disease-specific features.

    Groups recommendations by disease for bulk operations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disease_keys: Dict[str, set] = {}

    def set_with_disease(
        self,
        key: str,
        recommendation: str,
        disease_name: str
    ) -> None:
        """
        Cache recommendation with disease tracking.

        Args:
            key: Cache key
            recommendation: Recommendation text
            disease_name: Disease name for grouping
        """
        self.set(key, recommendation)

        # Track key by disease
        if disease_name not in self.disease_keys:
            self.disease_keys[disease_name] = set()
        self.disease_keys[disease_name].add(key)

    def invalidate_disease(self, disease_name: str) -> int:
        """
        Invalidate all cache entries for a disease.

        Useful when treatment protocols change.

        Args:
            disease_name: Disease name to invalidate

        Returns:
            Number of entries invalidated
        """
        keys = self.disease_keys.get(disease_name, set())
        count = 0

        for key in keys:
            if self.invalidate(key):
                count += 1

        self.disease_keys[disease_name] = set()
        return count

    def get_disease_count(self, disease_name: str) -> int:
        """Get number of cached entries for a disease."""
        return len(self.disease_keys.get(disease_name, set()))


if __name__ == '__main__':
    # Quick test
    logging.basicConfig(level=logging.DEBUG)

    cache = RecommendationCache(ttl_hours=1, max_entries=5)

    # Test caching
    for i in range(7):
        key = cache.build_key(f"disease_{i}", 0.85)
        cache.set(key, f"Recommendation for disease {i}")

    print(f"Cache stats: {cache.get_stats()}")

    # Test retrieval
    key = cache.build_key("disease_5", 0.85)
    result = cache.get(key)
    print(f"Retrieved: {result}")

    print(f"Final stats: {cache.get_stats()}")
