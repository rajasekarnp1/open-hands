#!/usr/bin/env python3
"""
Intelligent Caching Layer

Provides smart caching for OpenHands operations with automatic invalidation.
"""

import asyncio
import time
import hashlib
import json
from typing import Any, Optional, Dict, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    created_at: float
    access_count: int
    ttl: float
    key_hash: str

class IntelligentCache:
    """Smart caching system with automatic optimization."""

    def __init__(self, max_size: int = 10000, default_ttl: float = 3600):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.hit_count = 0
        self.miss_count = 0

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""

        key_hash = self._hash_key(key)

        if key_hash in self.cache:
            entry = self.cache[key_hash]

            # Check if expired
            if time.time() - entry.created_at > entry.ttl:
                del self.cache[key_hash]
                self.miss_count += 1
                return None

            # Update access count
            entry.access_count += 1
            self.hit_count += 1

            logger.debug(f"Cache hit for key: {key}")
            return entry.value

        self.miss_count += 1
        logger.debug(f"Cache miss for key: {key}")
        return None

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""

        key_hash = self._hash_key(key)
        ttl = ttl or self.default_ttl

        # Evict if cache is full
        if len(self.cache) >= self.max_size:
            await self._evict_lru()

        entry = CacheEntry(
            value=value,
            created_at=time.time(),
            access_count=1,
            ttl=ttl,
            key_hash=key_hash
        )

        self.cache[key_hash] = entry
        logger.debug(f"Cached value for key: {key}")

    async def invalidate(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern."""

        invalidated = 0
        keys_to_remove = []

        for key_hash, entry in self.cache.items():
            if pattern in key_hash:  # Simple pattern matching
                keys_to_remove.append(key_hash)

        for key_hash in keys_to_remove:
            del self.cache[key_hash]
            invalidated += 1

        logger.info(f"Invalidated {invalidated} cache entries")
        return invalidated

    def _hash_key(self, key: str) -> str:
        """Create hash for cache key."""
        return hashlib.md5(key.encode()).hexdigest()

    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""

        if not self.cache:
            return

        # Find entry with lowest access count
        lru_key = min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
        del self.cache[lru_key]

        logger.debug("Evicted LRU cache entry")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""

        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0

        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }

# Global cache instance
global_cache = IntelligentCache()

def cached(ttl: Optional[float] = None, key_func: Optional[Callable] = None):
    """Decorator for caching function results."""

    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"

            # Try to get from cache
            cached_result = await global_cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = await func(*args, **kwargs)
            await global_cache.set(cache_key, result, ttl)

            return result

        return wrapper
    return decorator
