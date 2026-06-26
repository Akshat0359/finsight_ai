"""
FinSight AI — diskcache wrapper with TTL helpers.
Provides a simple key-value cache backed by the filesystem.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import diskcache

from app.config import get_settings

logger = logging.getLogger(__name__)

# Default TTL values (seconds)
TTL_SHORT = 3_600       # 1 hour  — market prices
TTL_MEDIUM = 86_400     # 1 day   — LLM responses, news
TTL_LONG = 604_800      # 7 days  — SEC filings, embeddings
TTL_EMBEDDING = 2_592_000  # 30 days — embedding vectors


@lru_cache(maxsize=1)
def get_cache() -> diskcache.Cache:
    """Return a singleton diskcache.Cache instance."""
    settings = get_settings()
    cache = diskcache.Cache(
        directory=settings.CACHE_DIR,
        size_limit=2 ** 30,  # 1 GB max disk usage
        eviction_policy="least-recently-used",
    )
    logger.debug("diskcache initialized at %s", settings.CACHE_DIR)
    return cache


def cache_get(key: str, default: Any = None) -> Any:
    """Get a value from cache. Returns default if missing or expired."""
    cache = get_cache()
    try:
        value = cache.get(key, default=default)
        return value
    except Exception as exc:
        logger.warning("Cache get error for key %s: %s", key, exc)
        return default


def cache_set(key: str, value: Any, ttl: int = TTL_MEDIUM) -> bool:
    """Set a value in cache with TTL in seconds."""
    cache = get_cache()
    try:
        cache.set(key, value, expire=ttl)
        return True
    except Exception as exc:
        logger.warning("Cache set error for key %s: %s", key, exc)
        return False


def cache_delete(key: str) -> bool:
    """Delete a key from cache."""
    cache = get_cache()
    try:
        cache.delete(key)
        return True
    except Exception as exc:
        logger.warning("Cache delete error for key %s: %s", key, exc)
        return False


def cache_exists(key: str) -> bool:
    """Check if a key exists and is not expired."""
    cache = get_cache()
    try:
        return key in cache
    except Exception:
        return False


def build_cache_key(*parts: str) -> str:
    """Build a namespaced cache key from parts."""
    return ":".join(str(p) for p in parts)


def clear_ticker_cache(ticker: str) -> int:
    """Delete all cache entries for a given ticker. Returns count deleted."""
    cache = get_cache()
    deleted = 0
    try:
        for key in list(cache.iterkeys()):
            if isinstance(key, str) and ticker.upper() in key.upper():
                cache.delete(key)
                deleted += 1
    except Exception as exc:
        logger.warning("Error clearing cache for ticker %s: %s", ticker, exc)
    return deleted
