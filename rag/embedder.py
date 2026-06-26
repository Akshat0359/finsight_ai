"""
FinSight AI — Gemini Embedding Wrapper
Batched embedding with diskcache to avoid redundant API calls.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any

import google.generativeai as genai

from app.config import get_settings
from cache.disk_cache import TTL_EMBEDDING, build_cache_key, cache_get, cache_set

logger = logging.getLogger(__name__)

settings = get_settings()


def _configure_genai() -> None:
    """Configure the Gemini SDK (idempotent)."""
    genai.configure(api_key=settings.GEMINI_API_KEY)


def _text_hash(text: str) -> str:
    """Return SHA-256 hash of text for cache keying."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:32]


def embed_single(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """
    Embed a single text string using Gemini text-embedding-004.
    Returns a list of floats (embedding vector).
    Caches results to avoid redundant API calls.
    """
    _configure_genai()
    cache_key = build_cache_key("embed", task_type, _text_hash(text))
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        result = genai.embed_content(
            model=settings.GEMINI_EMBEDDING_MODEL,
            content=text,
            task_type=task_type,
        )
        vector: list[float] = result["embedding"]
        cache_set(cache_key, vector, ttl=TTL_EMBEDDING)
        return vector
    except Exception as exc:
        logger.error("Embedding error for text (len=%d): %s", len(text), exc)
        # Return zero vector as fallback so pipeline doesn't crash
        return [0.0] * 768


def embed_batch(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    batch_size: int | None = None,
) -> list[list[float]]:
    """
    Embed a list of texts in batches.
    Returns list of embedding vectors in the same order as input texts.
    Caches each embedding individually.
    """
    _configure_genai()
    _batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE

    if not texts:
        return []

    all_vectors: list[list[float]] = []

    # Check cache first for each text
    uncached_indices: list[int] = []
    result_map: dict[int, list[float]] = {}

    for i, text in enumerate(texts):
        cache_key = build_cache_key("embed", task_type, _text_hash(text))
        cached = cache_get(cache_key)
        if cached is not None:
            result_map[i] = cached
        else:
            uncached_indices.append(i)

    # Batch-embed uncached texts
    if uncached_indices:
        uncached_texts = [texts[i] for i in uncached_indices]
        logger.info(
            "Embedding %d texts (batch_size=%d, %d cached)",
            len(uncached_texts), _batch_size, len(result_map),
        )

        # Process in batches
        for batch_start in range(0, len(uncached_texts), _batch_size):
            batch = uncached_texts[batch_start: batch_start + _batch_size]
            try:
                result = genai.embed_content(
                    model=settings.GEMINI_EMBEDDING_MODEL,
                    content=batch,
                    task_type=task_type,
                )
                batch_vectors: list[list[float]] = result["embedding"]
                # Cache each result
                for j, vector in enumerate(batch_vectors):
                    original_idx = uncached_indices[batch_start + j]
                    result_map[original_idx] = vector
                    cache_key = build_cache_key(
                        "embed", task_type, _text_hash(texts[original_idx])
                    )
                    cache_set(cache_key, vector, ttl=TTL_EMBEDDING)
            except Exception as exc:
                logger.error(
                    "Batch embedding error (batch %d-%d): %s",
                    batch_start, batch_start + _batch_size, exc,
                )
                # Fill with zero vectors for this batch
                for j in range(len(batch)):
                    original_idx = uncached_indices[batch_start + j]
                    result_map[original_idx] = [0.0] * 768

    # Reassemble in original order
    for i in range(len(texts)):
        all_vectors.append(result_map.get(i, [0.0] * 768))

    return all_vectors


def embed_query(query: str) -> list[float]:
    """Embed a search query (uses RETRIEVAL_QUERY task type for better results)."""
    return embed_single(query, task_type="RETRIEVAL_QUERY")
