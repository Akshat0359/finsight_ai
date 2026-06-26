"""
FinSight AI — Hybrid Semantic + BM25 Retriever
Combines ChromaDB cosine similarity with BM25 re-ranking.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from rag.embedder import embed_query
from rag.vectorstore import semantic_search

logger = logging.getLogger(__name__)

# Weight for combining semantic and BM25 scores
SEMANTIC_WEIGHT = 0.7
BM25_WEIGHT = 0.3


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


def _bm25_scores(
    query: str, documents: list[str]
) -> list[float]:
    """Compute BM25 scores for all documents against query."""
    if not documents:
        return []
    tokenized_docs = [_tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(_tokenize(query))
    return [float(s) for s in scores]


def _normalize(scores: list[float]) -> list[float]:
    """Min-max normalize a list of scores to [0, 1]."""
    arr = np.array(scores, dtype=float)
    min_v = arr.min()
    max_v = arr.max()
    if max_v == min_v:
        return [0.5] * len(scores)
    return ((arr - min_v) / (max_v - min_v)).tolist()


def retrieve(
    query: str,
    collection: str,
    ticker: str,
    top_k: int = 5,
    section_filter: str | None = None,
    candidate_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Hybrid retrieval: semantic search → BM25 re-ranking.

    Args:
        query: Natural language search query
        collection: ChromaDB collection name (e.g., "sec_filings")
        ticker: Company ticker for collection namespacing
        top_k: Number of final results to return
        section_filter: Optionally filter by SEC section name
        candidate_k: How many candidates to pull from ChromaDB before re-ranking

    Returns:
        List of {text, metadata, score, bm25_score, combined_score} dicts
        sorted by combined_score descending.
    """
    _candidate_k = candidate_k or min(top_k * 4, 50)

    # Build ChromaDB where filter
    where: dict[str, Any] | None = None
    if section_filter:
        where = {"section": {"$eq": section_filter}}

    # Embed query and run semantic search
    try:
        query_embedding = embed_query(query)
    except Exception as exc:
        logger.error("Failed to embed query: %s", exc)
        return []

    candidates = semantic_search(
        query_embedding=query_embedding,
        collection=collection,
        ticker=ticker,
        top_k=_candidate_k,
        where=where,
    )

    if not candidates:
        return []

    # BM25 re-ranking
    docs = [c["text"] for c in candidates]
    bm25_raw = _bm25_scores(query, docs)
    sem_scores = [c["score"] for c in candidates]

    # Normalize both score lists
    norm_bm25 = _normalize(bm25_raw) if bm25_raw else [0.0] * len(candidates)
    norm_sem = _normalize(sem_scores)

    # Combine scores
    combined: list[float] = [
        SEMANTIC_WEIGHT * s + BM25_WEIGHT * b
        for s, b in zip(norm_sem, norm_bm25)
    ]

    # Attach scores and sort
    for i, candidate in enumerate(candidates):
        candidate["bm25_score"] = round(bm25_raw[i], 4) if bm25_raw else 0.0
        candidate["combined_score"] = round(combined[i], 4)

    ranked = sorted(candidates, key=lambda x: x["combined_score"], reverse=True)
    return ranked[:top_k]


def retrieve_multi_collection(
    query: str,
    collections: list[str],
    ticker: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Retrieve from multiple collections and merge results.
    Returns top_k results by combined_score across all collections.
    """
    all_results: list[dict[str, Any]] = []
    per_collection_k = max(top_k, 3)

    for col in collections:
        try:
            results = retrieve(
                query=query,
                collection=col,
                ticker=ticker,
                top_k=per_collection_k,
            )
            for r in results:
                r["collection"] = col
            all_results.extend(results)
        except Exception as exc:
            logger.warning("Retrieval failed for collection '%s': %s", col, exc)

    # Re-rank merged results
    all_results.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)
    return all_results[:top_k]
