"""
FinSight AI — Vector Store MCP Tools
Bridge between MCP server and the rag.vectorstore / rag.embedder modules.
"""
from __future__ import annotations

import logging
from typing import Any

from rag.embedder import embed_batch, embed_query
from rag.vectorstore import (
    collection_exists,
    get_collection_count,
    semantic_search as vs_search,
    upsert_chunks,
)

logger = logging.getLogger(__name__)


def embed_and_store(
    texts: list[str],
    metadatas: list[dict[str, Any]],
    collection: str,
    ticker: str,
) -> int:
    """
    Embed texts using Gemini and store them in ChromaDB.

    Args:
        texts: List of text chunks to embed and store
        metadatas: Parallel list of metadata dicts for each chunk
        collection: Collection name (e.g., 'sec_filings', 'news')
        ticker: Company ticker for collection namespacing

    Returns:
        Number of documents stored
    """
    if not texts:
        return 0

    if len(texts) != len(metadatas):
        logger.error(
            "texts (%d) and metadatas (%d) must have same length",
            len(texts), len(metadatas),
        )
        return 0

    try:
        embeddings = embed_batch(texts, task_type="RETRIEVAL_DOCUMENT")
        count = upsert_chunks(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            collection=collection,
            ticker=ticker,
        )
        logger.info(
            "embed_and_store: stored %d chunks in '%s' for %s", count, collection, ticker
        )
        return count
    except Exception as exc:
        logger.error("embed_and_store error: %s", exc)
        return 0


def semantic_search(
    query: str,
    collection: str,
    ticker: str,
    top_k: int = 5,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Perform semantic search against a ChromaDB collection.

    Args:
        query: Natural language search query
        collection: Collection name
        ticker: Company ticker
        top_k: Number of results to return
        where: Optional ChromaDB metadata filter

    Returns:
        List of {text, metadata, distance, score} dicts
    """
    try:
        query_embedding = embed_query(query)
        results = vs_search(
            query_embedding=query_embedding,
            collection=collection,
            ticker=ticker,
            top_k=top_k,
            where=where,
        )
        return results
    except Exception as exc:
        logger.error("semantic_search error: %s", exc)
        return []


def check_collection_exists(collection: str, ticker: str) -> bool:
    """
    Check whether a ChromaDB collection exists and has documents.

    Args:
        collection: Collection name
        ticker: Company ticker

    Returns:
        True if collection exists and has at least one document
    """
    return collection_exists(collection, ticker)


def get_collection_document_count(collection: str, ticker: str) -> int:
    """
    Return the number of documents in a collection.

    Args:
        collection: Collection name
        ticker: Company ticker

    Returns:
        Document count (0 if collection doesn't exist)
    """
    return get_collection_count(collection, ticker)
