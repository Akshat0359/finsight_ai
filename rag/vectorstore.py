"""
FinSight AI — ChromaDB Vector Store Client
Wraps ChromaDB with collection management and upsert helpers.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings

logger = logging.getLogger(__name__)

app_settings = get_settings()


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.PersistentClient:
    """Return singleton ChromaDB persistent client."""
    client = chromadb.PersistentClient(
        path=app_settings.CHROMA_PERSIST_DIR,
        settings=ChromaSettings(
            anonymized_telemetry=False,
            allow_reset=True,
        ),
    )
    logger.info("ChromaDB client initialized at %s", app_settings.CHROMA_PERSIST_DIR)
    return client


def _collection_name(collection: str, ticker: str) -> str:
    """Build a sanitized collection name: {ticker_lower}_{collection}."""
    safe_ticker = ticker.lower().replace("-", "_").replace(".", "_")
    safe_collection = collection.lower().replace("-", "_").replace(" ", "_")
    return f"{safe_ticker}_{safe_collection}"


def get_or_create_collection(
    collection: str,
    ticker: str,
    metadata: dict[str, Any] | None = None,
) -> chromadb.Collection:
    """Get existing collection or create a new one."""
    client = get_chroma_client()
    name = _collection_name(collection, ticker)
    col = client.get_or_create_collection(
        name=name,
        metadata=metadata or {"hnsw:space": "cosine"},
    )
    return col


def upsert_chunks(
    texts: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]],
    collection: str,
    ticker: str,
) -> int:
    """
    Upsert text chunks with their embeddings into a collection.
    Returns number of documents stored.
    """
    if not texts:
        return 0

    col = get_or_create_collection(collection, ticker)

    # Build IDs from metadata or position
    ids: list[str] = []
    for i, meta in enumerate(metadatas):
        section = meta.get("section", "doc")
        chunk_idx = meta.get("chunk_index", i)
        form = meta.get("form_type", "doc")
        ids.append(f"{ticker}_{form}_{section}_{chunk_idx}_{i}")

    # Deduplicate by ID
    unique: dict[str, tuple[str, list[float], dict[str, Any]]] = {}
    for doc_id, text, emb, meta in zip(ids, texts, embeddings, metadatas):
        unique[doc_id] = (text, emb, meta)

    final_ids = list(unique.keys())
    final_texts = [unique[i][0] for i in final_ids]
    final_embeddings = [unique[i][1] for i in final_ids]
    final_metadatas = [unique[i][2] for i in final_ids]

    try:
        col.upsert(
            ids=final_ids,
            documents=final_texts,
            embeddings=final_embeddings,
            metadatas=final_metadatas,
        )
        logger.info(
            "Upserted %d documents into collection '%s'",
            len(final_ids), _collection_name(collection, ticker),
        )
        return len(final_ids)
    except Exception as exc:
        logger.error("ChromaDB upsert error: %s", exc)
        return 0


def semantic_search(
    query_embedding: list[float],
    collection: str,
    ticker: str,
    top_k: int = 5,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Perform semantic search against a collection.
    Returns list of {text, metadata, distance, score} dicts.
    """
    try:
        col = get_or_create_collection(collection, ticker)
        count = col.count()
        if count == 0:
            return []

        k = min(top_k, count)
        query_params: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_params["where"] = where

        results = col.query(**query_params)
        output: list[dict[str, Any]] = []

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            # Convert cosine distance [0,2] to similarity score [0,1]
            score = round(1.0 - dist / 2.0, 4)
            output.append({
                "text": doc,
                "metadata": meta,
                "distance": round(float(dist), 4),
                "score": score,
            })

        return output

    except Exception as exc:
        logger.error("ChromaDB search error: %s", exc)
        return []


def collection_exists(collection: str, ticker: str) -> bool:
    """Check if a collection exists and has documents."""
    try:
        client = get_chroma_client()
        name = _collection_name(collection, ticker)
        existing = [c.name for c in client.list_collections()]
        if name not in existing:
            return False
        col = client.get_collection(name)
        return col.count() > 0
    except Exception:
        return False


def get_collection_count(collection: str, ticker: str) -> int:
    """Return document count in a collection."""
    try:
        col = get_or_create_collection(collection, ticker)
        return col.count()
    except Exception:
        return 0


def delete_collection(collection: str, ticker: str) -> bool:
    """Delete a collection entirely."""
    try:
        client = get_chroma_client()
        name = _collection_name(collection, ticker)
        client.delete_collection(name)
        logger.info("Deleted collection '%s'", name)
        return True
    except Exception as exc:
        logger.warning("Could not delete collection: %s", exc)
        return False
