"""
FinSight AI — Tests for rag/retriever.py
Tests hybrid retrieval with mocked ChromaDB and BM25 re-ranking.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag.retriever import _bm25_scores, _normalize, _tokenize, retrieve


class TestTokenize:
    def test_basic_tokenize(self):
        result = _tokenize("Hello World")
        assert result == ["hello", "world"]

    def test_empty_string(self):
        assert _tokenize("") == [""]

    def test_lowercase_conversion(self):
        assert _tokenize("UPPERCASE") == ["uppercase"]


class TestNormalize:
    def test_normalize_all_same(self):
        result = _normalize([1.0, 1.0, 1.0])
        assert all(v == 0.5 for v in result)

    def test_normalize_range(self):
        result = _normalize([0.0, 0.5, 1.0])
        assert result[0] == 0.0
        assert result[-1] == 1.0
        assert 0.0 < result[1] < 1.0

    def test_normalize_empty(self):
        assert _normalize([]) == []


class TestBM25Scores:
    def test_relevant_doc_scores_higher(self):
        query = "cloud computing revenue"
        docs = [
            "Cloud revenue grew 30% driven by Azure computing services",
            "The company hired new executives last quarter",
            "Revenue from cloud computing services increased significantly",
        ]
        scores = _bm25_scores(query, docs)
        assert len(scores) == 3
        # First and third docs should score higher than second
        assert scores[0] > scores[1]
        assert scores[2] > scores[1]

    def test_empty_docs(self):
        assert _bm25_scores("query", []) == []

    def test_returns_correct_count(self):
        scores = _bm25_scores("test query", ["doc1", "doc2", "doc3"])
        assert len(scores) == 3


class TestRetrieve:
    """Tests for the main retrieve() function with mocked dependencies."""

    def _mock_candidates(self) -> list[dict]:
        return [
            {
                "text": "Cloud revenue grew significantly driven by Azure adoption",
                "metadata": {"section": "ITEM_7_MDA", "form_type": "10-K"},
                "distance": 0.1,
                "score": 0.95,
            },
            {
                "text": "The company faces regulatory challenges in Europe",
                "metadata": {"section": "ITEM_1A_RISK_FACTORS", "form_type": "10-K"},
                "distance": 0.3,
                "score": 0.70,
            },
            {
                "text": "Revenue grew 16% year over year to reach $245 billion",
                "metadata": {"section": "ITEM_7_MDA", "form_type": "10-K"},
                "distance": 0.2,
                "score": 0.80,
            },
        ]

    @patch("rag.retriever.embed_query")
    @patch("rag.retriever.semantic_search")
    def test_retrieve_returns_results(self, mock_sem_search, mock_embed):
        mock_embed.return_value = [0.1] * 768
        mock_sem_search.return_value = self._mock_candidates()

        results = retrieve(
            query="revenue growth cloud",
            collection="sec_filings",
            ticker="MSFT",
            top_k=3,
        )

        assert len(results) <= 3
        assert all("combined_score" in r for r in results)
        assert all("bm25_score" in r for r in results)

    @patch("rag.retriever.embed_query")
    @patch("rag.retriever.semantic_search")
    def test_retrieve_sorted_by_combined_score(self, mock_sem_search, mock_embed):
        mock_embed.return_value = [0.1] * 768
        mock_sem_search.return_value = self._mock_candidates()

        results = retrieve(
            query="revenue cloud growth",
            collection="sec_filings",
            ticker="MSFT",
            top_k=3,
        )

        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i]["combined_score"] >= results[i + 1]["combined_score"]

    @patch("rag.retriever.embed_query")
    @patch("rag.retriever.semantic_search")
    def test_retrieve_empty_collection(self, mock_sem_search, mock_embed):
        mock_embed.return_value = [0.1] * 768
        mock_sem_search.return_value = []

        results = retrieve(
            query="any query",
            collection="sec_filings",
            ticker="EMPTY",
            top_k=5,
        )
        assert results == []

    @patch("rag.retriever.embed_query")
    @patch("rag.retriever.semantic_search")
    def test_retrieve_respects_top_k(self, mock_sem_search, mock_embed):
        mock_embed.return_value = [0.1] * 768
        # Return 5 candidates
        mock_sem_search.return_value = self._mock_candidates() + self._mock_candidates()[:2]

        results = retrieve(
            query="revenue",
            collection="sec_filings",
            ticker="MSFT",
            top_k=2,
        )
        assert len(results) <= 2

    @patch("rag.retriever.embed_query")
    def test_retrieve_handles_embed_failure(self, mock_embed):
        mock_embed.side_effect = Exception("API error")

        results = retrieve(
            query="any query",
            collection="sec_filings",
            ticker="MSFT",
            top_k=5,
        )
        assert results == []
