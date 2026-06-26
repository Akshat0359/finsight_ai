"""
FinSight AI — MCP Client Wrapper
Direct Python calls to tool functions (bypasses HTTP for same-process use).
When the MCP server is separate, this can be swapped for HTTP client calls.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Thin wrapper around MCP tool functions.
    Calls tool functions directly (in-process) for reliability.
    All methods mirror the MCP tool signatures exactly.
    """

    # ------------------------------------------------------------------ #
    # EDGAR
    # ------------------------------------------------------------------ #
    def search_company_cik(self, company_name: str) -> dict[str, str]:
        from mcp_server.tools.edgar_tools import search_company_cik
        return search_company_cik(company_name)

    def get_recent_filings(
        self, cik: str, form_types: list[str], limit: int = 10
    ) -> list[dict[str, Any]]:
        from mcp_server.tools.edgar_tools import get_recent_filings
        return get_recent_filings(cik, form_types, limit)

    def download_filing_text(self, cik: str, accession_number: str) -> str:
        from mcp_server.tools.edgar_tools import download_filing_text
        return download_filing_text(cik, accession_number)

    def get_company_facts(self, cik: str) -> dict[str, Any]:
        from mcp_server.tools.edgar_tools import get_company_facts
        return get_company_facts(cik)

    # ------------------------------------------------------------------ #
    # Market Data
    # ------------------------------------------------------------------ #
    def get_price_history(self, ticker: str, period: str = "1y") -> dict[str, Any]:
        from mcp_server.tools.market_tools import get_price_history
        return get_price_history(ticker, period)

    def get_financial_statements(self, ticker: str) -> dict[str, Any]:
        from mcp_server.tools.market_tools import get_financial_statements
        return get_financial_statements(ticker)

    def get_company_info(self, ticker: str) -> dict[str, Any]:
        from mcp_server.tools.market_tools import get_company_info
        return get_company_info(ticker)

    def compute_financial_ratios(self, ticker: str) -> dict[str, Any]:
        from mcp_server.tools.market_tools import compute_financial_ratios
        return compute_financial_ratios(ticker)

    # ------------------------------------------------------------------ #
    # News
    # ------------------------------------------------------------------ #
    def get_news_articles(
        self, query: str, ticker: str, days: int = 30
    ) -> list[dict[str, Any]]:
        from mcp_server.tools.news_tools import get_news_articles
        return get_news_articles(query, ticker, days)

    # ------------------------------------------------------------------ #
    # Vector Store
    # ------------------------------------------------------------------ #
    def embed_and_store(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]],
        collection: str,
        ticker: str,
    ) -> int:
        from mcp_server.tools.vector_tools import embed_and_store
        return embed_and_store(texts, metadatas, collection, ticker)

    def semantic_search(
        self,
        query: str,
        collection: str,
        ticker: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        from mcp_server.tools.vector_tools import semantic_search
        return semantic_search(query, collection, ticker, top_k, where)

    def collection_exists(self, collection: str, ticker: str) -> bool:
        from mcp_server.tools.vector_tools import check_collection_exists
        return check_collection_exists(collection, ticker)

    def get_collection_count(self, collection: str, ticker: str) -> int:
        from mcp_server.tools.vector_tools import get_collection_document_count
        return get_collection_document_count(collection, ticker)

    # ------------------------------------------------------------------ #
    # PDF
    # ------------------------------------------------------------------ #
    def generate_pdf_report(
        self, report_data: dict[str, Any], run_id: str, ticker: str
    ) -> dict[str, str]:
        from mcp_server.tools.pdf_tools import generate_pdf_report
        return generate_pdf_report(report_data, run_id, ticker)

    def get_pdf_path(self, run_id: str) -> str:
        from mcp_server.tools.pdf_tools import get_pdf_path
        return get_pdf_path(run_id)


# Singleton client instance
_client: MCPClient | None = None


def get_mcp_client() -> MCPClient:
    """Return singleton MCPClient instance."""
    global _client
    if _client is None:
        _client = MCPClient()
    return _client
