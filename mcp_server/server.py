"""
FinSight AI — FastMCP Server
Exposes all tools to LangGraph agents via MCP protocol.
"""
from __future__ import annotations

import logging
from typing import Any

from fastmcp import FastMCP

from mcp_server.tools.edgar_tools import (
    download_filing_text,
    get_company_facts,
    get_recent_filings,
    search_company_cik,
)
from mcp_server.tools.market_tools import (
    compute_financial_ratios,
    get_company_info,
    get_financial_statements,
    get_price_history,
)
from mcp_server.tools.news_tools import get_news_articles
from mcp_server.tools.pdf_tools import generate_pdf_report, get_pdf_path
from mcp_server.tools.vector_tools import (
    check_collection_exists,
    embed_and_store,
    get_collection_document_count,
    semantic_search,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="FinSight AI Tools",
    instructions=(
        "Financial intelligence tools for SEC EDGAR, market data, "
        "news retrieval, vector storage, and PDF generation."
    ),
)


# ------------------------------------------------------------------ #
# EDGAR Tools
# ------------------------------------------------------------------ #
@mcp.tool()
def tool_search_company_cik(company_name: str) -> dict[str, str]:
    """Search SEC EDGAR for a company by name or ticker. Returns ticker, name, CIK."""
    return search_company_cik(company_name)


@mcp.tool()
def tool_get_recent_filings(
    cik: str,
    form_types: list[str],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Get recent SEC filings for a company. form_types: ['10-K', '10-Q', '8-K']"""
    return get_recent_filings(cik, form_types, limit)


@mcp.tool()
def tool_download_filing_text(cik: str, accession_number: str) -> str:
    """Download and extract text from an SEC filing. Returns cleaned plain text."""
    return download_filing_text(cik, accession_number)


@mcp.tool()
def tool_get_company_facts(cik: str) -> dict[str, Any]:
    """Fetch XBRL structured financial facts from SEC EDGAR."""
    return get_company_facts(cik)


# ------------------------------------------------------------------ #
# Market Data Tools
# ------------------------------------------------------------------ #
@mcp.tool()
def tool_get_price_history(ticker: str, period: str = "1y") -> dict[str, Any]:
    """Fetch OHLCV price history. period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y."""
    return get_price_history(ticker, period)


@mcp.tool()
def tool_get_financial_statements(ticker: str) -> dict[str, Any]:
    """Fetch income statement, balance sheet, and cash flow statement."""
    return get_financial_statements(ticker)


@mcp.tool()
def tool_get_company_info(ticker: str) -> dict[str, Any]:
    """Fetch company profile: sector, industry, description, market cap."""
    return get_company_info(ticker)


@mcp.tool()
def tool_compute_financial_ratios(ticker: str) -> dict[str, Any]:
    """Compute all financial ratios: P/E, P/B, EV/EBITDA, ROE, ROA, margins, FCF, returns."""
    return compute_financial_ratios(ticker)


# ------------------------------------------------------------------ #
# News Tools
# ------------------------------------------------------------------ #
@mcp.tool()
def tool_get_news_articles(
    query: str, ticker: str, days: int = 30
) -> list[dict[str, Any]]:
    """Fetch news articles from Google News RSS and Yahoo Finance RSS."""
    return get_news_articles(query, ticker, days)


# ------------------------------------------------------------------ #
# Vector Store Tools
# ------------------------------------------------------------------ #
@mcp.tool()
def tool_embed_and_store(
    texts: list[str],
    metadatas: list[dict[str, Any]],
    collection: str,
    ticker: str,
) -> int:
    """Embed texts with Gemini and store in ChromaDB. Returns count stored."""
    return embed_and_store(texts, metadatas, collection, ticker)


@mcp.tool()
def tool_semantic_search(
    query: str,
    collection: str,
    ticker: str,
    top_k: int = 5,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Semantic search in ChromaDB. Returns list of {text, metadata, score}."""
    return semantic_search(query, collection, ticker, top_k, where)


@mcp.tool()
def tool_collection_exists(collection: str, ticker: str) -> bool:
    """Check if a ChromaDB collection exists and has documents."""
    return check_collection_exists(collection, ticker)


@mcp.tool()
def tool_get_collection_count(collection: str, ticker: str) -> int:
    """Return document count in a ChromaDB collection."""
    return get_collection_document_count(collection, ticker)


# ------------------------------------------------------------------ #
# PDF Tools
# ------------------------------------------------------------------ #
@mcp.tool()
def tool_generate_pdf_report(
    report_data: dict[str, Any], run_id: str, ticker: str
) -> dict[str, str]:
    """Generate a PDF report for a completed analysis run."""
    return generate_pdf_report(report_data, run_id, ticker)


@mcp.tool()
def tool_get_pdf_path(run_id: str) -> str:
    """Return the expected PDF file path for a run_id."""
    return get_pdf_path(run_id)


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    logger.info("Starting FinSight AI MCP server on port %d", port)
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
