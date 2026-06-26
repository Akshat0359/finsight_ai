"""
FinSight AI — Test Fixtures (conftest.py)
Shared fixtures for all test modules.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test environment variables BEFORE any app imports
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("CHROMA_PERSIST_DIR", "/tmp/test_chroma")
os.environ.setdefault("CACHE_DIR", "/tmp/test_cache")
os.environ.setdefault("PDF_OUTPUT_DIR", "/tmp/test_reports")

from db.database import Base


# ---- In-memory SQLite test engine ----
@pytest.fixture(scope="session")
def test_engine():
    """Create an in-memory SQLite engine for tests."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture()
def test_db(test_engine) -> Generator[Session, None, None]:
    """Provide a transactional test DB session that rolls back after each test."""
    TestSession = sessionmaker(bind=test_engine)
    session = TestSession()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


# ---- Mock Gemini client ----
@pytest.fixture()
def mock_gemini():
    """Mock the Gemini generativeai SDK to avoid real API calls."""
    with patch("google.generativeai.configure"), \
         patch("google.generativeai.GenerativeModel") as mock_model_cls, \
         patch("google.generativeai.embed_content") as mock_embed:

        # Mock embed_content
        mock_embed.return_value = {"embedding": [[0.1] * 768]}

        # Mock GenerativeModel.generate_content
        mock_response = MagicMock()
        mock_response.text = '{"health_score": 7.5, "health_rationale": "Test", "strengths": ["s1"], "weaknesses": ["w1"]}'
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_model_cls.return_value = mock_model

        yield {"model": mock_model, "embed": mock_embed}


# ---- Mock MCP client ----
@pytest.fixture()
def mock_mcp_client():
    """Mock MCPClient to avoid real network calls."""
    mock = MagicMock()

    mock.search_company_cik.return_value = {
        "ticker": "MSFT",
        "name": "Microsoft Corporation",
        "cik": "0000789019",
    }

    mock.get_recent_filings.return_value = [
        {
            "form_type": "10-K",
            "filing_date": "2024-07-30",
            "accession_number": "0000789019-24-000088",
            "period_of_report": "2024-06-30",
        }
    ]

    mock.download_filing_text.return_value = (
        "ITEM 1. BUSINESS\n\nMicrosoft Corporation is a technology company.\n\n"
        "ITEM 1A. RISK FACTORS\n\nThe company faces regulatory risks in various jurisdictions.\n\n"
        "ITEM 7. MANAGEMENT DISCUSSION AND ANALYSIS\n\nRevenue grew 16% year over year."
    )

    mock.get_price_history.return_value = {
        "ticker": "MSFT",
        "period": "1y",
        "data": [{"date": "2024-01-01", "close": 370.0}],
    }

    mock.get_financial_statements.return_value = {
        "ticker": "MSFT",
        "income_statement": {},
        "balance_sheet": {},
        "cash_flow": {},
    }

    mock.get_company_info.return_value = {
        "ticker": "MSFT",
        "longName": "Microsoft Corporation",
        "sector": "Technology",
        "industry": "Software—Infrastructure",
        "marketCap": 3_000_000_000_000,
    }

    mock.compute_financial_ratios.return_value = {
        "pe_ratio": 35.2,
        "pb_ratio": 12.1,
        "ev_ebitda": 25.0,
        "debt_to_equity": 0.31,
        "current_ratio": 1.77,
        "roe": 38.5,
        "roa": 17.8,
        "gross_margin": 70.1,
        "operating_margin": 44.6,
        "free_cash_flow_ttm": 70_000_000_000,
        "revenue_cagr_3y": 14.2,
        "return_1m": 3.2,
        "return_3m": 8.5,
        "return_6m": 15.1,
        "return_1y": 28.3,
        "return_ytd": 12.4,
        "volatility_30d": 18.5,
    }

    mock.get_news_articles.return_value = [
        {
            "title": "Microsoft beats earnings estimates",
            "summary": "Microsoft reported strong quarterly results.",
            "date": "2024-10-30",
            "source": "Reuters",
            "url": "https://example.com/1",
        },
        {
            "title": "Azure cloud growth accelerates",
            "summary": "Azure revenue grew 33% year over year.",
            "date": "2024-10-28",
            "source": "Bloomberg",
            "url": "https://example.com/2",
        },
    ]

    mock.embed_and_store.return_value = 5
    mock.semantic_search.return_value = [
        {
            "text": "Revenue grew significantly driven by cloud adoption.",
            "metadata": {"section": "ITEM_7_MDA", "form_type": "10-K"},
            "score": 0.92,
        }
    ]
    mock.collection_exists.return_value = True
    mock.get_collection_count.return_value = 42
    mock.generate_pdf_report.return_value = {"pdf_path": "/tmp/test.pdf", "status": "success"}

    return mock


# ---- Sample report data fixture ----
@pytest.fixture()
def sample_report() -> dict:
    """A minimal valid FinSightReport dict for testing."""
    return {
        "metadata": {
            "ticker": "MSFT",
            "company_name": "Microsoft Corporation",
            "run_id": "test-run-id-001",
            "generated_at": "2024-11-01T12:00:00",
            "data_sources": ["SEC EDGAR", "yfinance"],
        },
        "overall_signal": {
            "signal": "BULLISH",
            "confidence": "HIGH",
            "rationale": "Strong fundamentals and growing cloud business.",
        },
        "executive_summary": "Microsoft shows excellent financial health.",
        "investment_thesis": ["Cloud growth", "AI monetization", "Strong FCF"],
        "key_risks": ["Regulatory pressure", "Competition", "Macro slowdown"],
        "financial_analysis": {
            "health_score": 8.5,
            "health_rationale": "Strong balance sheet and margins.",
            "ratios": {
                "pe_ratio": 35.2,
                "pb_ratio": 12.1,
                "roe": 38.5,
                "gross_margin": 70.1,
            },
            "price_performance": {
                "return_1m": 3.2,
                "return_1y": 28.3,
                "volatility_30d": 18.5,
            },
            "strengths": ["High margins", "Cloud leader"],
            "weaknesses": ["High valuation", "Regulatory risk"],
        },
        "risk_analysis": {
            "overall_risk_score": 4.2,
            "risk_factors": [
                {
                    "category": "REGULATORY",
                    "description": "Antitrust scrutiny from EU.",
                    "severity": 3,
                    "likelihood": 3,
                    "is_new": False,
                    "source_filing": "10-K",
                }
            ],
        },
        "sentiment_analysis": {
            "aggregate_score": 0.45,
            "trend": "IMPROVING",
            "article_count": 25,
            "positive_count": 18,
            "negative_count": 3,
            "neutral_count": 4,
            "top_events": [],
            "earnings_tone": "POSITIVE",
        },
        "sources": [],
        "conclusion": "Microsoft remains a top-tier technology investment.",
    }
