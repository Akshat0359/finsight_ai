"""
FinSight AI — Orchestrator Agent Node
Resolves ticker to CIK, validates company info, and sets up state for ingestion.
"""
from __future__ import annotations

import logging

from agents.state import FinSightState
from mcp_server.client import get_mcp_client

logger = logging.getLogger(__name__)


def orchestrator_node(state: FinSightState) -> FinSightState:
    """
    Entry node: resolve CIK, validate ticker, prepare state.
    """
    ticker = state.get("ticker", "").upper().strip()
    run_id = state.get("run_id", "")
    errors: list[str] = []

    logger.info("[orchestrator] Starting analysis for ticker=%s run_id=%s", ticker, run_id)

    try:
        client = get_mcp_client()

        # Resolve company info from EDGAR
        company_info = client.search_company_cik(ticker)
        cik = company_info.get("cik", "")
        company_name = company_info.get("name", ticker)

        # Fallback: use ticker as company name if EDGAR lookup fails
        if not company_name or company_name == ticker:
            try:
                market_info = client.get_company_info(ticker)
                company_name = market_info.get("longName") or market_info.get("shortName") or ticker
            except Exception as exc:
                logger.warning("Could not get company name from market data: %s", exc)
                company_name = ticker

        logger.info(
            "[orchestrator] Resolved: ticker=%s name='%s' cik=%s",
            ticker, company_name, cik,
        )

        return {
            **state,
            "ticker": ticker,
            "company_name": company_name,
            "cik": cik,
            "status": "RUNNING",
            "errors": errors,
            "cache_used": False,
            "total_tokens": 0,
        }

    except Exception as exc:
        msg = f"Orchestrator error: {exc}"
        logger.error(msg)
        errors.append(msg)
        return {
            **state,
            "ticker": ticker,
            "company_name": ticker,
            "cik": "",
            "status": "RUNNING",
            "errors": errors,
        }
