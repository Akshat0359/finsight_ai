"""
FinSight AI — LangGraph State Definition
TypedDict state shared across all graph nodes.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class FinSightState(TypedDict, total=False):
    """
    Shared state passed through the LangGraph StateGraph.
    All fields are optional (total=False) to allow partial updates per node.
    """
    # ---- Input ----
    ticker: str
    company_name: str
    cik: str
    run_id: str
    force_refresh: bool

    # ---- Ingestor outputs ----
    ingestor_result: dict[str, Any]

    # ---- Specialist agent outputs ----
    financial_result: dict[str, Any]
    risk_result: dict[str, Any]
    sentiment_result: dict[str, Any]

    # ---- Synthesis output ----
    synthesis_result: dict[str, Any]

    # ---- Error accumulation (uses operator.add to merge across parallel nodes) ----
    errors: Annotated[list[str], operator.add]

    # ---- Pipeline metadata ----
    status: str        # PENDING | RUNNING | COMPLETE | FAILED
    cache_used: bool
    total_tokens: int
    news_articles: list[dict[str, Any]]
