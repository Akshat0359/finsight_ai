"""
FinSight AI — Risk Analysis Agent Node
RAG retrieval of risk factors from Item 1A + Gemini extraction and scoring.
"""
from __future__ import annotations

import logging
from typing import Any

import google.generativeai as genai

from agents._llm import get_cached_or_call
from agents.state import FinSightState
from mcp_server.client import get_mcp_client
from prompts.risk import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)

# Gemini response schema for risk analysis
_RISK_FACTOR_SCHEMA = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "category": genai.protos.Schema(type=genai.protos.Type.STRING),
        "description": genai.protos.Schema(type=genai.protos.Type.STRING),
        "severity": genai.protos.Schema(type=genai.protos.Type.INTEGER),
        "likelihood": genai.protos.Schema(type=genai.protos.Type.INTEGER),
        "is_new": genai.protos.Schema(type=genai.protos.Type.BOOLEAN),
        "source_filing": genai.protos.Schema(type=genai.protos.Type.STRING),
    },
    required=["category", "description", "severity", "likelihood", "is_new", "source_filing"],
)

_RISK_SCHEMA = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "overall_risk_score": genai.protos.Schema(type=genai.protos.Type.NUMBER),
        "risk_factors": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=_RISK_FACTOR_SCHEMA,
        ),
    },
    required=["overall_risk_score", "risk_factors"],
)


def risk_node(state: FinSightState) -> FinSightState:
    """
    Risk analysis node:
    1. Retrieve Item 1A (Risk Factors) chunks from ChromaDB
    2. Call Gemini to extract and score risk factors
    3. Return structured RiskAnalysis
    """
    ticker = state.get("ticker", "")
    company_name = state.get("company_name", ticker)
    errors: list[str] = []

    logger.info("[risk] Starting risk analysis for %s", ticker)

    try:
        client = get_mcp_client()

        # 1. Retrieve risk factor chunks (ITEM_1A section specifically)
        risk_chunks = client.semantic_search(
            query="risk factors regulatory competitive market operational financial legal",
            collection="sec_filings",
            ticker=ticker,
            top_k=8,
            where={"section": {"$eq": "ITEM_1A_RISK_FACTORS"}},
        )

        # Fallback: search without section filter
        if not risk_chunks:
            risk_chunks = client.semantic_search(
                query="risk factors regulatory competitive market risk management",
                collection="sec_filings",
                ticker=ticker,
                top_k=8,
            )

        # 2. Build prompt and call Gemini
        user_prompt = build_user_prompt(
            ticker=ticker,
            company_name=company_name,
            risk_chunks=risk_chunks,
        )

        llm_result = get_cached_or_call(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=_RISK_SCHEMA,
        )

        # 3. Validate and normalize scores
        risk_factors = llm_result.get("risk_factors", [])
        for rf in risk_factors:
            rf["severity"] = max(1, min(5, int(rf.get("severity", 3))))
            rf["likelihood"] = max(1, min(5, int(rf.get("likelihood", 3))))

        overall_score = float(llm_result.get("overall_risk_score", 5.0))
        overall_score = max(1.0, min(10.0, overall_score))

        risk_result: dict[str, Any] = {
            "overall_risk_score": overall_score,
            "risk_factors": risk_factors,
        }

        logger.info(
            "[risk] Complete: risk_score=%.1f, %d factors for %s",
            overall_score, len(risk_factors), ticker,
        )

        return {**state, "risk_result": risk_result, "errors": errors}

    except Exception as exc:
        msg = f"Risk agent error: {exc}"
        logger.error(msg)
        errors.append(msg)

        return {
            **state,
            "risk_result": {
                "overall_risk_score": 5.0,
                "risk_factors": [],
            },
            "errors": errors,
        }
