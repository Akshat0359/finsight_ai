"""
FinSight AI — Financial Analysis Agent Node
Computes ratios analytically, retrieves MD&A context, calls Gemini for interpretation.
"""
from __future__ import annotations

import logging
from typing import Any

import google.generativeai as genai

from agents._llm import get_cached_or_call
from agents.state import FinSightState
from mcp_server.client import get_mcp_client
from prompts.financial import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)

# Gemini response schema for financial analysis
_FINANCIAL_SCHEMA = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "health_score": genai.protos.Schema(type=genai.protos.Type.NUMBER),
        "health_rationale": genai.protos.Schema(type=genai.protos.Type.STRING),
        "strengths": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(type=genai.protos.Type.STRING),
        ),
        "weaknesses": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(type=genai.protos.Type.STRING),
        ),
    },
    required=["health_score", "health_rationale", "strengths", "weaknesses"],
)


def financial_node(state: FinSightState) -> FinSightState:
    """
    Financial analysis node:
    1. Compute ratios via MCP
    2. Retrieve MD&A chunks via RAG
    3. Call Gemini for LLM interpretation
    4. Merge analytical ratios into result
    """
    ticker = state.get("ticker", "")
    company_name = state.get("company_name", ticker)
    errors: list[str] = []

    logger.info("[financial] Starting financial analysis for %s", ticker)

    try:
        client = get_mcp_client()

        # 1. Compute financial ratios (analytical, no LLM)
        ratios = client.compute_financial_ratios(ticker)

        # 2. Retrieve MD&A context from ChromaDB
        mda_chunks = client.semantic_search(
            query="revenue growth earnings profitability management discussion analysis",
            collection="sec_filings",
            ticker=ticker,
            top_k=5,
            where={"section": {"$in": ["ITEM_7_MDA", "ITEM_7A_QUANT", "ITEM_8_FINANCIALS"]}},
        )

        # Fallback: search without section filter if no results
        if not mda_chunks:
            mda_chunks = client.semantic_search(
                query="revenue growth earnings profitability",
                collection="sec_filings",
                ticker=ticker,
                top_k=5,
            )

        # 3. Build prompt and call Gemini
        user_prompt = build_user_prompt(
            ticker=ticker,
            company_name=company_name,
            ratios=ratios,
            context_chunks=mda_chunks,
        )

        llm_result = get_cached_or_call(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=_FINANCIAL_SCHEMA,
        )

        # 4. Extract price performance separately
        price_perf = {
            "return_1m": ratios.get("return_1m"),
            "return_3m": ratios.get("return_3m"),
            "return_6m": ratios.get("return_6m"),
            "return_1y": ratios.get("return_1y"),
            "return_ytd": ratios.get("return_ytd"),
            "volatility_30d": ratios.get("volatility_30d"),
        }

        # 5. Build FinancialAnalysis-compatible result
        financial_result: dict[str, Any] = {
            "health_score": float(llm_result.get("health_score", 5.0)),
            "health_rationale": llm_result.get("health_rationale", ""),
            "ratios": {
                "pe_ratio": ratios.get("pe_ratio"),
                "pb_ratio": ratios.get("pb_ratio"),
                "ev_ebitda": ratios.get("ev_ebitda"),
                "debt_to_equity": ratios.get("debt_to_equity"),
                "current_ratio": ratios.get("current_ratio"),
                "roe": ratios.get("roe"),
                "roa": ratios.get("roa"),
                "gross_margin": ratios.get("gross_margin"),
                "operating_margin": ratios.get("operating_margin"),
                "free_cash_flow_ttm": ratios.get("free_cash_flow_ttm"),
                "revenue_cagr_3y": ratios.get("revenue_cagr_3y"),
            },
            "price_performance": price_perf,
            "strengths": llm_result.get("strengths", []),
            "weaknesses": llm_result.get("weaknesses", []),
        }

        logger.info(
            "[financial] Complete: health_score=%.1f for %s",
            financial_result["health_score"], ticker,
        )

        return {**state, "financial_result": financial_result, "errors": errors}

    except Exception as exc:
        msg = f"Financial agent error: {exc}"
        logger.error(msg)
        errors.append(msg)

        # Return partial result so pipeline continues
        return {
            **state,
            "financial_result": {
                "health_score": 5.0,
                "health_rationale": "Analysis unavailable due to error.",
                "ratios": {},
                "price_performance": {},
                "strengths": [],
                "weaknesses": [msg],
            },
            "errors": errors,
        }
