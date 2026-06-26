"""
FinSight AI — Orchestrator Prompts
"""
from __future__ import annotations

SYSTEM_PROMPT = """You are FinSight AI's orchestration agent. Your role is to validate company information and plan the analysis pipeline. Always respond with structured JSON."""


def build_user_prompt(company_name: str, ticker: str, cik: str) -> str:
    return f"""Validate and confirm the following company details for analysis:
Company Name: {company_name}
Ticker: {ticker}
CIK: {cik}

Confirm the details are correct and specify which SEC filing types to fetch: 10-K (annual), 10-Q (quarterly), 8-K (material events).
Output JSON with: confirmed_ticker, confirmed_name, cik, filing_types_to_fetch (list), analysis_scope."""
