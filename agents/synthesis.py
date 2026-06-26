"""
FinSight AI — Synthesis Agent Node
Generates the final FinSightReport by combining all specialist outputs + RAG context.
Saves to DB and triggers PDF generation.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import google.generativeai as genai

from agents._llm import get_cached_or_call
from agents.state import FinSightState
from mcp_server.client import get_mcp_client
from prompts.synthesis import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)

# Gemini response schema for synthesis
_OVERALL_SIGNAL_SCHEMA = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "signal": genai.protos.Schema(type=genai.protos.Type.STRING),
        "confidence": genai.protos.Schema(type=genai.protos.Type.STRING),
        "rationale": genai.protos.Schema(type=genai.protos.Type.STRING),
    },
    required=["signal", "confidence", "rationale"],
)

_SYNTHESIS_SCHEMA = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "overall_signal": _OVERALL_SIGNAL_SCHEMA,
        "executive_summary": genai.protos.Schema(type=genai.protos.Type.STRING),
        "investment_thesis": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(type=genai.protos.Type.STRING),
        ),
        "key_risks": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(type=genai.protos.Type.STRING),
        ),
        "conclusion": genai.protos.Schema(type=genai.protos.Type.STRING),
    },
    required=[
        "overall_signal", "executive_summary",
        "investment_thesis", "key_risks", "conclusion",
    ],
)


def synthesis_node(state: FinSightState) -> FinSightState:
    """
    Synthesis node:
    1. Retrieve cross-collection context (top 5 chunks)
    2. Build comprehensive prompt with all agent outputs
    3. Call Gemini for final synthesis
    4. Assemble FinSightReport
    5. Save to DB + trigger PDF generation
    """
    ticker = state.get("ticker", "")
    company_name = state.get("company_name", ticker)
    run_id = state.get("run_id", "")
    errors: list[str] = []

    logger.info("[synthesis] Building final report for %s run_id=%s", ticker, run_id)

    financial_result = state.get("financial_result", {})
    risk_result = state.get("risk_result", {})
    sentiment_result = state.get("sentiment_result", {})

    try:
        client = get_mcp_client()

        # 1. Cross-collection retrieval
        context_chunks: list[dict[str, Any]] = []
        for col in ["sec_filings", "news"]:
            try:
                chunks = client.semantic_search(
                    query=f"{company_name} business outlook future strategy",
                    collection=col,
                    ticker=ticker,
                    top_k=3,
                )
                context_chunks.extend(chunks)
            except Exception:
                pass

        # 2. Build prompt
        user_prompt = build_user_prompt(
            ticker=ticker,
            company_name=company_name,
            financial_analysis=financial_result,
            risk_analysis=risk_result,
            sentiment_analysis=sentiment_result,
            context_chunks=context_chunks[:5],
        )

        # 3. Call Gemini
        llm_result = get_cached_or_call(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=_SYNTHESIS_SCHEMA,
        )

        # 4. Assemble sources list
        sources: list[dict[str, Any]] = []
        for col in ["sec_filings", "news"]:
            try:
                chunks = client.semantic_search(
                    query="company overview",
                    collection=col,
                    ticker=ticker,
                    top_k=3,
                )
                for chunk in chunks:
                    meta = chunk.get("metadata", {})
                    sources.append({
                        "type": meta.get("form_type", col),
                        "title": meta.get("title", meta.get("section", "SEC Filing")),
                        "date": meta.get("filing_date", ""),
                        "url": meta.get("document_url", ""),
                    })
            except Exception:
                pass

        # 5. Build full FinSightReport
        overall_signal = llm_result.get("overall_signal", {})
        report: dict[str, Any] = {
            "metadata": {
                "ticker": ticker,
                "company_name": company_name,
                "run_id": run_id,
                "generated_at": datetime.utcnow().isoformat(),
                "data_sources": ["SEC EDGAR", "yfinance", "Google News RSS"],
            },
            "overall_signal": {
                "signal": overall_signal.get("signal", "NEUTRAL"),
                "confidence": overall_signal.get("confidence", "LOW"),
                "rationale": overall_signal.get("rationale", ""),
            },
            "executive_summary": llm_result.get("executive_summary", ""),
            "investment_thesis": llm_result.get("investment_thesis", [])[:3],
            "key_risks": llm_result.get("key_risks", [])[:3],
            "financial_analysis": {
                "health_score": financial_result.get("health_score", 5.0),
                "health_rationale": financial_result.get("health_rationale", ""),
                "ratios": financial_result.get("ratios", {}),
                "price_performance": financial_result.get("price_performance", {}),
                "strengths": financial_result.get("strengths", []),
                "weaknesses": financial_result.get("weaknesses", []),
            },
            "risk_analysis": {
                "overall_risk_score": risk_result.get("overall_risk_score", 5.0),
                "risk_factors": risk_result.get("risk_factors", []),
            },
            "sentiment_analysis": {
                "aggregate_score": sentiment_result.get("aggregate_score", 0.0),
                "trend": sentiment_result.get("trend", "STABLE"),
                "article_count": sentiment_result.get("article_count", 0),
                "positive_count": sentiment_result.get("positive_count", 0),
                "negative_count": sentiment_result.get("negative_count", 0),
                "neutral_count": sentiment_result.get("neutral_count", 0),
                "top_events": sentiment_result.get("top_events", []),
                "earnings_tone": sentiment_result.get("earnings_tone", "N/A"),
            },
            "sources": sources[:10],
            "conclusion": llm_result.get("conclusion", ""),
        }

        # 6. Save to DB
        _save_report_to_db(run_id=run_id, ticker=ticker, report=report)

        # 7. Generate PDF
        try:
            pdf_result = client.generate_pdf_report(
                report_data=report, run_id=run_id, ticker=ticker
            )
            report["pdf_path"] = pdf_result.get("pdf_path", "")
        except Exception as exc:
            logger.warning("PDF generation failed: %s", exc)
            errors.append(f"PDF generation failed: {exc}")

        logger.info(
            "[synthesis] Complete: signal=%s confidence=%s for %s",
            report["overall_signal"]["signal"],
            report["overall_signal"]["confidence"],
            ticker,
        )

        return {
            **state,
            "synthesis_result": report,
            "status": "COMPLETE",
            "errors": errors,
        }

    except Exception as exc:
        msg = f"Synthesis agent error: {exc}"
        logger.error(msg)
        errors.append(msg)
        return {
            **state,
            "synthesis_result": {},
            "status": "FAILED",
            "errors": errors,
        }


def _save_report_to_db(run_id: str, ticker: str, report: dict[str, Any]) -> None:
    """Persist report to the SQLite database."""
    try:
        from db.database import get_sync_db
        from db.models import Report, Run
        from datetime import datetime

        overall_signal = report.get("overall_signal", {})
        financial = report.get("financial_analysis", {})
        risk = report.get("risk_analysis", {})
        sentiment = report.get("sentiment_analysis", {})

        with get_sync_db() as db:
            # Update run status
            run = db.query(Run).filter(Run.id == run_id).first()
            if run:
                run.status = "COMPLETE"
                run.completed_at = datetime.utcnow()

            # Create report record
            existing = db.query(Report).filter(Report.run_id == run_id).first()
            if not existing:
                db_report = Report(
                    run_id=run_id,
                    ticker=ticker,
                    report_json=json.dumps(report),
                    overall_signal=overall_signal.get("signal", "NEUTRAL"),
                    signal_confidence=overall_signal.get("confidence", "LOW"),
                    financial_score=float(financial.get("health_score", 5.0)),
                    risk_score=float(risk.get("overall_risk_score", 5.0)),
                    sentiment_score=float(sentiment.get("aggregate_score", 0.0)),
                )
                db.add(db_report)
                db.flush()

    except Exception as exc:
        logger.error("Failed to save report to DB: %s", exc)
