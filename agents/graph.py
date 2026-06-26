"""
FinSight AI — LangGraph StateGraph
Full pipeline: orchestrator → ingestor → [financial, risk, sentiment] (parallel) → synthesis.
"""
from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.financial import financial_node
from agents.ingestor import ingestor_node
from agents.orchestrator import orchestrator_node
from agents.risk import risk_node
from agents.sentiment import sentiment_node
from agents.state import FinSightState
from agents.synthesis import synthesis_node

logger = logging.getLogger(__name__)


def route_after_ingest(state: FinSightState) -> list[Send]:
    """
    Fan-out to parallel specialist agents after ingestor completes.
    Uses LangGraph's Send API for parallel dispatch.
    """
    return [
        Send("financial_analysis", state),
        Send("risk_analysis", state),
        Send("sentiment_news", state),
    ]


def _merge_parallel_results(states: list[FinSightState]) -> FinSightState:
    """
    Merge results from parallel nodes into a single state.
    Called by LangGraph's join after all parallel nodes complete.
    """
    if not states:
        return {}  # type: ignore[return-value]

    # Start with the first state as base
    merged = dict(states[0])

    # Merge specialist results from other states
    for s in states[1:]:
        if "financial_result" in s and s["financial_result"]:
            merged["financial_result"] = s["financial_result"]
        if "risk_result" in s and s["risk_result"]:
            merged["risk_result"] = s["risk_result"]
        if "sentiment_result" in s and s["sentiment_result"]:
            merged["sentiment_result"] = s["sentiment_result"]

        # Accumulate errors
        existing_errors = merged.get("errors", [])
        new_errors = s.get("errors", [])
        merged["errors"] = existing_errors + new_errors

    return merged  # type: ignore[return-value]


def build_graph() -> Any:
    """
    Build and compile the FinSight AI LangGraph StateGraph.

    Graph topology:
    START → orchestrator → ingestor → [fan-out] → financial_analysis
                                                 → risk_analysis
                                                 → sentiment_news
                                     [join] → synthesis → END
    """
    builder = StateGraph(FinSightState)

    # Register nodes
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("ingestor", ingestor_node)
    builder.add_node("financial_analysis", financial_node)
    builder.add_node("risk_analysis", risk_node)
    builder.add_node("sentiment_news", sentiment_node)
    builder.add_node("synthesis", synthesis_node)

    # Sequential edges
    builder.add_edge(START, "orchestrator")
    builder.add_edge("orchestrator", "ingestor")

    # Parallel fan-out after ingestor
    builder.add_conditional_edges(
        "ingestor",
        route_after_ingest,
        ["financial_analysis", "risk_analysis", "sentiment_news"],
    )

    # Join parallel nodes → synthesis
    builder.add_edge("financial_analysis", "synthesis")
    builder.add_edge("risk_analysis", "synthesis")
    builder.add_edge("sentiment_news", "synthesis")

    # Terminal edge
    builder.add_edge("synthesis", END)

    # Compile without checkpointer for simplicity
    graph = builder.compile()
    logger.info("LangGraph StateGraph compiled successfully")
    return graph


# Singleton compiled graph
_graph = None


def get_graph() -> Any:
    """Return singleton compiled graph."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


async def run_analysis(
    ticker: str,
    run_id: str,
    company_name: str = "",
    force_refresh: bool = False,
) -> dict[str, Any]:
    """
    Execute the full analysis pipeline for a ticker.
    This is the main entry point called by the FastAPI background task.
    """
    from db.database import get_sync_db
    from db.models import Run

    graph = get_graph()

    initial_state: FinSightState = {
        "ticker": ticker.upper(),
        "company_name": company_name or ticker.upper(),
        "cik": "",
        "run_id": run_id,
        "force_refresh": force_refresh,
        "errors": [],
        "status": "RUNNING",
        "cache_used": False,
        "total_tokens": 0,
        "news_articles": [],
        "ingestor_result": {},
        "financial_result": {},
        "risk_result": {},
        "sentiment_result": {},
        "synthesis_result": {},
    }

    logger.info("Starting analysis pipeline: ticker=%s run_id=%s", ticker, run_id)

    try:
        # Update run status to RUNNING
        with get_sync_db() as db:
            run = db.query(Run).filter(Run.id == run_id).first()
            if run:
                run.status = "RUNNING"

        final_state = await graph.ainvoke(initial_state)

        logger.info(
            "Analysis complete: ticker=%s run_id=%s status=%s errors=%d",
            ticker, run_id,
            final_state.get("status", "UNKNOWN"),
            len(final_state.get("errors", [])),
        )
        return dict(final_state)

    except Exception as exc:
        logger.error("Graph execution error for %s: %s", ticker, exc)
        # Mark run as failed
        try:
            with get_sync_db() as db:
                run = db.query(Run).filter(Run.id == run_id).first()
                if run:
                    run.status = "FAILED"
                    run.error_message = str(exc)
        except Exception:
            pass
        return {"status": "FAILED", "errors": [str(exc)]}
