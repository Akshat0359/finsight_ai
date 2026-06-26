"""
FinSight AI — Analyze Router
POST /api/v1/analyze — launch analysis run
GET  /api/v1/runs/{run_id} — check run status
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db
from app.schemas.run import AnalyzeRequest, AnalyzeResponse, RunStatus
from db.models import Run

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Analysis"])


async def _run_analysis_background(
    ticker: str,
    run_id: str,
    company_name: str,
    force_refresh: bool,
) -> None:
    """Background task: run the LangGraph pipeline."""
    try:
        from agents.graph import run_analysis
        await run_analysis(
            ticker=ticker,
            run_id=run_id,
            company_name=company_name,
            force_refresh=force_refresh,
        )
    except Exception as exc:
        logger.error("Background analysis failed for run %s: %s", run_id, exc)
        # Update DB with failure status
        from db.database import get_sync_db
        from db.models import Run as RunModel
        with get_sync_db() as db:
            run = db.query(RunModel).filter(RunModel.id == run_id).first()
            if run:
                run.status = "FAILED"
                run.error_message = str(exc)


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=202,
    summary="Launch financial analysis for a ticker",
)
async def analyze(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> AnalyzeResponse:
    """
    Launch a new financial analysis pipeline for the given ticker.

    - Returns immediately with run_id and PENDING status
    - Use GET /api/v1/runs/{run_id} to poll status
    - When COMPLETE, use GET /api/v1/reports/{run_id} for results
    """
    ticker = request.ticker.upper().strip()

    # Check for recent cached run (last 24 hours) unless force_refresh
    if not request.force_refresh:
        from sqlalchemy import and_
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(hours=24)

        result = await db.execute(
            select(Run)
            .where(
                and_(
                    Run.ticker == ticker,
                    Run.status == "COMPLETE",
                    Run.completed_at >= cutoff,
                )
            )
            .order_by(Run.completed_at.desc())
            .limit(1)
        )
        existing_run = result.scalar_one_or_none()

        if existing_run:
            logger.info("Returning cached run %s for %s", existing_run.id, ticker)
            return AnalyzeResponse(
                run_id=existing_run.id,
                status="COMPLETE",
                ticker=ticker,
                message="Returning cached analysis (use force_refresh=true to re-run)",
            )

    # Create new run record
    run_id = str(uuid.uuid4())
    new_run = Run(
        id=run_id,
        ticker=ticker,
        status="PENDING",
        triggered_by="api",
        started_at=datetime.utcnow(),
    )
    db.add(new_run)
    await db.flush()

    # Upsert company record
    from db.models import Company
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert

    company_result = await db.execute(
        select(Company).where(Company.ticker == ticker)
    )
    if not company_result.scalar_one_or_none():
        db.add(Company(ticker=ticker, name=ticker))
        await db.flush()

    # Launch background analysis
    background_tasks.add_task(
        _run_analysis_background,
        ticker=ticker,
        run_id=run_id,
        company_name=ticker,
        force_refresh=request.force_refresh,
    )

    logger.info("Launched analysis run %s for ticker %s", run_id, ticker)
    return AnalyzeResponse(
        run_id=run_id,
        status="PENDING",
        ticker=ticker,
        message=f"Analysis started. Poll GET /api/v1/runs/{run_id} for status.",
    )


@router.get(
    "/runs/{run_id}",
    response_model=RunStatus,
    summary="Get analysis run status",
)
async def get_run_status(
    run_id: str,
    db: AsyncSession = Depends(get_db),
) -> RunStatus:
    """Get the current status of an analysis run by run_id."""
    result = await db.execute(select(Run).where(Run.id == run_id))
    run = result.scalar_one_or_none()

    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return RunStatus.model_validate(run)
