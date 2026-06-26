"""
FinSight AI — Reports Router
GET /api/v1/reports/{run_id}     — get full JSON report
GET /api/v1/reports/{run_id}/pdf — download PDF report
"""
from __future__ import annotations

import json
import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db
from db.models import Report, Run

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/reports", tags=["Reports"])


@router.get(
    "/{run_id}",
    summary="Get full analysis report JSON",
)
async def get_report(
    run_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Retrieve the full FinSightReport JSON for a completed run.
    Returns 404 if run doesn't exist or 202 if analysis is still running.
    """
    # Check run exists
    run_result = await db.execute(select(Run).where(Run.id == run_id))
    run = run_result.scalar_one_or_none()

    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    if run.status in ("PENDING", "RUNNING"):
        raise HTTPException(
            status_code=202,
            detail=f"Analysis is still {run.status}. Try again later.",
        )

    if run.status == "FAILED":
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {run.error_message or 'Unknown error'}",
        )

    # Fetch report
    report_result = await db.execute(
        select(Report).where(Report.run_id == run_id)
    )
    report = report_result.scalar_one_or_none()

    if not report:
        raise HTTPException(status_code=404, detail="Report not found for this run")

    try:
        return json.loads(report.report_json)
    except json.JSONDecodeError as exc:
        logger.error("Invalid report JSON for run %s: %s", run_id, exc)
        raise HTTPException(status_code=500, detail="Report data is corrupted")


@router.get(
    "/{run_id}/pdf",
    summary="Download PDF report",
    response_class=FileResponse,
)
async def get_report_pdf(
    run_id: str,
    db: AsyncSession = Depends(get_db),
) -> FileResponse:
    """Download the PDF report for a completed run."""
    # Check run exists and is complete
    run_result = await db.execute(select(Run).where(Run.id == run_id))
    run = run_result.scalar_one_or_none()

    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    if run.status != "COMPLETE":
        raise HTTPException(
            status_code=202,
            detail=f"Analysis is {run.status}. PDF not available yet.",
        )

    # Fetch report for PDF path
    report_result = await db.execute(
        select(Report).where(Report.run_id == run_id)
    )
    report = report_result.scalar_one_or_none()

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    # Check PDF path
    pdf_path = report.pdf_path
    if not pdf_path:
        from app.config import get_settings
        settings = get_settings()
        pdf_path = f"{settings.PDF_OUTPUT_DIR}/{run_id}.pdf"

    if not os.path.exists(pdf_path):
        # Try to generate on the fly
        try:
            report_data = json.loads(report.report_json)
            from delivery.pdf_renderer import render_pdf
            pdf_path = render_pdf(
                report_data=report_data,
                run_id=run_id,
                ticker=report.ticker,
            )
        except Exception as exc:
            logger.error("PDF generation failed: %s", exc)
            raise HTTPException(
                status_code=500,
                detail="PDF generation failed. Try /api/v1/reports/{run_id} for JSON.",
            )

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=f"finsight_{report.ticker}_{run_id[:8]}.pdf",
    )


@router.get(
    "/ticker/{ticker}",
    summary="List all reports for a ticker",
)
async def list_reports_for_ticker(
    ticker: str,
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """List all completed reports for a ticker, newest first."""
    result = await db.execute(
        select(Report)
        .where(Report.ticker == ticker.upper())
        .order_by(Report.created_at.desc())
        .limit(10)
    )
    reports = result.scalars().all()

    return [
        {
            "report_id": r.id,
            "run_id": r.run_id,
            "ticker": r.ticker,
            "overall_signal": r.overall_signal,
            "signal_confidence": r.signal_confidence,
            "financial_score": r.financial_score,
            "risk_score": r.risk_score,
            "sentiment_score": r.sentiment_score,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in reports
    ]
