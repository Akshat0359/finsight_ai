"""
FinSight AI — PDF Generation MCP Tool
Triggers PDF rendering via the delivery module.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def generate_pdf_report(
    report_data: dict[str, Any],
    run_id: str,
    ticker: str,
) -> dict[str, str]:
    """
    Generate a PDF report for a completed analysis run.

    Args:
        report_data: The full FinSightReport dict
        run_id: The analysis run ID (used as filename)
        ticker: Company ticker symbol

    Returns:
        {pdf_path: str, status: str}
    """
    try:
        from delivery.pdf_renderer import render_pdf
        pdf_path = render_pdf(report_data=report_data, run_id=run_id, ticker=ticker)
        logger.info("PDF generated at %s for run %s", pdf_path, run_id)
        return {"pdf_path": pdf_path, "status": "success"}
    except Exception as exc:
        logger.error("PDF generation failed for run %s: %s", run_id, exc)
        return {"pdf_path": "", "status": "error", "error": str(exc)}


def get_pdf_path(run_id: str) -> str:
    """
    Return the expected PDF file path for a given run_id.
    Does not check if file exists.
    """
    from app.config import get_settings
    settings = get_settings()
    return f"{settings.PDF_OUTPUT_DIR}/{run_id}.pdf"
