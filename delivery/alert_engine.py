"""
FinSight AI — Alert Engine with APScheduler
Monitors for new filings and sentiment spikes. Triggers alerts based on configs.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_scheduler: BackgroundScheduler | None = None


def check_new_filings(ticker: str, cik: str) -> list[str]:
    """
    Check if there are new filings since the last ingested filing for ticker.
    Returns list of new accession numbers found.
    """
    from db.database import get_sync_db
    from db.models import Filing
    from mcp_server.tools.edgar_tools import get_recent_filings

    try:
        # Get most recent ingested accession
        with get_sync_db() as db:
            latest = (
                db.query(Filing)
                .filter(Filing.ticker == ticker)
                .order_by(Filing.ingested_at.desc())
                .first()
            )
            last_date = latest.filing_date if latest else "2020-01-01"

        # Fetch recent filings from EDGAR
        recent = get_recent_filings(cik=cik, form_types=["10-K", "10-Q", "8-K"], limit=5)
        new_accessions = [
            f["accession_number"]
            for f in recent
            if f.get("filing_date", "") > last_date
        ]
        return new_accessions

    except Exception as exc:
        logger.warning("Error checking new filings for %s: %s", ticker, exc)
        return []


def check_sentiment_change(ticker: str, threshold: float = 0.3) -> dict[str, Any]:
    """
    Compare latest sentiment score vs stored baseline.
    Returns {is_spike: bool, direction: str, delta: float}.
    """
    from db.database import get_sync_db
    from db.models import Report

    try:
        with get_sync_db() as db:
            # Get last 2 reports
            reports = (
                db.query(Report)
                .filter(Report.ticker == ticker)
                .order_by(Report.created_at.desc())
                .limit(2)
                .all()
            )

        if len(reports) < 2:
            return {"is_spike": False, "direction": "NONE", "delta": 0.0}

        current_score = reports[0].sentiment_score or 0.0
        baseline_score = reports[1].sentiment_score or 0.0
        delta = current_score - baseline_score

        is_spike = abs(delta) >= threshold
        direction = "POSITIVE" if delta > 0 else "NEGATIVE" if delta < 0 else "NONE"

        return {"is_spike": is_spike, "direction": direction, "delta": round(delta, 4)}

    except Exception as exc:
        logger.warning("Error checking sentiment change for %s: %s", ticker, exc)
        return {"is_spike": False, "direction": "NONE", "delta": 0.0}


def _create_alert_event(
    config_id: int,
    ticker: str,
    message: str,
    data: dict[str, Any] | None = None,
) -> None:
    """Persist an alert event to the database."""
    from db.database import get_sync_db
    from db.models import AlertEvent

    try:
        with get_sync_db() as db:
            event = AlertEvent(
                config_id=config_id,
                ticker=ticker,
                triggered_at=datetime.utcnow(),
                message=message,
                data_json=json.dumps(data) if data else None,
                acknowledged=False,
            )
            db.add(event)
        logger.info("Alert event created: ticker=%s message='%s'", ticker, message)
    except Exception as exc:
        logger.error("Failed to create alert event: %s", exc)


def evaluate_alert_configs() -> None:
    """
    Evaluate all enabled alert configs and create AlertEvents if triggered.
    Called by the scheduler every ALERT_INTERVAL_HOURS.
    """
    from db.database import get_sync_db
    from db.models import AlertConfig, Company

    logger.info("Evaluating alert configs...")

    try:
        with get_sync_db() as db:
            configs = db.query(AlertConfig).filter(AlertConfig.enabled == True).all()  # noqa: E712
            config_data = [
                {
                    "id": c.id,
                    "ticker": c.ticker,
                    "alert_type": c.alert_type,
                    "threshold": c.threshold or 0.3,
                }
                for c in configs
            ]

        for config in config_data:
            config_id = config["id"]
            ticker = config["ticker"]
            alert_type = config["alert_type"]
            threshold = config["threshold"]

            try:
                # Get CIK for this ticker
                with get_sync_db() as db:
                    company = db.query(Company).filter(Company.ticker == ticker).first()
                    cik = company.cik if company else ""

                if alert_type == "NEW_FILING":
                    if cik:
                        new_filings = check_new_filings(ticker, cik)
                        if new_filings:
                            _create_alert_event(
                                config_id=config_id,
                                ticker=ticker,
                                message=f"New SEC filings detected for {ticker}: {', '.join(new_filings[:3])}",
                                data={"new_accessions": new_filings},
                            )

                elif alert_type == "SENTIMENT_SPIKE":
                    result = check_sentiment_change(ticker, threshold=threshold)
                    if result["is_spike"]:
                        _create_alert_event(
                            config_id=config_id,
                            ticker=ticker,
                            message=(
                                f"Sentiment spike detected for {ticker}: "
                                f"{result['direction']} change of {result['delta']:.2f}"
                            ),
                            data=result,
                        )

                elif alert_type == "SIGNAL_CHANGE":
                    # Check if signal changed between last two reports
                    from db.database import get_sync_db
                    from db.models import Report
                    with get_sync_db() as db:
                        reports = (
                            db.query(Report)
                            .filter(Report.ticker == ticker)
                            .order_by(Report.created_at.desc())
                            .limit(2)
                            .all()
                        )
                    if len(reports) >= 2:
                        current_signal = reports[0].overall_signal
                        prev_signal = reports[1].overall_signal
                        if current_signal != prev_signal:
                            _create_alert_event(
                                config_id=config_id,
                                ticker=ticker,
                                message=(
                                    f"Signal changed for {ticker}: "
                                    f"{prev_signal} → {current_signal}"
                                ),
                                data={"from": prev_signal, "to": current_signal},
                            )

            except Exception as exc:
                logger.warning(
                    "Error evaluating alert config %d for %s: %s", config_id, ticker, exc
                )

    except Exception as exc:
        logger.error("Error loading alert configs: %s", exc)

    logger.info("Alert evaluation complete")


def start_scheduler() -> None:
    """Start the APScheduler background scheduler for alert evaluation."""
    global _scheduler

    if _scheduler is not None and _scheduler.running:
        logger.info("Alert scheduler already running")
        return

    _scheduler = BackgroundScheduler(
        job_defaults={"coalesce": True, "max_instances": 1},
    )

    _scheduler.add_job(
        evaluate_alert_configs,
        trigger="interval",
        hours=settings.ALERT_INTERVAL_HOURS,
        id="evaluate_alerts",
        name="Evaluate Alert Configs",
        replace_existing=True,
    )

    _scheduler.start()
    logger.info(
        "Alert scheduler started — checking every %d hours", settings.ALERT_INTERVAL_HOURS
    )


def stop_scheduler() -> None:
    """Stop the background scheduler."""
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Alert scheduler stopped")
