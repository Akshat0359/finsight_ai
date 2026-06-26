"""
FinSight AI — Alerts Router
CRUD for alert configurations and alert event acknowledgement.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db
from app.schemas.alert import (
    AlertAcknowledge,
    AlertConfigCreate,
    AlertConfigResponse,
    AlertEventResponse,
)
from db.models import AlertConfig, AlertEvent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/alerts", tags=["Alerts"])


@router.post(
    "/configs",
    response_model=AlertConfigResponse,
    status_code=201,
    summary="Create alert configuration",
)
async def create_alert_config(
    config: AlertConfigCreate,
    db: AsyncSession = Depends(get_db),
) -> AlertConfigResponse:
    """Create a new alert configuration for a ticker."""
    new_config = AlertConfig(
        ticker=config.ticker.upper(),
        alert_type=config.alert_type,
        threshold=config.threshold,
        delivery_method=config.delivery_method,
        enabled=config.enabled,
    )
    db.add(new_config)
    await db.flush()
    await db.refresh(new_config)
    return AlertConfigResponse.model_validate(new_config)


@router.get(
    "/configs",
    response_model=list[AlertConfigResponse],
    summary="List all alert configurations",
)
async def list_alert_configs(
    ticker: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> list[AlertConfigResponse]:
    """List all alert configurations, optionally filtered by ticker."""
    query = select(AlertConfig).order_by(AlertConfig.created_at.desc())
    if ticker:
        query = query.where(AlertConfig.ticker == ticker.upper())
    result = await db.execute(query)
    configs = result.scalars().all()
    return [AlertConfigResponse.model_validate(c) for c in configs]


@router.get(
    "/configs/{config_id}",
    response_model=AlertConfigResponse,
    summary="Get alert configuration by ID",
)
async def get_alert_config(
    config_id: int,
    db: AsyncSession = Depends(get_db),
) -> AlertConfigResponse:
    result = await db.execute(
        select(AlertConfig).where(AlertConfig.id == config_id)
    )
    config = result.scalar_one_or_none()
    if not config:
        raise HTTPException(status_code=404, detail="Alert config not found")
    return AlertConfigResponse.model_validate(config)


@router.delete(
    "/configs/{config_id}",
    status_code=204,
    summary="Delete alert configuration",
)
async def delete_alert_config(
    config_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    result = await db.execute(
        select(AlertConfig).where(AlertConfig.id == config_id)
    )
    config = result.scalar_one_or_none()
    if not config:
        raise HTTPException(status_code=404, detail="Alert config not found")
    await db.delete(config)


@router.patch(
    "/configs/{config_id}/toggle",
    response_model=AlertConfigResponse,
    summary="Toggle alert enabled/disabled",
)
async def toggle_alert_config(
    config_id: int,
    db: AsyncSession = Depends(get_db),
) -> AlertConfigResponse:
    result = await db.execute(
        select(AlertConfig).where(AlertConfig.id == config_id)
    )
    config = result.scalar_one_or_none()
    if not config:
        raise HTTPException(status_code=404, detail="Alert config not found")
    config.enabled = not config.enabled
    await db.flush()
    await db.refresh(config)
    return AlertConfigResponse.model_validate(config)


@router.get(
    "/events",
    response_model=list[AlertEventResponse],
    summary="List alert events",
)
async def list_alert_events(
    ticker: str | None = None,
    unacknowledged_only: bool = False,
    db: AsyncSession = Depends(get_db),
) -> list[AlertEventResponse]:
    """List alert events, optionally filtered by ticker and acknowledgement status."""
    query = select(AlertEvent).order_by(AlertEvent.triggered_at.desc()).limit(100)
    if ticker:
        query = query.where(AlertEvent.ticker == ticker.upper())
    if unacknowledged_only:
        query = query.where(AlertEvent.acknowledged == False)  # noqa: E712
    result = await db.execute(query)
    events = result.scalars().all()
    return [AlertEventResponse.model_validate(e) for e in events]


@router.patch(
    "/events/{event_id}/acknowledge",
    response_model=AlertEventResponse,
    summary="Acknowledge an alert event",
)
async def acknowledge_event(
    event_id: int,
    body: AlertAcknowledge,
    db: AsyncSession = Depends(get_db),
) -> AlertEventResponse:
    result = await db.execute(
        select(AlertEvent).where(AlertEvent.id == event_id)
    )
    event = result.scalar_one_or_none()
    if not event:
        raise HTTPException(status_code=404, detail="Alert event not found")
    event.acknowledged = body.acknowledged
    await db.flush()
    await db.refresh(event)
    return AlertEventResponse.model_validate(event)
