"""
FinSight AI — Alert Config & Event Pydantic Schemas
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class AlertConfigCreate(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    alert_type: Literal["NEW_FILING", "SENTIMENT_SPIKE", "SIGNAL_CHANGE"]
    threshold: float | None = None
    delivery_method: Literal["dashboard", "log"] = "dashboard"
    enabled: bool = True


class AlertConfigResponse(BaseModel):
    id: int
    ticker: str
    alert_type: str
    threshold: float | None = None
    delivery_method: str
    enabled: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class AlertEventResponse(BaseModel):
    id: int
    config_id: int
    ticker: str
    triggered_at: datetime
    message: str
    data_json: str | None = None
    acknowledged: bool

    model_config = {"from_attributes": True}


class AlertAcknowledge(BaseModel):
    acknowledged: bool = True
