"""
FinSight AI — Run Status Pydantic Schemas
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10, description="Company ticker symbol")
    force_refresh: bool = Field(default=False, description="Force re-analysis even if cached")


class RunCreate(BaseModel):
    ticker: str
    triggered_by: str = "api"


class RunStatus(BaseModel):
    run_id: str
    ticker: str
    status: Literal["PENDING", "RUNNING", "COMPLETE", "FAILED"]
    triggered_by: str = "api"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    total_tokens_used: int = 0
    error_message: str | None = None

    model_config = {"from_attributes": True}


class AnalyzeResponse(BaseModel):
    run_id: str
    status: Literal["PENDING", "RUNNING", "COMPLETE", "FAILED"]
    ticker: str
    message: str = ""
