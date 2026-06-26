"""
FinSight AI — Health Check Router
"""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["Health"])


@router.get("/health", summary="Health check")
async def health_check() -> dict:
    """Returns service health status."""
    return {
        "status": "healthy",
        "service": "FinSight AI",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
    }
