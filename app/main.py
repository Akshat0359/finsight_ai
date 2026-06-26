"""
FinSight AI — FastAPI Application Entry Point
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings

# ------------------------------------------------------------------ #
# Logging setup
# ------------------------------------------------------------------ #
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Lifespan (startup / shutdown)
# ------------------------------------------------------------------ #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize DB, ChromaDB, and alert scheduler."""
    logger.info("FinSight AI starting up...")

    # Initialize database tables
    from db.database import init_db
    init_db()
    logger.info("Database initialized")

    # Ensure ChromaDB client is ready
    try:
        from rag.vectorstore import get_chroma_client
        get_chroma_client()
        logger.info("ChromaDB initialized at %s", settings.CHROMA_PERSIST_DIR)
    except Exception as exc:
        logger.warning("ChromaDB init warning: %s", exc)

    # Start alert scheduler
    try:
        from delivery.alert_engine import start_scheduler
        start_scheduler()
        logger.info("Alert scheduler started")
    except Exception as exc:
        logger.warning("Alert scheduler start warning: %s", exc)

    yield

    # Shutdown
    logger.info("FinSight AI shutting down...")
    try:
        from delivery.alert_engine import stop_scheduler
        stop_scheduler()
    except Exception:
        pass


# ------------------------------------------------------------------ #
# App factory
# ------------------------------------------------------------------ #
def create_app() -> FastAPI:
    app = FastAPI(
        title="FinSight AI",
        description=(
            "Automated multi-agent financial intelligence platform. "
            "Fetches SEC filings, market data, and news to generate "
            "structured investment analysis reports."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ---- CORS ----
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- Request logging middleware ----
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        logger.info("→ %s %s", request.method, request.url.path)
        response = await call_next(request)
        logger.info("← %s %s %d", request.method, request.url.path, response.status_code)
        return response

    # ---- Exception handlers ----
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        return JSONResponse(
            status_code=404,
            content={"error": "Not found", "path": str(request.url.path)},
        )

    @app.exception_handler(500)
    async def internal_error_handler(request: Request, exc):
        logger.error("Internal server error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
        )

    # ---- Routers ----
    from app.routers.alerts import router as alerts_router
    from app.routers.analyze import router as analyze_router
    from app.routers.health import router as health_router
    from app.routers.reports import router as reports_router

    app.include_router(health_router)
    app.include_router(analyze_router)
    app.include_router(reports_router)
    app.include_router(alerts_router)

    # ---- Static files for PDF downloads ----
    pdf_dir = Path(settings.PDF_OUTPUT_DIR)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/pdfs", StaticFiles(directory=str(pdf_dir)), name="pdfs")

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "FinSight AI",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )
