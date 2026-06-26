"""
FinSight AI — Application Configuration
Uses pydantic-settings for type-safe environment variable loading.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # LLM
    # ------------------------------------------------------------------ #
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"
    GEMINI_EMBEDDING_MODEL: str = "models/text-embedding-004"

    # ------------------------------------------------------------------ #
    # Optional API keys
    # ------------------------------------------------------------------ #
    NEWSAPI_KEY: str = ""
    FRED_API_KEY: str = ""

    # ------------------------------------------------------------------ #
    # Infrastructure
    # ------------------------------------------------------------------ #
    DATABASE_URL: str = "sqlite:///./finsight.db"
    CHROMA_PERSIST_DIR: str = ".chroma_data"
    CACHE_DIR: str = ".cache_data"
    PDF_OUTPUT_DIR: str = ".reports"
    MCP_SERVER_URL: str = "http://localhost:8001"

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    LOG_LEVEL: str = "INFO"

    # ------------------------------------------------------------------ #
    # FastAPI
    # ------------------------------------------------------------------ #
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # ------------------------------------------------------------------ #
    # Streamlit
    # ------------------------------------------------------------------ #
    STREAMLIT_SERVER_PORT: int = 8501

    # ------------------------------------------------------------------ #
    # EDGAR
    # ------------------------------------------------------------------ #
    EDGAR_USER_AGENT: str = "FinSight AI dev@finsight.ai"
    EDGAR_BASE_URL: str = "https://data.sec.gov"
    EDGAR_SEARCH_URL: str = "https://efts.sec.gov/LATEST/search-index"

    # ------------------------------------------------------------------ #
    # RAG
    # ------------------------------------------------------------------ #
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    EMBEDDING_BATCH_SIZE: int = 100
    MAX_FILING_CHARS: int = 100_000

    # ------------------------------------------------------------------ #
    # Alert scheduler
    # ------------------------------------------------------------------ #
    ALERT_INTERVAL_HOURS: int = 6

    def ensure_dirs(self) -> None:
        """Create all local storage directories if they don't exist."""
        for d in [
            self.CHROMA_PERSIST_DIR,
            self.CACHE_DIR,
            self.PDF_OUTPUT_DIR,
        ]:
            Path(d).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings singleton."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
