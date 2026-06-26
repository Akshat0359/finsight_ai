"""
FinSight AI — Database Engine & Session Management
SQLAlchemy with SQLite, both sync and async support.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator

from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# ------------------------------------------------------------------ #
# Sync engine (used for init and sync operations)
# ------------------------------------------------------------------ #
_sync_url = settings.DATABASE_URL
if _sync_url.startswith("sqlite:///"):
    _sync_url = _sync_url  # keep as-is
_async_url = _sync_url.replace("sqlite:///", "sqlite+aiosqlite:///")

sync_engine = create_engine(
    _sync_url,
    connect_args={"check_same_thread": False},
    echo=False,
)

# Enable WAL mode for better concurrent reads
@event.listens_for(sync_engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):  # type: ignore[no-untyped-def]
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SyncSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=sync_engine,
)

# ------------------------------------------------------------------ #
# Async engine (used by FastAPI routes)
# ------------------------------------------------------------------ #
async_engine = create_async_engine(
    _async_url,
    connect_args={"check_same_thread": False},
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


# ------------------------------------------------------------------ #
# Base model
# ------------------------------------------------------------------ #
class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


# ------------------------------------------------------------------ #
# Session dependency helpers
# ------------------------------------------------------------------ #
@contextmanager
def get_sync_db() -> Generator[Session, None, None]:
    """Sync context manager for database sessions."""
    db = SyncSessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@asynccontextmanager
async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager for database sessions."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def init_db() -> None:
    """Create all tables in the database. Call once on startup."""
    from db.models import (  # noqa: F401 — import to register models
        AlertConfig,
        AlertEvent,
        Company,
        Filing,
        LLMCache,
        Report,
        Run,
    )

    Base.metadata.create_all(bind=sync_engine)
    logger.info("Database tables created/verified.")
