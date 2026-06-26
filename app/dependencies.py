"""
FinSight AI — FastAPI Dependency Injection
DB session, cache, and MCP client factories for use with Depends().
"""
from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from cache.disk_cache import get_cache
from db.database import AsyncSessionLocal
from mcp_server.client import MCPClient, get_mcp_client


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: async SQLAlchemy session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_cache_dep():
    """FastAPI dependency: diskcache Cache instance."""
    return get_cache()


def get_mcp_dep() -> MCPClient:
    """FastAPI dependency: MCP client singleton."""
    return get_mcp_client()
