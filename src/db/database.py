"""
Async database engine.
Supports PostgreSQL (production) and SQLite (development fallback).

PostgreSQL: set DATABASE_URL=postgresql+asyncpg://user:pass@localhost/hospital_ai
SQLite:     automatic fallback to data/hospital.db if PostgreSQL not configured.
"""

import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

_raw_url = os.getenv("DATABASE_URL", "")

if _raw_url.startswith("postgresql"):
    # PostgreSQL
    DATABASE_URL = _raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    connect_args = {}
else:
    # SQLite fallback — zero setup, works out of the box
    os.makedirs("data", exist_ok=True)
    DATABASE_URL = "sqlite+aiosqlite:///data/hospital.db"
    connect_args = {"check_same_thread": False}

engine = create_async_engine(DATABASE_URL, echo=False, connect_args=connect_args)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    """Create all tables."""
    from src.db import models  # noqa: F401 — ensure models are imported before create_all
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
