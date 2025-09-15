"""Database configuration and session management."""

from typing import AsyncGenerator
import os
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

# Get database URL from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://aiopslab:aiopslab@localhost:5432/aiopslab"
)

# Check if we're in testing mode
TESTING = os.getenv("TESTING", "false").lower() == "true"

# Create async engine - lazy initialization for testing
def _create_engine():
    if TESTING:
        # Use in-memory SQLite for tests
        return create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            echo=False,
            connect_args={"check_same_thread": False},
        )
    else:
        return create_async_engine(
            DATABASE_URL,
            echo=os.getenv("SQL_ECHO", "false").lower() == "true",
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )

# Only create engine if not importing for tests
engine = None
if not TESTING or os.getenv("CREATE_ENGINE", "true").lower() == "true":
    engine = _create_engine()

# Create async session factory
async_session = None
if engine:
    async_session = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

# Create declarative base
Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session."""
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Close database connections."""
    await engine.dispose()