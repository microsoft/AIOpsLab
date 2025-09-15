"""Pytest configuration and fixtures for API tests."""

import os
import sys
import pytest
import pytest_asyncio
import asyncio
from typing import AsyncGenerator, Generator
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool
import uuid
from datetime import datetime, timedelta

# Setup test environment
os.environ['TESTING'] = 'true'
os.environ['AUTO_START_WORKERS'] = 'false'
os.environ['DATABASE_URL'] = 'sqlite+aiosqlite:///:memory:'

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.main import create_app
from src.models import Base
from src.models.task import Task, TaskStatus
from src.models.worker import Worker, WorkerStatus


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def app():
    """Create test application."""
    # Override settings for testing
    os.environ['AUTO_START_WORKERS'] = 'false'
    test_app = create_app()
    yield test_app


@pytest_asyncio.fixture
async def client(app):
    """Create async HTTP client for testing."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def test_db():
    """Create test database."""
    DATABASE_URL = "sqlite+aiosqlite:///:memory:"

    engine = create_async_engine(
        DATABASE_URL,
        poolclass=NullPool,
        echo=False
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    yield async_session

    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_db):
    """Get database session for tests."""
    async with test_db() as session:
        yield session
        await session.rollback()


# Mock data fixtures that don't require database
@pytest.fixture
def sample_task_data():
    """Create sample task data."""
    return {
        "id": str(uuid.uuid4()),
        "problem_id": "test-problem-1",
        "status": "pending",
        "parameters": {
            "max_steps": 30,
            "timeout_minutes": 30,
            "priority": 5
        },
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }


@pytest.fixture
def sample_worker_data():
    """Create sample worker data."""
    return {
        "id": "worker-001-kind",
        "backend_type": "default",
        "status": "idle",
        "last_heartbeat": datetime.utcnow().isoformat(),
        "registered_at": datetime.utcnow().isoformat(),
        "tasks_completed": 0,
        "tasks_failed": 0
    }


# Mock API responses for testing
@pytest_asyncio.fixture
async def mock_api(monkeypatch):
    """Mock API responses."""

    async def mock_get_task(*args, **kwargs):
        return {
            "id": str(uuid.uuid4()),
            "problem_id": "test-problem",
            "status": "pending",
            "parameters": {},
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }

    async def mock_list_tasks(*args, **kwargs):
        return {
            "tasks": [],
            "total": 0
        }

    async def mock_create_task(*args, **kwargs):
        return {
            "id": str(uuid.uuid4()),
            "problem_id": kwargs.get("problem_id", "test"),
            "status": "pending",
            "parameters": kwargs.get("parameters", {}),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }

    # Can add monkeypatch here if needed
    return {
        "get_task": mock_get_task,
        "list_tasks": mock_list_tasks,
        "create_task": mock_create_task
    }