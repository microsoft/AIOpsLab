"""Pytest configuration and fixtures for API tests."""

import pytest
import pytest_asyncio
import asyncio
from typing import AsyncGenerator, Generator
from httpx import AsyncClient
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool
import uuid
from datetime import datetime, timedelta


from src.main import app
from src.models import Base, get_db, engine
from src.models.task import Task
from src.models.worker import Worker


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def test_db():
    """Create test database."""
    # For now, return a mock database URL
    # In real implementation, this would create a test database
    DATABASE_URL = "postgresql+asyncpg://test:test@localhost:5432/test_aiopslab"

    engine = create_async_engine(
        DATABASE_URL,
        poolclass=NullPool,
        echo=False
    )

    # async with engine.begin() as conn:
    #     await conn.run_sync(Base.metadata.drop_all)
    #     await conn.run_sync(Base.metadata.create_all)

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


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture
async def client():
    """Create async HTTP client for testing."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def sample_task(db_session):
    """Create a sample task for testing."""
    task_data = {
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
    # In real implementation, save to database
    # task = Task(**task_data)
    # db_session.add(task)
    # await db_session.commit()
    return task_data


@pytest_asyncio.fixture
async def sample_tasks(db_session):
    """Create multiple sample tasks."""
    tasks = []
    for i in range(5):
        task_data = {
            "id": str(uuid.uuid4()),
            "problem_id": f"test-problem-{i}",
            "status": "pending" if i < 3 else "running",
            "worker_id": f"worker-00{i}-kind" if i >= 3 else None,
            "parameters": {
                "priority": i,
                "max_steps": 30
            },
            "created_at": (datetime.utcnow() - timedelta(minutes=i)).isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        tasks.append(task_data)
    return tasks


@pytest_asyncio.fixture
async def pending_task(db_session):
    """Create a pending task."""
    return await sample_task(db_session)


@pytest_asyncio.fixture
async def running_task(db_session):
    """Create a running task."""
    task_data = {
        "id": str(uuid.uuid4()),
        "problem_id": "test-problem",
        "status": "running",
        "worker_id": "worker-001-kind",
        "started_at": datetime.utcnow().isoformat(),
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }
    return task_data


@pytest_asyncio.fixture
async def completed_task(db_session):
    """Create a completed task."""
    task_data = {
        "id": str(uuid.uuid4()),
        "problem_id": "test-problem",
        "status": "completed",
        "worker_id": "worker-001-kind",
        "result": {
            "logs": ["Task completed successfully"],
            "metrics": {"duration": 120},
            "output": {"solution": "Problem solved"}
        },
        "started_at": (datetime.utcnow() - timedelta(minutes=2)).isoformat(),
        "completed_at": datetime.utcnow().isoformat(),
        "created_at": (datetime.utcnow() - timedelta(minutes=3)).isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }
    return task_data


@pytest_asyncio.fixture
async def failed_task(db_session):
    """Create a failed task."""
    task_data = {
        "id": str(uuid.uuid4()),
        "problem_id": "test-problem",
        "status": "failed",
        "error_details": "Task execution failed: Error details",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }
    return task_data


@pytest_asyncio.fixture
async def timeout_task(db_session):
    """Create a timed out task."""
    task_data = {
        "id": str(uuid.uuid4()),
        "problem_id": "test-problem",
        "status": "timeout",
        "error_details": "Task exceeded timeout limit of 30 minutes",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }
    return task_data


@pytest_asyncio.fixture
async def task_with_logs(db_session):
    """Create a task with logs."""
    task_data = await sample_task(db_session)
    # In real implementation, create log entries
    return task_data


@pytest_asyncio.fixture
async def task_without_logs(db_session):
    """Create a task without logs."""
    return await sample_task(db_session)


@pytest_asyncio.fixture
async def sample_task_with_logs(db_session):
    """Create a task with associated logs."""
    return await task_with_logs(db_session)


@pytest_asyncio.fixture
async def registered_worker(db_session):
    """Create a registered worker."""
    worker_data = {
        "id": "worker-001-kind",
        "backend_type": "default",
        "status": "idle",
        "last_heartbeat": datetime.utcnow().isoformat(),
        "registered_at": datetime.utcnow().isoformat(),
        "tasks_completed": 0,
        "tasks_failed": 0
    }
    return worker_data


@pytest_asyncio.fixture
async def registered_workers(db_session):
    """Create multiple registered workers."""
    workers = []
    for i in range(3):
        worker_data = {
            "id": f"worker-00{i}-kind",
            "backend_type": "default" if i < 2 else "gpu",
            "status": "idle" if i < 2 else "busy",
            "last_heartbeat": datetime.utcnow().isoformat(),
            "registered_at": datetime.utcnow().isoformat(),
            "tasks_completed": i * 5,
            "tasks_failed": i
        }
        workers.append(worker_data)
    return workers


@pytest_asyncio.fixture
async def busy_worker(db_session):
    """Create a busy worker."""
    worker_data = {
        "id": "worker-002-kind",
        "backend_type": "default",
        "status": "busy",
        "current_task_id": str(uuid.uuid4()),
        "last_heartbeat": datetime.utcnow().isoformat(),
        "registered_at": datetime.utcnow().isoformat(),
        "tasks_completed": 10,
        "tasks_failed": 1
    }
    return worker_data


@pytest_asyncio.fixture
async def offline_worker(db_session):
    """Create an offline worker."""
    worker_data = {
        "id": "worker-003-kind",
        "backend_type": "default",
        "status": "offline",
        "last_heartbeat": (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
        "registered_at": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
        "tasks_completed": 20,
        "tasks_failed": 2
    }
    return worker_data


@pytest_asyncio.fixture
async def worker_with_capabilities(db_session):
    """Create a worker with capabilities."""
    worker_data = await registered_worker(db_session)
    worker_data["capabilities"] = {
        "max_parallel_tasks": 3,
        "supported_problems": ["misconfig", "stress", "delay"]
    }
    return worker_data


@pytest_asyncio.fixture
async def worker_with_metadata(db_session):
    """Create a worker with metadata."""
    worker_data = await registered_worker(db_session)
    worker_data["metadata"] = {
        "host": "server-01",
        "version": "1.0.0",
        "kind_cluster": worker_data["id"]
    }
    return worker_data