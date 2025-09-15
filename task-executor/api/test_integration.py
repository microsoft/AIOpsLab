#!/usr/bin/env python3
"""Integration test for the API with internal workers."""

import asyncio
import time
from uuid import uuid4

# Test database connection
from src.models import get_db, init_db, close_db
from src.models.task import Task, TaskStatus
from src.models.worker import Worker, WorkerStatus
from src.services.task_service import TaskService
from src.services.worker_service import WorkerService
from src.schemas.task import TaskCreate
from src.schemas.worker import WorkerRegister
from src.workers.manager import WorkerManager
from src.config.logging import get_logger

logger = get_logger(__name__)


async def test_integration():
    """Test core functionality."""
    print("\n=== Integration Test ===\n")

    # Initialize database
    print("1. Initializing database...")
    await init_db()
    print("   ✓ Database initialized")

    # Test task creation
    print("\n2. Testing task creation...")
    async for session in get_db():
        service = TaskService(session)

        task_data = TaskCreate(
            problem_id="test-problem-001",
            parameters={"max_steps": 10, "test": True},
            priority=5
        )

        task = await service.create_task(task_data)
        print(f"   ✓ Task created: {task.id}")

        # Verify task in database
        retrieved = await service.get_task(task.id)
        assert retrieved is not None
        assert retrieved.problem_id == "test-problem-001"
        print("   ✓ Task verified in database")
        break

    # Test worker registration
    print("\n3. Testing worker registration...")
    async for session in get_db():
        service = WorkerService(session)

        worker_data = WorkerRegister(
            worker_id="test-worker-001-kind",
            backend_type="test",
            capabilities={"max_parallel_tasks": 1},
            metadata={"test": True}
        )

        worker = await service.register_worker(worker_data)
        print(f"   ✓ Worker registered: {worker.id}")

        # List workers
        workers = await service.list_workers()
        assert len(workers) > 0
        print(f"   ✓ Found {len(workers)} workers")
        break

    # Test worker manager
    print("\n4. Testing worker manager...")
    manager = WorkerManager(num_workers=2)

    # Start workers
    await manager.start()
    print(f"   ✓ Started {manager.num_workers} workers")

    # Check status
    status = manager.get_status()
    assert status["running"] == True
    assert status["num_workers"] == 2
    print("   ✓ Worker manager status verified")

    # Submit tasks for processing
    print("\n5. Submitting tasks for processing...")
    task_ids = []
    async for session in get_db():
        service = TaskService(session)

        for i in range(3):
            task_data = TaskCreate(
                problem_id=f"test-problem-{i:03d}",
                parameters={"index": i},
                priority=10 - i
            )
            task = await service.create_task(task_data)
            task_ids.append(task.id)
            print(f"   ✓ Task {i+1} created: {task.id}")
        break

    # Wait for tasks to be processed
    print("\n6. Waiting for task processing...")
    await asyncio.sleep(5)  # Give workers time to process

    # Check task status
    async for session in get_db():
        service = TaskService(session)

        completed = 0
        for task_id in task_ids:
            task = await service.get_task(task_id)
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                completed += 1
                print(f"   ✓ Task {task_id} status: {task.status}")

        print(f"   Completed: {completed}/{len(task_ids)}")
        break

    # Test scaling
    print("\n7. Testing worker scaling...")
    await manager.scale(4)
    status = manager.get_status()
    assert status["num_workers"] == 4
    print("   ✓ Scaled to 4 workers")

    await manager.scale(1)
    status = manager.get_status()
    assert status["num_workers"] == 1
    print("   ✓ Scaled down to 1 worker")

    # Stop workers
    print("\n8. Stopping workers...")
    await manager.stop()
    print("   ✓ Workers stopped")

    # Close database
    await close_db()
    print("\n=== All tests passed! ===\n")


if __name__ == "__main__":
    asyncio.run(test_integration())