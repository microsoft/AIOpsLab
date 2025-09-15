"""Worker manager that runs background workers within the API process."""

import asyncio
import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Worker, WorkerStatus, Task, TaskStatus, get_db
from ..lib.task_queue import TaskQueue
from ..services.worker_service import WorkerService
from ..services.task_service import TaskService
from .executor import TaskExecutor
from ..config.logging import get_logger
from ..config.settings import settings

logger = get_logger(__name__)


class WorkerManager:
    """Manages internal workers that run as background tasks."""

    def __init__(self, num_workers: int = 3):
        """Initialize worker manager."""
        self.num_workers = num_workers
        self.workers: Dict[str, asyncio.Task] = {}
        self.executors: Dict[str, TaskExecutor] = {}
        self.running = False
        self.shutdown_event = asyncio.Event()

    async def start(self):
        """Start all workers as background tasks."""
        if self.running:
            logger.warning("worker_manager.already_running")
            return

        self.running = True
        logger.info("worker_manager.starting", num_workers=self.num_workers)

        # Create and start workers
        for i in range(1, self.num_workers + 1):
            worker_id = f"worker-{i:03d}-internal"
            await self._start_worker(worker_id)

        logger.info("worker_manager.started", active_workers=len(self.workers))

    async def _start_worker(self, worker_id: str):
        """Start a single worker as a background task."""
        # Create executor (using LLM executor for better logging)
        # We'll pass the session when executing tasks
        executor = TaskExecutor(worker_id, backend_type="internal")
        self.executors[worker_id] = executor

        # Register worker in database
        async for session in get_db():
            try:
                service = WorkerService(session)
                from ..schemas.worker import WorkerRegister

                worker_data = WorkerRegister(
                    worker_id=worker_id.replace("-internal", "-kind"),  # Match expected pattern
                    backend_type="internal",
                    capabilities={
                        "max_parallel_tasks": 1,
                        "supported_problems": []
                    },
                    metadata={
                        "type": "internal",
                        "process": "api",
                        "version": "1.0.0"
                    }
                )
                await service.register_worker(worker_data)

            except Exception as e:
                logger.error(
                    "worker.registration.failed",
                    worker_id=worker_id,
                    error=str(e)
                )
            finally:
                break  # Exit the async generator

        # Create and start worker task
        task = asyncio.create_task(
            self._worker_loop(worker_id, executor),
            name=f"worker-{worker_id}"
        )
        self.workers[worker_id] = task

        logger.info("worker.started", worker_id=worker_id)

    async def _worker_loop(self, worker_id: str, executor: TaskExecutor):
        """Main loop for a worker."""
        logger.info("worker.loop.started", worker_id=worker_id)

        # Use the pattern-compliant ID for database operations
        db_worker_id = worker_id.replace("-internal", "-kind")
        consecutive_errors = 0
        max_consecutive_errors = 5

        while self.running:
            try:
                # Check for shutdown
                if self.shutdown_event.is_set():
                    break

                # Get database session and claim a task
                async for session in get_db():
                    try:
                        # Claim next task
                        queue = TaskQueue(session)
                        task = await queue.claim_next_task(db_worker_id)

                        if task:
                            consecutive_errors = 0  # Reset error counter
                            logger.info(
                                "worker.task.claimed",
                                worker_id=worker_id,
                                task_id=str(task.id),
                                problem_id=task.problem_id
                            )

                            # Execute the task using real orchestrator
                            # Always use real orchestrator with Kind cluster
                            from .orchestrator_executor import OrchestratorExecutor
                            orch_executor = OrchestratorExecutor(worker_id, session, backend_type="orchestrator")
                            result = await orch_executor.execute(task)

                            # Update task status
                            if result.get("success"):
                                await queue.complete_task(task.id, result)
                                logger.info(
                                    "worker.task.completed",
                                    worker_id=worker_id,
                                    task_id=str(task.id)
                                )
                            else:
                                await queue.fail_task(
                                    task.id,
                                    result.get("error", "Unknown error")
                                )
                                logger.error(
                                    "worker.task.failed",
                                    worker_id=worker_id,
                                    task_id=str(task.id),
                                    error=result.get("error")
                                )

                            # Update worker heartbeat
                            await self._update_heartbeat(session, db_worker_id, idle=True)

                        else:
                            # No tasks available, wait
                            await asyncio.sleep(settings.WORKER_POLL_INTERVAL)

                    except Exception as e:
                        logger.error(
                            "worker.loop.error",
                            worker_id=worker_id,
                            error=str(e)
                        )
                        consecutive_errors += 1

                        if consecutive_errors >= max_consecutive_errors:
                            logger.error(
                                "worker.too_many_errors",
                                worker_id=worker_id,
                                errors=consecutive_errors
                            )
                            break

                        await asyncio.sleep(settings.WORKER_POLL_INTERVAL * 2)
                    finally:
                        break  # Exit the async generator

            except asyncio.CancelledError:
                logger.info("worker.cancelled", worker_id=worker_id)
                break
            except Exception as e:
                logger.error(
                    "worker.unexpected_error",
                    worker_id=worker_id,
                    error=str(e)
                )
                await asyncio.sleep(settings.WORKER_POLL_INTERVAL)

        logger.info("worker.loop.stopped", worker_id=worker_id)

    async def _update_heartbeat(self, session: AsyncSession, worker_id: str, idle: bool = True):
        """Update worker heartbeat."""
        try:
            from sqlalchemy import update
            await session.execute(
                update(Worker)
                .where(Worker.id == worker_id)
                .values(
                    last_heartbeat=datetime.utcnow(),
                    status=WorkerStatus.IDLE if idle else WorkerStatus.BUSY
                )
            )
            await session.commit()
        except Exception as e:
            logger.error(
                "worker.heartbeat.failed",
                worker_id=worker_id,
                error=str(e)
            )

    async def stop(self):
        """Stop all workers gracefully."""
        if not self.running:
            return

        logger.info("worker_manager.stopping")
        self.running = False
        self.shutdown_event.set()

        # Cancel all worker tasks
        for worker_id, task in self.workers.items():
            if not task.done():
                task.cancel()
                logger.info("worker.cancelling", worker_id=worker_id)

        # Wait for all tasks to complete
        if self.workers:
            await asyncio.gather(*self.workers.values(), return_exceptions=True)

        # Mark all workers as offline in database
        async for session in get_db():
            try:
                from sqlalchemy import update
                for worker_id in self.workers.keys():
                    db_worker_id = worker_id.replace("-internal", "-kind")
                    await session.execute(
                        update(Worker)
                        .where(Worker.id == db_worker_id)
                        .values(status=WorkerStatus.OFFLINE)
                    )
                await session.commit()
            except Exception as e:
                logger.error("worker_manager.cleanup.failed", error=str(e))
            finally:
                break

        self.workers.clear()
        self.executors.clear()
        logger.info("worker_manager.stopped")

    async def scale(self, num_workers: int):
        """Scale the number of workers up or down."""
        current = len(self.workers)

        if num_workers > current:
            # Scale up
            for i in range(current + 1, num_workers + 1):
                worker_id = f"worker-{i:03d}-internal"
                if worker_id not in self.workers:
                    await self._start_worker(worker_id)
            logger.info("worker_manager.scaled_up", from_workers=current, to_workers=num_workers)

        elif num_workers < current:
            # Scale down
            workers_to_remove = list(self.workers.keys())[num_workers:]
            for worker_id in workers_to_remove:
                if worker_id in self.workers:
                    self.workers[worker_id].cancel()
                    del self.workers[worker_id]
                    del self.executors[worker_id]
            logger.info("worker_manager.scaled_down", from_workers=current, to_workers=num_workers)

    def get_status(self) -> Dict:
        """Get current status of all workers."""
        return {
            "running": self.running,
            "num_workers": len(self.workers),
            "workers": {
                worker_id: {
                    "status": "running" if not task.done() else "stopped",
                    "has_executor": worker_id in self.executors
                }
                for worker_id, task in self.workers.items()
            }
        }