"""Worker service for managing worker processes."""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, func

from ..models import Worker, WorkerStatus, Task, TaskStatus
from ..lib.task_queue import TaskQueue
from ..schemas.worker import WorkerRegister, WorkerHeartbeat
from ..config.logging import get_logger

logger = get_logger(__name__)


class WorkerService:
    """Service layer for worker operations."""

    def __init__(self, session: AsyncSession):
        """Initialize worker service with database session."""
        self.session = session
        self.queue = TaskQueue(session)

    async def register_worker(self, worker_data: WorkerRegister) -> Worker:
        """Register a new worker or update existing one."""
        worker = await self.get_worker(worker_data.worker_id)

        if worker:
            worker.backend_type = worker_data.backend_type
            worker.capabilities = worker_data.capabilities
            worker.worker_metadata = worker_data.metadata
            worker.status = WorkerStatus.IDLE
            worker.last_heartbeat = datetime.utcnow()
            worker.current_task_id = None

            logger.info(
                "worker.reregistered",
                worker_id=worker_data.worker_id,
                backend_type=worker_data.backend_type
            )
        else:
            worker = Worker(
                id=worker_data.worker_id,
                backend_type=worker_data.backend_type,
                status=WorkerStatus.IDLE,
                capabilities=worker_data.capabilities,
                worker_metadata=worker_data.metadata
            )
            self.session.add(worker)

            logger.info(
                "worker.registered",
                worker_id=worker_data.worker_id,
                backend_type=worker_data.backend_type
            )

        await self.session.commit()
        await self.session.refresh(worker)

        return worker

    async def get_worker(self, worker_id: str) -> Optional[Worker]:
        """Get worker by ID."""
        query = select(Worker).where(Worker.id == worker_id)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def list_workers(
        self,
        status: Optional[WorkerStatus] = None,
        backend_type: Optional[str] = None
    ) -> List[Worker]:
        """List all workers with optional filtering."""
        query = select(Worker)

        conditions = []
        if status:
            conditions.append(Worker.status == status)
        if backend_type:
            conditions.append(Worker.backend_type == backend_type)

        if conditions:
            query = query.where(and_(*conditions))

        query = query.order_by(Worker.id)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def heartbeat(
        self,
        worker_id: str,
        heartbeat_data: WorkerHeartbeat
    ) -> Worker:
        """Update worker heartbeat and status."""
        worker = await self.get_worker(worker_id)
        if not worker:
            raise ValueError(f"Worker {worker_id} not found")

        worker.last_heartbeat = datetime.utcnow()
        worker.status = heartbeat_data.status
        worker.current_task_id = heartbeat_data.current_task_id

        await self.session.commit()
        await self.session.refresh(worker)

        logger.debug(
            "worker.heartbeat",
            worker_id=worker_id,
            status=heartbeat_data.status.value
        )

        return worker

    async def claim_task(self, worker_id: str) -> Optional[Task]:
        """Claim next available task for worker."""
        worker = await self.get_worker(worker_id)
        if not worker:
            raise ValueError(f"Worker {worker_id} not found")

        if worker.status != WorkerStatus.IDLE:
            logger.warning(
                "worker.claim_task.not_idle",
                worker_id=worker_id,
                status=worker.status.value
            )
            return None

        task = await self.queue.claim_next_task(worker_id)

        if task:
            worker.status = WorkerStatus.BUSY
            worker.current_task_id = task.id
            await self.session.commit()

            logger.info(
                "task.claimed",
                task_id=str(task.id),
                worker_id=worker_id,
                problem_id=task.problem_id
            )

        return task

    async def complete_task(
        self,
        worker_id: str,
        task_id: str,
        result: Dict[str, Any]
    ) -> Task:
        """Mark task as completed by worker."""
        worker = await self.get_worker(worker_id)
        if not worker:
            raise ValueError(f"Worker {worker_id} not found")

        task = await self.queue.complete_task(task_id, result)

        worker.status = WorkerStatus.IDLE
        worker.current_task_id = None
        worker.tasks_completed += 1

        await self.session.commit()

        logger.info(
            "task.completed.by_worker",
            task_id=str(task_id),
            worker_id=worker_id
        )

        return task

    async def fail_task(
        self,
        worker_id: str,
        task_id: str,
        error_details: str
    ) -> Task:
        """Mark task as failed by worker."""
        worker = await self.get_worker(worker_id)
        if not worker:
            raise ValueError(f"Worker {worker_id} not found")

        task = await self.queue.fail_task(task_id, error_details)

        worker.status = WorkerStatus.IDLE
        worker.current_task_id = None
        worker.tasks_failed += 1

        await self.session.commit()

        logger.error(
            "task.failed.by_worker",
            task_id=str(task_id),
            worker_id=worker_id,
            error=error_details[:200]
        )

        return task

    async def get_worker_stats(self, worker_id: str) -> Dict[str, Any]:
        """Get detailed statistics for a worker."""
        worker = await self.get_worker(worker_id)
        if not worker:
            raise ValueError(f"Worker {worker_id} not found")

        total_tasks = worker.tasks_completed + worker.tasks_failed
        success_rate = (
            worker.tasks_completed / total_tasks
            if total_tasks > 0
            else 0
        )

        uptime = datetime.utcnow() - worker.registered_at.replace(tzinfo=None)

        query = select(func.avg(Task.completed_at - Task.started_at)).where(
            and_(
                Task.worker_id == worker_id,
                Task.status == TaskStatus.COMPLETED,
                Task.started_at != None,
                Task.completed_at != None
            )
        )
        result = await self.session.execute(query)
        avg_duration = result.scalar_one_or_none()

        return {
            "worker_id": worker_id,
            "total_tasks": total_tasks,
            "success_rate": success_rate,
            "average_task_duration": (
                avg_duration.total_seconds() if avg_duration else None
            ),
            "uptime_seconds": uptime.total_seconds(),
            "current_status": worker.status
        }

    async def check_worker_health(
        self,
        timeout_seconds: int = 60
    ) -> Dict[str, List[Worker]]:
        """Check health of all workers based on heartbeat."""
        workers = await self.list_workers()

        healthy = []
        unhealthy = []
        cutoff = datetime.utcnow() - timedelta(seconds=timeout_seconds)

        for worker in workers:
            if worker.last_heartbeat.replace(tzinfo=None) > cutoff:
                healthy.append(worker)
            else:
                unhealthy.append(worker)

                if worker.status != WorkerStatus.OFFLINE:
                    worker.status = WorkerStatus.OFFLINE
                    if worker.current_task_id:
                        await self._release_worker_task(worker)

        await self.session.commit()

        return {
            "healthy": healthy,
            "unhealthy": unhealthy
        }

    async def _release_worker_task(self, worker: Worker):
        """Release task from offline worker back to pending."""
        if not worker.current_task_id:
            return

        query = (
            update(Task)
            .where(
                and_(
                    Task.id == worker.current_task_id,
                    Task.status == TaskStatus.RUNNING
                )
            )
            .values(
                status=TaskStatus.PENDING,
                worker_id=None,
                started_at=None,
                updated_at=datetime.utcnow()
            )
        )

        await self.session.execute(query)

        logger.warning(
            "task.released",
            task_id=str(worker.current_task_id),
            worker_id=worker.id,
            reason="worker_offline"
        )

        worker.current_task_id = None