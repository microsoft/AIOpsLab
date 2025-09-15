"""Health check and monitoring endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from datetime import datetime
import time

from ..models import get_db, Worker, WorkerStatus
from ..services.task_service import TaskService
from ..services.worker_service import WorkerService
from ..schemas.common import HealthResponse, QueueStatsResponse
from ..config.settings import settings
from ..config.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    session: AsyncSession = Depends(get_db)
) -> HealthResponse:
    """Health check endpoint with database and worker status."""
    try:
        db_start = time.time()
        result = await session.execute(text("SELECT 1"))
        _ = result.scalar()
        db_latency = (time.time() - db_start) * 1000

        db_status = {
            "connected": True,
            "latency_ms": round(db_latency, 2)
        }

    except Exception as e:
        logger.error("health.database.error", error=str(e))
        db_status = {
            "connected": False,
            "error": str(e)
        }

    try:
        worker_service = WorkerService(session)
        workers = await worker_service.list_workers()

        worker_status = {
            "total": len(workers),
            "idle": sum(1 for w in workers if w.status == WorkerStatus.IDLE),
            "busy": sum(1 for w in workers if w.status == WorkerStatus.BUSY),
            "offline": sum(1 for w in workers if w.status == WorkerStatus.OFFLINE)
        }

    except Exception as e:
        logger.error("health.workers.error", error=str(e))
        worker_status = {
            "error": str(e)
        }

    try:
        task_service = TaskService(session)
        queue_stats = await task_service.get_queue_stats()

    except Exception as e:
        logger.error("health.queue.error", error=str(e))
        queue_stats = {
            "error": str(e)
        }

    overall_status = "healthy"
    if not db_status.get("connected", False):
        overall_status = "unhealthy"
    elif worker_status.get("total", 0) == 0:
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.VERSION,
        timestamp=datetime.utcnow(),
        database=db_status,
        workers=worker_status,
        queue=queue_stats
    )


@router.get("/queue/stats", response_model=QueueStatsResponse)
async def get_queue_stats(
    session: AsyncSession = Depends(get_db)
) -> QueueStatsResponse:
    """Get detailed task queue statistics."""
    try:
        from ..models import Task, TaskStatus
        from sqlalchemy import func, and_

        task_service = TaskService(session)
        stats = await task_service.get_queue_stats()

        total = sum(stats.values())

        completed = stats.get(TaskStatus.COMPLETED.value, 0)
        failed = stats.get(TaskStatus.FAILED.value, 0)
        finished = completed + failed

        success_rate = completed / finished if finished > 0 else None

        wait_time_query = select(
            func.avg(Task.started_at - Task.created_at)
        ).where(
            and_(
                Task.status.in_([TaskStatus.RUNNING, TaskStatus.COMPLETED]),
                Task.started_at != None
            )
        )
        wait_result = await session.execute(wait_time_query)
        avg_wait = wait_result.scalar_one_or_none()

        exec_time_query = select(
            func.avg(Task.completed_at - Task.started_at)
        ).where(
            and_(
                Task.status == TaskStatus.COMPLETED,
                Task.started_at != None,
                Task.completed_at != None
            )
        )
        exec_result = await session.execute(exec_time_query)
        avg_exec = exec_result.scalar_one_or_none()

        return QueueStatsResponse(
            pending=stats.get(TaskStatus.PENDING.value, 0),
            running=stats.get(TaskStatus.RUNNING.value, 0),
            completed=stats.get(TaskStatus.COMPLETED.value, 0),
            failed=stats.get(TaskStatus.FAILED.value, 0),
            timeout=stats.get(TaskStatus.TIMEOUT.value, 0),
            cancelled=stats.get(TaskStatus.CANCELLED.value, 0),
            total=total,
            success_rate=success_rate,
            average_wait_time=avg_wait.total_seconds() if avg_wait else None,
            average_execution_time=avg_exec.total_seconds() if avg_exec else None
        )

    except Exception as e:
        logger.error("api.queue.stats.error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AIOpsLab Task Execution API",
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health"
    }