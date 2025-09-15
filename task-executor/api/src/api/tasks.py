"""Task-related API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from typing import Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import get_db, TaskStatus
from ..services.task_service import TaskService
from ..schemas.task import (
    TaskCreate,
    TaskResponse,
    TaskListResponse,
    TaskLogsResponse,
    TaskLogEntry,
    TaskStatistics
)
from ..schemas.common import PaginationParams
from ..config.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/tasks", response_model=TaskResponse, status_code=201)
async def create_task(
    task_data: TaskCreate,
    session: AsyncSession = Depends(get_db)
) -> TaskResponse:
    """Create a new task and add it to the queue."""
    try:
        service = TaskService(session)
        task = await service.create_task(task_data)

        logger.info(
            "api.task.created",
            task_id=str(task.id),
            problem_id=task_data.problem_id
        )

        return TaskResponse.model_validate(task)

    except Exception as e:
        logger.error(
            "api.task.create.error",
            error=str(e),
            problem_id=task_data.problem_id
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks", response_model=TaskListResponse)
async def list_tasks(
    status: Optional[TaskStatus] = Query(None, description="Filter by status"),
    problem_id: Optional[str] = Query(None, description="Filter by problem ID"),
    worker_id: Optional[str] = Query(None, description="Filter by worker ID"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    sort: str = Query("-created_at", pattern=r'^-?(created_at|updated_at|started_at|completed_at)$'),
    session: AsyncSession = Depends(get_db)
) -> TaskListResponse:
    """List tasks with filtering and pagination."""
    try:
        service = TaskService(session)
        tasks, total = await service.list_tasks(
            status=status,
            problem_id=problem_id,
            worker_id=worker_id,
            page=page,
            page_size=page_size,
            sort=sort
        )

        return TaskListResponse(
            tasks=[TaskResponse.model_validate(task) for task in tasks],
            total=total,
            page=page,
            page_size=page_size
        )

    except Exception as e:
        logger.error("api.task.list.error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/stats", response_model=TaskStatistics)
async def get_task_statistics(
    session: AsyncSession = Depends(get_db)
) -> TaskStatistics:
    """Get task execution statistics."""
    try:
        service = TaskService(session)
        stats = await service.get_statistics()

        return TaskStatistics.model_validate(stats)

    except Exception as e:
        logger.error(
            "api.task.stats.error",
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: UUID = Path(..., description="Task ID"),
    session: AsyncSession = Depends(get_db)
) -> TaskResponse:
    """Get task details by ID."""
    try:
        service = TaskService(session)
        task = await service.get_task(task_id)

        if not task:
            raise HTTPException(
                status_code=404,
                detail=f"Task {task_id} not found"
            )

        return TaskResponse.model_validate(task)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "api.task.get.error",
            error=str(e),
            task_id=str(task_id)
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/cancel", response_model=TaskResponse)
async def cancel_task(
    task_id: UUID = Path(..., description="Task ID"),
    session: AsyncSession = Depends(get_db)
) -> TaskResponse:
    """Cancel a pending or running task."""
    try:
        service = TaskService(session)
        task = await service.cancel_task(task_id)

        logger.info(
            "api.task.cancelled",
            task_id=str(task_id)
        )

        return TaskResponse.model_validate(task)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            "api.task.cancel.error",
            error=str(e),
            task_id=str(task_id)
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}/logs", response_model=TaskLogsResponse)
async def get_task_logs(
    task_id: UUID = Path(..., description="Task ID"),
    level: Optional[str] = Query(None, description="Filter by log level"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum logs to return"),
    session: AsyncSession = Depends(get_db)
) -> TaskLogsResponse:
    """Get logs for a specific task."""
    try:
        service = TaskService(session)

        task = await service.get_task(task_id)
        if not task:
            raise HTTPException(
                status_code=404,
                detail=f"Task {task_id} not found"
            )

        from ..models import LogLevel
        log_level = LogLevel(level) if level else None

        logs = await service.get_task_logs(
            task_id,
            level=log_level,
            limit=limit
        )

        return TaskLogsResponse(
            logs=[TaskLogEntry.model_validate(log) for log in logs],
            total=len(logs)
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            "api.task.logs.error",
            error=str(e),
            task_id=str(task_id)
        )
        raise HTTPException(status_code=500, detail=str(e))