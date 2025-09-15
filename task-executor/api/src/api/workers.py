"""Worker-related API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Request
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import get_db, WorkerStatus, Task
from ..services.worker_service import WorkerService
from ..schemas.worker import (
    WorkerRegister,
    WorkerResponse,
    WorkerListResponse,
    WorkerHeartbeat,
    WorkerStatsResponse
)
from ..schemas.task import TaskResponse
from ..config.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/workers/register", response_model=WorkerResponse, status_code=201)
async def register_worker(
    worker_data: WorkerRegister,
    session: AsyncSession = Depends(get_db)
) -> WorkerResponse:
    """Register a new worker or update existing one."""
    try:
        service = WorkerService(session)
        worker = await service.register_worker(worker_data)

        logger.info(
            "api.worker.registered",
            worker_id=worker_data.worker_id,
            backend_type=worker_data.backend_type
        )

        return WorkerResponse.model_validate(worker)

    except Exception as e:
        logger.error(
            "api.worker.register.error",
            error=str(e),
            worker_id=worker_data.worker_id
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workers", response_model=WorkerListResponse)
async def list_workers(
    status: Optional[WorkerStatus] = Query(None, description="Filter by status"),
    backend_type: Optional[str] = Query(None, description="Filter by backend type"),
    session: AsyncSession = Depends(get_db)
) -> WorkerListResponse:
    """List all workers with optional filtering."""
    try:
        service = WorkerService(session)
        workers = await service.list_workers(
            status=status,
            backend_type=backend_type
        )

        return WorkerListResponse(
            workers=[WorkerResponse.model_validate(w) for w in workers],
            total=len(workers)
        )

    except Exception as e:
        logger.error("api.worker.list.error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workers/{worker_id}", response_model=WorkerResponse)
async def get_worker(
    worker_id: str = Path(..., description="Worker ID"),
    session: AsyncSession = Depends(get_db)
) -> WorkerResponse:
    """Get worker details by ID."""
    try:
        service = WorkerService(session)
        worker = await service.get_worker(worker_id)

        if not worker:
            raise HTTPException(
                status_code=404,
                detail=f"Worker {worker_id} not found"
            )

        return WorkerResponse.model_validate(worker)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "api.worker.get.error",
            error=str(e),
            worker_id=worker_id
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workers/{worker_id}/heartbeat", response_model=WorkerResponse)
async def worker_heartbeat(
    worker_id: str = Path(..., description="Worker ID"),
    heartbeat_data: WorkerHeartbeat = ...,
    session: AsyncSession = Depends(get_db)
) -> WorkerResponse:
    """Update worker heartbeat and status."""
    try:
        service = WorkerService(session)
        worker = await service.heartbeat(worker_id, heartbeat_data)

        return WorkerResponse.model_validate(worker)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(
            "api.worker.heartbeat.error",
            error=str(e),
            worker_id=worker_id
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workers/{worker_id}/claim", response_model=Optional[TaskResponse])
async def claim_task(
    worker_id: str = Path(..., description="Worker ID"),
    session: AsyncSession = Depends(get_db)
) -> Optional[TaskResponse]:
    """Claim next available task for worker."""
    try:
        service = WorkerService(session)
        task = await service.claim_task(worker_id)

        if task:
            logger.info(
                "api.worker.claimed_task",
                worker_id=worker_id,
                task_id=str(task.id)
            )
            return TaskResponse.model_validate(task)

        return None

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(
            "api.worker.claim.error",
            error=str(e),
            worker_id=worker_id
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workers/{worker_id}/tasks/{task_id}/complete", response_model=TaskResponse)
async def complete_task(
    worker_id: str = Path(..., description="Worker ID"),
    task_id: str = Path(..., description="Task ID"),
    result: dict = ...,
    session: AsyncSession = Depends(get_db)
) -> TaskResponse:
    """Mark task as completed by worker."""
    try:
        service = WorkerService(session)
        task = await service.complete_task(worker_id, task_id, result)

        logger.info(
            "api.worker.completed_task",
            worker_id=worker_id,
            task_id=task_id
        )

        return TaskResponse.model_validate(task)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            "api.worker.complete.error",
            error=str(e),
            worker_id=worker_id,
            task_id=task_id
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workers/{worker_id}/tasks/{task_id}/fail", response_model=TaskResponse)
async def fail_task(
    worker_id: str = Path(..., description="Worker ID"),
    task_id: str = Path(..., description="Task ID"),
    error_details: dict = ...,
    session: AsyncSession = Depends(get_db)
) -> TaskResponse:
    """Mark task as failed by worker."""
    try:
        service = WorkerService(session)

        error_str = error_details.get("error", "Unknown error")
        if isinstance(error_details, dict):
            error_str = str(error_details)

        task = await service.fail_task(worker_id, task_id, error_str)

        logger.info(
            "api.worker.failed_task",
            worker_id=worker_id,
            task_id=task_id
        )

        return TaskResponse.model_validate(task)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            "api.worker.fail.error",
            error=str(e),
            worker_id=worker_id,
            task_id=task_id
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workers/{worker_id}/stats", response_model=WorkerStatsResponse)
async def get_worker_stats(
    worker_id: str = Path(..., description="Worker ID"),
    session: AsyncSession = Depends(get_db)
) -> WorkerStatsResponse:
    """Get detailed statistics for a worker."""
    try:
        service = WorkerService(session)
        stats = await service.get_worker_stats(worker_id)

        return WorkerStatsResponse(**stats)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(
            "api.worker.stats.error",
            error=str(e),
            worker_id=worker_id
        )
        raise HTTPException(status_code=500, detail=str(e))