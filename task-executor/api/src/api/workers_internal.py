"""Internal worker management endpoints."""

from fastapi import APIRouter, HTTPException, Query, Request
from typing import Dict, Any

from ..config.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/workers/internal/status")
async def get_internal_workers_status(
    request: Request
) -> Dict[str, Any]:
    """Get status of internal worker manager."""
    if not hasattr(request.app.state, "worker_manager"):
        return {
            "enabled": False,
            "message": "Internal workers not enabled"
        }

    manager = request.app.state.worker_manager
    return manager.get_status()


@router.post("/workers/internal/scale")
async def scale_internal_workers(
    request: Request,
    num_workers: int = Query(..., ge=0, le=50, description="Number of workers to scale to")
) -> Dict[str, Any]:
    """Scale internal workers up or down."""
    if not hasattr(request.app.state, "worker_manager"):
        raise HTTPException(
            status_code=400,
            detail="Internal workers not enabled"
        )

    manager = request.app.state.worker_manager
    current = len(manager.workers)

    await manager.scale(num_workers)

    logger.info(
        "api.workers.scaled",
        from_workers=current,
        to_workers=num_workers
    )

    return {
        "previous_workers": current,
        "current_workers": num_workers,
        "status": "scaled"
    }


@router.post("/workers/internal/stop")
async def stop_internal_workers(
    request: Request
) -> Dict[str, Any]:
    """Stop all internal workers."""
    if not hasattr(request.app.state, "worker_manager"):
        raise HTTPException(
            status_code=400,
            detail="Internal workers not enabled"
        )

    manager = request.app.state.worker_manager
    await manager.stop()

    logger.info("api.workers.stopped")

    return {
        "status": "stopped",
        "message": "All internal workers stopped"
    }


@router.post("/workers/internal/start")
async def start_internal_workers(
    request: Request
) -> Dict[str, Any]:
    """Start internal workers if stopped."""
    if not hasattr(request.app.state, "worker_manager"):
        raise HTTPException(
            status_code=400,
            detail="Internal workers not enabled"
        )

    manager = request.app.state.worker_manager
    await manager.start()

    logger.info("api.workers.started", workers=len(manager.workers))

    return {
        "status": "started",
        "workers": len(manager.workers)
    }