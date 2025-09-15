"""Worker-related Pydantic schemas."""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
import re

from ..models import WorkerStatus


class WorkerRegister(BaseModel):
    """Request schema for worker registration."""

    worker_id: str = Field(
        ...,
        description="Worker identifier (format: worker-XXX-kind)",
        pattern=r'^worker-[0-9]{3}-kind$',
        examples=["worker-001-kind"]
    )

    backend_type: str = Field(
        ...,
        description="Backend configuration type",
        min_length=1,
        max_length=100,
        examples=["default", "gpt4", "claude"]
    )

    capabilities: Dict[str, Any] = Field(
        default_factory=dict,
        description="Worker capabilities",
        examples=[{
            "max_parallel_tasks": 1,
            "supported_problems": ["sock-shop", "hotel-reservation"]
        }]
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Worker metadata",
        examples=[{
            "host": "worker-001.cluster.local",
            "version": "1.0.0",
            "kind_cluster": "worker-001-kind"
        }]
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "worker_id": "worker-001-kind",
                "backend_type": "default",
                "capabilities": {
                    "max_parallel_tasks": 1,
                    "supported_problems": []
                },
                "metadata": {
                    "host": "worker-001.cluster.local",
                    "version": "1.0.0"
                }
            }
        }
    )

    @field_validator('worker_id')
    @classmethod
    def validate_worker_id(cls, v: str) -> str:
        """Validate worker ID format."""
        if not re.match(r'^worker-[0-9]{3}-kind$', v):
            raise ValueError('Worker ID must follow pattern: worker-XXX-kind')
        return v


class WorkerResponse(BaseModel):
    """Response schema for worker details."""

    id: str = Field(..., description="Worker identifier")
    backend_type: str = Field(..., description="Backend type")
    status: WorkerStatus = Field(..., description="Current worker status")
    last_heartbeat: datetime = Field(..., description="Last heartbeat timestamp")

    current_task_id: Optional[UUID] = Field(None, description="Currently executing task")
    capabilities: Dict[str, Any] = Field(..., description="Worker capabilities")
    metadata: Dict[str, Any] = Field(..., description="Worker metadata")

    tasks_completed: int = Field(..., description="Total completed tasks")
    tasks_failed: int = Field(..., description="Total failed tasks")
    registered_at: datetime = Field(..., description="Registration timestamp")

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "worker-001-kind",
                "backend_type": "default",
                "status": "busy",
                "last_heartbeat": "2025-09-14T10:30:00Z",
                "current_task_id": "550e8400-e29b-41d4-a716-446655440000",
                "capabilities": {"max_parallel_tasks": 1},
                "metadata": {"host": "worker-001.cluster.local"},
                "tasks_completed": 10,
                "tasks_failed": 2,
                "registered_at": "2025-09-14T09:00:00Z"
            }
        }
    )


class WorkerListResponse(BaseModel):
    """Response schema for worker list."""

    workers: List[WorkerResponse] = Field(..., description="List of workers")
    total: int = Field(..., description="Total number of workers")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "workers": [
                    {
                        "id": "worker-001-kind",
                        "backend_type": "default",
                        "status": "idle",
                        "last_heartbeat": "2025-09-14T10:30:00Z",
                        "tasks_completed": 10,
                        "tasks_failed": 2,
                        "registered_at": "2025-09-14T09:00:00Z"
                    }
                ],
                "total": 3
            }
        }
    )


class WorkerHeartbeat(BaseModel):
    """Request schema for worker heartbeat."""

    status: WorkerStatus = Field(..., description="Current worker status")
    current_task_id: Optional[UUID] = Field(None, description="Currently executing task")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "busy",
                "current_task_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }
    )


class WorkerStatsResponse(BaseModel):
    """Response schema for worker statistics."""

    worker_id: str = Field(..., description="Worker identifier")
    total_tasks: int = Field(..., description="Total tasks processed")
    success_rate: float = Field(..., description="Success rate (0-1)")
    average_task_duration: Optional[float] = Field(
        None,
        description="Average task duration in seconds"
    )
    uptime_seconds: float = Field(..., description="Worker uptime in seconds")
    current_status: WorkerStatus = Field(..., description="Current status")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "worker_id": "worker-001-kind",
                "total_tasks": 12,
                "success_rate": 0.833,
                "average_task_duration": 300.5,
                "uptime_seconds": 3600,
                "current_status": "idle"
            }
        }
    )