"""Task-related Pydantic schemas."""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from uuid import UUID
import re

from ..models import TaskStatus


class TaskCreate(BaseModel):
    """Request schema for creating a new task."""

    problem_id: str = Field(
        ...,
        description="Problem identifier to execute",
        min_length=1,
        max_length=255,
        examples=["sock-shop-chaos"]
    )

    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Task execution parameters",
        examples=[{
            "max_steps": 30,
            "timeout_minutes": 30,
            "priority": 5,
            "backend_config": "default"
        }]
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "problem_id": "sock-shop-chaos",
                "parameters": {
                    "max_steps": 30,
                    "timeout_minutes": 30,
                    "priority": 5
                }
            }
        }
    )

    @field_validator('problem_id')
    @classmethod
    def validate_problem_id(cls, v: str) -> str:
        """Validate problem ID format."""
        if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9\-_]*$', v):
            raise ValueError('Problem ID must start with alphanumeric and contain only alphanumeric, hyphens, and underscores')
        return v


class TaskResponse(BaseModel):
    """Response schema for task details."""

    id: UUID = Field(..., description="Unique task identifier")
    problem_id: str = Field(..., description="Problem identifier")
    status: TaskStatus = Field(..., description="Current task status")
    parameters: Dict[str, Any] = Field(..., description="Task parameters")

    worker_id: Optional[str] = Field(None, description="Assigned worker ID")
    result: Optional[Dict[str, Any]] = Field(None, description="Task execution result")
    error_details: Optional[str] = Field(None, description="Error details if failed")

    created_at: datetime = Field(..., description="Task creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    started_at: Optional[datetime] = Field(None, description="Execution start time")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "problem_id": "sock-shop-chaos",
                "status": "running",
                "parameters": {
                    "max_steps": 30,
                    "timeout_minutes": 30,
                    "priority": 5
                },
                "worker_id": "worker-001-kind",
                "created_at": "2025-09-14T10:00:00Z",
                "updated_at": "2025-09-14T10:01:00Z",
                "started_at": "2025-09-14T10:01:00Z"
            }
        }
    )


class TaskListResponse(BaseModel):
    """Response schema for paginated task list."""

    tasks: List[TaskResponse] = Field(..., description="List of tasks")
    total: int = Field(..., description="Total number of tasks matching filters")
    page: int = Field(..., description="Current page number (1-based)")
    page_size: int = Field(..., description="Number of items per page")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tasks": [
                    {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "problem_id": "sock-shop-chaos",
                        "status": "completed",
                        "parameters": {"max_steps": 30},
                        "created_at": "2025-09-14T10:00:00Z",
                        "updated_at": "2025-09-14T10:30:00Z"
                    }
                ],
                "total": 42,
                "page": 1,
                "page_size": 20
            }
        }
    )


class TaskStatusUpdate(BaseModel):
    """Request schema for updating task status (internal use)."""

    status: TaskStatus = Field(..., description="New task status")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result (for completion)")
    error_details: Optional[str] = Field(None, description="Error details (for failure)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "completed",
                "result": {
                    "solution": "restart service",
                    "confidence": 0.95
                }
            }
        }
    )


class TaskLogEntry(BaseModel):
    """Schema for a single task log entry."""

    id: UUID = Field(..., description="Log entry ID")
    task_id: UUID = Field(..., description="Associated task ID")
    timestamp: datetime = Field(..., description="Log timestamp")
    level: Literal["debug", "info", "warning", "error", "critical"] = Field(
        ...,
        description="Log level"
    )
    message: str = Field(..., description="Log message")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context"
    )

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "650e8400-e29b-41d4-a716-446655440001",
                "task_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2025-09-14T10:15:00Z",
                "level": "info",
                "message": "Executing kubectl get pods",
                "context": {
                    "step": 5,
                    "action": "exec"
                }
            }
        }
    )


class TaskLogsResponse(BaseModel):
    """Response schema for task logs."""

    logs: List[TaskLogEntry] = Field(..., description="List of log entries")
    total: int = Field(..., description="Total number of log entries")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "logs": [
                    {
                        "id": "650e8400-e29b-41d4-a716-446655440001",
                        "task_id": "550e8400-e29b-41d4-a716-446655440000",
                        "timestamp": "2025-09-14T10:15:00Z",
                        "level": "info",
                        "message": "Task started",
                        "context": {}
                    }
                ],
                "total": 25
            }
        }
    )


class TaskStatistics(BaseModel):
    """Task execution statistics."""
    total_tasks: int = Field(..., description="Total number of tasks")
    pending_tasks: int = Field(..., description="Number of pending tasks")
    running_tasks: int = Field(..., description="Number of running tasks")
    completed_tasks: int = Field(..., description="Number of completed tasks")
    failed_tasks: int = Field(..., description="Number of failed tasks")
    timeout_tasks: int = Field(..., description="Number of timed out tasks")
    cancelled_tasks: int = Field(..., description="Number of cancelled tasks")
    avg_execution_time: Optional[float] = Field(None, description="Average execution time in seconds")
    success_rate: float = Field(..., description="Task success rate (0-1)")
    tasks_by_problem: Dict[str, int] = Field(default_factory=dict, description="Task count by problem ID")
    tasks_by_worker: Dict[str, int] = Field(default_factory=dict, description="Task count by worker ID")

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "total_tasks": 100,
                "pending_tasks": 5,
                "running_tasks": 2,
                "completed_tasks": 85,
                "failed_tasks": 5,
                "timeout_tasks": 2,
                "cancelled_tasks": 1,
                "avg_execution_time": 124.5,
                "success_rate": 0.85,
                "tasks_by_problem": {
                    "sock-shop-chaos": 45,
                    "microservice-debug": 30,
                    "network-failure": 25
                },
                "tasks_by_worker": {
                    "worker-001-kind": 40,
                    "worker-002-kind": 35,
                    "worker-003-kind": 25
                }
            }
        }
    )