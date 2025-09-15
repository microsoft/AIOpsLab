"""Common Pydantic schemas used across the API."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, Literal
from datetime import datetime


class PaginationParams(BaseModel):
    """Common pagination parameters."""

    page: int = Field(
        default=1,
        ge=1,
        description="Page number (1-based)"
    )

    page_size: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of items per page"
    )

    sort: Optional[str] = Field(
        default="-created_at",
        description="Sort field (prefix with - for descending)",
        pattern=r'^-?(created_at|updated_at|started_at|completed_at|priority)$'
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "page": 1,
                "page_size": 20,
                "sort": "-created_at"
            }
        }
    )


class ErrorResponse(BaseModel):
    """Standard error response schema."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )
    request_id: Optional[str] = Field(
        None,
        description="Request tracking ID"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "ValidationError",
                "message": "Invalid problem_id format",
                "details": {
                    "field": "problem_id",
                    "value": "invalid@id"
                },
                "timestamp": "2025-09-14T10:30:00Z",
                "request_id": "req-12345"
            }
        }
    )


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ...,
        description="Overall system health"
    )

    version: str = Field(..., description="API version")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )

    database: Dict[str, Any] = Field(
        ...,
        description="Database health status"
    )

    workers: Dict[str, Any] = Field(
        ...,
        description="Workers health summary"
    )

    queue: Dict[str, Any] = Field(
        ...,
        description="Task queue statistics"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2025-09-14T10:30:00Z",
                "database": {
                    "connected": True,
                    "latency_ms": 2.5
                },
                "workers": {
                    "total": 3,
                    "idle": 2,
                    "busy": 1,
                    "offline": 0
                },
                "queue": {
                    "pending": 5,
                    "running": 1,
                    "completed": 42
                }
            }
        }
    )


class QueueStatsResponse(BaseModel):
    """Task queue statistics response."""

    pending: int = Field(..., description="Number of pending tasks")
    running: int = Field(..., description="Number of running tasks")
    completed: int = Field(..., description="Number of completed tasks")
    failed: int = Field(..., description="Number of failed tasks")
    timeout: int = Field(..., description="Number of timed out tasks")
    cancelled: int = Field(..., description="Number of cancelled tasks")

    total: int = Field(..., description="Total number of tasks")
    success_rate: Optional[float] = Field(
        None,
        description="Overall success rate (0-1)"
    )

    average_wait_time: Optional[float] = Field(
        None,
        description="Average wait time in queue (seconds)"
    )

    average_execution_time: Optional[float] = Field(
        None,
        description="Average execution time (seconds)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "pending": 5,
                "running": 2,
                "completed": 100,
                "failed": 10,
                "timeout": 3,
                "cancelled": 1,
                "total": 121,
                "success_rate": 0.826,
                "average_wait_time": 15.5,
                "average_execution_time": 300.2
            }
        }
    )