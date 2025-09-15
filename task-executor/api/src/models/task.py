"""Task model for storing AIOpsLab task execution requests."""

from sqlalchemy import (
    Column, String, DateTime, Text, Enum, Index, CheckConstraint, func
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from datetime import datetime

from .database import Base
from .enums import TaskStatus


class Task(Base):
    """Task model representing an AIOpsLab problem execution request."""

    __tablename__ = "tasks"

    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False
    )

    # Core fields
    problem_id = Column(
        String(255),
        nullable=False,
        index=True,
        comment="AIOpsLab problem identifier"
    )

    status = Column(
        Enum(TaskStatus, name="task_status", values_callable=lambda obj: [e.value for e in obj]),
        nullable=False,
        default=TaskStatus.PENDING,
        index=True,
        comment="Current task status"
    )

    # Task parameters (stored as JSONB for flexibility)
    parameters = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Task execution parameters including agent_config, max_steps, etc."
    )

    # Worker assignment
    worker_id = Column(
        String(255),
        nullable=True,
        index=True,
        comment="Assigned worker's cluster name"
    )

    # Results and errors
    result = Column(
        JSONB,
        nullable=True,
        comment="Execution results including logs, metrics, and output"
    )

    error_details = Column(
        Text,
        nullable=True,
        comment="Error message if task failed"
    )

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True
    )

    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now()
    )

    started_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When task execution started"
    )

    completed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When task execution completed"
    )

    # Table configuration
    __table_args__ = (
        # Indexes for common queries
        Index('idx_tasks_status_created', 'status', 'created_at'),
        Index('idx_tasks_worker_status', 'worker_id', 'status'),
        Index('idx_tasks_problem_id_status', 'problem_id', 'status'),

        # GIN indexes for JSONB queries
        Index('idx_tasks_parameters', 'parameters', postgresql_using='gin'),
        Index('idx_tasks_result', 'result', postgresql_using='gin'),

        # Constraints
        CheckConstraint(
            "status != 'running' OR worker_id IS NOT NULL",
            name='check_running_has_worker'
        ),
        CheckConstraint(
            "status NOT IN ('completed', 'failed', 'timeout') OR completed_at IS NOT NULL",
            name='check_terminal_has_completed_at'
        ),
        CheckConstraint(
            "started_at IS NULL OR created_at <= started_at",
            name='check_started_after_created'
        ),
        CheckConstraint(
            "completed_at IS NULL OR started_at <= completed_at",
            name='check_completed_after_started'
        ),
    )

    def to_dict(self) -> dict:
        """Convert task to dictionary."""
        return {
            "id": str(self.id),
            "problem_id": self.problem_id,
            "status": self.status.value if self.status else None,
            "parameters": self.parameters or {},
            "worker_id": self.worker_id,
            "result": self.result,
            "error_details": self.error_details,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    def can_transition_to(self, new_status: TaskStatus) -> bool:
        """Check if task can transition to new status."""
        if self.status == TaskStatus.PENDING:
            return new_status in [TaskStatus.RUNNING, TaskStatus.CANCELLED]
        elif self.status == TaskStatus.RUNNING:
            return new_status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TIMEOUT, TaskStatus.CANCELLED]
        else:
            # Terminal states cannot transition
            return False

    def __repr__(self) -> str:
        return f"<Task {self.id}: {self.problem_id} ({self.status})>"