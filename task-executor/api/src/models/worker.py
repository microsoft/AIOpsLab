"""Worker model for managing worker processes."""

from sqlalchemy import (
    Column, String, DateTime, Integer, Enum, Index, CheckConstraint, func
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from datetime import datetime

from .database import Base
from .enums import WorkerStatus


class Worker(Base):
    """Worker model representing a task execution process."""

    __tablename__ = "workers"

    # Primary key - worker ID follows pattern worker-XXX-kind
    id = Column(
        String(255),
        primary_key=True,
        nullable=False,
        comment="Worker identifier (cluster name)"
    )

    # Backend configuration
    backend_type = Column(
        String(100),
        nullable=False,
        index=True,
        comment="Backend configuration type"
    )

    # Status
    status = Column(
        Enum(WorkerStatus, name="worker_status", values_callable=lambda obj: [e.value for e in obj]),
        nullable=False,
        default=WorkerStatus.IDLE,
        index=True,
        comment="Current worker status"
    )

    # Heartbeat tracking
    last_heartbeat = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
        comment="Last health check time"
    )

    # Current task
    current_task_id = Column(
        UUID(as_uuid=True),
        nullable=True,
        comment="Currently executing task ID"
    )

    # Capabilities (stored as JSONB)
    capabilities = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Worker capabilities including max_parallel_tasks and supported_problems"
    )

    # Metadata (stored as JSONB)
    worker_metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Worker metadata including host, version, and kind_cluster"
    )

    # Statistics
    tasks_completed = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Total completed tasks"
    )

    tasks_failed = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Total failed tasks"
    )

    # Timestamps
    registered_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="Registration time"
    )

    # Table configuration
    __table_args__ = (
        # Indexes for common queries
        Index('idx_workers_status_heartbeat', 'status', 'last_heartbeat'),
        Index('idx_workers_backend_status', 'backend_type', 'status'),

        # GIN indexes for JSONB queries
        Index('idx_workers_capabilities', 'capabilities', postgresql_using='gin'),
        Index('idx_workers_worker_metadata', 'worker_metadata', postgresql_using='gin'),

        # Constraints
        CheckConstraint(
            "id ~ '^worker-[0-9]{3}-kind$'",
            name='check_worker_id_pattern'
        ),
        CheckConstraint(
            "status != 'busy' OR current_task_id IS NOT NULL",
            name='check_busy_has_task'
        ),
        CheckConstraint(
            "status = 'busy' OR current_task_id IS NULL",
            name='check_not_busy_no_task'
        ),
        CheckConstraint(
            "tasks_completed >= 0",
            name='check_tasks_completed_positive'
        ),
        CheckConstraint(
            "tasks_failed >= 0",
            name='check_tasks_failed_positive'
        ),
    )

    def to_dict(self) -> dict:
        """Convert worker to dictionary."""
        return {
            "id": self.id,
            "backend_type": self.backend_type,
            "status": self.status.value if self.status else None,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "current_task_id": str(self.current_task_id) if self.current_task_id else None,
            "capabilities": self.capabilities or {},
            "metadata": self.worker_metadata or {},
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "registered_at": self.registered_at.isoformat() if self.registered_at else None,
        }

    def is_available(self) -> bool:
        """Check if worker is available for new tasks."""
        return self.status == WorkerStatus.IDLE

    def is_online(self, timeout_seconds: int = 60) -> bool:
        """Check if worker is online based on heartbeat."""
        if not self.last_heartbeat:
            return False

        from datetime import timedelta
        now = datetime.utcnow()
        return (now - self.last_heartbeat.replace(tzinfo=None)) < timedelta(seconds=timeout_seconds)

    def can_handle_problem(self, problem_id: str) -> bool:
        """Check if worker can handle specific problem type."""
        if not self.capabilities:
            return True  # No restrictions

        supported = self.capabilities.get("supported_problems", [])
        if not supported:
            return True  # No restrictions

        # Check if problem matches any supported pattern
        for pattern in supported:
            if pattern in problem_id:
                return True
        return False

    def __repr__(self) -> str:
        return f"<Worker {self.id}: {self.backend_type} ({self.status})>"