"""TaskLog model for storing task execution logs."""

from sqlalchemy import (
    Column, String, DateTime, Text, Enum, Index, ForeignKey, func
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime

from .database import Base
from .enums import LogLevel


class TaskLog(Base):
    """TaskLog model for detailed execution logs."""

    __tablename__ = "task_logs"

    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False
    )

    # Foreign key to task
    task_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tasks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Associated task ID"
    )

    # Log entry details
    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
        comment="Log entry time"
    )

    level = Column(
        Enum(LogLevel, name="log_level", values_callable=lambda obj: [e.value for e in obj]),
        nullable=False,
        default=LogLevel.INFO,
        index=True,
        comment="Log severity level"
    )

    message = Column(
        Text,
        nullable=False,
        comment="Log message"
    )

    # Additional context (stored as JSONB)
    context = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional context including step, action, and details"
    )

    # Table configuration
    __table_args__ = (
        # Composite indexes for common queries
        Index('idx_task_logs_task_timestamp', 'task_id', 'timestamp'),
        Index('idx_task_logs_task_level', 'task_id', 'level'),
        Index('idx_task_logs_level_timestamp', 'level', 'timestamp'),

        # GIN index for JSONB queries
        Index('idx_task_logs_context', 'context', postgresql_using='gin'),
    )

    def to_dict(self) -> dict:
        """Convert log entry to dictionary."""
        return {
            "id": str(self.id),
            "task_id": str(self.task_id),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "level": self.level.value if self.level else None,
            "message": self.message,
            "context": self.context or {},
        }

    def __repr__(self) -> str:
        return f"<TaskLog {self.id}: [{self.level}] {self.message[:50]}...>"