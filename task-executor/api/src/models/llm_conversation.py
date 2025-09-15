"""LLM Conversation model for storing AI agent interactions."""

from sqlalchemy import (
    Column, String, DateTime, Text, Enum, Index, ForeignKey, func, Integer
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
import enum

from .database import Base


class MessageRole(enum.Enum):
    """Role of the message in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class LLMConversation(Base):
    """Model for storing complete LLM conversation history."""

    __tablename__ = "llm_conversations"

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

    # Conversation metadata
    session_id = Column(
        UUID(as_uuid=True),
        nullable=False,
        default=uuid.uuid4,
        index=True,
        comment="Unique session identifier for this conversation"
    )

    started_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="When conversation started"
    )

    ended_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When conversation ended"
    )

    # Model information
    model_name = Column(
        String(100),
        nullable=True,
        comment="LLM model used (e.g., gpt-4, claude-3)"
    )

    model_config = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Model configuration (temperature, max_tokens, etc.)"
    )

    # Conversation statistics
    total_messages = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Total number of messages in conversation"
    )

    total_tokens = Column(
        Integer,
        nullable=True,
        comment="Total tokens consumed"
    )

    total_cost = Column(
        JSONB,
        nullable=True,
        comment="Cost breakdown (input_tokens, output_tokens, total_cost)"
    )

    # Complete conversation history
    messages = Column(
        JSONB,
        nullable=False,
        default=list,
        comment="Complete message history"
    )

    # Additional metadata
    conversation_metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional metadata (tools used, errors, etc.)"
    )

    # Table configuration
    __table_args__ = (
        # Composite indexes for common queries
        Index('idx_llm_conversations_task_session', 'task_id', 'session_id'),
        Index('idx_llm_conversations_task_started', 'task_id', 'started_at'),

        # GIN indexes for JSONB queries
        Index('idx_llm_conversations_messages', 'messages', postgresql_using='gin'),
        Index('idx_llm_conversations_metadata', 'conversation_metadata', postgresql_using='gin'),
    )

    def add_message(self, role: MessageRole, content: str, **kwargs):
        """Add a message to the conversation."""
        if not isinstance(self.messages, list):
            self.messages = []

        message = {
            "timestamp": datetime.utcnow().isoformat(),
            "role": role.value,
            "content": content,
            **kwargs
        }

        self.messages.append(message)
        self.total_messages += 1

        return message

    def to_dict(self) -> dict:
        """Convert conversation to dictionary."""
        return {
            "id": str(self.id),
            "task_id": str(self.task_id),
            "session_id": str(self.session_id),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "model_name": self.model_name,
            "model_config": self.model_config or {},
            "total_messages": self.total_messages,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "messages": self.messages or [],
            "metadata": self.conversation_metadata or {}
        }

    def __repr__(self) -> str:
        return f"<LLMConversation {self.id}: {self.total_messages} messages>"

