"""Database models for AIOpsLab Task Execution API."""

# Import enums directly as they don't require database
from .enums import TaskStatus, WorkerStatus, LogLevel

# Lazy imports for database-dependent objects
def __getattr__(name):
    """Lazy import database objects to avoid initialization at import time."""
    if name == "Base":
        from .database import Base
        return Base
    elif name == "engine":
        from .database import engine
        return engine
    elif name == "async_session":
        from .database import async_session
        return async_session
    elif name == "get_db":
        from .database import get_db
        return get_db
    elif name == "init_db":
        from .database import init_db
        return init_db
    elif name == "close_db":
        from .database import close_db
        return close_db
    elif name == "Task":
        from .task import Task
        return Task
    elif name == "Worker":
        from .worker import Worker
        return Worker
    elif name == "TaskLog":
        from .task_log import TaskLog
        return TaskLog
    elif name == "LLMConversation":
        from .llm_conversation import LLMConversation
        return LLMConversation
    elif name == "MessageRole":
        from .llm_conversation import MessageRole
        return MessageRole
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Database
    "Base",
    "engine",
    "async_session",
    "get_db",
    "init_db",
    "close_db",
    # Models
    "Task",
    "TaskStatus",
    "Worker",
    "WorkerStatus",
    "TaskLog",
    "LogLevel",
    "LLMConversation",
    "MessageRole",
]