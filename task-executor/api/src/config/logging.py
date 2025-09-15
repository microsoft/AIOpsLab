"""Logging configuration with structlog."""

import logging
import sys
from typing import Any, Dict

import structlog
from structlog.types import EventDict, Processor


def add_task_context(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add task context to log entries."""
    from contextvars import ContextVar
    
    task_id_var: ContextVar[str] = ContextVar("task_id", default="")
    worker_id_var: ContextVar[str] = ContextVar("worker_id", default="")
    
    if task_id := task_id_var.get():
        event_dict["task_id"] = task_id
    if worker_id := worker_id_var.get():
        event_dict["worker_id"] = worker_id
    
    return event_dict


def setup_logging(log_level: str = "INFO", json_logs: bool = False) -> None:
    """Configure structlog for the application."""
    
    timestamper = structlog.processors.TimeStamper(fmt="iso")
    
    shared_processors: list[Processor] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        add_task_context,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        timestamper,
    ]
    
    if json_logs:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=renderer,
            foreign_pre_chain=shared_processors,
        )
    )
    
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Silence noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a configured logger instance."""
    return structlog.get_logger(name)


# Context managers for request/task tracking
from contextvars import ContextVar
from contextlib import contextmanager
from typing import Generator

task_id_var: ContextVar[str] = ContextVar("task_id", default="")
worker_id_var: ContextVar[str] = ContextVar("worker_id", default="")
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


@contextmanager
def task_context(task_id: str) -> Generator[None, None, None]:
    """Context manager to set task ID for logging."""
    token = task_id_var.set(task_id)
    try:
        yield
    finally:
        task_id_var.reset(token)


@contextmanager
def worker_context(worker_id: str) -> Generator[None, None, None]:
    """Context manager to set worker ID for logging."""
    token = worker_id_var.set(worker_id)
    try:
        yield
    finally:
        worker_id_var.reset(token)


@contextmanager
def request_context(request_id: str) -> Generator[None, None, None]:
    """Context manager to set request ID for logging."""
    token = request_id_var.set(request_id)
    try:
        yield
    finally:
        request_id_var.reset(token)