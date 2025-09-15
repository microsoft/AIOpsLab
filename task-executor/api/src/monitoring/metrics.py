"""Prometheus metrics for monitoring."""

from fastapi import Request, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time
from typing import Callable

from ..config.logging import get_logger

logger = get_logger(__name__)

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"]
)

task_queue_size = Gauge(
    "task_queue_size",
    "Number of tasks in queue by status",
    ["status"]
)

worker_count = Gauge(
    "worker_count",
    "Number of workers by status",
    ["status"]
)


class PrometheusMiddleware:
    """Middleware to collect Prometheus metrics."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        path = scope["path"]
        method = scope["method"]

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status = message["status"]
                duration = time.time() - start_time

                http_requests_total.labels(
                    method=method,
                    endpoint=path,
                    status=status
                ).inc()

                http_request_duration_seconds.labels(
                    method=method,
                    endpoint=path
                ).observe(duration)

            await send(message)

        await self.app(scope, receive, send_wrapper)


async def metrics_endpoint(request: Request) -> Response:
    """Endpoint to expose Prometheus metrics."""
    try:
        from ..models import get_db
        from ..services.task_service import TaskService
        from ..services.worker_service import WorkerService

        async with get_db() as session:
            task_service = TaskService(session)
            queue_stats = await task_service.get_queue_stats()

            for status, count in queue_stats.items():
                task_queue_size.labels(status=status).set(count)

            worker_service = WorkerService(session)
            workers = await worker_service.list_workers()

            from ..models import WorkerStatus
            for status in WorkerStatus:
                count = sum(1 for w in workers if w.status == status)
                worker_count.labels(status=status.value).set(count)

    except Exception as e:
        logger.error("metrics.collection.error", error=str(e))

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )