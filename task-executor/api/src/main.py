"""Main FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
from typing import AsyncGenerator

from .config.settings import settings
from .config.logging import get_logger
from .models import init_db, close_db
from .api import tasks, workers, workers_internal, health, llm_conversations
from .middleware.error_handler import error_handler_middleware
from .middleware.request_id import request_id_middleware
from .monitoring.metrics import PrometheusMiddleware, metrics_endpoint

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifecycle manager."""
    logger.info("application.startup", version=settings.VERSION)

    await init_db()
    logger.info("database.connected")

    # Start internal workers
    if settings.AUTO_START_WORKERS:
        from .workers.manager import WorkerManager
        app.state.worker_manager = WorkerManager(num_workers=settings.NUM_INTERNAL_WORKERS)
        await app.state.worker_manager.start()
        logger.info("internal.workers.started", num_workers=settings.NUM_INTERNAL_WORKERS)

    if settings.ENABLE_BACKGROUND_TASKS:
        app.state.timeout_checker = asyncio.create_task(check_timeouts_periodically())
        logger.info("background.tasks.started")

    yield

    # Stop workers
    if hasattr(app.state, "worker_manager"):
        await app.state.worker_manager.stop()
        logger.info("internal.workers.stopped")

    if hasattr(app.state, "timeout_checker"):
        app.state.timeout_checker.cancel()
        try:
            await app.state.timeout_checker
        except asyncio.CancelledError:
            pass

    await close_db()
    logger.info("application.shutdown")


async def check_timeouts_periodically():
    """Background task to check for timed out tasks."""
    from .models import get_db
    from .services.task_service import TaskService

    while True:
        try:
            async with get_db() as session:
                service = TaskService(session)
                timeout_tasks = await service.check_timeouts()

                if timeout_tasks:
                    logger.info(
                        "timeout.checker.processed",
                        count=len(timeout_tasks)
                    )

            await asyncio.sleep(settings.TIMEOUT_CHECK_INTERVAL)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(
                "timeout.checker.error",
                error=str(e)
            )
            await asyncio.sleep(settings.TIMEOUT_CHECK_INTERVAL)


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="AIOpsLab Task Execution API",
        description="RESTful API for scalable AIOpsLab task execution",
        version=settings.VERSION,
        lifespan=lifespan,
        docs_url="/docs" if settings.ENABLE_DOCS else None,
        redoc_url="/redoc" if settings.ENABLE_DOCS else None,
    )

    app.middleware("http")(error_handler_middleware)
    app.middleware("http")(request_id_middleware)

    if settings.ENABLE_METRICS:
        app.add_middleware(PrometheusMiddleware)
        app.add_api_route("/metrics", metrics_endpoint, methods=["GET"])

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, tags=["Health"])
    app.include_router(tasks.router, prefix="/api/v1", tags=["Tasks"])
    app.include_router(workers.router, prefix="/api/v1", tags=["Workers"])
    app.include_router(workers_internal.router, prefix="/api/v1", tags=["Internal Workers"])
    app.include_router(llm_conversations.router, tags=["LLM Conversations"])

    return app


app = create_app()