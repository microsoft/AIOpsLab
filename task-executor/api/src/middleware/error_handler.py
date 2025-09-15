"""Error handling middleware."""

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import traceback
import uuid
from datetime import datetime

from ..config.logging import get_logger

logger = get_logger(__name__)


async def error_handler_middleware(request: Request, call_next):
    """Global error handler middleware."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    try:
        response = await call_next(request)
        return response

    except Exception as e:
        logger.error(
            "unhandled_exception",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            error=str(e),
            traceback=traceback.format_exc()
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )