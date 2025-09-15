"""Application settings and configuration."""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Settings
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AIOpsLab Task Execution API"

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://aiopslab:aiopslab@localhost:5432/aiopslab"
    )
    DATABASE_ECHO: bool = False

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # CORS
    CORS_ORIGINS: List[str] = ["*"]

    # Features
    ENABLE_DOCS: bool = True
    ENABLE_METRICS: bool = True
    ENABLE_BACKGROUND_TASKS: bool = True

    # Task Settings
    DEFAULT_MAX_STEPS: int = 30
    DEFAULT_TIMEOUT_MINUTES: int = 30
    DEFAULT_PRIORITY: int = 5
    TIMEOUT_CHECK_INTERVAL: int = 60  # seconds

    # Worker Settings
    WORKER_HEARTBEAT_TIMEOUT: int = 60  # seconds
    WORKER_OFFLINE_THRESHOLD: int = 120  # seconds
    WORKER_POLL_INTERVAL: int = 5  # seconds
    NUM_INTERNAL_WORKERS: int = 3  # Number of internal workers to start
    AUTO_START_WORKERS: bool = True  # Auto-start internal workers

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json or text

    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    model_config = {
        "env_file": ".env",
        "case_sensitive": True
    }


settings = Settings()