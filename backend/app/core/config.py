"""
Configuration settings for the application.
"""

from typing import List
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, field_validator
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings."""

    # Environment
    ENVIRONMENT: str = "development"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "ticktrader"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""

    # File Storage
    UPLOAD_DIR: str = "../aiend/data/raw"
    EXTRACTED_DIR: str = "../aiend/data/extracted"
    MAX_FILE_SIZE: int = 524288000  # 500MB (increased for zip files)
    ALLOWED_EXTENSIONS: str = ".csv,.parquet,.json,.txt,.zip,.gz,.tar,.tar.gz"

    # AI Engine
    AIEND_PATH: str = "../aiend"
    PYTHON_VENV_PATH: str = "../aiend/venv"

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Logging
    LOG_LEVEL: str = "INFO"

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v

    @property
    def database_url(self) -> str:
        """Get database URL."""
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def async_database_url(self) -> str:
        """Get async database URL."""
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def redis_url(self) -> str:
        """Get Redis URL."""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}"

    @property
    def upload_directory(self) -> Path:
        """Get upload directory path."""
        return Path(self.UPLOAD_DIR).resolve()

    @property
    def extracted_directory(self) -> Path:
        """Get extracted files directory path."""
        return Path(self.EXTRACTED_DIR).resolve()

    @property
    def aiend_directory(self) -> Path:
        """Get aiend directory path."""
        return Path(self.AIEND_PATH).resolve()

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
