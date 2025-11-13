"""
Database service for managing connections.
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class DatabaseService:
    """Database service."""

    def __init__(self):
        self.engine = None
        self.async_session = None

    async def connect(self):
        """Connect to database."""
        self.engine = create_async_engine(
            settings.async_database_url,
            echo=settings.ENVIRONMENT == "development",
            pool_size=20,
            max_overflow=0,
        )

        self.async_session = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

        logger.info("Database engine created")

    async def disconnect(self):
        """Disconnect from database."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connection closed")

    async def get_session(self) -> AsyncSession:
        """Get database session."""
        async with self.async_session() as session:
            yield session


# Global instance
database_service = DatabaseService()
