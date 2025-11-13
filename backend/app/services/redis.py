"""
Redis service for caching and job queue.
"""

import redis.asyncio as redis
from app.core.config import settings
import logging
import json

logger = logging.getLogger(__name__)


class RedisService:
    """Redis service."""

    def __init__(self):
        self.client = None

    async def connect(self):
        """Connect to Redis."""
        self.client = await redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True
        )

        # Test connection
        await self.client.ping()
        logger.info("Redis connected")

    async def disconnect(self):
        """Disconnect from Redis."""
        if self.client:
            await self.client.close()
            logger.info("Redis connection closed")

    # Job queue methods
    async def enqueue_job(self, job_data: dict):
        """Add job to queue."""
        await self.client.rpush('jobs:pending', json.dumps(job_data))

    async def dequeue_job(self) -> dict:
        """Get job from queue."""
        job = await self.client.lpop('jobs:pending')
        return json.loads(job) if job else None

    async def set_job_status(self, job_id: str, status: dict):
        """Set job status."""
        await self.client.setex(
            f'job:{job_id}:status',
            86400,  # 24 hours
            json.dumps(status)
        )

    async def get_job_status(self, job_id: str) -> dict:
        """Get job status."""
        status = await self.client.get(f'job:{job_id}:status')
        return json.loads(status) if status else None

    async def add_job_log(self, job_id: str, log_entry: str):
        """Add log entry for job."""
        await self.client.rpush(f'job:{job_id}:logs', log_entry)

    async def get_job_logs(self, job_id: str) -> list:
        """Get job logs."""
        logs = await self.client.lrange(f'job:{job_id}:logs', 0, -1)
        return logs


# Global instance
redis_service = RedisService()
