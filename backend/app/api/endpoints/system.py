"""
System monitoring endpoints.
"""

from fastapi import APIRouter, HTTPException
import psutil
import logging

from app.models.schemas import SystemStatus

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/status", response_model=SystemStatus)
async def get_system_status():
    """
    Get system resource usage.

    Returns CPU, memory, and disk usage.
    """
    try:
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Get running jobs count from Redis
        from app.services.redis import redis_service
        running_jobs = await redis_service.client.llen('jobs:running') if redis_service.client else 0

        return SystemStatus(
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            disk_usage=disk.percent,
            running_jobs=running_jobs
        )

    except Exception as e:
        logger.error(f"System status error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system status")


@router.get("/health")
async def health_check():
    """Detailed health check."""
    try:
        from app.services.database import database_service
        from app.services.redis import redis_service

        # Check database
        db_healthy = database_service.engine is not None

        # Check Redis
        redis_healthy = False
        if redis_service.client:
            try:
                await redis_service.client.ping()
                redis_healthy = True
            except:
                pass

        return {
            "status": "healthy" if (db_healthy and redis_healthy) else "degraded",
            "services": {
                "database": "ok" if db_healthy else "error",
                "redis": "ok" if redis_healthy else "error"
            }
        }

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
