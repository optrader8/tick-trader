"""
FastAPI main application entry point.

Backend server for Tick Trader web platform.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.routes import router as api_router

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Tick Trader API",
    description="Backend API for Tick Trader web platform",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Tick Trader API server...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"CORS origins: {settings.BACKEND_CORS_ORIGINS}")

    # Import here to avoid circular imports
    from app.services.database import database_service
    from app.services.redis import redis_service

    # Initialize database
    await database_service.connect()
    logger.info("Database connected")

    # Initialize Redis
    await redis_service.connect()
    logger.info("Redis connected")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Tick Trader API server...")

    from app.services.database import database_service
    from app.services.redis import redis_service

    await database_service.disconnect()
    await redis_service.disconnect()
    logger.info("Server shutdown complete")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(
        content={
            "status": "ok",
            "environment": settings.ENVIRONMENT,
        }
    )


# Include API routes
app.include_router(api_router, prefix="/api")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL.lower(),
    )
