"""
Analysis management endpoints.
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from typing import Dict
import logging
import json

from app.models.schemas import (
    StartAnalysisRequest,
    AnalysisStatusResponse,
    ApiResponse
)
from app.services.aiengine import ai_engine_service
from app.services.redis import redis_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/start")
async def start_analysis(request: StartAnalysisRequest):
    """
    Start AI analysis job.

    Creates a new analysis job and adds it to the queue.
    """
    try:
        # TODO: Validate dataset exists in database

        # Start analysis
        job_id = await ai_engine_service.start_analysis(
            dataset_path=f"../aiend/data/raw/{request.dataset_id}",
            model_type=request.model_type,
            config=request.config.model_dump()
        )

        logger.info(f"Analysis started: {job_id}")

        return {
            "success": True,
            "data": {
                "job_id": job_id,
                "status": "pending"
            }
        }

    except Exception as e:
        logger.error(f"Start analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{job_id}/status")
async def get_analysis_status(job_id: str):
    """
    Get analysis job status.

    Returns current status, progress, and logs.
    """
    try:
        status = await ai_engine_service.get_job_status(job_id)

        if not status:
            raise HTTPException(status_code=404, detail="Job not found")

        # Get logs
        logs = await redis_service.get_job_logs(job_id)

        return {
            "success": True,
            "data": {
                **status,
                "logs": logs[-50:]  # Last 50 log entries
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get status error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get status")


@router.post("/{job_id}/cancel")
async def cancel_analysis(job_id: str):
    """Cancel running analysis job."""
    try:
        await ai_engine_service.cancel_job(job_id)

        return {
            "success": True,
            "message": "Job cancelled"
        }

    except Exception as e:
        logger.error(f"Cancel job error: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel job")


@router.websocket("/ws/{job_id}")
async def websocket_analysis_status(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time analysis status updates.
    """
    await websocket.accept()
    logger.info(f"WebSocket connected for job {job_id}")

    try:
        while True:
            # Get current status
            status = await ai_engine_service.get_job_status(job_id)

            if status:
                await websocket.send_json(status)

                # Stop if job completed or failed
                if status.get('status') in ['completed', 'failed', 'cancelled']:
                    break

            # Wait before next update
            import asyncio
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()
