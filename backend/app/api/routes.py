"""
API routes aggregation.
"""

from fastapi import APIRouter
from app.api.endpoints import files, analysis, system

router = APIRouter()

router.include_router(files.router, prefix="/files", tags=["files"])
router.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
router.include_router(system.router, prefix="/system", tags=["system"])
