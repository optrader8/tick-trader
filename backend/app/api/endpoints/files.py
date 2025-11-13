"""
File management endpoints.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import Optional, List
import logging
from uuid import UUID

from app.core.config import settings
from app.models.schemas import (
    FileUploadResponse,
    FileMetadata,
    ApiResponse,
    FileListResponse
)
from app.services.file_handler import file_handler

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload data file with support for multiple formats.

    Supported formats:
    - Data files: CSV, Parquet, JSON, TXT
    - Archives: ZIP, TAR, TAR.GZ, GZ

    Archives are automatically extracted and individual files are saved.
    """
    try:
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()

        # Handle .tar.gz specially
        if file.filename.lower().endswith('.tar.gz'):
            file_ext = '.tar.gz'

        allowed_exts = settings.ALLOWED_EXTENSIONS.split(',')

        if file_ext not in allowed_exts:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} not allowed. Allowed: {allowed_exts}"
            )

        # Check file size
        content = await file.read()
        file_size = len(content)

        if file_size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {settings.MAX_FILE_SIZE / 1024 / 1024:.0f}MB"
            )

        # Save file and extract if needed
        result = await file_handler.save_file(
            content=content,
            original_filename=file.filename
        )

        logger.info(f"File uploaded successfully: {file.filename}")

        return ApiResponse(
            success=True,
            data=result
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"File upload failed: {str(e)}"
        )


@router.get("/list")
async def list_files(
    parent_id: Optional[str] = Query(None, description="Filter by parent file ID"),
    file_type: Optional[str] = Query(None, description="Filter by file type")
):
    """
    List all uploaded files with optional filters.

    Query parameters:
    - parent_id: Show only files extracted from this archive
    - file_type: Filter by file type (csv, json, zip, etc.)
    """
    try:
        parent_uuid = UUID(parent_id) if parent_id else None
        files = await file_handler.list_files(
            parent_id=parent_uuid,
            file_type=file_type
        )

        file_list = []
        for file_record in files:
            file_list.append({
                "id": str(file_record.id),
                "filename": file_record.filename,
                "original_name": file_record.original_name,
                "file_size": file_record.file_size,
                "file_type": file_record.file_type,
                "is_extracted": bool(file_record.is_extracted),
                "parent_file_id": str(file_record.parent_file_id) if file_record.parent_file_id else None,
                "metadata": file_record.metadata,
                "uploaded_at": file_record.uploaded_at.isoformat()
            })

        return ApiResponse(
            success=True,
            data={"files": file_list, "count": len(file_list)}
        )

    except Exception as e:
        logger.error(f"List files error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list files")


@router.get("/{file_id}")
async def get_file(file_id: str):
    """Get file metadata by ID."""
    try:
        file_uuid = UUID(file_id)
        file_record = await file_handler.get_file_by_id(file_uuid)

        if not file_record:
            raise HTTPException(status_code=404, detail="File not found")

        return ApiResponse(
            success=True,
            data={
                "id": str(file_record.id),
                "filename": file_record.filename,
                "original_name": file_record.original_name,
                "file_size": file_record.file_size,
                "file_type": file_record.file_type,
                "checksum": file_record.checksum,
                "is_extracted": bool(file_record.is_extracted),
                "parent_file_id": str(file_record.parent_file_id) if file_record.parent_file_id else None,
                "metadata": file_record.metadata,
                "uploaded_at": file_record.uploaded_at.isoformat()
            }
        )

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file ID format")
    except Exception as e:
        logger.error(f"Get file error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get file")


@router.delete("/{file_id}")
async def delete_file(file_id: str):
    """
    Delete uploaded file and all extracted files if it's an archive.
    """
    try:
        file_uuid = UUID(file_id)
        success = await file_handler.delete_file(file_uuid)

        if not success:
            raise HTTPException(status_code=404, detail="File not found")

        logger.info(f"File deleted: {file_id}")

        return ApiResponse(
            success=True,
            data={"message": "File and extracted files deleted successfully"}
        )

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete file error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete file")
