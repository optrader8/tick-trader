"""
File management endpoints.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import hashlib
from datetime import datetime
import logging
from uuid import uuid4

from app.core.config import settings
from app.models.schemas import FileUploadResponse, ApiResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload tick data file.

    Accepts CSV, Parquet, and JSON files.
    """
    try:
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
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
                detail=f"File too large. Max size: {settings.MAX_FILE_SIZE} bytes"
            )

        # Generate unique filename
        file_id = str(uuid4())
        filename = f"{file_id}{file_ext}"

        # Save file
        upload_dir = settings.upload_directory
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_path = upload_dir / filename
        with open(file_path, 'wb') as f:
            f.write(content)

        # Calculate checksum
        checksum = hashlib.sha256(content).hexdigest()

        logger.info(f"File uploaded: {filename} ({file_size} bytes)")

        # TODO: Save to database

        return FileUploadResponse(
            file_id=file_id,
            filename=filename,
            original_name=file.filename,
            file_size=file_size,
            uploaded_at=datetime.now()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(status_code=500, detail="File upload failed")


@router.get("/list")
async def list_files():
    """List all uploaded files."""
    try:
        upload_dir = settings.upload_directory

        if not upload_dir.exists():
            return {"files": []}

        files = []
        for file_path in upload_dir.glob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    "filename": file_path.name,
                    "size": stat.st_size,
                    "uploaded_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })

        return {"files": files}

    except Exception as e:
        logger.error(f"List files error: {e}")
        raise HTTPException(status_code=500, detail="Failed to list files")


@router.delete("/{file_id}")
async def delete_file(file_id: str):
    """Delete uploaded file."""
    try:
        upload_dir = settings.upload_directory

        # Find file
        file_found = False
        for file_path in upload_dir.glob(f"{file_id}*"):
            if file_path.is_file():
                file_path.unlink()
                file_found = True
                logger.info(f"File deleted: {file_path.name}")

        if not file_found:
            raise HTTPException(status_code=404, detail="File not found")

        return {"success": True, "message": "File deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete file error: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete file")
