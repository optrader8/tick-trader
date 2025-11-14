"""
File handling service for uploads, extraction, and metadata management.
"""

import zipfile
import tarfile
import gzip
import shutil
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from datetime import datetime
import pandas as pd

from app.core.config import settings
from app.models.database import File
from app.services.database import database_service

logger = logging.getLogger(__name__)


class FileHandler:
    """Handle file operations including upload, extraction, and metadata extraction."""

    def __init__(self):
        self.upload_dir = settings.upload_directory
        self.extracted_dir = settings.extracted_directory

    async def save_file(
        self,
        content: bytes,
        original_filename: str,
        file_id: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """
        Save uploaded file and extract if it's an archive.

        Args:
            content: File content in bytes
            original_filename: Original filename from upload
            file_id: Optional UUID for the file

        Returns:
            Dict with file metadata and extracted files info
        """
        if file_id is None:
            file_id = uuid4()

        # Determine file type
        file_ext = Path(original_filename).suffix.lower()
        file_type = self._get_file_type(file_ext)

        # Calculate checksum
        checksum = hashlib.sha256(content).hexdigest()
        file_size = len(content)

        # Save main file
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{file_id}{file_ext}"
        file_path = self.upload_dir / filename

        with open(file_path, 'wb') as f:
            f.write(content)

        logger.info(f"File saved: {filename} ({file_size} bytes, type: {file_type})")

        # Extract metadata
        metadata = await self._extract_metadata(file_path, file_type)

        # Save to database
        file_record = await self._save_to_db(
            file_id=file_id,
            filename=filename,
            original_name=original_filename,
            file_path=str(file_path),
            file_size=file_size,
            file_type=file_type,
            checksum=checksum,
            metadata=metadata
        )

        result = {
            "file_id": str(file_id),
            "filename": filename,
            "original_name": original_filename,
            "file_size": file_size,
            "file_type": file_type,
            "checksum": checksum,
            "metadata": metadata,
            "uploaded_at": datetime.now().isoformat()
        }

        # Extract if archive
        if file_type in ['zip', 'tar', 'gz', 'tar.gz']:
            extracted_files = await self._extract_archive(
                file_path, file_id, file_type
            )
            result["extracted_files"] = extracted_files

            # Update is_extracted flag
            await self._update_extracted_flag(file_id)

        return result

    def _get_file_type(self, file_ext: str) -> str:
        """Determine file type from extension."""
        ext_map = {
            '.csv': 'csv',
            '.json': 'json',
            '.txt': 'txt',
            '.parquet': 'parquet',
            '.zip': 'zip',
            '.gz': 'gz',
            '.tar': 'tar',
            '.tar.gz': 'tar.gz',
            '.tgz': 'tar.gz'
        }
        return ext_map.get(file_ext, 'unknown')

    async def _extract_metadata(
        self,
        file_path: Path,
        file_type: str
    ) -> Dict[str, Any]:
        """Extract metadata from file based on type."""
        metadata = {}

        try:
            if file_type == 'csv':
                # Read CSV and extract metadata
                df = pd.read_csv(file_path, nrows=5)
                metadata = {
                    "columns": list(df.columns),
                    "column_count": len(df.columns),
                    "preview_rows": df.head().to_dict('records')
                }

                # Get row count efficiently
                with open(file_path, 'r') as f:
                    metadata["row_count"] = sum(1 for _ in f) - 1  # Exclude header

            elif file_type == 'json':
                # Read JSON and extract structure
                with open(file_path, 'r') as f:
                    data = json.load(f)

                if isinstance(data, list):
                    metadata = {
                        "type": "array",
                        "item_count": len(data),
                        "preview": data[:3] if len(data) > 0 else []
                    }
                elif isinstance(data, dict):
                    metadata = {
                        "type": "object",
                        "keys": list(data.keys()),
                        "key_count": len(data.keys())
                    }

            elif file_type == 'parquet':
                # Read Parquet and extract metadata
                df = pd.read_parquet(file_path)
                metadata = {
                    "columns": list(df.columns),
                    "column_count": len(df.columns),
                    "row_count": len(df),
                    "preview_rows": df.head().to_dict('records')
                }

            elif file_type == 'txt':
                # Get line count for text files
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    metadata = {
                        "line_count": len(lines),
                        "preview": lines[:10]
                    }

            elif file_type == 'zip':
                # List files in ZIP
                with zipfile.ZipFile(file_path, 'r') as zf:
                    metadata = {
                        "file_count": len(zf.namelist()),
                        "files": zf.namelist()
                    }

        except Exception as e:
            logger.warning(f"Failed to extract metadata from {file_path}: {e}")
            metadata["error"] = str(e)

        return metadata

    async def _extract_archive(
        self,
        archive_path: Path,
        parent_id: UUID,
        archive_type: str
    ) -> List[Dict[str, Any]]:
        """Extract archive and save extracted files to database."""
        extracted_files = []
        extract_dir = self.extracted_dir / str(parent_id)
        extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            if archive_type == 'zip':
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    zf.extractall(extract_dir)
                    file_list = zf.namelist()

            elif archive_type in ['tar', 'tar.gz']:
                mode = 'r:gz' if archive_type == 'tar.gz' else 'r'
                with tarfile.open(archive_path, mode) as tf:
                    tf.extractall(extract_dir)
                    file_list = tf.getnames()

            elif archive_type == 'gz':
                # Single file gzip
                output_path = extract_dir / archive_path.stem
                with gzip.open(archive_path, 'rb') as f_in:
                    with open(output_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                file_list = [archive_path.stem]

            else:
                logger.error(f"Unsupported archive type: {archive_type}")
                return []

            # Process each extracted file
            for filename in file_list:
                file_path = extract_dir / filename

                # Skip directories
                if not file_path.is_file():
                    continue

                # Read file content
                with open(file_path, 'rb') as f:
                    content = f.read()

                file_id = uuid4()
                file_ext = file_path.suffix.lower()
                file_type = self._get_file_type(file_ext)
                checksum = hashlib.sha256(content).hexdigest()
                file_size = len(content)

                # Extract metadata for extracted file
                metadata = await self._extract_metadata(file_path, file_type)

                # Save to database with parent reference
                await self._save_to_db(
                    file_id=file_id,
                    filename=filename,
                    original_name=filename,
                    file_path=str(file_path),
                    file_size=file_size,
                    file_type=file_type,
                    checksum=checksum,
                    parent_file_id=parent_id,
                    metadata=metadata
                )

                extracted_files.append({
                    "file_id": str(file_id),
                    "filename": filename,
                    "file_size": file_size,
                    "file_type": file_type,
                    "metadata": metadata
                })

                logger.info(f"Extracted and saved: {filename}")

        except Exception as e:
            logger.error(f"Failed to extract archive {archive_path}: {e}")
            raise

        return extracted_files

    async def _save_to_db(
        self,
        file_id: UUID,
        filename: str,
        original_name: str,
        file_path: str,
        file_size: int,
        file_type: str,
        checksum: str,
        parent_file_id: Optional[UUID] = None,
        metadata: Optional[Dict] = None
    ) -> File:
        """Save file metadata to database."""
        async with database_service.get_session() as session:
            file_record = File(
                id=file_id,
                filename=filename,
                original_name=original_name,
                file_path=file_path,
                file_size=file_size,
                file_type=file_type,
                checksum=checksum,
                parent_file_id=parent_file_id,
                is_extracted=0,
                metadata=metadata or {}
            )

            session.add(file_record)
            await session.commit()
            await session.refresh(file_record)

            return file_record

    async def _update_extracted_flag(self, file_id: UUID):
        """Update is_extracted flag for archive file."""
        async with database_service.get_session() as session:
            result = await session.execute(
                f"UPDATE files SET is_extracted = 1 WHERE id = '{file_id}'"
            )
            await session.commit()

    async def get_file_by_id(self, file_id: UUID) -> Optional[File]:
        """Get file metadata from database."""
        async with database_service.get_session() as session:
            result = await session.execute(
                f"SELECT * FROM files WHERE id = '{file_id}'"
            )
            return result.first()

    async def list_files(
        self,
        parent_id: Optional[UUID] = None,
        file_type: Optional[str] = None
    ) -> List[File]:
        """List files with optional filters."""
        async with database_service.get_session() as session:
            query = "SELECT * FROM files WHERE 1=1"

            if parent_id:
                query += f" AND parent_file_id = '{parent_id}'"
            else:
                query += " AND parent_file_id IS NULL"  # Only root files

            if file_type:
                query += f" AND file_type = '{file_type}'"

            query += " ORDER BY uploaded_at DESC"

            result = await session.execute(query)
            return result.fetchall()

    async def delete_file(self, file_id: UUID) -> bool:
        """Delete file and its extracted files."""
        async with database_service.get_session() as session:
            # Get file record
            result = await session.execute(
                f"SELECT * FROM files WHERE id = '{file_id}'"
            )
            file_record = result.first()

            if not file_record:
                return False

            # Delete physical file
            file_path = Path(file_record.file_path)
            if file_path.exists():
                file_path.unlink()

            # Delete extracted files if any
            if file_record.is_extracted:
                extract_dir = self.extracted_dir / str(file_id)
                if extract_dir.exists():
                    shutil.rmtree(extract_dir)

                # Delete extracted file records
                await session.execute(
                    f"DELETE FROM files WHERE parent_file_id = '{file_id}'"
                )

            # Delete file record
            await session.execute(
                f"DELETE FROM files WHERE id = '{file_id}'"
            )
            await session.commit()

            return True


# Global instance
file_handler = FileHandler()
