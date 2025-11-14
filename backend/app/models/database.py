"""
Database models using SQLAlchemy.
"""

from sqlalchemy import Column, String, Integer, BigInteger, DateTime, Text, JSON, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid
import enum

Base = declarative_base()


class AnalysisStatus(str, enum.Enum):
    """Analysis status enum."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class File(Base):
    """File metadata table."""
    __tablename__ = "files"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    original_name = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    file_size = Column(BigInteger, nullable=False)
    file_type = Column(String(50))  # csv, json, txt, zip, parquet, etc.
    mime_type = Column(String(100))
    checksum = Column(String(64))
    is_extracted = Column(Integer, default=0)  # 0: no, 1: yes (for zip files)
    parent_file_id = Column(UUID(as_uuid=True), nullable=True)  # Reference to parent zip file
    metadata = Column(JSON)  # Additional metadata (rows, columns, date range, etc.)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    created_by = Column(UUID(as_uuid=True))


class Analysis(Base):
    """Analysis table."""
    __tablename__ = "analyses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), nullable=False)
    model_type = Column(String(50), nullable=False)
    config = Column(JSON, nullable=False)
    status = Column(
        Enum(AnalysisStatus),
        default=AnalysisStatus.PENDING,
        nullable=False
    )
    progress = Column(Integer, default=0)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    error_message = Column(Text)
    results_path = Column(Text)
    created_by = Column(UUID(as_uuid=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AnalysisResult(Base):
    """Analysis results table."""
    __tablename__ = "analysis_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_id = Column(UUID(as_uuid=True), nullable=False)
    metrics = Column(JSON, nullable=False)
    predictions_path = Column(Text)
    charts_data = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
