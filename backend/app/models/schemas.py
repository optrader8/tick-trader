"""
Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID
from enum import Enum


class AnalysisStatus(str, Enum):
    """Analysis status enum."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# File schemas
class FileUploadResponse(BaseModel):
    """File upload response."""
    file_id: str
    filename: str
    original_name: str
    file_size: int
    uploaded_at: datetime

    class Config:
        from_attributes = True


class FileMetadata(BaseModel):
    """File metadata."""
    id: UUID
    filename: str
    original_name: str
    file_size: int
    mime_type: Optional[str] = None
    uploaded_at: datetime


# Analysis schemas
class ModelConfig(BaseModel):
    """Model configuration."""
    sequence_length: Optional[int] = None
    lstm_units: Optional[List[int]] = None
    dropout_rate: Optional[float] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    d_model: Optional[int] = None
    n_heads: Optional[int] = None
    num_layers: Optional[int] = None


class StartAnalysisRequest(BaseModel):
    """Start analysis request."""
    dataset_id: str
    model_type: str = Field(..., description="Model type: lstm, transformer, ensemble, cnn_lstm")
    config: ModelConfig

    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, v):
        allowed = ['lstm', 'transformer', 'ensemble', 'cnn_lstm']
        if v not in allowed:
            raise ValueError(f"model_type must be one of {allowed}")
        return v


class AnalysisStatusResponse(BaseModel):
    """Analysis status response."""
    id: UUID
    status: AnalysisStatus
    progress: int
    current_step: Optional[str] = None
    estimated_time_remaining: Optional[int] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class AnalysisResultsResponse(BaseModel):
    """Analysis results response."""
    analysis_id: UUID
    metrics: Dict[str, Any]
    charts_data: Optional[Dict[str, Any]] = None
    download_urls: Dict[str, str]


# System schemas
class SystemStatus(BaseModel):
    """System status."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    running_jobs: int


# API Response
class ApiResponse(BaseModel):
    """Standard API response."""
    success: bool
    data: Optional[Any] = None
    error: Optional[Dict[str, str]] = None
