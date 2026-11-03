from typing import Any
from pydantic import BaseModel, Field
from app.domain.models import DocumentMetadata, FinancialChunk, TaskStatusEnum


class UploadResponse(BaseModel):
    message: str = Field(..., json_schema_extra={"example": "Financial document uploaded successfully. Ingestion queued."})
    task_id: str
    doc_id: str
    filename: str
    file_hash: str
    page_count: int
    status_url: str
    chunks_url: str


class TaskStatusResponse(BaseModel):
    task_id: str
    doc_id: str
    filename: str
    status: TaskStatusEnum
    current_step: str
    progress_percent: float
    started_at: str
    completed_at: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None


class DocumentListResponse(BaseModel):
    total_documents: int
    documents: list[DocumentMetadata]


class ChunksResponse(BaseModel):
    doc_id: str
    total_chunks: int
    count: int
    offset: int
    limit: int
    chunks: list[FinancialChunk]


class InstantParseResponse(BaseModel):
    doc_id: str
    filename: str
    stats: dict[str, Any]
    total_chunks: int
    table_chunks: int
    text_chunks: int
    chunks: list[FinancialChunk]


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    worker_backend: str
    celery_connected: bool
    upload_dir_writable: bool
