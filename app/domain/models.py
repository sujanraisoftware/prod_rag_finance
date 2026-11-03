from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    MIXED = "mixed"
    HEADER = "header"


class TaskStatusEnum(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class FinancialChunk(BaseModel):
    """
    Production-grade financial chunk structure with rich provenance metadata
    to eliminate hallucinations during RAG generation.
    """
    chunk_id: str = Field(..., description="Unique chunk identifier (e.g. {doc_id}_p{page}_c{index})")
    doc_id: str = Field(..., description="Parent document identifier")
    filename: str = Field(..., description="Original filename")
    page_number: int = Field(..., description="1-indexed page number in the original PDF")
    total_pages: int = Field(..., description="Total pages in the document")
    section_header: str | None = Field(default=None, description="Immediate heading under which this chunk falls")
    section_breadcrumb: list[str] = Field(default_factory=list, description="Full hierarchy stack (e.g. ['Item 8', 'Consolidated Statements'])")
    chunk_type: ChunkType = Field(default=ChunkType.TEXT, description="Type of content in this chunk")
    content: str = Field(..., description="Enriched chunk content formatted with context headers for LLM retrieval")
    raw_content: str = Field(..., description="Original extracted content without prompt metadata headers")
    is_table: bool = Field(default=False, description="Flag indicating if this chunk contains financial tabular data")
    token_estimate: int = Field(default=0, description="Estimated token count (approx 4 chars per token)")
    char_count: int = Field(default=0, description="Character count")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional custom metadata")


class DocumentMetadata(BaseModel):
    """
    Metadata record for an ingested financial document.
    """
    doc_id: str
    filename: str
    file_hash: str = Field(..., description="SHA-256 hash of the original PDF")
    page_count: int
    file_size_bytes: int
    uploaded_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    processed_at: str | None = None
    processing_time_seconds: float | None = None
    total_chunks: int = 0
    table_chunks_count: int = 0
    text_chunks_count: int = 0
    status: TaskStatusEnum = TaskStatusEnum.PENDING
    error_message: str | None = None


class TaskProgress(BaseModel):
    """
    Real-time progress record for asynchronous ingestion tasks.
    """
    task_id: str
    doc_id: str
    filename: str
    status: TaskStatusEnum = TaskStatusEnum.PENDING
    current_step: str = "Initialized"
    progress_percent: float = 0.0
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None

