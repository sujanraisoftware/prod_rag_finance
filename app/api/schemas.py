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


class VectorSearchRequest(BaseModel):
    query: str = Field(..., description="Financial question or search query (e.g. 'Operating cash flow FY 2025')")
    limit: int = Field(default=5, ge=1, le=50, description="Number of chunks to retrieve")
    doc_id: str | None = Field(default=None, description="Optional document ID filter for document-specific queries")
    is_table_only: bool = Field(default=False, description="Filter only tabular balance sheets/income statements")
    page_number: int | None = Field(default=None, description="Filter to a specific page number")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum cosine similarity threshold")


class SearchResultItem(BaseModel):
    chunk_id: str
    similarity_score: float
    doc_id: str
    filename: str
    page_number: int
    total_pages: int
    chunk_type: str
    is_table: bool
    section_header: str | None = None
    section_breadcrumb: list[str] = Field(default_factory=list)
    content: str
    raw_content: str
    token_estimate: int


class VectorSearchResponse(BaseModel):
    query: str
    total_results: int
    execution_time_ms: float
    results: list[SearchResultItem]


class AnswerRequest(BaseModel):
    query: str = Field(..., description="Financial question (e.g. 'What was the total revenue in FY 2025?')")
    doc_id: str | None = Field(default=None, description="Optional document ID to isolate search to a specific filing")
    top_k: int = Field(default=4, ge=1, le=10, description="Final number of top reranked chunks fed to the LLM")
    initial_fetch_k: int = Field(default=15, ge=5, le=50, description="Number of candidate chunks retrieved from Milvus before reranking")
    is_table_only: bool = Field(default=False, description="Filter only tabular balance sheets and financial statements")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum vector similarity threshold")
    model: str | None = Field(default=None, description="Groq model override (e.g. 'llama-3.3-70b-versatile', 'llama-3.1-8b-instant')")
    temperature: float | None = Field(default=None, ge=0.0, le=1.0, description="LLM sampling temperature (default: 0.0 for deterministic answers)")


class SourceCitation(BaseModel):
    chunk_id: str
    page_number: int
    filename: str
    section_header: str | None = None
    section_breadcrumb: list[str] = Field(default_factory=list)
    is_table: bool
    rerank_score: float
    content_snippet: str


class AnswerResponse(BaseModel):
    query: str
    doc_id: str | None = None
    answer: str
    model_used: str
    context_tokens_estimated: int
    sources: list[SourceCitation]
    latency_ms: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    worker_backend: str
    celery_connected: bool
    milvus_connected: bool
    milvus_total_vectors: int
    upload_dir_writable: bool


