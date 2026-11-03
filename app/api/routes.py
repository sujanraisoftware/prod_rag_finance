import os
from pathlib import Path
import shutil
import time
import uuid
from typing import Annotated
from fastapi import APIRouter, File, UploadFile, Query, status
from fastapi.responses import JSONResponse

from app.api.schemas import (
    UploadResponse,
    TaskStatusResponse,
    DocumentListResponse,
    ChunksResponse,
    InstantParseResponse,
    VectorSearchRequest,
    VectorSearchResponse,
    SearchResultItem,
    HealthResponse,
)
from app.config import settings
from app.core.exceptions import InvalidPDFException
from app.core.logging import logger
from app.core.security import sanitize_filename, validate_pdf_file
from app.domain.embeddings import embedding_service
from app.domain.models import ChunkType, DocumentMetadata
from app.domain.rag.chunker import FinancialChunker
from app.domain.rag.extractor import FastFinanceExtractor
from app.domain.storage import document_store
from app.domain.vector_store import milvus_store
from app.domain.worker import task_manager, UnifiedTaskManager

router = APIRouter(prefix="/api/v1", tags=["Financial Ingestion"])


@router.post(
    "/documents/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload financial document for asynchronous parsing",
    description="Accepts PDF financial filings (10-K, 10-Q, 8-K, earnings reports), validates integrity, and dispatches background parsing via PyMuPDF4LLM.",
)
async def upload_document(file: Annotated[UploadFile, File(...)]):
    # Ensure upload directory exists
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Sanitize filename & generate IDs
    clean_name = sanitize_filename(file.filename or "financial_document.pdf")
    doc_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())
    temp_file_path = settings.UPLOAD_DIR / f"{doc_id}_{clean_name}"

    # 2. Stream upload to disk to maintain low memory profile
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to write uploaded file to disk: {str(e)}")
        raise InvalidPDFException(f"Failed to receive uploaded file: {str(e)}")

    # 3. Validate PDF integrity, magic bytes, encryption, size
    try:
        validation_info = validate_pdf_file(
            file_path=temp_file_path,
            filename=clean_name,
            max_size_mb=settings.MAX_UPLOAD_SIZE_MB
        )
    except Exception as validation_err:
        # Cleanup invalid file immediately
        if temp_file_path.exists():
            os.remove(temp_file_path)
        raise validation_err

    # 4. Dispatch background ingestion task
    task_manager.dispatch_task(
        file_path=str(temp_file_path),
        doc_id=doc_id,
        filename=clean_name,
        task_id=task_id,
        file_hash=validation_info["file_hash"]
    )

    return UploadResponse(
        message="Financial document uploaded successfully. Ingestion queued.",
        task_id=task_id,
        doc_id=doc_id,
        filename=clean_name,
        file_hash=validation_info["file_hash"],
        page_count=validation_info["page_count"],
        status_url=f"/api/v1/tasks/{task_id}",
        chunks_url=f"/api/v1/documents/{doc_id}/chunks"
    )


@router.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
    summary="Get ingestion task status",
    description="Check the real-time parsing progress, status, and completion metrics of an ingestion task.",
)
async def get_task_status(task_id: str):
    progress = task_manager.get_task_status(task_id)
    return TaskStatusResponse(
        task_id=progress.task_id,
        doc_id=progress.doc_id,
        filename=progress.filename,
        status=progress.status,
        current_step=progress.current_step,
        progress_percent=progress.progress_percent,
        started_at=progress.started_at,
        completed_at=progress.completed_at,
        error=progress.error,
        result=progress.result
    )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all ingested financial documents",
)
async def list_documents():
    docs = document_store.list_documents()
    return DocumentListResponse(total_documents=len(docs), documents=docs)


@router.get(
    "/documents/{doc_id}",
    response_model=DocumentMetadata,
    summary="Get document metadata",
)
async def get_document(doc_id: str):
    return document_store.get_document(doc_id)


@router.get(
    "/documents/{doc_id}/chunks",
    response_model=ChunksResponse,
    summary="Get extracted financial chunks",
    description="Retrieve parsed and structured chunks for vector database insertion or LLM inspection. Supports filtering by chunk type (table vs text) and pagination.",
)
async def get_document_chunks(
    doc_id: str,
    chunk_type: ChunkType | None = Query(default=None, description="Filter chunks by type: 'table' or 'text'"),
    page_number: int | None = Query(default=None, description="Filter chunks by 1-indexed page number"),
    limit: int = Query(default=50, ge=1, le=500, description="Page limit"),
    offset: int = Query(default=0, ge=0, description="Offset index")
):
    chunks, total = document_store.get_chunks(
        doc_id=doc_id,
        chunk_type=chunk_type,
        page_number=page_number,
        limit=limit,
        offset=offset
    )
    return ChunksResponse(
        doc_id=doc_id,
        total_chunks=total,
        count=len(chunks),
        offset=offset,
        limit=limit,
        chunks=chunks
    )


@router.post(
    "/documents/parse-instant",
    response_model=InstantParseResponse,
    summary="Synchronous instant parse",
    description="Directly parses and chunks an uploaded financial PDF in real time. Ideal for low-latency queries or testing.",
)
async def parse_instant(file: Annotated[UploadFile, File(...)]):
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    clean_name = sanitize_filename(file.filename or "instant_document.pdf")
    doc_id = str(uuid.uuid4())
    temp_file_path = settings.UPLOAD_DIR / f"instant_{doc_id}_{clean_name}"

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        validate_pdf_file(
            file_path=temp_file_path,
            filename=clean_name,
            max_size_mb=settings.MAX_UPLOAD_SIZE_MB
        )

        extractor = FastFinanceExtractor(
            enable_layout=settings.ENABLE_LAYOUT_PARSER,
            table_strategy=settings.TABLE_STRATEGY,
            table_output=settings.TABLE_OUTPUT,
            top_margin=settings.HEADER_MARGIN,
            bottom_margin=settings.FOOTER_MARGIN,
        )
        pages, stats = extractor.extract_document(temp_file_path)

        chunker = FinancialChunker(
            max_tokens=settings.CHUNK_MAX_TOKENS,
            overlap_tokens=settings.CHUNK_OVERLAP_TOKENS,
            preserve_tables=settings.PRESERVE_TABLES_ATOMICALLY,
            inject_grounding_header=settings.INJECT_GROUNDING_HEADER
        )
        chunks = chunker.chunk_pages(doc_id=doc_id, filename=clean_name, pages=pages)

        table_chunks = sum(1 for c in chunks if c.chunk_type == ChunkType.TABLE)
        text_chunks = len(chunks) - table_chunks

        return InstantParseResponse(
            doc_id=doc_id,
            filename=clean_name,
            stats=stats,
            total_chunks=len(chunks),
            table_chunks=table_chunks,
            text_chunks=text_chunks,
            chunks=chunks
        )
    finally:
        if temp_file_path.exists():
            os.remove(temp_file_path)


@router.post(
    "/search",
    response_model=VectorSearchResponse,
    summary="Semantic vector search across financial chunks",
    description="Embeds the user query and searches the Milvus vector database with optional scalar filtering by doc_id, page_number, and table_only flag.",
)
async def vector_search(request: VectorSearchRequest):
    t0 = time.perf_counter()

    # 1. Generate query embedding
    query_vector = embedding_service.embed_query(request.query)

    # 2. Search Milvus with metadata filters
    hits = milvus_store.search(
        query_vector=query_vector,
        limit=request.limit,
        doc_id=request.doc_id,
        is_table_only=request.is_table_only,
        page_number=request.page_number,
        min_score=request.min_score
    )

    duration_ms = (time.perf_counter() - t0) * 1000

    results = [SearchResultItem.model_validate(h) for h in hits]
    return VectorSearchResponse(
        query=request.query,
        total_results=len(results),
        execution_time_ms=round(duration_ms, 2),
        results=results
    )


@router.delete(
    "/documents/{doc_id}",
    summary="Delete an ingested document, its audit chunks, and Milvus vectors",
)
async def delete_document(doc_id: str):
    # 1. Delete from local metadata store
    document_store.delete_document(doc_id)
    # 2. Delete vectors from Milvus collection
    deleted_vectors = milvus_store.delete_by_doc_id(doc_id)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "message": f"Document {doc_id} and its associated Milvus vectors deleted successfully.",
            "vectors_removed": deleted_vectors
        }
    )

