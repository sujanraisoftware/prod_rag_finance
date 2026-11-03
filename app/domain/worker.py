import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import os
from pathlib import Path
import time
from typing import Any
from celery import Celery
import redis

from app.config import settings
from app.core.logging import logger
from app.domain.embeddings import embedding_service
from app.domain.models import (
    DocumentMetadata,
    TaskProgress,
    TaskStatusEnum,
    ChunkType
)
from app.domain.rag.chunker import FinancialChunker
from app.domain.rag.extractor import FastFinanceExtractor
from app.domain.storage import document_store
from app.domain.vector_store import milvus_store


# Celery App setup for distributed production workers
celery_app = Celery(
    "fintech_rag",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=settings.CELERY_TASK_TIMEOUT_SECONDS,
    broker_connection_retry_on_startup=True,
)

# In-process worker thread pool for standalone / dev / test modes
_thread_pool = ThreadPoolExecutor(max_workers=4)


def process_financial_document_sync(
    file_path: str,
    doc_id: str,
    filename: str,
    task_id: str,
    file_hash: str = ""
) -> dict[str, Any]:
    """
    Synchronous core processing logic for financial document parsing and chunking.
    Used by both Celery workers and in-process background tasks.
    """
    path_obj = Path(file_path)
    start_time = time.perf_counter()

    try:
        # Step 1: Extraction via PyMuPDF4LLM
        document_store.update_task_progress(
            task_id=task_id,
            status=TaskStatusEnum.PROCESSING,
            step="Extracting text and tables via PyMuPDF4LLM",
            percent=25.0
        )

        extractor = FastFinanceExtractor(
            enable_layout=settings.ENABLE_LAYOUT_PARSER,
            table_strategy=settings.TABLE_STRATEGY,
            table_output=settings.TABLE_OUTPUT,
            top_margin=settings.HEADER_MARGIN,
            bottom_margin=settings.FOOTER_MARGIN,
        )

        pages, stats = extractor.extract_document(path_obj)
        page_count = len(pages)

        # Step 2: Financial Semantic Chunking & Table Preservation
        document_store.update_task_progress(
            task_id=task_id,
            status=TaskStatusEnum.PROCESSING,
            step="Executing financial table preservation and semantic chunking",
            percent=50.0
        )

        chunker = FinancialChunker(
            max_tokens=settings.CHUNK_MAX_TOKENS,
            overlap_tokens=settings.CHUNK_OVERLAP_TOKENS,
            preserve_tables=settings.PRESERVE_TABLES_ATOMICALLY,
            inject_grounding_header=settings.INJECT_GROUNDING_HEADER
        )

        chunks = chunker.chunk_pages(doc_id=doc_id, filename=filename, pages=pages)
        total_chunks = len(chunks)
        table_chunks = sum(1 for c in chunks if c.chunk_type == ChunkType.TABLE)
        text_chunks = total_chunks - table_chunks

        # Step 3: Generate Dense Vector Embeddings
        document_store.update_task_progress(
            task_id=task_id,
            status=TaskStatusEnum.PROCESSING,
            step="Generating dense vector embeddings with ONNX",
            percent=70.0
        )

        chunk_texts = [c.content for c in chunks]
        embeddings = embedding_service.embed_documents(chunk_texts)

        # Step 4: Index into Milvus Vector Database & Local Audit Storage
        document_store.update_task_progress(
            task_id=task_id,
            status=TaskStatusEnum.PROCESSING,
            step="Indexing financial vectors into Milvus collection",
            percent=88.0
        )

        indexed_vectors = milvus_store.insert_chunks(chunks=chunks, embeddings=embeddings)

        duration = time.perf_counter() - start_time
        doc_metadata = DocumentMetadata(
            doc_id=doc_id,
            filename=filename,
            file_hash=file_hash or "N/A",
            page_count=page_count,
            file_size_bytes=path_obj.stat().st_size if path_obj.exists() else 0,
            processed_at=datetime.now(timezone.utc).isoformat(),
            processing_time_seconds=round(duration, 4),
            total_chunks=total_chunks,
            table_chunks_count=table_chunks,
            text_chunks_count=text_chunks,
            status=TaskStatusEnum.COMPLETED
        )

        document_store.save_document(doc_metadata, chunks)

        result_payload = {
            "doc_id": doc_id,
            "filename": filename,
            "page_count": page_count,
            "total_chunks": total_chunks,
            "table_chunks": table_chunks,
            "text_chunks": text_chunks,
            "vectors_indexed": indexed_vectors,
            "milvus_collection": milvus_store.collection_name,
            "duration_seconds": round(duration, 4),
            "pages_per_sec": round(page_count / max(duration, 0.0001), 2)
        }

        document_store.update_task_progress(
            task_id=task_id,
            status=TaskStatusEnum.COMPLETED,
            step="Completed successfully",
            percent=100.0,
            result=result_payload
        )

        logger.info(
            f"Successfully processed financial document '{filename}' ({doc_id}) "
            f"into {total_chunks} chunks ({table_chunks} tables) in {duration:.3f}s"
        )
        return result_payload

    except Exception as e:
        logger.error(f"Error processing document '{filename}' ({doc_id}): {str(e)}", exc_info=True)
        document_store.update_task_progress(
            task_id=task_id,
            status=TaskStatusEnum.FAILED,
            step="Processing failed",
            percent=100.0,
            error=str(e)
        )
        raise e

    finally:
        # Step 4: Clean up temporary uploaded file
        if path_obj.exists():
            try:
                os.remove(path_obj)
                logger.debug(f"Removed temporary file {path_obj}")
            except Exception as cleanup_err:
                logger.warning(f"Failed to remove temporary file {path_obj}: {str(cleanup_err)}")


@celery_app.task(name="process_financial_document_celery", bind=True)
def process_financial_document_celery(
    self,
    file_path: str,
    doc_id: str,
    filename: str,
    task_id: str,
    file_hash: str = ""
) -> dict[str, Any]:
    """Celery task entry point for distributed workers."""
    return process_financial_document_sync(
        file_path=file_path,
        doc_id=doc_id,
        filename=filename,
        task_id=task_id,
        file_hash=file_hash
    )


class UnifiedTaskManager:
    """
    Manages task dispatching across Celery and in-process background workers.
    Ensures seamless operation regardless of whether an external Redis server is running.
    """

    @staticmethod
    def is_celery_available() -> bool:
        """Checks if Celery broker (Redis) is actively responsive."""
        if settings.WORKER_BACKEND == "in_process":
            return False
        try:
            r = redis.Redis.from_url(settings.CELERY_BROKER_URL, socket_connect_timeout=0.5)
            return bool(r.ping())
        except Exception:
            return False

    @classmethod
    def dispatch_task(
        cls,
        file_path: str,
        doc_id: str,
        filename: str,
        task_id: str,
        file_hash: str = ""
    ) -> str:
        """
        Dispatches document parsing task to either Celery or In-Process ThreadPool.
        """
        # Initialize task progress in store
        task_progress = TaskProgress(
            task_id=task_id,
            doc_id=doc_id,
            filename=filename,
            status=TaskStatusEnum.PENDING,
            current_step="Queued for ingestion",
            progress_percent=0.0
        )
        document_store.save_task(task_progress)

        use_celery = (settings.WORKER_BACKEND == "celery") or (
            settings.WORKER_BACKEND == "auto" and cls.is_celery_available()
        )

        if use_celery:
            logger.info(f"Dispatching task {task_id} to Celery worker cluster")
            process_financial_document_celery.apply_async(
                args=[file_path, doc_id, filename, task_id, file_hash],
                task_id=task_id
            )
        else:
            logger.info(f"Dispatching task {task_id} to local asynchronous worker pool")
            _thread_pool.submit(
                process_financial_document_sync,
                file_path,
                doc_id,
                filename,
                task_id,
                file_hash
            )

        return task_id

    @staticmethod
    def get_task_status(task_id: str) -> TaskProgress:
        """Retrieves real-time status of a parsing task."""
        return document_store.get_task(task_id)


task_manager = UnifiedTaskManager()
