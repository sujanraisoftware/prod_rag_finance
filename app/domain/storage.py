import json
import threading
from pathlib import Path
from typing import Any
from app.config import settings
from app.core.exceptions import DocumentNotFoundException, TaskNotFoundException
from app.core.logging import logger
from app.domain.models import DocumentMetadata, FinancialChunk, TaskProgress, TaskStatusEnum, ChunkType


class DocumentStorage:
    """
    Thread-safe storage manager for financial documents, chunks, and task states.
    Maintains an in-memory hot cache and persists records to JSON files on disk.
    """

    def __init__(self, base_storage_dir: Path = settings.STORAGE_DIR):
        self.storage_dir = base_storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

        # In-memory indexes
        self._documents: dict[str, DocumentMetadata] = {}
        self._chunks: dict[str, list[FinancialChunk]] = {}
        self._tasks: dict[str, TaskProgress] = {}

        self._load_existing_from_disk()

    def _load_existing_from_disk(self) -> None:
        """Loads previously processed documents from storage directory on startup."""
        for file in self.storage_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    meta = DocumentMetadata.model_validate(data["metadata"])
                    chunks = [FinancialChunk.model_validate(c) for c in data.get("chunks", [])]
                    self._documents[meta.doc_id] = meta
                    self._chunks[meta.doc_id] = chunks
            except Exception as e:
                logger.warning(f"Could not load previous document from {file.name}: {str(e)}")

    def save_task(self, task: TaskProgress) -> None:
        with self._lock:
            self._tasks[task.task_id] = task

    def update_task_progress(
        self,
        task_id: str,
        status: TaskStatusEnum,
        step: str,
        percent: float,
        error: str | None = None,
        result: dict[str, Any] | None = None
    ) -> TaskProgress:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                task = TaskProgress(
                    task_id=task_id,
                    doc_id="unknown",
                    filename="unknown"
                )
                self._tasks[task_id] = task

            task.status = status
            task.current_step = step
            task.progress_percent = round(percent, 2)
            if error:
                task.error = error
            if result:
                task.result = result
            return task

    def get_task(self, task_id: str) -> TaskProgress:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                raise TaskNotFoundException(task_id)
            return task

    def save_document(self, metadata: DocumentMetadata, chunks: list[FinancialChunk]) -> None:
        with self._lock:
            self._documents[metadata.doc_id] = metadata
            self._chunks[metadata.doc_id] = chunks

            # Persist to disk
            file_path = self.storage_dir / f"{metadata.doc_id}.json"
            try:
                payload = {
                    "metadata": metadata.model_dump(),
                    "chunks": [c.model_dump() for c in chunks]
                }
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to persist document {metadata.doc_id} to disk: {str(e)}")

    def get_document(self, doc_id: str) -> DocumentMetadata:
        with self._lock:
            doc = self._documents.get(doc_id)
            if not doc:
                raise DocumentNotFoundException(doc_id)
            return doc

    def list_documents(self) -> list[DocumentMetadata]:
        with self._lock:
            return list(self._documents.values())

    def get_chunks(
        self,
        doc_id: str,
        chunk_type: ChunkType | None = None,
        page_number: int | None = None,
        limit: int = 50,
        offset: int = 0
    ) -> tuple[list[FinancialChunk], int]:
        with self._lock:
            if doc_id not in self._chunks:
                raise DocumentNotFoundException(doc_id)

            chunks = self._chunks[doc_id]

            if chunk_type is not None:
                chunks = [c for c in chunks if c.chunk_type == chunk_type]

            if page_number is not None:
                chunks = [c for c in chunks if c.page_number == page_number]

            total_count = len(chunks)
            paginated = chunks[offset : offset + limit]
            return paginated, total_count

    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id in self._documents:
                del self._documents[doc_id]
            if doc_id in self._chunks:
                del self._chunks[doc_id]

            disk_file = self.storage_dir / f"{doc_id}.json"
            if disk_file.exists():
                disk_file.unlink(missing_ok=True)
            return True


# Global document storage singleton
document_store = DocumentStorage()

