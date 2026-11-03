from contextlib import asynccontextmanager
import os
from pathlib import Path
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router as api_router
from app.api.schemas import HealthResponse
from app.config import settings
from app.core.exceptions import (
    FinTechRAGException,
    InvalidPDFException,
    FileSizeExceededException,
    DocumentNotFoundException,
    TaskNotFoundException,
)
from app.core.logging import logger
from app.domain.worker import UnifiedTaskManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Starting {settings.APP_NAME} v{settings.APP_VERSION} "
        f"[Worker Backend: {settings.WORKER_BACKEND}, Celery Broker: {settings.CELERY_BROKER_URL}]"
    )
    yield
    # Shutdown actions
    logger.info(f"Shutting down {settings.APP_NAME}")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
# Production Financial RAG Ingestion Engine
High-throughput, C-accelerated financial PDF parsing pipeline powered by **PyMuPDF4LLM**.

### Core Capabilities:
- **Sub-second Ingestion**: Replaces heavy OCR/layout vision pipelines (Docling) with C-native PyMuPDF4LLM, eliminating CPU/memory spikes.
- **Financial Table Preservation**: Extracts tabular balance sheets and income statements atomically or with header preservation across sub-chunks.
- **Section Hierarchy & Zero-Hallucination Grounding**: Automatically detects SEC Items/Notes and injects provenance headers into chunk metadata.
- **Dual Worker Architecture**: Supports distributed Celery clusters for scale, with transparent fallback to asynchronous background workers.
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for FinTech frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Exception Handlers
@app.exception_handler(FinTechRAGException)
async def handle_fintech_exception(request: Request, exc: FinTechRAGException):
    logger.warning(f"Domain exception on {request.url.path}: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "details": exc.details
        }
    )


@app.exception_handler(Exception)
async def handle_unexpected_exception(request: Request, exc: Exception):
    logger.error(f"Unexpected server error on {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected server error occurred during document processing.",
            "details": {"error_type": exc.__class__.__name__}
        }
    )


# Mount API Routes
app.include_router(api_router)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    celery_ok = UnifiedTaskManager.is_celery_available()
    upload_writable = settings.UPLOAD_DIR.exists() and os.access(settings.UPLOAD_DIR, os.W_OK)

    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        worker_backend=settings.WORKER_BACKEND,
        celery_connected=celery_ok,
        upload_dir_writable=upload_writable
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
