import os
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, Request, Response, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from app.api.routes import router as api_router
from app.api.schemas import HealthResponse
from app.config import settings
from app.core.exceptions import DocumentNotFoundException, FinTechRAGException
from app.core.llm_service import LLMError, llm_service
from app.core.logging import logger
from app.domain.embeddings import embedding_service
from app.domain.reranker import reranker_service
from app.domain.storage import document_store
from app.domain.vector_store import milvus_store
from app.domain.worker import UnifiedTaskManager

MAX_PROMPT_CHARS = int(getattr(settings, "MAX_PROMPT_CHARS", 2000))
MAX_UPLOAD_MB = int(getattr(settings, "MAX_UPLOAD_MB", 50))
ENABLE_DOCS = bool(getattr(settings, "ENABLE_DOCS", True))
CORS_ORIGINS: list[str] = list(getattr(settings, "CORS_ALLOW_ORIGINS", []) or [])

# Doc IDs already verified as indexed in Milvus (avoids a probe query on every request)
_synced_docs: set[str] = set()
_sync_lock = threading.Lock()

_CSP = (
    "default-src 'self'; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src https://fonts.gstatic.com; "
    "script-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; connect-src 'self'; "
    "frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    if not llm_service.is_configured:
        logger.error("GROQ_API_KEY is not set: queries will fail until it is configured.")
    logger.info(
        f"Starting {settings.APP_NAME} v{settings.APP_VERSION} "
        f"[Worker Backend: {settings.WORKER_BACKEND}, Milvus URI: {settings.MILVUS_URI}]"
    )
    yield
    logger.info(f"Shutting down {settings.APP_NAME}")
    try:
        await llm_service.aclose()
    finally:
        milvus_store.close()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
# Financial RAG Ingestion & Query Engine
Financial PDF parsing pipeline powered by **PyMuPDF4LLM**, with Milvus retrieval, FlashRank reranking and Groq inference.

- **Fast ingestion** via C-native PyMuPDF4LLM
- **Table preservation** for balance sheets and income statements
- **Section provenance** (SEC Items/Notes) injected into chunk metadata
- **Dual worker architecture**: Celery with automatic fallback to background workers
    """,
    lifespan=lifespan,
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None,
    openapi_url="/openapi.json" if ENABLE_DOCS else None,
)

app.add_middleware(GZipMiddleware, minimum_size=1024)

# The bundled UI is same-origin. Only enable CORS for explicitly configured origins.
if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials="*" not in CORS_ORIGINS,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.error(f"[{request_id}] unhandled error {request.method} {request.url.path}", exc_info=True)
        raise
    elapsed_ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    # Swagger/ReDoc load assets from a CDN, so skip the strict CSP there.
    if request.url.path not in ("/docs", "/redoc"):
        response.headers["Content-Security-Policy"] = _CSP
    if request.url.path not in ("/health", "/health/live"):
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} -> {response.status_code} ({elapsed_ms:.0f}ms)"
        )
    return response


# ---------------------------------------------------------------- Exception handlers
@app.exception_handler(FinTechRAGException)
async def handle_fintech_exception(request: Request, exc: FinTechRAGException):
    logger.warning(f"Domain exception on {request.url.path}: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.__class__.__name__, "message": exc.message, "details": exc.details},
    )


@app.exception_handler(Exception)
async def handle_unexpected_exception(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "-")
    logger.error(f"[{request_id}] Unexpected error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected server error occurred.",
            "request_id": request_id,
        },
    )


app.include_router(api_router)

# ---------------------------------------------------------------- UI
templates = Jinja2Templates(directory="templates")


def _render(request: Request, status_code: int = 200, **ctx) -> Response:
    base = {
        "documents": document_store.list_documents(),
        "selected_doc_id": "",
        "user_prompt": "",
        "result": None,
        "model_used": None,
        "context_tokens": None,
        "sources": [],
        "timing": None,
        "is_table_only": False,
        "error": None,
        "truncated": False,
        "max_prompt_chars": MAX_PROMPT_CHARS,
        "max_upload_mb": MAX_UPLOAD_MB,
        "request_id": getattr(request.state, "request_id", None),
    }
    base.update(ctx)
    return templates.TemplateResponse(
        request=request, name="index.html", context=base, status_code=status_code
    )


def _ensure_indexed(doc_id: str) -> None:
    """Blocking. Verify the doc exists and its chunks are in Milvus; backfill if not."""
    if doc_id in _synced_docs:
        return
    with _sync_lock:
        if doc_id in _synced_docs:
            return
        document_store.get_document(doc_id)  # raises DocumentNotFoundException
        try:
            dummy_vec = [0.0] * settings.EMBEDDING_DIMENSION
            existing = milvus_store.search(
                query_vector=dummy_vec, limit=1, doc_id=doc_id, min_score=-1.0
            )
            if not existing:
                chunks, _ = document_store.get_chunks(doc_id, limit=2000)
                if chunks:
                    embs = embedding_service.embed_documents([c.content for c in chunks])
                    milvus_store.insert_chunks(chunks=chunks, embeddings=embs)
                    logger.info(f"Backfilled {len(chunks)} chunks into Milvus for doc {doc_id}")
            _synced_docs.add(doc_id)
        except Exception as e:
            # Don't mark as synced so the next request retries.
            logger.warning(f"Milvus auto-sync failed for doc {doc_id}: {e}")


def _retrieve(prompt: str, doc_id: str | None, is_table_only: bool) -> dict:
    """Blocking retrieval pipeline: embed -> vector search -> rerank -> compress."""
    if doc_id:
        _ensure_indexed(doc_id)

    t0 = time.perf_counter()
    query_vec = embedding_service.embed_query(prompt)
    hits = milvus_store.search(
        query_vector=query_vec,
        limit=settings.RERANKER_INITIAL_TOP_K,
        doc_id=doc_id,
        is_table_only=is_table_only,
    )
    retrieval_ms = (time.perf_counter() - t0) * 1000
    if not hits:
        return {"hits": [], "retrieval_ms": retrieval_ms}

    t0 = time.perf_counter()
    reranked = reranker_service.rerank(
        query=prompt, candidates=hits, top_k=settings.RERANKER_FINAL_TOP_K
    )
    rerank_ms = (time.perf_counter() - t0) * 1000

    context, est_tokens = reranker_service.build_compressed_context(
        reranked, max_tokens=settings.MAX_CONTEXT_TOKENS
    )
    return {
        "hits": hits,
        "reranked": reranked,
        "context": context,
        "est_tokens": est_tokens,
        "retrieval_ms": retrieval_ms,
        "rerank_ms": rerank_ms,
    }


def _build_sources(reranked: list[dict]) -> list[dict]:
    sources = []
    for c in reranked:
        raw = (c.get("raw_content") or c.get("content") or "").strip()
        sources.append(
            {
                "chunk_id": c.get("chunk_id", ""),
                "page_number": int(c.get("page_number", 1)),
                "filename": c.get("filename", ""),
                "section_header": c.get("section_header"),
                "is_table": bool(c.get("is_table", False)),
                "rerank_score": float(c.get("rerank_score", 0.0)),
                "content_snippet": raw[:300] + ("..." if len(raw) > 300 else ""),
            }
        )
    return sources


@app.get("/", tags=["UI"])
async def render_query_ui(request: Request):
    return _render(request)


@app.post("/", tags=["UI"])
async def handle_query_form(
    request: Request,
    prompt: str = Form(...),
    doc_id: str | None = Form(None),
    is_table_only: bool = Form(False),
):
    """Query pipeline: (auto-sync) -> Milvus search -> FlashRank rerank -> compress -> Groq."""
    request_id = getattr(request.state, "request_id", "-")
    prompt = prompt.strip()
    target_doc_id = doc_id.strip() if doc_id and doc_id.strip() else None
    form_state = {
        "selected_doc_id": target_doc_id or "",
        "user_prompt": prompt,
        "is_table_only": is_table_only,
    }

    if not prompt:
        return _render(request, 400, error="Please enter a question.", **form_state)
    if len(prompt) > MAX_PROMPT_CHARS:
        return _render(
            request, 400,
            error=f"Your question is too long ({len(prompt)} characters). The limit is {MAX_PROMPT_CHARS}.",
            **form_state,
        )

    t0 = time.perf_counter()
    try:
        # Embedding, Milvus and reranking are blocking: keep them off the event loop.
        retrieved = await run_in_threadpool(_retrieve, prompt, target_doc_id, is_table_only)

        if not retrieved["hits"]:
            total_ms = (time.perf_counter() - t0) * 1000
            return _render(
                request,
                result="The document does not contain any relevant information regarding your query.",
                timing={
                    "retrieval_ms": round(retrieved["retrieval_ms"], 1),
                    "rerank_ms": 0.0,
                    "llm_ms": 0.0,
                    "total_ms": round(total_ms, 1),
                },
                **form_state,
            )

        t0_llm = time.perf_counter()
        llm_out = await llm_service.generate_financial_answer(
            query=prompt, context=retrieved["context"]
        )
        llm_ms = (time.perf_counter() - t0_llm) * 1000
        total_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            f"[{request_id}] query ok doc={target_doc_id or 'ALL'} prompt_chars={len(prompt)} "
            f"sources={len(retrieved['reranked'])} total_ms={total_ms:.0f}"
        )
        return _render(
            request,
            result=llm_out["answer"],
            model_used=llm_out["model"],
            truncated=llm_out.get("truncated", False),
            context_tokens=retrieved["est_tokens"],
            sources=_build_sources(retrieved["reranked"]),
            timing={
                "retrieval_ms": round(retrieved["retrieval_ms"], 1),
                "rerank_ms": round(retrieved["rerank_ms"], 1),
                "llm_ms": round(llm_ms, 1),
                "total_ms": round(total_ms, 1),
            },
            **form_state,
        )

    except DocumentNotFoundException as exc:
        return _render(request, 404, error=exc.message, **{**form_state, "selected_doc_id": ""})
    except LLMError as exc:
        return _render(request, 503, error=exc.user_message, **form_state)
    except Exception:
        logger.error(f"[{request_id}] query pipeline failed", exc_info=True)
        return _render(
            request, 500,
            error="Something went wrong while processing your question. Please try again.",
            **form_state,
        )


# ---------------------------------------------------------------- Health
@app.get("/health/live", tags=["Health"])
async def liveness():
    """Process is up. Use for container liveness probes."""
    return {"status": "alive"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(response: Response):
    """Readiness: returns 503 if Milvus or the upload directory is unavailable."""

    def _probe():
        celery_ok = UnifiedTaskManager.is_celery_available()
        milvus_ok = milvus_store.is_healthy()
        count = milvus_store.get_count() if milvus_ok else 0
        return celery_ok, milvus_ok, count

    celery_ok, milvus_ok, milvus_count = await run_in_threadpool(_probe)
    upload_writable = settings.UPLOAD_DIR.exists() and os.access(settings.UPLOAD_DIR, os.W_OK)

    ready = milvus_ok and upload_writable
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="healthy" if ready else "degraded",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        worker_backend=settings.WORKER_BACKEND,
        celery_connected=celery_ok,
        milvus_connected=milvus_ok,
        milvus_total_vectors=milvus_count,
        upload_dir_writable=upload_writable,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=getattr(settings, "HOST", "0.0.0.0"),
        port=int(getattr(settings, "PORT", 8000)),
        reload=bool(getattr(settings, "DEBUG", False)),
    )