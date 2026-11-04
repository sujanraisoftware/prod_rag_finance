from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Application
    APP_NAME: str = "FinTech Fast RAG Ingestion Engine"
    APP_VERSION: str = "1.0.0"
    DEBUG_MODE: bool = False

    # Directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    UPLOAD_DIR: Path = Path("./temp_uploads")
    STORAGE_DIR: Path = Path("./storage/processed")

    # Security & File Limits
    MAX_UPLOAD_SIZE_MB: int = 100
    ALLOWED_MIME_TYPES: list[str] = ["application/pdf"]

    # Worker Backend Configuration
    # Options: "auto" (Celery if Redis available, else in_process), "in_process", "celery"
    WORKER_BACKEND: str = "auto"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    CELERY_TASK_TIMEOUT_SECONDS: int = 600

    # PyMuPDF4LLM & Extraction Parameters
    ENABLE_LAYOUT_PARSER: bool = True
    TABLE_STRATEGY: str = "lines_strict"
    TABLE_OUTPUT: str = "markdown"  # "markdown" or "html"
    HEADER_MARGIN: float = 36.0    # 0.5 inch (36 pt) top margin to clip running headers
    FOOTER_MARGIN: float = 36.0    # 0.5 inch (36 pt) bottom margin to clip running footers
    LEFT_MARGIN: float = 0.0
    RIGHT_MARGIN: float = 0.0
    WRITE_IMAGES: bool = False      # Discard heavy raster payloads for pure text/table speed
    DETECT_BG_COLOR: bool = True

    # Financial Chunking Parameters
    CHUNK_MAX_TOKENS: int = 1000
    CHUNK_OVERLAP_TOKENS: int = 100
    PRESERVE_TABLES_ATOMICALLY: bool = True
    INJECT_GROUNDING_HEADER: bool = True

    # Vector Database Configuration
    # Options: "milvus" (default standalone via Milvus Lite) or "chromadb"
    VECTOR_STORE_BACKEND: str = "milvus"
    CHROMA_PATH: str = "./storage/chroma_db"
    CHROMA_COLLECTION_NAME: str = "financial_rag_chunks"

    # Milvus Vector Database Configuration (Standalone embedded via Milvus Lite)
    # Stored locally at ./storage/milvus_finance.db with zero external daemons needed
    MILVUS_URI: str = "./storage/milvus_finance.db"
    MILVUS_TOKEN: str = ""
    MILVUS_COLLECTION_NAME: str = "financial_rag_chunks"

    # Embedding Service Configuration (CPU-optimized ONNX runtime)
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 32

    # Reranker Configuration (Ultra-fast CPU ONNX Cross-Encoder)
    RERANKER_MODEL_NAME: str = "ms-marco-TinyBERT-L-2-v2"
    RERANKER_INITIAL_TOP_K: int = 15
    RERANKER_FINAL_TOP_K: int = 4
    MAX_CONTEXT_TOKENS: int = 2000

    # Groq LLM Configuration
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_TEMPERATURE: float = 0.0
    GROQ_MAX_TOKENS: int = 1024


settings = Settings()
