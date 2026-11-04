import threading
from pathlib import Path
from typing import Any

from pymilvus import MilvusClient, DataType
from pymilvus.exceptions import MilvusException

from app.config import settings
from app.core.logging import logger
from app.domain.models import FinancialChunk

# Milvus error code returned when a collection exists but is not loaded in memory
_ERR_COLLECTION_NOT_LOADED = 101


def _escape_filter_value(value: str) -> str:
    """Escapes a string so it can be safely embedded in a Milvus filter expression."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


class MilvusVectorStore:
    """
    Vector storage repository for financial chunks powered by Milvus.
    Supports both local embedded Milvus Lite (.db file) and remote Milvus clusters.
    Enables:
    - Cosine-similarity dense vector retrieval
    - Scalar filtering (doc_id, page number, is_table)
    - Dynamic metadata fields
    - Automatic (re)loading of the collection if it is released
    """

    def __init__(
        self,
        uri: str = settings.MILVUS_URI,
        token: str = settings.MILVUS_TOKEN,
        collection_name: str = settings.MILVUS_COLLECTION_NAME,
        dimension: int = settings.EMBEDDING_DIMENSION,
    ):
        self.uri = uri
        self.token = token
        self.collection_name = collection_name
        self.dimension = dimension
        self._lock = threading.RLock()
        self._client: MilvusClient | None = None

        # Ensure parent directory exists if using local Milvus Lite
        if not uri.startswith("http://") and not uri.startswith("https://"):
            db_path = Path(uri)
            db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_collection()

    # ------------------------------------------------------------------
    # Connection / collection lifecycle
    # ------------------------------------------------------------------
    def _get_client(self) -> MilvusClient:
        if self._client is None:
            kwargs: dict[str, Any] = {"uri": self.uri}
            if self.token:
                kwargs["token"] = self.token
            self._client = MilvusClient(**kwargs)
        return self._client

    def _is_loaded(self, client: MilvusClient) -> bool:
        try:
            state = client.get_load_state(self.collection_name)
            return "Loaded" in str(state.get("state", ""))
        except Exception as e:
            logger.warning(f"Could not read Milvus load state: {str(e)}")
            return False

    def _ensure_loaded(self) -> None:
        """Loads the collection into memory if it is not already loaded (idempotent)."""
        with self._lock:
            client = self._get_client()
            if not self._is_loaded(client):
                logger.info(f"Loading Milvus collection '{self.collection_name}' into memory")
                client.load_collection(self.collection_name)

    def _init_collection(self) -> None:
        """Initializes collection schema and vector index if not present, then loads it."""
        with self._lock:
            client = self._get_client()

            if client.has_collection(self.collection_name):
                logger.info(f"Connected to existing Milvus collection: '{self.collection_name}'")
                self._ensure_loaded()
                return

            logger.info(f"Creating new Milvus collection: '{self.collection_name}' (dim={self.dimension})")

            # 1. Define Schema
            schema = client.create_schema(
                auto_id=False,
                enable_dynamic_field=True,
                description="Financial RAG chunks collection with strict provenance metadata",
            )

            # Primary Key
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=128, is_primary=True)
            # Dense Vector
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dimension)
            # Scalar Filter Fields
            schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=128)
            schema.add_field(field_name="filename", datatype=DataType.VARCHAR, max_length=256)
            schema.add_field(field_name="page_number", datatype=DataType.INT64)
            schema.add_field(field_name="total_pages", datatype=DataType.INT64)
            schema.add_field(field_name="chunk_type", datatype=DataType.VARCHAR, max_length=32)
            schema.add_field(field_name="is_table", datatype=DataType.BOOL)
            schema.add_field(field_name="section_header", datatype=DataType.VARCHAR, max_length=512)
            schema.add_field(field_name="section_breadcrumb", datatype=DataType.VARCHAR, max_length=1024)
            schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="raw_content", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="token_estimate", datatype=DataType.INT64)
            schema.add_field(field_name="char_count", datatype=DataType.INT64)
            schema.add_field(field_name="file_hash", datatype=DataType.VARCHAR, max_length=64)

            # 2. Define Index
            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="AUTOINDEX",
                metric_type="COSINE",
            )

            # 3. Create Collection (with index_params it should auto-load, but we verify)
            client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
            )
            self._ensure_loaded()
            logger.info(f"Successfully initialized Milvus collection '{self.collection_name}'")

    def _call_with_reload(self, fn_name: str, **kwargs: Any) -> Any:
        """
        Calls a MilvusClient method (search/query/delete). If the collection has been
        released (error code 101), reloads it once and retries.
        """
        client = self._get_client()
        fn = getattr(client, fn_name)
        try:
            return fn(**kwargs)
        except MilvusException as e:
            if getattr(e, "code", None) == _ERR_COLLECTION_NOT_LOADED:
                logger.warning(
                    f"Collection '{self.collection_name}' was released; reloading and retrying {fn_name}"
                )
                client.load_collection(self.collection_name)
                return fn(**kwargs)
            raise

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------
    def insert_chunks(
        self,
        chunks: list[FinancialChunk],
        embeddings: list[list[float]],
    ) -> int:
        """Inserts financial chunks along with their vector embeddings into Milvus."""
        if not chunks or not embeddings:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")

        records: list[dict[str, Any]] = []
        for chunk, vec in zip(chunks, embeddings):
            # Ensure text stays within Milvus VARCHAR limit
            safe_content = chunk.content[:65000]
            safe_raw = chunk.raw_content[:65000]
            breadcrumb_str = " > ".join(chunk.section_breadcrumb)[:1000]

            records.append({
                "id": chunk.chunk_id,
                "vector": vec,
                "doc_id": chunk.doc_id,
                "filename": chunk.filename,
                "page_number": int(chunk.page_number),
                "total_pages": int(chunk.total_pages),
                "chunk_type": chunk.chunk_type.value,
                "is_table": bool(chunk.is_table),
                "section_header": (chunk.section_header or "")[:500],
                "section_breadcrumb": breadcrumb_str,
                "content": safe_content,
                "raw_content": safe_raw,
                "token_estimate": int(chunk.token_estimate),
                "char_count": int(chunk.char_count),
                "file_hash": chunk.metadata.get("file_hash", "")[:60],
            })

        with self._lock:
            client = self._get_client()
            res = client.insert(
                collection_name=self.collection_name,
                data=records,
            )
            inserted_count = res.get("insert_count", len(records)) if isinstance(res, dict) else len(records)
            logger.info(f"Indexed {inserted_count} financial chunks into Milvus collection '{self.collection_name}'")
            return inserted_count

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------
    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        doc_id: str | None = None,
        is_table_only: bool = False,
        page_number: int | None = None,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Dense vector search with scalar metadata filtering."""
        filter_exprs: list[str] = []
        if doc_id:
            filter_exprs.append(f'doc_id == "{_escape_filter_value(doc_id)}"')
        if is_table_only:
            filter_exprs.append("is_table == true")
        if page_number is not None:
            filter_exprs.append(f"page_number == {int(page_number)}")

        filter_str = " and ".join(filter_exprs) if filter_exprs else ""

        output_fields = [
            "id",
            "doc_id",
            "filename",
            "page_number",
            "total_pages",
            "chunk_type",
            "is_table",
            "section_header",
            "section_breadcrumb",
            "content",
            "raw_content",
            "token_estimate",
        ]

        with self._lock:
            self._ensure_loaded()
            raw_results = self._call_with_reload(
                "search",
                collection_name=self.collection_name,
                data=[query_vector],
                limit=limit,
                filter=filter_str,
                output_fields=output_fields,
            )

        formatted_results: list[dict[str, Any]] = []
        if raw_results and len(raw_results) > 0:
            for hit in raw_results[0]:
                # For COSINE metric, Milvus returns the similarity in "distance"
                score = float(hit.get("distance", 0.0))
                if score < min_score:
                    continue

                entity = hit.get("entity", {})
                breadcrumb = entity.get("section_breadcrumb", "")
                breadcrumb_list = (
                    [b.strip() for b in breadcrumb.split(">") if b.strip()] if breadcrumb else []
                )

                formatted_results.append({
                    "chunk_id": hit.get("id"),
                    "similarity_score": round(score, 4),
                    "doc_id": entity.get("doc_id"),
                    "filename": entity.get("filename"),
                    "page_number": entity.get("page_number"),
                    "total_pages": entity.get("total_pages"),
                    "chunk_type": entity.get("chunk_type"),
                    "is_table": entity.get("is_table"),
                    "section_header": entity.get("section_header"),
                    "section_breadcrumb": breadcrumb_list,
                    "content": entity.get("content"),
                    "raw_content": entity.get("raw_content"),
                    "token_estimate": entity.get("token_estimate"),
                })

        return formatted_results

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Deletes all chunks associated with a specific document from Milvus."""
        with self._lock:
            try:
                self._ensure_loaded()
                res = self._call_with_reload(
                    "delete",
                    collection_name=self.collection_name,
                    filter=f'doc_id == "{_escape_filter_value(doc_id)}"',
                )
                if isinstance(res, dict):
                    deleted = int(res.get("delete_count", 0))
                elif isinstance(res, list):
                    deleted = len(res)
                else:
                    deleted = 0
                logger.info(f"Deleted {deleted} vectors for document '{doc_id}' from Milvus")
                return deleted
            except Exception as e:
                logger.error(f"Error deleting vectors for doc '{doc_id}' in Milvus: {str(e)}")
                return 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def get_count(self) -> int:
        """Returns total number of indexed chunk vectors in the collection."""
        with self._lock:
            client = self._get_client()
            try:
                stats = client.get_collection_stats(self.collection_name)
                return int(stats.get("row_count", 0))
            except Exception as e:
                logger.warning(f"Could not fetch Milvus collection stats: {str(e)}")
                return 0

    def is_healthy(self) -> bool:
        """Checks connectivity to Milvus and that the collection is loaded (reloads if needed)."""
        try:
            with self._lock:
                client = self._get_client()
                if not client.has_collection(self.collection_name):
                    return False
                self._ensure_loaded()
                return self._is_loaded(client)
        except Exception:
            return False

    def close(self) -> None:
        with self._lock:
            if self._client:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None



# Backward compatibility alias
milvus_store = MilvusVectorStore()