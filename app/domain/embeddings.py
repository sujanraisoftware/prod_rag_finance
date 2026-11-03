from typing import Sequence
from fastembed import TextEmbedding

from app.config import settings
from app.core.logging import logger


class FinancialEmbeddingService:
    """
    High-speed, CPU-optimized dense vector embedding service powered by FastEmbed (ONNX Runtime).
    Produces 384-dimensional normalized embeddings with low latency and zero GPU / PyTorch overhead.
    """

    def __init__(
        self,
        model_name: str = settings.EMBEDDING_MODEL_NAME,
        batch_size: int = settings.EMBEDDING_BATCH_SIZE
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: TextEmbedding | None = None

    def _get_model(self) -> TextEmbedding:
        if self._model is None:
            logger.info(f"Loading FastEmbed ONNX model: {self.model_name}")
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Generates dense vector embeddings for a list of financial document chunks.
        """
        if not texts:
            return []

        model = self._get_model()
        # FastEmbed returns a generator of numpy arrays
        embeddings_generator = model.embed(
            documents=texts,
            batch_size=self.batch_size
        )
        return [emb.tolist() for emb in embeddings_generator]

    def embed_query(self, query: str) -> list[float]:
        """
        Generates dense vector embedding for a single user query.
        """
        model = self._get_model()
        # FastEmbed passage vs query embeddings
        query_generator = model.query_embed(query)
        for emb in query_generator:
            return emb.tolist()
        return []


embedding_service = FinancialEmbeddingService()

