import time
from typing import Any
from flashrank import Ranker, RerankRequest

from app.config import settings
from app.core.logging import logger


class FinancialReranker:
    """
    Ultra-fast cross-encoder reranker powered by FlashRank (ONNX runtime).
    Takes broad candidate chunks retrieved from Milvus dense vector search (top 15-20)
    and scores them with high semantic precision, pruning out irrelevant or noisy context.
    """

    def __init__(
        self,
        model_name: str = settings.RERANKER_MODEL_NAME,
        cache_dir: str = "/tmp/flashrank"
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._ranker: Ranker | None = None

    def _get_ranker(self) -> Ranker:
        if self._ranker is None:
            logger.info(f"Initializing FlashRank Cross-Encoder: {self.model_name}")
            self._ranker = Ranker(
                model_name=self.model_name,
                cache_dir=self.cache_dir
            )
        return self._ranker

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = settings.RERANKER_FINAL_TOP_K
    ) -> list[dict[str, Any]]:
        """
        Reranks Milvus candidate hits against the exact user query.
        Returns the top_k most relevant chunks sorted by cross-encoder score.
        """
        if not candidates:
            return []

        if len(candidates) <= 1:
            for c in candidates:
                c["rerank_score"] = c.get("similarity_score", 1.0)
            return candidates

        ranker = self._get_ranker()

        passages = []
        lookup = {}
        for idx, item in enumerate(candidates):
            cid = item.get("chunk_id") or str(idx)
            text = item.get("content") or item.get("raw_content", "")
            passages.append({"id": cid, "text": text})
            lookup[cid] = item

        rerank_req = RerankRequest(query=query, passages=passages)
        scored_results = ranker.rerank(rerank_req)

        # Merge rerank scores back into chunk metadata and sort
        reranked_chunks: list[dict[str, Any]] = []
        for scored in scored_results[:top_k]:
            cid = scored["id"]
            orig_item = dict(lookup[cid])
            orig_item["rerank_score"] = round(float(scored["score"]), 4)
            reranked_chunks.append(orig_item)

        return reranked_chunks

    @staticmethod
    def build_compressed_context(
        chunks: list[dict[str, Any]],
        max_tokens: int = settings.MAX_CONTEXT_TOKENS
    ) -> tuple[str, int]:
        """
        Builds a compact, low-noise context prompt for the LLM.
        Strips unnecessary repetitive boilerplate while preserving tables and provenance.
        """
        max_chars = max_tokens * 4
        context_parts = []
        total_chars = 0

        for i, c in enumerate(chunks):
            page = c.get("page_number", "N/A")
            filename = c.get("filename", "Doc")
            breadcrumb = " > ".join(c.get("section_breadcrumb", [])) if c.get("section_breadcrumb") else "General"
            raw_text = c.get("raw_content") or c.get("content", "")

            # Format source block concisely
            header = f"--- [Source {i+1} | Document: {filename} | Page {page} | Section: {breadcrumb}] ---"
            block = f"{header}\n{raw_text.strip()}"

            if total_chars + len(block) > max_chars and context_parts:
                break

            context_parts.append(block)
            total_chars += len(block) + 2

        compact_context = "\n\n".join(context_parts)
        estimated_tokens = len(compact_context) // 4
        return compact_context, estimated_tokens


reranker_service = FinancialReranker()

