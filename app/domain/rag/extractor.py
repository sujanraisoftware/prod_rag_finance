import time
from pathlib import Path
from typing import Any
import pymupdf
import pymupdf4llm

from app.config import settings
from app.core.exceptions import ExtractionException
from app.core.logging import logger


class FastFinanceExtractor:
    """
    High-throughput, C-accelerated financial PDF parsing engine powered by PyMuPDF4LLM.
    Configured specifically for FinTech documents:
    - Strips running headers/footers via margin geometry to prevent vector noise
    - High accuracy table extraction (markdown/HTML) without multi-gigabyte memory spikes
    - Fallback protection between GNN layout and pure C-RAG modes
    """

    def __init__(
        self,
        enable_layout: bool = settings.ENABLE_LAYOUT_PARSER,
        table_strategy: str = settings.TABLE_STRATEGY,
        table_output: str = settings.TABLE_OUTPUT,
        top_margin: float = settings.HEADER_MARGIN,
        bottom_margin: float = settings.FOOTER_MARGIN,
        left_margin: float = settings.LEFT_MARGIN,
        right_margin: float = settings.RIGHT_MARGIN,
    ):
        self.enable_layout = enable_layout
        self.table_strategy = table_strategy
        self.table_output = table_output
        # PyMuPDF margins format: (left, top, right, bottom)
        self.margins = (left_margin, top_margin, right_margin, bottom_margin)

    def extract_document(
        self,
        file_path: str | Path,
        pages: list[int] | None = None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Parses the document into page-level chunks with rich financial markdown.

        Returns:
            tuple: (list of page dictionaries, performance stats dictionary)
        """
        path_str = str(file_path)
        start_time = time.perf_counter()

        # Try with preferred layout setting first
        try:
            pymupdf4llm.use_layout(self.enable_layout)
            if self.enable_layout:
                raw_page_chunks = pymupdf4llm.to_markdown(
                    path_str,
                    pages=pages,
                    page_chunks=True,
                    header=False,
                    footer=False,
                    write_images=False,
                    embed_images=False,
                    margins=self.margins,
                    table_strategy=self.table_strategy,
                    table_output=self.table_output,
                    show_progress=False
                )
            else:
                raw_page_chunks = pymupdf4llm.to_markdown(
                    path_str,
                    pages=pages,
                    page_chunks=True,
                    write_images=False,
                    embed_images=False,
                    margins=self.margins,
                    table_strategy=self.table_strategy,
                    table_output=self.table_output,
                    show_progress=False
                )
        except Exception as e:
            if self.enable_layout:
                logger.warning(
                    f"Layout-mode extraction failed for {path_str}: {str(e)}. "
                    "Falling back to pure C-accelerated RAG mode."
                )
                try:
                    pymupdf4llm.use_layout(False)
                    raw_page_chunks = pymupdf4llm.to_markdown(
                        path_str,
                        pages=pages,
                        page_chunks=True,
                        write_images=False,
                        embed_images=False,
                        margins=self.margins,
                        table_strategy=self.table_strategy,
                        table_output=self.table_output,
                        show_progress=False
                    )
                except Exception as fallback_err:
                    logger.error(f"Fallback extraction also failed: {str(fallback_err)}")
                    raise ExtractionException(f"Failed to extract text from PDF: {str(fallback_err)}")
            else:
                logger.error(f"Extraction failed: {str(e)}")
                raise ExtractionException(f"Failed to extract text from PDF: {str(e)}")

        duration = time.perf_counter() - start_time

        # If pymupdf4llm returned a single string (when page_chunks was somehow false)
        if isinstance(raw_page_chunks, str):
            raw_page_chunks = [{"text": raw_page_chunks, "metadata": {"page": 1}}]

        total_pages = len(raw_page_chunks)
        pages_per_sec = total_pages / max(duration, 0.0001)

        logger.info(
            f"Extraction of {Path(file_path).name} complete: {total_pages} pages in "
            f"{duration:.3f}s ({pages_per_sec:.1f} pages/sec)"
        )

        formatted_pages: list[dict[str, Any]] = []
        for idx, page in enumerate(raw_page_chunks):
            page_meta = page.get("metadata", {})
            page_num = page_meta.get("page", idx + 1)
            text_content = page.get("text", "")

            formatted_pages.append({
                "page_number": int(page_num),
                "text": text_content,
                "tables": page.get("tables", []),
                "images": page.get("images", []),
                "toc_items": page.get("toc_items", []),
                "metadata": page_meta
            })

        stats = {
            "duration_seconds": round(duration, 4),
            "pages_per_sec": round(pages_per_sec, 2),
            "total_pages": total_pages,
            "engine": "pymupdf4llm-layout" if self.enable_layout else "pymupdf4llm-rag"
        }

        return formatted_pages, stats
