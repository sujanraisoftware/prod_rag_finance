import re
from typing import Any
from app.config import settings
from app.domain.models import FinancialChunk, ChunkType
from app.domain.rag.table_cleaner import is_markdown_table, clean_table_markdown, slice_large_table


# Common SEC / Financial Section regex patterns
HEADING_REGEX = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
SEC_ITEM_REGEX = re.compile(r"^\s*(ITEM\s+[0-9A-Z]+[\.\:\s]+.+)$", re.IGNORECASE | re.MULTILINE)
NOTE_HEADER_REGEX = re.compile(r"^\s*(NOTE\s+[0-9]+[\.\:\s]+.+)$", re.IGNORECASE | re.MULTILINE)
FOOTNOTE_REGEX = re.compile(r"^\s*(\*|\([0-9a-zA-Z]+\)|Note\s*:)\s+.+", re.MULTILINE)


class FinancialChunker:
    """
    Financial-aware semantic chunker designed for zero-hallucination RAG pipelines:
    1. Preserves financial balance sheets, income statements, and schedules atomically
    2. Dynamically tracks section breadcrumb hierarchy (Item, Note, Sub-heading)
    3. Injects audit/provenance headers into chunks for vector grounding
    4. Binds footnotes to preceding tabular data
    """

    def __init__(
        self,
        max_tokens: int = settings.CHUNK_MAX_TOKENS,
        overlap_tokens: int = settings.CHUNK_OVERLAP_TOKENS,
        preserve_tables: bool = settings.PRESERVE_TABLES_ATOMICALLY,
        inject_grounding_header: bool = settings.INJECT_GROUNDING_HEADER,
    ):
        self.max_tokens = max_tokens
        # Approx 4 characters per token
        self.max_chars = max_tokens * 4
        self.overlap_chars = overlap_tokens * 4
        self.preserve_tables = preserve_tables
        self.inject_grounding_header = inject_grounding_header

    def chunk_pages(
        self,
        doc_id: str,
        filename: str,
        pages: list[dict[str, Any]]
    ) -> list[FinancialChunk]:
        """
        Processes extracted page records into structured, audit-ready financial chunks.
        """
        all_chunks: list[FinancialChunk] = []
        total_pages = len(pages)
        section_stack: list[str] = []
        chunk_counter = 0

        for page in pages:
            page_num = page["page_number"]
            raw_text = page.get("text", "").strip()
            if not raw_text:
                continue

            # Break page into semantic blocks (paragraphs, tables, headers)
            blocks = self._split_page_into_semantic_blocks(raw_text)

            current_block_idx = 0
            while current_block_idx < len(blocks):
                block = blocks[current_block_idx]

                # 1. Check if block updates heading hierarchy
                new_heading = self._detect_heading(block)
                if new_heading:
                    level, title = new_heading
                    # Adjust stack depth according to level
                    if level <= len(section_stack):
                        section_stack = section_stack[: level - 1]
                    section_stack.append(title)

                breadcrumb = list(section_stack)
                immediate_header = section_stack[-1] if section_stack else None

                # 2. Check if block is a Table
                if is_markdown_table(block):
                    cleaned_table = clean_table_markdown(block)

                    # Check if next block is a footnote and bind it if present
                    if current_block_idx + 1 < len(blocks):
                        next_b = blocks[current_block_idx + 1]
                        if FOOTNOTE_REGEX.match(next_b):
                            cleaned_table += f"\n\n**Footnote:** {next_b}"
                            current_block_idx += 1  # Consumed footnote

                    # Check if table fits in chunk limit or needs row slicing
                    table_chars = len(cleaned_table)
                    if table_chars <= self.max_chars or not self.preserve_tables:
                        table_slices = [cleaned_table]
                    else:
                        table_slices = slice_large_table(
                            cleaned_table,
                            max_rows_per_slice=35,
                            context_prefix=immediate_header or filename
                        )

                    for t_slice in table_slices:
                        chunk_id = f"{doc_id}_p{page_num}_c{chunk_counter}"
                        enriched_content = self._format_enriched_chunk(
                            filename=filename,
                            page_num=page_num,
                            breadcrumb=breadcrumb,
                            body=t_slice
                        )

                        all_chunks.append(
                            FinancialChunk(
                                chunk_id=chunk_id,
                                doc_id=doc_id,
                                filename=filename,
                                page_number=page_num,
                                total_pages=total_pages,
                                section_header=immediate_header,
                                section_breadcrumb=breadcrumb,
                                chunk_type=ChunkType.TABLE,
                                content=enriched_content,
                                raw_content=t_slice,
                                is_table=True,
                                token_estimate=len(enriched_content) // 4,
                                char_count=len(enriched_content),
                                metadata={
                                    "is_table": True,
                                    "page": page_num,
                                    "source": filename
                                }
                            )
                        )
                        chunk_counter += 1

                else:
                    # 3. Standard Text Block Chunking
                    # Accumulate regular paragraphs up to max_chars
                    text_accumulator = [block]
                    accumulated_len = len(block)

                    while (
                        current_block_idx + 1 < len(blocks)
                        and not is_markdown_table(blocks[current_block_idx + 1])
                        and not self._detect_heading(blocks[current_block_idx + 1])
                    ):
                        next_len = len(blocks[current_block_idx + 1])
                        if accumulated_len + next_len + 2 > self.max_chars:
                            break
                        current_block_idx += 1
                        text_accumulator.append(blocks[current_block_idx])
                        accumulated_len += next_len + 2

                    combined_text = "\n\n".join(text_accumulator).strip()
                    if combined_text:
                        # If combined_text is still oversized (e.g. massive single paragraph)
                        text_slices = self._split_oversized_text(combined_text)
                        for slice_text in text_slices:
                            chunk_id = f"{doc_id}_p{page_num}_c{chunk_counter}"
                            enriched_content = self._format_enriched_chunk(
                                filename=filename,
                                page_num=page_num,
                                breadcrumb=breadcrumb,
                                body=slice_text
                            )

                            all_chunks.append(
                                FinancialChunk(
                                    chunk_id=chunk_id,
                                    doc_id=doc_id,
                                    filename=filename,
                                    page_number=page_num,
                                    total_pages=total_pages,
                                    section_header=immediate_header,
                                    section_breadcrumb=breadcrumb,
                                    chunk_type=ChunkType.TEXT,
                                    content=enriched_content,
                                    raw_content=slice_text,
                                    is_table=False,
                                    token_estimate=len(enriched_content) // 4,
                                    char_count=len(enriched_content),
                                    metadata={
                                        "is_table": False,
                                        "page": page_num,
                                        "source": filename
                                    }
                                )
                            )
                            chunk_counter += 1

                current_block_idx += 1

        return all_chunks

    def _split_page_into_semantic_blocks(self, text: str) -> list[str]:
        """
        Splits page text into logical markdown blocks, ensuring tables remain intact.
        """
        raw_lines = text.split("\n")
        blocks: list[str] = []
        current_block: list[str] = []
        in_table = False

        for line in raw_lines:
            stripped = line.strip()

            if stripped.startswith("|") and stripped.endswith("|"):
                if not in_table:
                    # Flush previous non-table block
                    if current_block:
                        joined = "\n".join(current_block).strip()
                        if joined:
                            blocks.append(joined)
                        current_block = []
                    in_table = True
                current_block.append(stripped)
            else:
                if in_table:
                    # Flush finished table
                    if current_block:
                        joined = "\n".join(current_block).strip()
                        if joined:
                            blocks.append(joined)
                        current_block = []
                    in_table = False

                if not stripped:
                    if current_block:
                        joined = "\n".join(current_block).strip()
                        if joined:
                            blocks.append(joined)
                        current_block = []
                else:
                    current_block.append(line)

        if current_block:
            joined = "\n".join(current_block).strip()
            if joined:
                blocks.append(joined)

        return blocks

    def _detect_heading(self, block: str) -> tuple[int, str] | None:
        """
        Detects markdown headings or SEC-style Item / Note headings.
        Returns: (heading_level, clean_title) or None
        """
        lines = block.strip().split("\n")
        first_line = lines[0].strip()

        # Markdown header: #, ##, ###
        md_match = HEADING_REGEX.match(first_line)
        if md_match:
            hashes, text = md_match.groups()
            clean_title = re.sub(r"^#+\s*", "", text).strip()
            return len(hashes), clean_title

        # SEC Item: ITEM 1A. RISK FACTORS
        sec_match = SEC_ITEM_REGEX.match(first_line)
        if sec_match:
            return 1, sec_match.group(1).strip()

        # Financial Note: NOTE 3. REVENUE
        note_match = NOTE_HEADER_REGEX.match(first_line)
        if note_match:
            return 2, note_match.group(1).strip()

        return None

    def _split_oversized_text(self, text: str) -> list[str]:
        """
        Recursively splits text on sentence or newline boundaries when exceeding max_chars.
        """
        if len(text) <= self.max_chars:
            return [text]

        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)
        slices = []
        current_slice: list[str] = []
        current_len = 0

        for s in sentences:
            s_len = len(s)
            if current_len + s_len > self.max_chars and current_slice:
                slices.append(" ".join(current_slice))
                current_slice = [s]
                current_len = s_len
            else:
                current_slice.append(s)
                current_len += s_len + 1

        if current_slice:
            slices.append(" ".join(current_slice))

        return slices

    def _format_enriched_chunk(
        self,
        filename: str,
        page_num: int,
        breadcrumb: list[str],
        body: str
    ) -> str:
        """
        Enriches chunk content with explicit financial provenance metadata.
        This provides instant context to embedding models and LLMs during RAG generation.
        """
        if not self.inject_grounding_header:
            return body

        section_str = " > ".join(breadcrumb) if breadcrumb else "General Financial Disclosures"
        header = f"<!-- Financial Context | Document: {filename} | Page: {page_num} | Section: {section_str} -->\n"
        return header + body

