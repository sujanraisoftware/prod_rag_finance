from pathlib import Path
from app.domain.models import ChunkType
from app.domain.rag.extractor import FastFinanceExtractor
from app.domain.rag.chunker import FinancialChunker
from app.domain.rag.table_cleaner import is_markdown_table, slice_large_table


def test_chunker_section_and_table_preservation(sample_financial_pdf: Path):
    extractor = FastFinanceExtractor(enable_layout=True, top_margin=30.0, bottom_margin=30.0)
    pages, _ = extractor.extract_document(sample_financial_pdf)

    chunker = FinancialChunker(
        max_tokens=500,
        preserve_tables=True,
        inject_grounding_header=True
    )

    chunks = chunker.chunk_pages(
        doc_id="doc_test_123",
        filename="sample_10k_filing.pdf",
        pages=pages
    )

    assert len(chunks) > 0

    # Ensure table chunks exist and are marked as ChunkType.TABLE
    table_chunks = [c for c in chunks if c.chunk_type == ChunkType.TABLE]
    assert len(table_chunks) >= 1

    tbl = table_chunks[0]
    assert tbl.is_table is True
    assert tbl.page_number == 1
    # Check that provenance grounding header was injected
    assert "<!-- Financial Context" in tbl.content
    assert "sample_10k_filing.pdf" in tbl.content

    # Check that note chunks exist on page 2
    note_chunks = [c for c in chunks if c.page_number == 2]
    assert len(note_chunks) >= 1
    assert any("Revenue Recognition" in c.content or "Accounting Policies" in c.content for c in note_chunks)


def test_table_cleaner_and_slicing():
    # Construct a 50-row financial table
    table_lines = [
        "| Account | Q1 | Q2 |",
        "|---|---|---|"
    ]
    for i in range(50):
        table_lines.append(f"| Debt Tranche {i} | ${i*10}M | ${i*12}M |")
    large_table = "\n".join(table_lines)

    assert is_markdown_table(large_table) is True

    slices = slice_large_table(large_table, max_rows_per_slice=20, context_prefix="Long-term Debt")
    assert len(slices) == 3

    # Every slice must retain the header and columns
    for s in slices:
        assert "| Account | Q1 | Q2 |" in s
        assert "|---|---|---|" in s

