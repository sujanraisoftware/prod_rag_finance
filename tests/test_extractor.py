from pathlib import Path
from app.domain.rag.extractor import FastFinanceExtractor


def test_extractor_speed_and_accuracy(sample_financial_pdf: Path):
    extractor = FastFinanceExtractor(
        enable_layout=True,
        top_margin=30.0,
        bottom_margin=30.0
    )

    pages, stats = extractor.extract_document(sample_financial_pdf)

    assert len(pages) == 2
    assert stats["duration_seconds"] >= 0
    assert stats["pages_per_sec"] > 0

    p1 = pages[0]
    assert p1["page_number"] == 1
    # Check that running header / footer outside margins were omitted
    assert "UNITED STATES SECURITIES AND EXCHANGE COMMISSION" not in p1["text"]
    assert "Page 54 of 120" not in p1["text"]

    # Check that financial line items exist
    assert "Net Product Sales" in p1["text"] or "Product Sales" in p1["text"]
    assert "389,900" in p1["text"]


def test_extractor_pure_rag_fallback(sample_financial_pdf: Path):
    """Ensure pure C-accelerated RAG mode works flawlessly if layout model is disabled."""
    extractor = FastFinanceExtractor(
        enable_layout=False,
        top_margin=30.0,
        bottom_margin=30.0
    )

    pages, stats = extractor.extract_document(sample_financial_pdf)
    assert len(pages) == 2
    assert stats["engine"] == "pymupdf4llm-rag"
    assert "Revenue Recognition" in pages[1]["text"]

