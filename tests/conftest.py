import pytest
from pathlib import Path
import pymupdf


@pytest.fixture
def sample_financial_pdf(tmp_path: Path) -> Path:
    """
    Creates a realistic SEC 10-K financial statement PDF for testing:
    - Page 1: Income Statement with vector lines and headers/footers
    - Page 2: Balance Sheet with multi-column table
    - Page 3: Notes to Financial Statements with footnotes
    """
    pdf_path = tmp_path / "sample_10k_filing.pdf"
    doc = pymupdf.open()

    # --- Page 1: Income Statement ---
    p1 = doc.new_page(width=612, height=792)
    # Running header (to test margin stripping)
    p1.insert_text((50, 20), "UNITED STATES SECURITIES AND EXCHANGE COMMISSION - FORM 10-K", fontsize=8)

    # Document Heading
    p1.insert_text((50, 60), "# ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA", fontsize=14)
    p1.insert_text((50, 85), "## Consolidated Statements of Operations", fontsize=12)
    p1.insert_text((50, 105), "Years Ended December 31, 2025 and 2024 (in millions except per share amounts)", fontsize=9)

    # Table with lines
    p1.draw_rect(pymupdf.Rect(50, 120, 550, 240), color=(0, 0, 0), width=1)
    p1.draw_line(pymupdf.Point(50, 150), pymupdf.Point(550, 150), color=(0, 0, 0), width=1)
    p1.draw_line(pymupdf.Point(320, 120), pymupdf.Point(320, 240), color=(0, 0, 0), width=1)
    p1.draw_line(pymupdf.Point(440, 120), pymupdf.Point(440, 240), color=(0, 0, 0), width=1)

    p1.insert_text((60, 140), "Financial Line Item", fontsize=10)
    p1.insert_text((330, 140), "FY 2025", fontsize=10)
    p1.insert_text((450, 140), "FY 2024", fontsize=10)

    p1.insert_text((60, 170), "Net Product Sales", fontsize=10)
    p1.insert_text((330, 170), "$284,500", fontsize=10)
    p1.insert_text((450, 170), "$260,200", fontsize=10)

    p1.insert_text((60, 200), "Net Service Revenues", fontsize=10)
    p1.insert_text((330, 200), "$105,400", fontsize=10)
    p1.insert_text((450, 200), "$85,200", fontsize=10)

    p1.insert_text((60, 230), "Total Net Sales", fontsize=10)
    p1.insert_text((330, 230), "$389,900", fontsize=10)
    p1.insert_text((450, 230), "$345,400", fontsize=10)

    # Footnote below table
    p1.insert_text((50, 260), "* Note: Service revenues include cloud subscription services and enterprise contracts.", fontsize=9)

    # Running footer
    p1.insert_text((50, 770), "Page 54 of 120 - Acme Global Inc.", fontsize=8)

    # --- Page 2: Notes to Financial Statements ---
    p2 = doc.new_page(width=612, height=792)
    p2.insert_text((50, 20), "UNITED STATES SECURITIES AND EXCHANGE COMMISSION - FORM 10-K", fontsize=8)

    p2.insert_text((50, 60), "### Note 1. Summary of Significant Accounting Policies", fontsize=12)
    p2.insert_text(
        (50, 90),
        "Basis of Presentation: The consolidated financial statements include the accounts of the Company and its wholly-owned subsidiaries. All intercompany accounts and transactions have been eliminated in consolidation.",
        fontsize=10
    )

    p2.insert_text((50, 140), "### Note 2. Revenue Recognition", fontsize=12)
    p2.insert_text(
        (50, 170),
        "The Company accounts for revenue in accordance with ASC Topic 606. Revenue is measured based on the consideration specified in a contract with a customer and excludes amounts collected on behalf of third parties.",
        fontsize=10
    )

    p2.insert_text((50, 770), "Page 55 of 120 - Acme Global Inc.", fontsize=8)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path

