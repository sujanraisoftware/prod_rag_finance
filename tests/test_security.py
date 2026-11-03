from pathlib import Path
from starlette.testclient import TestClient
import pymupdf

from app.main import app

client = TestClient(app)


def test_reject_non_pdf_file(tmp_path: Path):
    fake_txt = tmp_path / "report.txt"
    fake_txt.write_text("This is not a pdf file")

    with open(fake_txt, "rb") as f:
        resp = client.post(
            "/api/v1/documents/upload",
            files={"file": ("report.txt", f, "text/plain")}
        )

    assert resp.status_code == 422
    data = resp.json()
    assert "InvalidPDFException" in data["error"] or "magic bytes" in data["message"]


def test_reject_empty_file(tmp_path: Path):
    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.touch()

    with open(empty_pdf, "rb") as f:
        resp = client.post(
            "/api/v1/documents/upload",
            files={"file": ("empty.pdf", f, "application/pdf")}
        )

    assert resp.status_code == 422


def test_reject_encrypted_pdf(tmp_path: Path):
    pdf_path = tmp_path / "encrypted.pdf"
    doc = pymupdf.open()
    doc.new_page()
    # Save with encryption/password
    doc.save(str(pdf_path), encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret123", owner_pw="owner123")
    doc.close()

    with open(pdf_path, "rb") as f:
        resp = client.post(
            "/api/v1/documents/upload",
            files={"file": ("encrypted.pdf", f, "application/pdf")}
        )

    assert resp.status_code == 422
    assert "password-protected" in resp.json()["message"]

