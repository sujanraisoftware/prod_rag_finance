import hashlib
import os
import re
from pathlib import Path
from typing import Any
import pymupdf

from app.core.exceptions import InvalidPDFException, FileSizeExceededException
from app.core.logging import logger


PDF_MAGIC_BYTES = b"%PDF-"


def sanitize_filename(filename: str) -> str:
    """
    Sanitizes filename to prevent directory traversal and illegal characters.
    """
    # Remove path separators and null bytes
    clean = os.path.basename(filename)
    clean = clean.replace("\x00", "").strip()
    # Replace any non-alphanumeric (except . - _) with _
    clean = re.sub(r"[^a-zA-Z0-9._-]", "_", clean)
    if not clean.lower().endswith(".pdf"):
        clean += ".pdf"
    return clean


def calculate_sha256(file_path: Path) -> str:
    """Computes SHA-256 checksum of a file for regulatory audit and deduplication."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def validate_pdf_file(file_path: Path, filename: str, max_size_mb: int = 100) -> dict[str, Any]:
    """
    Validates PDF file integrity, size, encryption status, and magic bytes.
    Returns metadata if valid, otherwise raises an exception.
    """
    if not file_path.exists():
        raise InvalidPDFException("File does not exist on disk.")

    file_size = file_path.stat().st_size
    max_bytes = max_size_mb * 1024 * 1024

    if file_size == 0:
        raise InvalidPDFException("Uploaded PDF file is empty (0 bytes).")

    if file_size > max_bytes:
        raise FileSizeExceededException(
            f"File size ({file_size / (1024 * 1024):.2f} MB) exceeds maximum allowed limit of {max_size_mb} MB."
        )

    # Magic byte check
    with open(file_path, "rb") as f:
        header = f.read(1024)
        if PDF_MAGIC_BYTES not in header:
            raise InvalidPDFException(
                "Invalid file format: File header does not contain valid PDF magic bytes (%PDF-)."
            )

    # Deep validation via PyMuPDF
    try:
        doc = pymupdf.open(str(file_path))
    except Exception as e:
        logger.error(f"Failed to open PDF '{filename}': {str(e)}")
        raise InvalidPDFException(f"Corrupted or malformed PDF file: {str(e)}")

    try:
        if doc.is_encrypted:
            raise InvalidPDFException(
                "PDF is password-protected or encrypted. Please remove encryption before ingesting financial documents."
            )

        page_count = doc.page_count
        if page_count == 0:
            raise InvalidPDFException("PDF contains 0 pages.")

        file_hash = calculate_sha256(file_path)

        return {
            "page_count": page_count,
            "file_size_bytes": file_size,
            "file_hash": file_hash,
            "is_pdf": True
        }
    finally:
        doc.close()

