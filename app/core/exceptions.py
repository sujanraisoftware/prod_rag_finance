class FinTechRAGException(Exception):
    """Base exception for financial RAG system."""
    def __init__(self, message: str, status_code: int = 500, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class InvalidPDFException(FinTechRAGException):
    """Raised when PDF file is invalid, corrupt, or encrypted."""
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message=message, status_code=422, details=details)


class FileSizeExceededException(FinTechRAGException):
    """Raised when file exceeds size limit."""
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message=message, status_code=413, details=details)


class ExtractionException(FinTechRAGException):
    """Raised when PyMuPDF4LLM extraction fails."""
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message=message, status_code=500, details=details)


class DocumentNotFoundException(FinTechRAGException):
    """Raised when document is not found."""
    def __init__(self, doc_id: str):
        super().__init__(
            message=f"Document with ID '{doc_id}' not found.",
            status_code=404,
            details={"doc_id": doc_id}
        )


class TaskNotFoundException(FinTechRAGException):
    """Raised when task is not found."""
    def __init__(self, task_id: str):
        super().__init__(
            message=f"Task with ID '{task_id}' not found.",
            status_code=404,
            details={"task_id": task_id}
        )

