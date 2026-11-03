import time
from pathlib import Path
from starlette.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "FinTech Fast RAG" in data["app_name"]


def test_instant_parse_endpoint(sample_financial_pdf: Path):
    with open(sample_financial_pdf, "rb") as f:
        response = client.post(
            "/api/v1/documents/parse-instant",
            files={"file": ("sample_10k_filing.pdf", f, "application/pdf")}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["total_chunks"] > 0
    assert data["table_chunks"] >= 1
    assert "stats" in data
    assert data["stats"]["pages_per_sec"] > 0


def test_async_upload_and_task_polling(sample_financial_pdf: Path):
    with open(sample_financial_pdf, "rb") as f:
        upload_resp = client.post(
            "/api/v1/documents/upload",
            files={"file": ("sample_10k_filing.pdf", f, "application/pdf")}
        )

    assert upload_resp.status_code == 202
    upload_data = upload_resp.json()
    task_id = upload_data["task_id"]
    doc_id = upload_data["doc_id"]
    assert task_id is not None
    assert doc_id is not None

    # Poll task until completion (max 10 seconds)
    completed = False
    for _ in range(50):
        task_resp = client.get(f"/api/v1/tasks/{task_id}")
        assert task_resp.status_code == 200
        task_data = task_resp.json()
        if task_data["status"] == "COMPLETED":
            completed = True
            break
        elif task_data["status"] == "FAILED":
            pytest.fail(f"Task failed: {task_data.get('error')}")
        time.sleep(0.1)

    assert completed, "Task did not complete within timeout"

    # Query document metadata
    doc_resp = client.get(f"/api/v1/documents/{doc_id}")
    assert doc_resp.status_code == 200
    doc_data = doc_resp.json()
    assert doc_data["page_count"] == 2
    assert doc_data["total_chunks"] > 0

    # Query chunks with table filter
    chunks_resp = client.get(f"/api/v1/documents/{doc_id}/chunks?chunk_type=table")
    assert chunks_resp.status_code == 200
    chunks_data = chunks_resp.json()
    assert chunks_data["total_chunks"] >= 1
    assert all(c["chunk_type"] == "table" for c in chunks_data["chunks"])

    # Test Milvus Vector Search endpoint
    search_resp = client.post(
        "/api/v1/search",
        json={
            "query": "Net product sales and service revenues",
            "limit": 5,
            "doc_id": doc_id,
            "is_table_only": True
        }
    )
    assert search_resp.status_code == 200
    search_data = search_resp.json()
    assert search_data["total_results"] >= 1
    assert search_data["results"][0]["doc_id"] == doc_id
    assert search_data["results"][0]["is_table"] is True
    assert search_data["execution_time_ms"] > 0

    # Test Grounded Financial Q&A with FlashRank Reranking & Low Context
    answer_resp = client.post(
        "/api/v1/documents/query/answer",
        json={
            "query": "What were the total net sales in FY 2025 and 2024?",
            "doc_id": doc_id,
            "top_k": 3,
            "initial_fetch_k": 10
        }
    )
    assert answer_resp.status_code == 200
    answer_data = answer_resp.json()
    assert answer_data["query"] == "What were the total net sales in FY 2025 and 2024?"
    assert len(answer_data["sources"]) >= 1
    assert answer_data["sources"][0]["rerank_score"] >= 0
    assert answer_data["context_tokens_estimated"] > 0
    assert "latency_ms" in answer_data
    assert answer_data["latency_ms"]["rerank_ms"] >= 0

    # Test document deletion (removes from store and Milvus)
    del_resp = client.delete(f"/api/v1/documents/{doc_id}")
    assert del_resp.status_code == 200
    assert "deleted successfully" in del_resp.json()["message"]

