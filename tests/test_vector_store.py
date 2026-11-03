import pytest
from app.domain.models import FinancialChunk, ChunkType
from app.domain.embeddings import embedding_service
from app.domain.vector_store import MilvusVectorStore


@pytest.fixture
def temp_milvus_store(tmp_path):
    db_file = tmp_path / "test_store.db"
    store = MilvusVectorStore(
        uri=str(db_file),
        collection_name="test_financial_chunks",
        dimension=384
    )
    yield store
    store.close()


def test_milvus_insert_and_search(temp_milvus_store):
    chunks = [
        FinancialChunk(
            chunk_id="chunk_test_1",
            doc_id="doc_fin_1",
            filename="tesla_10k.pdf",
            page_number=12,
            total_pages=100,
            section_header="Item 7. MD&A",
            section_breadcrumb=["Item 7. MD&A", "Automotive Revenues"],
            chunk_type=ChunkType.TEXT,
            content="Total automotive revenue increased by 15% due to higher vehicle deliveries.",
            raw_content="Total automotive revenue increased by 15%...",
            is_table=False,
            token_estimate=15,
            char_count=70
        ),
        FinancialChunk(
            chunk_id="chunk_test_2",
            doc_id="doc_fin_1",
            filename="tesla_10k.pdf",
            page_number=45,
            total_pages=100,
            section_header="Item 8. Consolidated Statements of Operations",
            section_breadcrumb=["Item 8. Consolidated Statements of Operations"],
            chunk_type=ChunkType.TABLE,
            content="| Revenue Stream | FY 2025 | FY 2024 |\n|---|---|---|\n| Automotive | $82,400M | $71,200M |",
            raw_content="| Revenue Stream | FY 2025 | FY 2024 |...",
            is_table=True,
            token_estimate=25,
            char_count=100
        )
    ]

    # Generate embeddings
    embeddings = embedding_service.embed_documents([c.content for c in chunks])
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384

    # Insert into Milvus
    inserted = temp_milvus_store.insert_chunks(chunks, embeddings)
    assert inserted == 2

    # Query 1: Search for table chunks only
    query_vec = embedding_service.embed_query("Automotive revenue statement table")
    table_results = temp_milvus_store.search(
        query_vector=query_vec,
        limit=5,
        is_table_only=True
    )
    assert len(table_results) == 1
    assert table_results[0]["is_table"] is True
    assert table_results[0]["chunk_id"] == "chunk_test_2"
    assert table_results[0]["similarity_score"] > 0

    # Query 2: Search with doc_id filter
    all_results = temp_milvus_store.search(
        query_vector=query_vec,
        limit=5,
        doc_id="doc_fin_1"
    )
    assert len(all_results) == 2

    # Test deletion
    deleted = temp_milvus_store.delete_by_doc_id("doc_fin_1")
    assert deleted >= 1

    # Search again should yield 0
    after_del = temp_milvus_store.search(query_vector=query_vec, limit=5, doc_id="doc_fin_1")
    assert len(after_del) == 0

