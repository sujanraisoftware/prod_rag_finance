import os
import time
import psutil
import pymupdf
from app.domain.rag.extractor import FastFinanceExtractor
from app.domain.rag.chunker import FinancialChunker
from app.domain.embeddings import embedding_service
from app.domain.vector_store import MilvusVectorStore


def run_benchmark():
    # 1. Generate 20-page financial PDF
    pdf_path = "/tmp/bench_20p.pdf"
    doc = pymupdf.open()
    for i in range(20):
        page = doc.new_page(width=612, height=792)
        page.insert_text((50, 20), "ACME CORP 10-K ANNUAL REPORT", fontsize=8)
        page.insert_text((50, 60), f"# ITEM {i+1}. FINANCIAL SECTION NOTE {i+1}", fontsize=14)
        page.insert_text((50, 85), f"## Schedule of Debt and Obligations Part {i+1}", fontsize=11)

        # Table
        page.draw_rect(pymupdf.Rect(50, 100, 550, 220), color=(0, 0, 0), width=1)
        page.draw_line(pymupdf.Point(50, 130), pymupdf.Point(550, 130), color=(0, 0, 0), width=1)
        page.draw_line(pymupdf.Point(300, 100), pymupdf.Point(300, 220), color=(0, 0, 0), width=1)
        page.insert_text((60, 120), "Maturity Bucket", fontsize=10)
        page.insert_text((310, 120), "Principal Amount ($M)", fontsize=10)
        page.insert_text((60, 150), f"Due within {i+1} year(s)", fontsize=10)
        page.insert_text((310, 150), f"${(i+1)*250:,.2f}", fontsize=10)
        page.insert_text((60, 180), f"Due in {i+2} to {i+5} years", fontsize=10)
        page.insert_text((310, 180), f"${(i+1)*750:,.2f}", fontsize=10)
        page.insert_text((60, 210), "Total Obligations", fontsize=10)
        page.insert_text((310, 210), f"${(i+1)*1000:,.2f}", fontsize=10)

        page.insert_text(
            (50, 250),
            "The contractual maturities above are based on scheduled principal payments. Variable rate interest obligations are based on current SOFR rates at the reporting period balance sheet date.",
            fontsize=9
        )
        page.insert_text((50, 770), f"Page {i+1} of 20", fontsize=8)

    doc.save(pdf_path)
    doc.close()

    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss / 1024 / 1024

    # 2. Benchmark PyMuPDF4LLM Extraction
    t0 = time.perf_counter()
    extractor = FastFinanceExtractor(enable_layout=False, top_margin=30.0, bottom_margin=30.0)
    pages, stats = extractor.extract_document(pdf_path)
    extract_time = time.perf_counter() - t0

    # 3. Benchmark Financial Chunking & Table Preservation
    chunker = FinancialChunker(max_tokens=600, preserve_tables=True)
    t1 = time.perf_counter()
    chunks = chunker.chunk_pages("bench_doc", "bench_20p.pdf", pages)
    chunk_time = time.perf_counter() - t1

    # 4. Benchmark ONNX Vector Embeddings Generation
    t2 = time.perf_counter()
    chunk_texts = [c.content for c in chunks]
    embeddings = embedding_service.embed_documents(chunk_texts)
    embedding_time = time.perf_counter() - t2

    # 5. Benchmark Milvus Vector Database Indexing
    bench_milvus = MilvusVectorStore(
        uri="/tmp/bench_milvus.db",
        collection_name="bench_financial_chunks",
        dimension=384
    )
    t3 = time.perf_counter()
    indexed_count = bench_milvus.insert_chunks(chunks, embeddings)
    indexing_time = time.perf_counter() - t3

    # 6. Benchmark Milvus Filtered Vector Search
    t4 = time.perf_counter()
    query_vec = embedding_service.embed_query("Total obligations scheduled debt table")
    search_results = bench_milvus.search(
        query_vector=query_vec,
        limit=5,
        is_table_only=True
    )
    search_time = time.perf_counter() - t4

    total_pipeline_time = extract_time + chunk_time + embedding_time + indexing_time
    mem_after = proc.memory_info().rss / 1024 / 1024
    mem_delta = mem_after - mem_before

    print(f"\n==================================================================")
    print(f"📊 FULL END-TO-END FINTECH RAG PIPELINE BENCHMARK (PyMuPDF + Milvus)")
    print(f"==================================================================")
    print(f"Pages Ingested       : {len(pages)} pages")
    print(f"Total Chunks Created : {len(chunks)} ({sum(1 for c in chunks if c.is_table)} tabular chunks)")
    print(f"------------------------------------------------------------------")
    print(f"1. PyMuPDF4LLM Parse : {extract_time*1000:.2f} ms ({len(pages)/extract_time:.1f} pages/sec)")
    print(f"2. Financial Chunking: {chunk_time*1000:.2f} ms")
    print(f"3. FastEmbed (ONNX)  : {embedding_time*1000:.2f} ms ({len(chunks)/embedding_time:.1f} chunks/sec)")
    print(f"4. Milvus Indexing   : {indexing_time*1000:.2f} ms ({indexed_count} vectors indexed)")
    print(f"5. Milvus Query Time : {search_time*1000:.2f} ms (Top hit similarity: {search_results[0]['similarity_score']:.4f})")
    print(f"------------------------------------------------------------------")
    print(f"Total Pipeline Time  : {total_pipeline_time:.3f} s ({len(pages)/total_pipeline_time:.1f} pages/sec end-to-end)")
    print(f"Peak RAM Usage       : {mem_after:.1f} MB (Delta: {mem_delta:.1f} MB)")
    print(f"==================================================================\n")

    bench_milvus.close()


if __name__ == "__main__":
    run_benchmark()
