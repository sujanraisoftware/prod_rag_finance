from pymilvus import MilvusClient

client = MilvusClient("./storage/milvus_finance.db")

collections = client.list_collections()
print(f"Collections found: {collections}")

collection_name = "financial_rag_chunks"

if collection_name in collections:
    # Load the collection into memory (fixes the 'released' state error)
    client.load_collection(collection_name)
    print(client.get_load_state(collection_name))

    stats = client.get_collection_stats(collection_name)
    print(f"\nStats for '{collection_name}': {stats}")

    print("\nFetching sample data...")
    results = client.query(
        collection_name=collection_name,
        filter="",              # empty filter + limit works if there's no int 'id' field
        limit=2,
        output_fields=["*"],
    )
    for i, res in enumerate(results):
        print(f"\n--- Record {i+1} ---")
        print(res)
else:
    print(f"Collection '{collection_name}' not found.")