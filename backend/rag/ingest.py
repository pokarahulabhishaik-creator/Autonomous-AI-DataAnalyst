"""
Ingestion script for AutoAnalyst AI Knowledge Base.
"""
from backend.rag.vector_store import VectorStore
from backend.rag.knowledge_base import KNOWLEDGE_DOCUMENTS

def run_ingest():
    store = VectorStore()
    print(f"Successfully initialized VectorStore. Active store: {'ChromaDB' if store.use_chroma else 'TF-IDF Fallback'}")
    test_results = store.search("correlation and causation", top_k=2)
    print(f"Test query retrieved {len(test_results)} documents.")
    for doc in test_results:
        print(f" - [{doc.get('category')}] {doc.get('title')}")

if __name__ == "__main__":
    run_ingest()
