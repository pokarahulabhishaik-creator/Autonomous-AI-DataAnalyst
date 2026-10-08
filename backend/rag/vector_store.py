import logging
from typing import List, Dict, Any, Optional
from backend.rag.knowledge_base import KNOWLEDGE_DOCUMENTS

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Knowledge Vector Store for AutoAnalyst AI.
    Uses ChromaDB when available, with a fast in-memory TF-IDF semantic fallback.
    """

    def __init__(self, collection_name: str = "autoanalyst_kb"):
        self.collection_name = collection_name
        self.chroma_client = None
        self.collection = None
        self.use_chroma = False
        self._init_store()

    def _init_store(self):
        try:
            import chromadb
            # Use Ephemeral in-memory client for serverless/cloud compatibility
            self.chroma_client = chromadb.Client()
            self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)
            
            # Ingest documents if collection is empty
            if self.collection.count() == 0:
                self.ingest_documents(KNOWLEDGE_DOCUMENTS)
            self.use_chroma = True
            logger.info("ChromaDB vector store initialized successfully.")
        except Exception as e:
            logger.warning(f"ChromaDB initialization failed ({e}). Using in-memory TF-IDF retriever.")
            self.use_chroma = False
            self._init_tfidf_fallback()

    def _init_tfidf_fallback(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        self.documents = KNOWLEDGE_DOCUMENTS
        self.corpus = [
            f"{doc['title']} {doc['category']} {doc['content']}" for doc in self.documents
        ]
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.doc_vectors = self.vectorizer.fit_transform(self.corpus)

    def ingest_documents(self, docs: List[Dict[str, Any]]):
        if not self.collection:
            return
        ids = [doc["id"] for doc in docs]
        documents = [f"{doc['title']}\n{doc['content']}" for doc in docs]
        metadatas = [{"category": doc.get("category", "general"), "title": doc["title"]} for doc in docs]
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search relevant analytical guidance documents for a query."""
        if self.use_chroma and self.collection:
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=min(top_k, len(KNOWLEDGE_DOCUMENTS))
                )
                hits = []
                if results and "documents" in results and results["documents"]:
                    docs = results["documents"][0]
                    metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
                    ids = results["ids"][0] if "ids" in results else [""] * len(docs)
                    for d_id, doc, meta in zip(ids, docs, metas):
                        hits.append({
                            "id": d_id,
                            "title": meta.get("title", ""),
                            "category": meta.get("category", ""),
                            "content": doc
                        })
                if hits:
                    return hits
            except Exception as e:
                logger.warning(f"Chroma search error ({e}), attempting fallback.")

        # Fallback to in-memory TF-IDF
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            query_vec = self.vectorizer.transform([query])
            sims = cosine_similarity(query_vec, self.doc_vectors).flatten()
            top_indices = sims.argsort()[::-1][:top_k]
            return [self.documents[i] for i in top_indices if sims[i] > 0.05] or [self.documents[0]]
        except Exception as e:
            logger.warning(f"TF-IDF fallback search failed: {e}")
            return KNOWLEDGE_DOCUMENTS[:top_k]
