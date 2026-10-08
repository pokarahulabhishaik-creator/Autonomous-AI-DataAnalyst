from typing import List, Dict, Any
from backend.rag.vector_store import VectorStore

_STORE_INSTANCE = None

def get_vector_store() -> VectorStore:
    global _STORE_INSTANCE
    if _STORE_INSTANCE is None:
        _STORE_INSTANCE = VectorStore()
    return _STORE_INSTANCE

class AnalyticsRetriever:
    """Retriever for analytical guidelines, statistical rules, and KPI standards."""

    def __init__(self):
        self.store = get_vector_store()

    def retrieve_guidelines(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant analytical guidance based on user question or analytical finding."""
        return self.store.search(query, top_k=top_k)

    def get_context_string(self, query: str, top_k: int = 2) -> str:
        """Format retrieved guidelines as concise string context for prompt injection."""
        docs = self.retrieve_guidelines(query, top_k=top_k)
        if not docs:
            return ""
        formatted = []
        for d in docs:
            formatted.append(f"[{d.get('title')}]: {d.get('content')}")
        return "\n".join(formatted)
