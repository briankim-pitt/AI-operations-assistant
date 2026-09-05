from typing import Protocol

from app.models import SearchResult


class VectorStore(Protocol):
    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        ...


class Retriever:
    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 3) -> list[SearchResult]:
        clean_query = query.strip()

        if not clean_query:
            return []

        return self.vector_store.search(clean_query, top_k)