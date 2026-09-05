from sklearn.metrics.pairwise import cosine_similarity

from app.embeddings import EmbeddingProvider
from app.models import Chunk, SearchResult


class LocalVectorStore:
    def __init__(
        self,
        chunks: list[Chunk],
        embedding_provider: EmbeddingProvider,
    ) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required")

        self.chunks = chunks
        self.embedding_provider = embedding_provider
        self.chunk_vectors = self.embedding_provider.embed(
            [chunk.content for chunk in chunks]
        )

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        query_vector = self.embedding_provider.embed([query])[0]
        similarities = cosine_similarity(
            [query_vector],
            self.chunk_vectors,
        )[0]

        best_indexes = similarities.argsort()[::-1][:top_k]

        return [
            SearchResult(
                chunk=self.chunks[index],
                score=float(similarities[index]),
            )
            for index in best_indexes
        ]