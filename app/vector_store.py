from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.embeddings import EmbeddingProvider
from app.models import Chunk, SearchResult
from app.normalization import normalize_for_search


class LocalVectorStore:
    def __init__(
        self,
        chunks: list[Chunk],
        embedding_provider: EmbeddingProvider,
        semantic_weight: float = 0.7,
    ) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required")
        if not 0 <= semantic_weight <= 1:
            raise ValueError("semantic_weight must be between 0 and 1")

        self.chunks = chunks
        self.embedding_provider = embedding_provider
        self.semantic_weight = semantic_weight
        self.lexical_weight = 1 - semantic_weight
        self.normalized_chunks = [
            normalize_for_search(chunk.content) for chunk in chunks
        ]
        self.chunk_vectors = self.embedding_provider.embed(
            self.normalized_chunks
        )
        self.lexical_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            sublinear_tf=True,
        )
        self.lexical_vectors = self.lexical_vectorizer.fit_transform(
            self.normalized_chunks
        )

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        normalized_query = normalize_for_search(query)
        query_vector = self.embedding_provider.embed([normalized_query])[0]
        semantic_similarities = cosine_similarity(
            [query_vector],
            self.chunk_vectors,
        )[0]
        lexical_query_vector = self.lexical_vectorizer.transform(
            [normalized_query]
        )
        lexical_similarities = cosine_similarity(
            lexical_query_vector,
            self.lexical_vectors,
        )[0]
        similarities = (
            self.semantic_weight * semantic_similarities
            + self.lexical_weight * lexical_similarities
        )

        best_indexes = similarities.argsort()[::-1][:top_k]

        return [
            SearchResult(
                chunk=self.chunks[index],
                score=float(similarities[index]),
            )
            for index in best_indexes
        ]
