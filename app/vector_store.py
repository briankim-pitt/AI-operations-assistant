from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models import Chunk, SearchResult


class LocalVectorStore:
    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required")

        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.chunk_vectors = self.vectorizer.fit_transform(
            chunk.content for chunk in chunks
        )

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.chunk_vectors)[0]
        best_indexes = similarities.argsort()[::-1][:top_k]

        return [
            SearchResult(
                chunk=self.chunks[index],
                score=float(similarities[index]),
            )
            for index in best_indexes
            if similarities[index] > 0
        ]