import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.bm25 import BM25Index
from app.embeddings import EmbeddingProvider
from app.models import Chunk, SearchResult
from app.normalization import normalize_for_search
from app.tokenization import SudachiTokenizer


class LocalVectorStore:
    def __init__(
        self,
        chunks: list[Chunk],
        embedding_provider: EmbeddingProvider,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.2,
    ) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required")
        if semantic_weight < 0 or bm25_weight < 0:
            raise ValueError("retrieval weights cannot be negative")
        if semantic_weight + bm25_weight > 1:
            raise ValueError("semantic_weight and bm25_weight cannot exceed 1")

        self.chunks = chunks
        self.embedding_provider = embedding_provider
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.character_weight = 1 - semantic_weight - bm25_weight
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
        self.tokenizer = SudachiTokenizer()
        self.bm25_index = BM25Index(
            [self.tokenizer.tokenize(content) for content in self.normalized_chunks]
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
        bm25_scores = np.asarray(
            self.bm25_index.scores(
                self.tokenizer.tokenize(normalized_query)
            )
        )
        maximum_bm25_score = bm25_scores.max(initial=0)
        if maximum_bm25_score > 0:
            bm25_scores = bm25_scores / maximum_bm25_score
        similarities = (
            self.semantic_weight * semantic_similarities
            + self.bm25_weight * bm25_scores
            + self.character_weight * lexical_similarities
        )

        best_indexes = similarities.argsort()[::-1][:top_k]

        return [
            SearchResult(
                chunk=self.chunks[index],
                score=float(similarities[index]),
            )
            for index in best_indexes
        ]
