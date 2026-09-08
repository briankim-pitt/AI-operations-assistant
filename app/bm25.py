import math
from collections import Counter


class BM25Index:
    def __init__(
        self,
        documents: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not documents:
            raise ValueError("At least one tokenized document is required")

        self.document_term_frequencies = [
            Counter(document) for document in documents
        ]
        self.document_lengths = [len(document) for document in documents]
        self.average_document_length = (
            sum(self.document_lengths) / len(self.document_lengths)
        )
        self.document_count = len(documents)
        self.k1 = k1
        self.b = b

        self.document_frequencies: Counter[str] = Counter()
        for document in documents:
            self.document_frequencies.update(set(document))

    def scores(self, query_tokens: list[str]) -> list[float]:
        return [
            self._score_document(query_tokens, document_index)
            for document_index in range(self.document_count)
        ]

    def _score_document(
        self,
        query_tokens: list[str],
        document_index: int,
    ) -> float:
        score = 0.0
        term_frequencies = self.document_term_frequencies[document_index]
        document_length = self.document_lengths[document_index]

        for token in set(query_tokens):
            term_frequency = term_frequencies[token]
            if term_frequency == 0:
                continue

            document_frequency = self.document_frequencies[token]
            inverse_document_frequency = math.log(
                1
                + (
                    self.document_count
                    - document_frequency
                    + 0.5
                )
                / (document_frequency + 0.5)
            )
            length_normalization = self.k1 * (
                1
                - self.b
                + self.b
                * document_length
                / max(self.average_document_length, 1)
            )
            score += inverse_document_frequency * (
                term_frequency * (self.k1 + 1)
                / (term_frequency + length_normalization)
            )

        return score
