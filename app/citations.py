from dataclasses import dataclass

from app.ingestion import split_sentences
from app.tokenization import SudachiTokenizer


IGNORED_HIGHLIGHT_TERMS = {
    "有る",
    "居る",
    "為る",
    "成る",
    "こと",
    "これ",
    "それ",
}


@dataclass(frozen=True)
class RelevantExcerpt:
    text: str
    highlights: list[str]


class CitationExcerptSelector:
    def __init__(self, tokenizer: SudachiTokenizer) -> None:
        self.tokenizer = tokenizer

    def select(
        self,
        content: str,
        question: str,
        answer: str,
    ) -> RelevantExcerpt:
        sentences = split_sentences(content)
        if not sentences:
            return RelevantExcerpt(text=content, highlights=[])

        target_tokens = {
            token
            for token in self.tokenizer.tokenize(f"{question} {answer}")
            if token not in IGNORED_HIGHLIGHT_TERMS
        }

        best_sentence = sentences[0]
        best_highlights: list[str] = []
        best_score = -1

        for sentence in sentences:
            sentence_terms = self.tokenizer.tokenize_with_surfaces(sentence)
            highlights = list(
                dict.fromkeys(
                    surface
                    for surface, normalized in sentence_terms
                    if normalized in target_tokens
                )
            )
            score = sum(max(len(highlight), 1) for highlight in highlights)

            if score > best_score:
                best_sentence = sentence
                best_highlights = highlights
                best_score = score

        return RelevantExcerpt(
            text=best_sentence,
            highlights=best_highlights,
        )
