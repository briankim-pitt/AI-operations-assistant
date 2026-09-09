from dataclasses import dataclass

from app.ingestion import split_sentences
from app.tokenization import SudachiTokenizer


IGNORED_HIGHLIGHT_TERMS = {
    "有る".casefold(),
    "居る".casefold(),
    "為る".casefold(),
    "成る".casefold(),
    "こと",
    "これ",
    "それ",
    "確認",
    "担当者",
    "作成",
    "連絡",
    "当番",
}

MAX_HIGHLIGHTS = 5


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

        question_tokens = {
            token
            for token in self.tokenizer.tokenize(question)
            if token not in IGNORED_HIGHLIGHT_TERMS
        }
        answer_tokens = {
            token
            for token in self.tokenizer.tokenize(answer)
            if token not in IGNORED_HIGHLIGHT_TERMS
        }
        target_tokens = question_tokens | answer_tokens

        best_sentence = sentences[0]
        best_highlights: list[str] = []
        best_score = -1

        for sentence in sentences:
            sentence_terms = self.tokenizer.tokenize_with_surfaces(sentence)
            matched_terms = [
                (surface, normalized)
                for surface, normalized in sentence_terms
                if normalized in target_tokens
                and normalized not in IGNORED_HIGHLIGHT_TERMS
            ]
            highlights = [surface for surface, _ in matched_terms]
            score = sum(max(len(highlight), 1) for highlight in highlights)

            if score > best_score:
                best_sentence = sentence
                best_highlights = self._select_highlights(
                    matched_terms,
                    question_tokens,
                )
                best_score = score

        return RelevantExcerpt(
            text=best_sentence,
            highlights=best_highlights,
        )

    def _select_highlights(
        self,
        matched_terms: list[tuple[str, str]],
        question_tokens: set[str],
    ) -> list[str]:
        unique_terms = list(dict.fromkeys(matched_terms))
        ranked_terms = sorted(
            unique_terms,
            key=lambda term: (
                term[1] in question_tokens,
                any(character.isdigit() for character in term[0]),
                len(term[0]),
            ),
            reverse=True,
        )
        selected_surfaces = {
            surface
            for surface, _ in ranked_terms[:MAX_HIGHLIGHTS]
        }

        return [
            surface
            for surface, _ in unique_terms
            if surface in selected_surfaces
        ]
