import re
from pathlib import Path

from app.models import Chunk, Document, SourceDocument


SENTENCE_BOUNDARY_PATTERN = re.compile(
    r"(?<=[。！？!?])|(?<=\.)(?=\s|$)|\n{2,}"
)


def load_documents(data_directory: Path) -> list[Document]:
    documents = []

    for file_path in sorted(data_directory.glob("*.md")):
        content = file_path.read_text(encoding="utf-8").strip()

        if content:
            documents.append(
                Document(
                    content=content,
                    source=file_path.name,
                )
            )

    return documents


def split_sentences(content: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", sentence).strip()
        for sentence in SENTENCE_BOUNDARY_PATTERN.split(content.strip())
        if sentence.strip()
    ]


def _join_sentences(sentences: list[str]) -> str:
    return " ".join(sentences)


def _overlap_sentences(sentences: list[str], overlap: int) -> list[str]:
    overlapping_sentences: list[str] = []

    for sentence in reversed(sentences):
        candidate = [sentence, *overlapping_sentences]
        if overlapping_sentences and len(_join_sentences(candidate)) > overlap:
            break
        overlapping_sentences = candidate

    return overlapping_sentences


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 600,
    overlap: int = 120,
) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if chunk_size <= 0 or overlap < 0:
        raise ValueError("chunk_size must be positive and overlap cannot be negative")

    chunks = []

    for document in documents:
        document_chunks: list[str] = []
        current_sentences: list[str] = []

        for sentence in split_sentences(document.content):
            candidate = _join_sentences([*current_sentences, sentence])

            if current_sentences and len(candidate) > chunk_size:
                document_chunks.append(_join_sentences(current_sentences))
                overlapping_sentences = _overlap_sentences(
                    current_sentences,
                    overlap,
                )
                overlapping_candidate = _join_sentences(
                    [*overlapping_sentences, sentence]
                )
                current_sentences = (
                    overlapping_sentences
                    if len(overlapping_candidate) <= chunk_size
                    else []
                )

            current_sentences.append(sentence)

        if current_sentences:
            document_chunks.append(_join_sentences(current_sentences))

        for chunk_index, content in enumerate(document_chunks):
            chunks.append(
                Chunk(
                    content=content,
                    source=document.source,
                    chunk_index=chunk_index,
                    source_url=document.source_url,
                    provider=document.provider,
                    external_id=document.external_id,
                )
            )

    return chunks

def normalize_source_documents(
    source_documents: list[SourceDocument],
) -> list[Document]:
    return [
        Document(
            content=document.content,
            source=document.title,
            source_url=document.source_url,
            provider=document.provider,
            external_id=document.external_id,
        )
        for document in source_documents
    ]
