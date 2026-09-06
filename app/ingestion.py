from pathlib import Path

from app.models import Chunk, Document, SourceDocument

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

def chunk_documents(
    documents: list[Document],
    chunk_size: int = 80,
    overlap: int = 20,
) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    step_size = chunk_size - overlap

    for document in documents:
        words = document.content.split()

        for chunk_index, start in enumerate(range(0, len(words), step_size)):
            chunk_words = words[start : start + chunk_size]

            if not chunk_words:
                continue

            chunks.append(
                Chunk(
                    content=" ".join(chunk_words),
                    source=document.source,
                    chunk_index=chunk_index,
                )
            )

            if start + chunk_size >= len(words):
                break

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