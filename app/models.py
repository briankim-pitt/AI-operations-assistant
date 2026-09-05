from dataclasses import dataclass

@dataclass(frozen=True)
class Document:
    content: str
    source: str

@dataclass(frozen=True)
class Chunk:
    content: str
    source: str
    chunk_index: int
    
@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float