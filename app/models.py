from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping

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
    
@dataclass(frozen=True)
class SourceDocument:
    external_id: str
    provider: str
    title: str
    content: str
    source_url: str
    updated_at: datetime | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)