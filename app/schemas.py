from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class SourceResponse(BaseModel):
    source: str
    provider: str
    source_url: str
    chunk_index: int
    score: float
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]