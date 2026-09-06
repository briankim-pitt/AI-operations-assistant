from typing import Literal

from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    history: list[ConversationMessage] = Field(
        default_factory=list,
        max_length=10,
    )


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
