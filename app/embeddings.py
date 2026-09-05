from typing import Protocol

from openai import OpenAI

from app.config import settings


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class OpenAIEmbeddingProvider:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_embedding_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        ordered_embeddings = sorted(
            response.data,
            key=lambda item: item.index,
        )

        return [item.embedding for item in ordered_embeddings]