from openai import OpenAI

from app.config import settings
from app.models import SearchResult


class AnswerGenerator:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.openai_api_key)

    def generate(self, question: str, results: list[SearchResult]) -> str:
        if not results:
            return "I could not find relevant information in the company documents."

        context = "\n\n".join(
            (
                f"[Source: {result.chunk.source}, "
                f"chunk {result.chunk.chunk_index}]\n"
                f"{result.chunk.content}"
            )
            for result in results
        )

        response = self.client.responses.create(
            model=settings.openai_model,
            instructions=(
                "You are an internal operations assistant. "
                "Answer only from the supplied context. "
                "If the context does not contain the answer, say so. "
                "Do not invent company policies or facts."
            ),
            input=f"Context:\n{context}\n\nQuestion:\n{question}",
            max_output_tokens=300,
            store=False,
        )

        return response.output_text