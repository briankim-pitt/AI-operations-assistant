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
                "Give a concise answer in 1-3 sentences."
            ),
            input=f"Context:\n{context}\n\nQuestion:\n{question}",
            reasoning={"effort": "minimal"},
            text={"verbosity": "low"},
            max_output_tokens=800,
            store=False,
        )

        return response.output_text