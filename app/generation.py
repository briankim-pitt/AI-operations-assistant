from openai import OpenAI

from app.config import settings
from app.models import SearchResult


class AnswerGenerator:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.openai_api_key)

    def generate(
        self,
        question: str,
        results: list[SearchResult],
        history: list[dict[str, str]] | None = None,
    ) -> str:
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

        conversation = "\n".join(
            f"{message['role'].title()}: {message['content']}"
            for message in history or []
        )

        response = self.client.responses.create(
            model=settings.openai_model,
            instructions=(
                "You are an internal operations assistant. "
                "Answer only from the supplied context. "
                "If the context does not contain the answer, say so. "
                "Give a concise answer in 1-3 sentences. "
                "Use conversation history only to understand references and follow-up questions; "
                "do not treat it as authoritative company information. "
                "Resolve pronouns and short follow-ups from the conversation history. "
                "Examine every supplied context chunk before deciding that information is unavailable."
            ),
            input=(
                f"Conversation history:\n{conversation or 'None'}\n\n"
                f"Context:\n{context}\n\n"
                f"Current question:\n{question}"
            ),
            reasoning={"effort": "minimal"},
            text={"verbosity": "low"},
            max_output_tokens=800,
            store=False,
        )

        return response.output_text
