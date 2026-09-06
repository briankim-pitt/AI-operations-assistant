import os
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from app.main import ask
from app.models import Chunk, SearchResult
from app.schemas import AskRequest, ConversationMessage


class FakeRetriever:
    def __init__(self) -> None:
        self.query = ""
        self.top_k = 0

    def retrieve(self, query: str, top_k: int = 3) -> list[SearchResult]:
        self.query = query
        self.top_k = top_k
        return [
            SearchResult(
                chunk=Chunk(
                    content="Friday deployments require Head of Engineering approval.",
                    source="deployment-guide.md",
                    chunk_index=0,
                ),
                score=0.9,
            )
        ]


class FakeGenerator:
    def __init__(self) -> None:
        self.history = []

    def generate(self, question, results, history=None) -> str:
        self.history = history or []
        return "The Head of Engineering provides approval."


def test_follow_up_uses_session_history() -> None:
    retriever = FakeRetriever()
    generator = FakeGenerator()
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                retriever=retriever,
                answer_generator=generator,
            )
        )
    )
    payload = AskRequest(
        question="Who approves it?",
        history=[
            ConversationMessage(
                role="user",
                content="Can we deploy on Friday?",
            ),
            ConversationMessage(
                role="assistant",
                content="Friday deployments require special approval.",
            ),
        ],
    )

    response = ask(payload, request)

    assert "Can we deploy on Friday?" in retriever.query
    assert "Friday deployments require special approval." not in retriever.query
    assert retriever.query.endswith("Who approves it?")
    assert retriever.top_k == 6
    assert generator.history == [
        {
            "role": "user",
            "content": "Can we deploy on Friday?",
        },
        {
            "role": "assistant",
            "content": "Friday deployments require special approval.",
        },
    ]
    assert response.answer == "The Head of Engineering provides approval."
