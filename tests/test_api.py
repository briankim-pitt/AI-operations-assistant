import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from fastapi.testclient import TestClient

from app.main import app


class FakeAnswerGenerator:
    def generate(self, question, results) -> str:
        return "Friday deployments require Head of Engineering approval."


def test_ask_returns_answer_and_sources() -> None:
    with TestClient(app) as client:
        app.state.answer_generator = FakeAnswerGenerator()

        response = client.post(
            "/ask",
            json={"question": "Are Friday deployments allowed?"},
        )

    assert response.status_code == 200

    body = response.json()
    assert "Head of Engineering" in body["answer"]
    assert body["sources"]
    assert body["sources"][0]["source"] == "deployment-guide.md"


def test_ask_rejects_empty_question() -> None:
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": ""})

    assert response.status_code == 422