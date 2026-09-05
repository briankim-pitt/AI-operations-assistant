from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request

from app.generation import AnswerGenerator
from app.ingestion import chunk_documents, load_documents
from app.retrieval import Retriever
from app.schemas import AskRequest, AskResponse, SourceResponse
from app.vector_store import LocalVectorStore

DATA_DIRECTORY = Path(__file__).parent.parent / "data"


@asynccontextmanager
async def lifespan(app: FastAPI):
    documents = load_documents(DATA_DIRECTORY)
    chunks = chunk_documents(documents)

    app.state.retriever = Retriever(LocalVectorStore(chunks))
    app.state.answer_generator = AnswerGenerator()

    yield


app = FastAPI(
    title="AI Operations Assistant",
    lifespan=lifespan,
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request) -> AskResponse:
    results = request.app.state.retriever.retrieve(payload.question)
    answer = request.app.state.answer_generator.generate(
        payload.question,
        results,
    )

    sources = [
        SourceResponse(
            source=result.chunk.source,
            chunk_index=result.chunk.chunk_index,
            score=round(result.score, 4),
            excerpt=result.chunk.content,
        )
        for result in results
    ]

    return AskResponse(answer=answer, sources=sources)