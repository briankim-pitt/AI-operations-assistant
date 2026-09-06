from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.generation import AnswerGenerator
from app.retrieval import Retriever
from app.schemas import AskRequest, AskResponse, SourceResponse
from app.vector_store import LocalVectorStore
from app.config import settings
from app.connectors.google_drive import GoogleDriveConnector
from app.embeddings import OpenAIEmbeddingProvider
from app.ingestion import (
    chunk_documents,
    load_documents,
    normalize_source_documents,
)

DATA_DIRECTORY = Path(__file__).parent.parent / "data"
TEMPLATES_DIRECTORY = Path(__file__).parent / "templates"
STATIC_DIRECTORY = Path(__file__).parent / "static"
templates = Jinja2Templates(directory=TEMPLATES_DIRECTORY)


@asynccontextmanager
async def lifespan(app: FastAPI):
    documents = load_documents(DATA_DIRECTORY)

    if settings.enable_google_drive:
        if (
            settings.google_service_account_file is None
            or settings.google_drive_folder_id is None
        ):
            raise RuntimeError(
                "Google Drive is enabled, but its configuration is missing"
            )

        connector = GoogleDriveConnector(
            credentials_path=settings.google_service_account_file,
            folder_id=settings.google_drive_folder_id,
        )
        drive_source_documents = connector.fetch_documents()
        documents.extend(
            normalize_source_documents(drive_source_documents)
        )

    chunks = chunk_documents(documents)
    embedding_provider = OpenAIEmbeddingProvider()
    vector_store = LocalVectorStore(chunks, embedding_provider)

    app.state.retriever = Retriever(vector_store)
    app.state.answer_generator = AnswerGenerator()

    yield


app = FastAPI(
    title="AI Operations Assistant",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
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

    sources = []
    seen_sources = set()

    for result in results:
        source_key = (result.chunk.provider, result.chunk.source)
        if source_key in seen_sources:
            continue

        seen_sources.add(source_key)
        sources.append(
            SourceResponse(
                source=result.chunk.source,
                provider=result.chunk.provider,
                source_url=result.chunk.source_url,
                chunk_index=result.chunk.chunk_index,
                score=round(result.score, 4),
                excerpt=result.chunk.content,
            )
        )

    return AskResponse(answer=answer, sources=sources)
