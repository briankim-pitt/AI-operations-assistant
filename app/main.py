from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.citations import CitationExcerptSelector
from app.generation import AnswerGenerator
from app.retrieval import Retriever
from app.schemas import AskRequest, AskResponse, SourceResponse
from app.vector_store import LocalVectorStore
from app.config import settings
from app.connectors.google_drive import GoogleDriveConnector
from app.embedding_cache import CachedEmbeddingProvider
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
    openai_embedding_provider = OpenAIEmbeddingProvider()
    embedding_provider = CachedEmbeddingProvider(
        provider=openai_embedding_provider,
        database_path=settings.embedding_cache_path,
        namespace=settings.openai_embedding_model,
    )
    vector_store = LocalVectorStore(chunks, embedding_provider)

    app.state.retriever = Retriever(vector_store)
    app.state.answer_generator = AnswerGenerator()
    app.state.citation_selector = CitationExcerptSelector(
        vector_store.tokenizer
    )

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
    history = [message.model_dump() for message in payload.history]
    recent_user_questions = [
        message.content
        for message in payload.history
        if message.role == "user"
    ][-2:]
    retrieval_query = "\n".join(
        [*recent_user_questions, payload.question]
    )

    results = request.app.state.retriever.retrieve(
        retrieval_query,
        top_k=6,
    )
    answer = request.app.state.answer_generator.generate(
        payload.question,
        results,
        history,
    )

    sources = []
    seen_sources = set()

    for result in results:
        source_key = (result.chunk.provider, result.chunk.source)
        if source_key in seen_sources:
            continue

        seen_sources.add(source_key)
        relevant_excerpt = request.app.state.citation_selector.select(
            result.chunk.content,
            payload.question,
            answer,
        )
        sources.append(
            SourceResponse(
                source=result.chunk.source,
                provider=result.chunk.provider,
                source_url=result.chunk.source_url,
                chunk_index=result.chunk.chunk_index,
                score=round(result.score, 4),
                excerpt=relevant_excerpt.text,
                highlights=relevant_excerpt.highlights,
            )
        )

        if len(sources) == 3:
            break

    return AskResponse(answer=answer, sources=sources)
