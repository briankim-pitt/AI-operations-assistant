from pathlib import Path

from app.ingestion import chunk_documents, load_documents
from app.retrieval import Retriever
from app.vector_store import LocalVectorStore

DATA_DIRECTORY = Path(__file__).parent.parent / "data"


def test_retrieves_deployment_policy() -> None:
    documents = load_documents(DATA_DIRECTORY)
    chunks = chunk_documents(documents)
    retriever = Retriever(LocalVectorStore(chunks))

    results = retriever.retrieve("Are Friday deployments allowed?")

    assert results
    assert results[0].chunk.source == "deployment-guide.md"
    assert results[0].score > 0