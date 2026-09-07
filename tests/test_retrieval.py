from pathlib import Path

from app.ingestion import chunk_documents, load_documents, split_sentences
from app.models import Chunk, Document
from app.normalization import normalize_for_search
from app.retrieval import Retriever
from app.vector_store import LocalVectorStore

DATA_DIRECTORY = Path(__file__).parent.parent / "data"


class RecordingEmbeddingProvider:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [
            [
                float(
                    "deploy" in text.lower()
                    or "friday" in text.lower()
                ),
                1.0,
            ]
            for text in texts
        ]


class ConstantEmbeddingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 1.0] for _ in texts]


def test_retrieves_deployment_policy() -> None:
    documents = load_documents(DATA_DIRECTORY)
    chunks = chunk_documents(documents)
    embedding_provider = RecordingEmbeddingProvider()
    retriever = Retriever(LocalVectorStore(chunks, embedding_provider))

    results = retriever.retrieve("Are Friday deployments allowed?")

    assert results
    assert results[0].chunk.source == "deployment-guide.md"
    assert results[0].score > 0


def test_nfkc_normalizes_japanese_search_variants() -> None:
    assert normalize_for_search("Ｔｏｋｙｏ　１２３") == "Tokyo 123"
    assert normalize_for_search("ｿﾌﾄｳｪｱ") == "ソフトウェア"


def test_vector_store_normalizes_index_and_query_text() -> None:
    chunk = Chunk(
        content="東京の製品コードはＡＢＣ－１２３です。",
        source="products.md",
        chunk_index=0,
    )
    embedding_provider = RecordingEmbeddingProvider()
    vector_store = LocalVectorStore([chunk], embedding_provider)

    vector_store.search("ＡＢＣ－１２３")

    assert embedding_provider.calls == [
        ["東京の製品コードはABC-123です。"],
        ["ABC-123"],
    ]
    assert vector_store.chunks[0].content == chunk.content


def test_hybrid_search_recovers_exact_japanese_error_code() -> None:
    chunks = [
        Chunk(
            content="認証エラーSRC-401ではシークレットの期限を確認します。",
            source="pipeline.md",
            chunk_index=0,
        ),
        Chunk(
            content="集計データベースの復旧後にジョブを再開します。",
            source="database.md",
            chunk_index=0,
        ),
    ]
    vector_store = LocalVectorStore(
        chunks,
        ConstantEmbeddingProvider(),
        semantic_weight=0.7,
    )

    results = vector_store.search("ＳＲＣ－４０１とは何ですか？")

    assert results[0].chunk.source == "pipeline.md"
    assert results[0].score > results[1].score


def test_splits_japanese_without_spaces_at_sentence_boundaries() -> None:
    content = "東京で障害が発生しました。担当者へ連絡してください！復旧しました。"

    assert split_sentences(content) == [
        "東京で障害が発生しました。",
        "担当者へ連絡してください！",
        "復旧しました。",
    ]


def test_chunks_overlap_using_complete_sentences() -> None:
    document = Document(
        content="第一文です。第二文は少し長いです。第三文です。",
        source="runbook.md",
    )

    chunks = chunk_documents([document], chunk_size=20, overlap=10)

    assert [chunk.content for chunk in chunks] == [
        "第一文です。 第二文は少し長いです。",
        "第二文は少し長いです。 第三文です。",
    ]


def test_keeps_single_long_japanese_sentence_intact() -> None:
    sentence = "これは空白を含まない非常に長い日本語の文章です。"
    document = Document(content=sentence, source="policy.md")

    chunks = chunk_documents([document], chunk_size=10, overlap=2)

    assert [chunk.content for chunk in chunks] == [sentence]
