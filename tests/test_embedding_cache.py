from pathlib import Path

from app.embedding_cache import CachedEmbeddingProvider


class RecordingEmbeddingProvider:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [
            [float(index), float(len(text))]
            for index, text in enumerate(texts)
        ]


def create_cached_provider(
    provider: RecordingEmbeddingProvider,
    database_path: Path,
    namespace: str = "test-model",
) -> CachedEmbeddingProvider:
    return CachedEmbeddingProvider(
        provider=provider,
        database_path=database_path,
        namespace=namespace,
    )


def test_reuses_embeddings_across_provider_instances(tmp_path: Path) -> None:
    database_path = tmp_path / "embeddings.sqlite3"
    first_provider = RecordingEmbeddingProvider()
    first_cache = create_cached_provider(first_provider, database_path)

    first_result = first_cache.embed(["東京", "大阪"])

    second_provider = RecordingEmbeddingProvider()
    second_cache = create_cached_provider(second_provider, database_path)
    second_result = second_cache.embed(["東京", "大阪"])

    assert first_result == second_result
    assert first_provider.calls == [["東京", "大阪"]]
    assert second_provider.calls == []


def test_only_embeds_missing_texts(tmp_path: Path) -> None:
    provider = RecordingEmbeddingProvider()
    cache = create_cached_provider(
        provider,
        tmp_path / "embeddings.sqlite3",
    )

    cache.embed(["既存の文書"])
    cache.embed(["既存の文書", "新しい文書"])

    assert provider.calls == [["既存の文書"], ["新しい文書"]]


def test_embedding_model_namespace_invalidates_cache(tmp_path: Path) -> None:
    database_path = tmp_path / "embeddings.sqlite3"
    first_provider = RecordingEmbeddingProvider()
    create_cached_provider(
        first_provider,
        database_path,
        namespace="model-a",
    ).embed(["同じ文書"])

    second_provider = RecordingEmbeddingProvider()
    create_cached_provider(
        second_provider,
        database_path,
        namespace="model-b",
    ).embed(["同じ文書"])

    assert first_provider.calls == [["同じ文書"]]
    assert second_provider.calls == [["同じ文書"]]
