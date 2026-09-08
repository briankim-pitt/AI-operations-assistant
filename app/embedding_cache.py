import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Protocol


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class CachedEmbeddingProvider:
    def __init__(
        self,
        provider: EmbeddingProvider,
        database_path: Path,
        namespace: str,
    ) -> None:
        self.provider = provider
        self.database_path = database_path
        self.namespace = namespace
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        keys = [self._cache_key(text) for text in texts]
        cached_embeddings = self._load_embeddings(keys)
        missing_by_key = {
            key: text
            for key, text in zip(keys, texts, strict=True)
            if key not in cached_embeddings
        }

        if missing_by_key:
            missing_keys = list(missing_by_key)
            new_embeddings = self.provider.embed(
                [missing_by_key[key] for key in missing_keys]
            )
            if len(new_embeddings) != len(missing_keys):
                raise ValueError("Embedding provider returned an unexpected result count")

            generated_embeddings = dict(
                zip(missing_keys, new_embeddings, strict=True)
            )
            self._store_embeddings(generated_embeddings)
            cached_embeddings.update(generated_embeddings)

        return [cached_embeddings[key] for key in keys]

    def _cache_key(self, text: str) -> str:
        content = f"{self.namespace}\0{text}".encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path, timeout=30)

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _load_embeddings(
        self,
        keys: list[str],
    ) -> dict[str, list[float]]:
        embeddings: dict[str, list[float]] = {}

        with self._connect() as connection:
            for start in range(0, len(keys), 500):
                batch = keys[start : start + 500]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"SELECT cache_key, embedding FROM embeddings "
                    f"WHERE cache_key IN ({placeholders})",
                    batch,
                )
                embeddings.update(
                    (cache_key, json.loads(embedding))
                    for cache_key, embedding in rows
                )

        return embeddings

    def _store_embeddings(
        self,
        embeddings: dict[str, list[float]],
    ) -> None:
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO embeddings (
                    cache_key,
                    namespace,
                    embedding
                ) VALUES (?, ?, ?)
                """,
                [
                    (
                        cache_key,
                        self.namespace,
                        json.dumps(embedding),
                    )
                    for cache_key, embedding in embeddings.items()
                ],
            )
