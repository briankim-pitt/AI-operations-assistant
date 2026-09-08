# AI Operations Assistant

The local retriever uses three-signal hybrid ranking:

- 70% semantic similarity from OpenAI embeddings
- 20% BM25 over Sudachi-tokenized Japanese
- 10% NFKC-normalized character n-gram similarity

Semantic retrieval handles paraphrases and Japanese politeness variations. The
Sudachi/BM25 signal handles meaningful Japanese terms, while character n-grams
improve exact matches for abbreviations and codes such as `SRC-401`, even when
the query uses full-width characters.

Embeddings are cached in `.cache/embeddings.sqlite3`. On startup, unchanged
normalized chunks reuse their saved vectors. Only new or edited chunks are sent
to the embedding API. Changing `OPENAI_EMBEDDING_MODEL` creates different cache
keys automatically.
