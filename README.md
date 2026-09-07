# AI Operations Assistant

The local retriever uses hybrid ranking:

- 70% semantic similarity from OpenAI embeddings
- 30% NFKC-normalized character n-gram similarity

Semantic retrieval handles paraphrases and Japanese politeness variations. The
lexical signal improves matches for exact product names, abbreviations, and codes
such as `SRC-401`, even when the query uses full-width characters.
