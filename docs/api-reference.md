# API Reference

## `Lnclite`

Main async client.

- `await Lnclite.new(...)`: create a new store.
- `await Lnclite.load(...)`: load an existing store.
- `await Lnclite.new_from_dir(...)`: create a store from text files.
- `await client.create_index()`: create tag and vector indexes.
- `await client.search(query, tags_any=None, tags_all=None, limit=5)`: semantic search.

## `client.documents`

- `await client.documents.create(document_create)`: add one document.
- `await client.documents.batch_create(document_creates)`: add many documents.
- `await client.documents.get(id)`: return a document or `None`.
- `await client.documents.retrieve(id)`: return a document or raise `LncliteNotFoundError`.
- `await client.documents.list(...)`: list documents with pagination and tag filters.

## Models

- `DocumentCreate`: input model with `content` and `tags`.
- `Document`: stored document with `id`, `content`, `md5`, `vector`, and `tags`.
- `SearchResult`: search hit with `document` and `distance`.
- `SearchResults`: wrapper with `results`.
- `ManifestModel`: database manifest metadata.

## Helpers

- `get_openai_embeddings_model(...)`: create the embeddings model wrapper.
- `get_model_settings(dimensions=1536)`: create default model settings.
