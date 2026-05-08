# API Reference

## `Lnclite`

Main async client.

- `await Lnclite.new(...)`: create a new store.
- `await Lnclite.load(...)`: load an existing store.
- `await Lnclite.new_from_dir(...)`: create a store from text files.
- `await client.create_index()`: create tag and vector indexes.
- `await client.search(query, tags_any=None, tags_all=None, limit=5, include_vector=False)`: semantic search.

## `LncliteManager`

Lightweight manager for named local datasets. It lazily opens clients, reuses
cached clients, and can close clients by name, idle TTL, or all at once.

`lnclite` does not provide HTTP, authentication, authorization, or routing.
Applications should implement those concerns outside the library.

## `client.documents`

- `await client.documents.create(document_create)`: add one document.
- `await client.documents.batch_create(document_creates)`: add many documents.
- `await client.documents.batch_insert(document_creates)`: add many documents and return only the inserted count.
- `await client.documents.batch_insert_embedded(document_creates, vectors, normalize_vectors=True)`: add documents with caller-supplied vectors and return only the inserted count.
- `await client.documents.get(id, include_vector=False)`: return a document or `None`.
- `await client.documents.retrieve(id, include_vector=False)`: return a document or raise `LncliteNotFoundError`.
- `await client.documents.list(..., include_vector=False)`: list documents with pagination and tag filters.
- `await client.documents.index_plan()`: inspect recommended index behavior.

Use `batch_insert()` or `batch_insert_embedded()` for large ingestion workloads
where returned `Document` objects are not needed. Pass `verbose=True` to
`batch_create()`, `batch_insert()`, or `batch_insert_embedded()` to log phase
timings for table access, embedding, normalization, row construction, LanceDB
append, and return-object construction when applicable.

## Models

- `DocumentCreate`: input model with `content` and `tags`.
- `Document`: stored document with `id`, `content`, `md5`, `vector`, and `tags`.
- `DocumentIndexPlan`: vector index recommendation details.
- `LncliteConfig`: configuration for opening a store through `LncliteManager`.
- `SearchResult`: search hit with `document` and `distance`.
- `SearchResults`: wrapper with `results`.
- `ManifestModel`: database manifest metadata.

## Helpers

- `get_openai_embeddings_model(...)`: create the embeddings model wrapper.
- `get_model_settings(dimensions=1536)`: create default model settings.
