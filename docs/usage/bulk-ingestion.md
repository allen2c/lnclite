# Bulk Ingestion

Use `batch_create()` when you need the inserted `Document` objects back. For
large ingestion workloads, prefer count-returning insert APIs so Python does
not build return models or convert every vector to a list.

## Embed And Insert

`batch_insert()` keeps the standard lnclite embedding path and returns the
number of inserted rows:

```python
inserted = await client.documents.batch_insert(
    [
        DocumentCreate(content="First corpus document", tags=["source:corpus"]),
        DocumentCreate(content="Second corpus document", tags=["source:corpus"]),
    ],
    verbose=True,
)
```

When `verbose=True`, lnclite logs sub-timings for the insert path. This helps
separate embedding cost from row construction and LanceDB append cost.

## Insert Precomputed Vectors

If your ingestion pipeline embeds documents outside lnclite, use
`batch_insert_embedded()`:

```python
inserted = await client.documents.batch_insert_embedded(
    [
        DocumentCreate(content="First corpus document", tags=["source:corpus"]),
        DocumentCreate(content="Second corpus document", tags=["source:corpus"]),
    ],
    vectors,
    verbose=True,
)
```

Vectors are normalized by default so search behavior matches the standard
embedding path. If the caller has already normalized vectors, skip that work:

```python
inserted = await client.documents.batch_insert_embedded(
    documents,
    normalized_vectors,
    normalize_vectors=False,
)
```

`batch_insert_embedded()` validates that the vector count matches the document
count and that vector dimensions match the client's model settings.

## Index After Ingestion

Create indexes after loading the corpus:

```python
await client.documents.create_index()
```

This keeps ingestion focused on appends and avoids measuring index construction
as part of per-batch insert timing.
