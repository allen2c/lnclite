# Indexing

Call `create_index()` after adding a batch of documents:

```python
await client.create_index()
```

`lnclite` always creates a tag index. For small document sets, vector search may stay brute-force because exact search is fast and avoids vector index training overhead.

Choose vector search preference when constructing the client:

```python
client = await Lnclite.new(
    lancedb_path="outputs/demo.lance",
    openai_embeddings_model=embeddings,
    model_settings=ModelSettings(dimensions=1536),
    vector_search_prefer="balanced",
)
```

Available preferences are `storage`, `balanced`, `accuracy`, and `latency`.
