# Directory Ingest

`Lnclite.new_from_dir()` reads text files from a directory and creates documents.

```python
client = await Lnclite.new_from_dir(
    dir_path="notes",
    lancedb_path="outputs/notes.lance",
    dataset_name="Notes",
    dataset_description="Local text notes",
    openai_embeddings_model=embeddings,
    model_settings=ModelSettings(dimensions=1536),
)
```

Each ingested file receives a path tag:

```text
path:subfolder/example.md
```

You can filter by path tag:

```python
page = await client.documents.list(
    tags_all=["path:subfolder/example.md"],
)
```
