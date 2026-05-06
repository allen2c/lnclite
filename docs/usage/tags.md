# Tags

Documents can carry tags:

```python
DocumentCreate(
    content="A short note about search.",
    tags=["type:note", "topic:search"],
)
```

Use `tags_any` to match at least one tag:

```python
results = await client.search(
    "search design",
    tags_any=["topic:search", "topic:indexing"],
)
```

Use `tags_all` to require every tag:

```python
page = await client.documents.list(
    tags_all=["type:note", "topic:search"],
)
```

A practical convention is `namespace:value`, such as `topic:search`, `type:note`, or `path:docs/example.md`.
