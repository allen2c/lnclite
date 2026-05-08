# Basic Usage

```python
import asyncio

from openai import AsyncOpenAI
from openai_embeddings_model import ModelSettings

from lnclite import DocumentCreate, Lnclite, get_openai_embeddings_model


async def main():
    embeddings = get_openai_embeddings_model(
        openai_client=AsyncOpenAI(),
    )

    client = await Lnclite.new(
        lancedb_path="outputs/demo.lance",
        openai_embeddings_model=embeddings,
        model_settings=ModelSettings(dimensions=1536),
        name="Demo",
        description="Local demo documents",
    )

    await client.documents.batch_create(
        [
            DocumentCreate(
                content="A note about async Python clients.",
                tags=["type:note", "topic:python"],
            ),
            DocumentCreate(
                content="A note about vector search and indexing.",
                tags=["type:note", "topic:search"],
            ),
        ]
    )

    await client.create_index()

    results = await client.search(
        "How should I design vector search?",
        tags_any=["topic:search"],
    )

    for result in results.results:
        print(result.document.content)
        print(result.document.tags)
        print(result.distance)


if __name__ == "__main__":
    asyncio.run(main())
```

Returned documents hide vectors by default to avoid unnecessary memory and
serialization cost. Request vectors only when you need them:

```python
results = await client.search("How should I design vector search?", include_vector=True)
```

For large ingestion jobs where you only need the final searchable table, use
`batch_insert()` instead of `batch_create()`:

```python
inserted = await client.documents.batch_insert(
    [
        DocumentCreate(content="Large corpus document 1", tags=["source:corpus"]),
        DocumentCreate(content="Large corpus document 2", tags=["source:corpus"]),
    ],
    verbose=True,
)
print(inserted)
```

Load an existing store with matching model settings:

```python
client = await Lnclite.load(
    lancedb_path="outputs/demo.lance",
    openai_embeddings_model=embeddings,
    model_settings=ModelSettings(dimensions=1536),
)
```
