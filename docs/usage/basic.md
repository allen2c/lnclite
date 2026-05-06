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

Load an existing store with matching model settings:

```python
client = await Lnclite.load(
    lancedb_path="outputs/demo.lance",
    openai_embeddings_model=embeddings,
    model_settings=ModelSettings(dimensions=1536),
)
```
