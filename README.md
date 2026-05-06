# lnclite

`lnclite` is a small async LanceDB document store for OpenAI-compatible embeddings. It gives you a compact API for creating a local vector database, adding documents, filtering by tags, and running semantic search.

## Installation

```bash
pip install lnclite
```

For local development from this repository:

```bash
poetry install --all-groups
```

## Quick Start

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

## Documentation

Full documentation is published with MkDocs Material from this repository's `docs/` directory.

## License

MIT
