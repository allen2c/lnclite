# lnclite

[![PyPI](https://img.shields.io/pypi/v/lnclite.svg)](https://pypi.org/project/lnclite/)
[![Python](https://img.shields.io/pypi/pyversions/lnclite.svg)](https://pypi.org/project/lnclite/)
[![License](https://img.shields.io/github/license/allen2c/lnclite.svg)](LICENSE)
[![CI](https://github.com/allen2c/lnclite/actions/workflows/ci.yml/badge.svg)](https://github.com/allen2c/lnclite/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://allen2c.github.io/lnclite/)

`lnclite` is a small async LanceDB document store for OpenAI-compatible
embeddings. It gives you a compact API for creating a local vector database,
adding documents, filtering by tags, and running semantic search.

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

    # Vectors are hidden in returned documents by default.
    results_with_vectors = await client.search(
        "How should I design vector search?",
        include_vector=True,
    )
    print(len(results_with_vectors.results[0].document.vector))


if __name__ == "__main__":
    asyncio.run(main())
```

## Documentation

Full documentation is published at
[`allen2c.github.io/lnclite`](https://allen2c.github.io/lnclite/).

Useful pages:

- [Installation](https://allen2c.github.io/lnclite/installation/)
- [Basic usage](https://allen2c.github.io/lnclite/usage/basic/)
- [Directory ingest](https://allen2c.github.io/lnclite/usage/ingest-directory/)
- [Managing multiple stores](https://allen2c.github.io/lnclite/usage/manager/)
- [API reference](https://allen2c.github.io/lnclite/api-reference/)

## Examples

Runnable end-to-end examples live in [`examples/`](examples/README.md). They
cover basic search, tag filtering, directory ingest, managing multiple stores,
and an optional downloaded open-dataset demo.

## License

MIT
