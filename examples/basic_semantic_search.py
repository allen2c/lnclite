"""Create a small lnclite store and run semantic search."""

import asyncio
import shutil
from pathlib import Path

from openai import AsyncOpenAI
from openai_embeddings_model import ModelSettings

from lnclite import DocumentCreate, Lnclite, get_openai_embeddings_model
from lnclite.examples.sample_data import (
    SAMPLE_DOCUMENTS,
    document_tags,
    format_document,
)

STORE_PATH = Path("outputs/examples/basic-semantic-search.lance")


async def main() -> None:
    if STORE_PATH.exists():
        shutil.rmtree(STORE_PATH)

    embeddings = get_openai_embeddings_model(openai_client=AsyncOpenAI())
    client = await Lnclite.new(
        lancedb_path=STORE_PATH,
        openai_embeddings_model=embeddings,
        model_settings=ModelSettings(dimensions=1536),
        name="Basic semantic search",
        description="A small store for the basic lnclite example.",
    )

    try:
        await client.documents.batch_create(
            [
                DocumentCreate(
                    content=format_document(document),
                    tags=document_tags(document),
                )
                for document in SAMPLE_DOCUMENTS
            ]
        )
        await client.create_index()

        results = await client.search("How do I run concurrent Python I/O?", limit=3)
        print("Top matches:")
        for index, result in enumerate(results.results, start=1):
            print(f"{index}. {result.document.content.splitlines()[0]}")
            print(f"   tags: {', '.join(result.document.tags)}")
            print(f"   distance: {result.distance:.4f}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
