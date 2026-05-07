"""Search and list lnclite documents with tag filters."""

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

STORE_PATH = Path("outputs/examples/tagged-search-and-listing.lance")


async def main() -> None:
    if STORE_PATH.exists():
        shutil.rmtree(STORE_PATH)

    embeddings = get_openai_embeddings_model(openai_client=AsyncOpenAI())
    client = await Lnclite.new(
        lancedb_path=STORE_PATH,
        openai_embeddings_model=embeddings,
        model_settings=ModelSettings(dimensions=1536),
        name="Tagged search and listing",
        description="A small store for tag filtering examples.",
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

        print("Search with tags_any=['topic:python', 'topic:http']:")
        results = await client.search(
            "How do clients communicate with services?",
            tags_any=["topic:python", "topic:http"],
            limit=3,
        )
        for result in results.results:
            print(f"- {result.document.content.splitlines()[0]}")
            print(f"  tags: {', '.join(result.document.tags)}")

        print("\nList guides with pagination:")
        first_page = await client.documents.list(tags_all=["type:guide"], limit=2)
        _print_page("page 1", first_page.data)

        if first_page.next_page_token is not None:
            second_page = await client.documents.list(
                tags_all=["type:guide"],
                limit=2,
                next_page_token=first_page.next_page_token,
            )
            _print_page("page 2", second_page.data)
    finally:
        await client.close()


def _print_page(label: str, documents) -> None:
    print(label)
    for document in documents:
        print(f"- {document.content.splitlines()[0]}")


if __name__ == "__main__":
    asyncio.run(main())
