"""Manage multiple lnclite stores with LncliteManager."""

import asyncio
import shutil
from pathlib import Path

from openai import AsyncOpenAI
from openai_embeddings_model import ModelSettings

from lnclite import (
    DocumentCreate,
    LncliteConfig,
    LncliteManager,
    get_openai_embeddings_model,
)
from lnclite.examples.sample_data import (
    SAMPLE_DOCUMENTS,
    document_tags,
    format_document,
)

ROOT_DIR = Path("outputs/examples/manager")


async def main() -> None:
    if ROOT_DIR.exists():
        shutil.rmtree(ROOT_DIR)
    ROOT_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = get_openai_embeddings_model(openai_client=AsyncOpenAI())
    manager = LncliteManager(max_clients=1, idle_ttl_seconds=60)

    try:
        engineering = await manager.open(
            "engineering",
            config=_config(ROOT_DIR / "engineering.lance", embeddings),
            create=True,
        )
        await engineering.documents.batch_create(
            [
                DocumentCreate(
                    content=format_document(document),
                    tags=document_tags(document),
                )
                for document in SAMPLE_DOCUMENTS
                if "topic:release" not in document["tags"]
            ]
        )
        await engineering.create_index()

        release = await manager.open(
            "release",
            config=_config(ROOT_DIR / "release.lance", embeddings),
            create=True,
        )
        await release.documents.batch_create(
            [
                DocumentCreate(
                    content=format_document(document),
                    tags=document_tags(document),
                )
                for document in SAMPLE_DOCUMENTS
                if "topic:release" in document["tags"]
            ]
        )
        await release.create_index()

        print("After opening two stores with max_clients=1:")
        _print_stats(manager)

        engineering = await manager.open("engineering")
        results = await engineering.search("How do branches help development?", limit=2)
        print("\nSearch in reopened engineering store:")
        for result in results.results:
            print(f"- {result.document.content.splitlines()[0]}")

        print("\nFinal manager stats:")
        _print_stats(manager)
    finally:
        await manager.close_all()


def _config(lancedb_path: Path, embeddings) -> LncliteConfig:
    return LncliteConfig(
        lancedb_path=lancedb_path,
        openai_embeddings_model=embeddings,
        model_settings=ModelSettings(dimensions=1536),
    )


def _print_stats(manager: LncliteManager) -> None:
    stats = manager.stats()
    print(f"active_clients={stats.active_clients}, max_clients={stats.max_clients}")
    for name, client in stats.clients.items():
        print(f"- {name}: is_open={client.is_open}")


if __name__ == "__main__":
    asyncio.run(main())
