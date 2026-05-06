"""Lifecycle tests for creating and loading stores."""

import pytest
from openai import AsyncOpenAI
from openai_embeddings_model import ModelSettings

from lnclite import (
    DocumentCreate,
    Lnclite,
    LncliteNotFoundError,
    get_openai_embeddings_model,
)


@pytest.mark.asyncio
async def test_new_creates_manifest_and_load_round_trips(tmp_path):
    embeddings = get_openai_embeddings_model(openai_client=AsyncOpenAI())
    model_settings = ModelSettings(dimensions=1536)
    path = tmp_path / "store.lance"

    created = await Lnclite.new(
        lancedb_path=path,
        openai_embeddings_model=embeddings,
        model_settings=model_settings,
        name="Lifecycle",
        description="Lifecycle test store",
    )
    await created.documents.create(DocumentCreate(content="round trip document"))

    loaded = await Lnclite.load(
        lancedb_path=path,
        openai_embeddings_model=embeddings,
        model_settings=ModelSettings(dimensions=1536),
    )

    manifest = await loaded.manifest.retrieve()
    assert manifest.name == "Lifecycle"
    assert manifest.description == "Lifecycle test store"
    assert manifest.model == embeddings.model
    assert manifest.dimensions == 1536


@pytest.mark.asyncio
async def test_load_without_manifest_raises_not_found(tmp_path):
    client = Lnclite(
        lancedb_path=tmp_path / "store.lance",
        openai_embeddings_model=get_openai_embeddings_model(
            openai_client=AsyncOpenAI()
        ),
        model_settings=ModelSettings(dimensions=1536),
    )
    await client.documents.create(DocumentCreate(content="document without manifest"))

    with pytest.raises(LncliteNotFoundError, match="Manifest not found"):
        await Lnclite.load(
            lancedb_path=tmp_path / "store.lance",
            openai_embeddings_model=get_openai_embeddings_model(
                openai_client=AsyncOpenAI()
            ),
            model_settings=ModelSettings(dimensions=1536),
        )
