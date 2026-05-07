"""LncliteManager tests."""

import pytest
from openai import AsyncOpenAI
from openai_embeddings_model import ModelSettings

from lnclite import (
    LncliteConfig,
    LncliteManager,
    LncliteNotFoundError,
    get_openai_embeddings_model,
)


def make_config(path):
    return LncliteConfig(
        lancedb_path=path,
        openai_embeddings_model=get_openai_embeddings_model(
            openai_client=AsyncOpenAI()
        ),
        model_settings=ModelSettings(dimensions=1536),
    )


@pytest.mark.asyncio
async def test_manager_opens_and_reuses_client(tmp_path):
    manager = LncliteManager(max_clients=2)
    config = make_config(tmp_path / "a.lance")

    first = await manager.open("a", config=config, create=True)
    second = await manager.open("a")

    assert first is second
    assert manager.stats().active_clients == 1


@pytest.mark.asyncio
async def test_manager_unknown_config_raises_not_found():
    manager = LncliteManager()

    with pytest.raises(LncliteNotFoundError, match="No config registered"):
        await manager.open("missing")


@pytest.mark.asyncio
async def test_manager_evicts_lru_client_when_max_clients_exceeded(tmp_path):
    manager = LncliteManager(max_clients=1)

    first = await manager.open(
        "a",
        config=make_config(tmp_path / "a.lance"),
        create=True,
    )
    second = await manager.open(
        "b",
        config=make_config(tmp_path / "b.lance"),
        create=True,
    )

    assert first is not second
    assert manager.stats().active_clients == 1
    assert manager.stats().clients["a"].is_open is False
    assert manager.stats().clients["b"].is_open is True


@pytest.mark.asyncio
async def test_manager_close_all_closes_clients(tmp_path):
    manager = LncliteManager(max_clients=2)
    await manager.open("a", config=make_config(tmp_path / "a.lance"), create=True)
    await manager.open("b", config=make_config(tmp_path / "b.lance"), create=True)

    await manager.close_all()

    assert manager.stats().active_clients == 0
