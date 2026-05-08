"""Document batch insert count test."""

import numpy as np
import pytest

from lnclite import DocumentCreate


@pytest.mark.asyncio
async def test_batch_insert_returns_count_without_documents(
    bulk_lnclite_client,
    monkeypatch,
):
    documents = [
        DocumentCreate(content="alpha", tags=["a"]),
        DocumentCreate(content="beta", tags=["b"]),
    ]

    async def embed(texts: list[str]) -> np.ndarray:
        assert texts == ["alpha", "beta"]
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

    monkeypatch.setattr(bulk_lnclite_client, "embed", embed)

    inserted = await bulk_lnclite_client.documents.batch_insert(documents)

    assert inserted == 2
    assert await bulk_lnclite_client.documents.count() == 2
