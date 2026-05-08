"""Document embedded batch insert concurrency test."""

import asyncio

import numpy as np
import pytest

from lnclite import DocumentCreate


@pytest.mark.asyncio
async def test_batch_insert_embedded_supports_concurrent_async_writers(
    bulk_lnclite_client,
):
    async def insert_batch(batch_index: int) -> int:
        documents = [
            DocumentCreate(
                content=f"document {batch_index}-{index}",
                tags=[f"batch:{batch_index}"],
            )
            for index in range(5)
        ]
        vectors = np.eye(4, dtype=np.float32)[
            [(batch_index + index) % 4 for index in range(5)]
        ]
        return await bulk_lnclite_client.documents.batch_insert_embedded(
            documents,
            vectors,
            normalize_vectors=False,
        )

    inserted = await asyncio.gather(*(insert_batch(index) for index in range(4)))

    assert inserted == [5, 5, 5, 5]
    assert await bulk_lnclite_client.documents.count() == 20
