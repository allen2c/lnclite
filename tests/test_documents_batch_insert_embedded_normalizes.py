"""Document embedded batch insert normalization test."""

import numpy as np
import pytest

from lnclite import DocumentCreate


@pytest.mark.asyncio
async def test_batch_insert_embedded_normalizes_vectors_by_default(
    bulk_lnclite_client,
):
    await bulk_lnclite_client.documents.batch_insert_embedded(
        [DocumentCreate(content="alpha")],
        np.array([[3.0, 4.0, 0.0, 0.0]], dtype=np.float32),
    )

    page = await bulk_lnclite_client.documents.list(limit=1, include_vector=True)

    assert page.data[0].vector == pytest.approx([0.6, 0.8, 0.0, 0.0])
