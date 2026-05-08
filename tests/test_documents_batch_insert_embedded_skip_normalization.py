"""Document embedded batch insert skip-normalization test."""

import numpy as np
import pytest

from lnclite import DocumentCreate


@pytest.mark.asyncio
async def test_batch_insert_embedded_can_skip_normalization(bulk_lnclite_client):
    await bulk_lnclite_client.documents.batch_insert_embedded(
        [DocumentCreate(content="alpha")],
        np.array([[3.0, 4.0, 0.0, 0.0]], dtype=np.float32),
        normalize_vectors=False,
    )

    page = await bulk_lnclite_client.documents.list(limit=1, include_vector=True)

    assert page.data[0].vector == pytest.approx([3.0, 4.0, 0.0, 0.0])
