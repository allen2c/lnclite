"""Document embedded batch insert count validation test."""

import numpy as np
import pytest

from lnclite import DocumentCreate


@pytest.mark.asyncio
async def test_batch_insert_embedded_rejects_count_mismatch(bulk_lnclite_client):
    with pytest.raises(ValueError, match="Expected 2 vectors, got 1"):
        await bulk_lnclite_client.documents.batch_insert_embedded(
            [
                DocumentCreate(content="alpha"),
                DocumentCreate(content="beta"),
            ],
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        )
