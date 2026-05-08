"""Document embedded batch insert dimension validation test."""

import numpy as np
import pytest

from lnclite import DocumentCreate


@pytest.mark.asyncio
async def test_batch_insert_embedded_rejects_dimension_mismatch(bulk_lnclite_client):
    with pytest.raises(ValueError, match="Expected vectors with 4 dimensions, got 3"):
        await bulk_lnclite_client.documents.batch_insert_embedded(
            [DocumentCreate(content="alpha")],
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        )
