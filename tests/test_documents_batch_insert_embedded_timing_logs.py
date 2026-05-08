"""Document embedded batch insert timing log test."""

import logging

import numpy as np
import pytest

from lnclite import DocumentCreate


@pytest.mark.asyncio
async def test_batch_insert_embedded_logs_verbose_timings(
    bulk_lnclite_client,
    caplog,
):
    caplog.set_level(logging.INFO, logger="lnclite.documents")

    await bulk_lnclite_client.documents.batch_insert_embedded(
        [DocumentCreate(content="alpha")],
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        verbose=True,
    )

    assert "batch_insert_embedded timings:" in caplog.text
    assert "get_table=" in caplog.text
    assert "normalize_vectors=" in caplog.text
    assert "row_construction=" in caplog.text
    assert "table_add=" in caplog.text
