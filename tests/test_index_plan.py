"""Index planning tests."""

import lancedb.index

from lnclite.indexing import (
    recommended_num_sub_vectors,
    recommended_vector_index_config,
)


def test_recommended_vector_index_config_skips_tiny_tables():
    assert recommended_vector_index_config(255, 1536) is None


def test_recommended_vector_index_config_uses_pq_for_large_storage_preference():
    config = recommended_vector_index_config(2_000_000, 1536, prefer="storage")

    assert isinstance(config, lancedb.index.IvfPq)


def test_recommended_num_sub_vectors_prefers_storage_compression():
    assert recommended_num_sub_vectors(1536, "storage") == 96
