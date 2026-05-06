"""Index planning helpers for LanceDB vector indexes."""

import lancedb.index
from pydantic import BaseModel

from lnclite.constants import VectorIndexPreference


class DocumentIndexPlan(BaseModel):
    row_count: int
    dimensions: int
    vector_search_prefer: VectorIndexPreference
    should_create_vector_index: bool
    vector_index_kind: str | None


def recommended_vector_index_config(
    row_count: int,
    dim: int,
    *,
    prefer: VectorIndexPreference = "balanced",
):
    """Return a LanceDB vector index config for dot-search."""

    if row_count < 256:
        return None

    if row_count < 10_000:
        if prefer in {"accuracy", "latency"}:
            return lancedb.index.IvfFlat(distance_type="dot", num_partitions=32)
        return None

    if row_count < 50_000:
        return lancedb.index.IvfFlat(distance_type="dot", num_partitions=128)

    if row_count < 100_000:
        if prefer == "storage":
            return lancedb.index.IvfPq(
                distance_type="dot",
                num_partitions=256,
                num_sub_vectors=recommended_num_sub_vectors(dim, prefer),
                num_bits=8,
            )
        return lancedb.index.IvfFlat(distance_type="dot", num_partitions=256)

    if row_count < 500_000:
        return lancedb.index.IvfPq(
            distance_type="dot",
            num_partitions=1024 if prefer == "storage" else 2048,
            num_sub_vectors=recommended_num_sub_vectors(dim, prefer),
            num_bits=8,
        )

    if row_count < 1_000_000:
        return lancedb.index.IvfPq(
            distance_type="dot",
            num_partitions=2048 if prefer == "storage" else 4096,
            num_sub_vectors=recommended_num_sub_vectors(dim, prefer),
            num_bits=8,
        )

    if prefer == "latency":
        return lancedb.index.HnswPq(
            distance_type="dot",
            m=20,
            ef_construction=300,
            num_sub_vectors=recommended_num_sub_vectors(dim, "balanced"),
            num_bits=8,
        )

    return lancedb.index.IvfPq(
        distance_type="dot",
        num_partitions=4096 if prefer in {"storage", "balanced"} else 8192,
        num_sub_vectors=recommended_num_sub_vectors(dim, prefer),
        num_bits=8,
    )


def recommended_num_sub_vectors(
    dim: int, prefer: VectorIndexPreference = "balanced"
) -> int:
    """Return a PQ subvector count."""

    if prefer == "storage":
        target_sub_dim = 16
    elif prefer == "accuracy":
        target_sub_dim = 8
    else:
        target_sub_dim = 12

    candidates = [x for x in range(1, dim + 1) if dim % x == 0]
    return min(candidates, key=lambda x: abs((dim / x) - target_sub_dim))


def build_document_index_plan(
    *,
    row_count: int,
    dimensions: int,
    vector_search_prefer: VectorIndexPreference,
) -> DocumentIndexPlan:
    config = recommended_vector_index_config(
        row_count,
        dimensions,
        prefer=vector_search_prefer,
    )
    return DocumentIndexPlan(
        row_count=row_count,
        dimensions=dimensions,
        vector_search_prefer=vector_search_prefer,
        should_create_vector_index=config is not None,
        vector_index_kind=type(config).__name__ if config is not None else None,
    )
