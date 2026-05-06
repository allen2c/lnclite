"""Index parameter helpers for LanceDB vector indexes."""

import math


def calculate_index_params(n_rows: int, dimension: int) -> tuple[int, int]:
    max_partitions = n_rows // 256
    num_partitions = min(int(math.sqrt(n_rows) * 8), max_partitions)
    num_partitions = max(num_partitions, 1)

    target = dimension // 8
    num_sub_vectors = 1
    for i in range(target, 0, -1):
        if dimension % i == 0:
            num_sub_vectors = i
            break

    return num_partitions, num_sub_vectors
