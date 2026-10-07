"""Contiguous worker-block partitioning for BRAVE."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def partition_annotations(
    labels: np.ndarray,
    n_blocks: int = 3,
) -> Tuple[List[np.ndarray], List[str]]:
    """Create a worker-side disjoint cover by contiguous worker columns.

    Input column order is preserved. The final block absorbs the remainder.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("labels must be a 2D array with shape (n_instances, n_workers).")
    if n_blocks <= 0:
        raise ValueError("n_blocks must be positive.")

    n_workers = labels.shape[1]
    n_blocks = min(int(n_blocks), max(1, n_workers))
    workers_per_block = n_workers // n_blocks
    block_labels: List[np.ndarray] = []
    block_ids: List[str] = []
    for i in range(n_blocks):
        start_idx = i * workers_per_block
        end_idx = n_workers if i == n_blocks - 1 else start_idx + workers_per_block
        block_labels.append(labels[:, start_idx:end_idx].copy())
        block_ids.append(f"block_{i + 1}")
    return block_labels, block_ids
