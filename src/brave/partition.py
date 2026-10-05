"""Worker-block partition helpers used by BRAVE experiments."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def partition_annotations(
    labels: np.ndarray,
    n_blocks: int = 3,
    split_method: str = "workers",
) -> Tuple[List[np.ndarray], List[str]]:
    """Split an annotation matrix into deterministic blocks.

    BRAVE uses ``split_method="workers"`` to create a worker-side disjoint
    cover by contiguous worker columns. The optional ``"instances"`` mode
    retains the existing item-side partition behavior.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("labels must be a 2D array with shape (n_instances, n_workers).")
    if n_blocks <= 0:
        raise ValueError("n_blocks must be positive.")

    n_instances, n_workers = labels.shape
    n_blocks = min(int(n_blocks), max(1, n_workers if split_method == "workers" else n_instances))

    if split_method == "workers":
        workers_per_block = n_workers // n_blocks
        block_labels: List[np.ndarray] = []
        block_ids: List[str] = []
        for i in range(n_blocks):
            start_idx = i * workers_per_block
            end_idx = n_workers if i == n_blocks - 1 else start_idx + workers_per_block
            block_labels.append(labels[:, start_idx:end_idx].copy())
            block_ids.append(f"block_{i + 1}")
        return block_labels, block_ids

    if split_method == "instances":
        instances_per_block = n_instances // n_blocks
        block_labels = []
        block_ids = []
        for i in range(n_blocks):
            start_idx = i * instances_per_block
            end_idx = n_instances if i == n_blocks - 1 else start_idx + instances_per_block
            block_data = np.full_like(labels, -1)
            block_data[start_idx:end_idx, :] = labels[start_idx:end_idx, :]
            block_labels.append(block_data)
            block_ids.append(f"block_{i + 1}")
        return block_labels, block_ids

    raise ValueError(f"Unknown split method: {split_method}")

