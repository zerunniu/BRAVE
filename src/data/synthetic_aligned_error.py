"""Synthetic crowdsourcing data with controllable aligned wrong-class errors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class SyntheticAlignedErrorConfig:
    n_instances: int = 2000
    n_workers: int = 200
    n_classes: int = 3
    n_groups: int = 3
    labels_per_item: int = 4
    gamma: float = 0.4
    seed: int = 0


def generate_synthetic_aligned_error(
    config: SyntheticAlignedErrorConfig,
) -> Tuple[np.ndarray, np.ndarray, int, Dict]:
    """Generate sparse labels with group-shared class-specific wrong-label tendencies."""
    rng = np.random.default_rng(int(config.seed))
    I = int(config.n_instances)
    M = int(config.n_workers)
    C = int(config.n_classes)
    G = int(config.n_groups)
    d = max(1, min(int(config.labels_per_item), M))
    gamma = float(np.clip(config.gamma, 0.0, 1.0))

    truth = rng.integers(0, C, size=I, dtype=int)
    labels = np.full((I, M), -1, dtype=int)
    worker_groups = np.arange(M) % G

    # Each group has a stable rival for every true class, creating correlated
    # wrong evidence when several sampled workers come from related groups.
    group_rivals = np.zeros((G, C), dtype=int)
    for g in range(G):
        for c in range(C):
            group_rivals[g, c] = (c + 1 + g) % C
            if group_rivals[g, c] == c:
                group_rivals[g, c] = (c + 1) % C

    base_correct = 0.78
    wrong_rate = 1.0 - base_correct
    for i in range(I):
        workers = rng.choice(M, size=d, replace=False)
        c = int(truth[i])
        for j in workers:
            if rng.random() < base_correct:
                labels[i, j] = c
                continue

            group = int(worker_groups[j])
            if rng.random() < gamma or C == 2:
                labels[i, j] = int(group_rivals[group, c])
            else:
                alternatives = [x for x in range(C) if x != c]
                labels[i, j] = int(rng.choice(alternatives))

    meta = {
        "worker_groups": worker_groups,
        "group_rivals": group_rivals,
        "config": config,
        "wrong_rate": wrong_rate,
    }
    return labels, truth, C, meta
