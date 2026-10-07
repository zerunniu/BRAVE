"""
NetEaseCrowd dataset loader.

This dataset is available via Crowd-Kit's dataset loaders:
  crowdkit.datasets.load_dataset('netease_crowd')

The raw data is a sparse table of (task, worker, label) plus a ground-truth label per task.
The loader converts the table into a dense annotation matrix (I,J) with subsampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .base_loader import BaseDataLoader


@dataclass(frozen=True)
class NetEaseCrowdConfig:
    max_instances: int = 5000
    max_workers: int = 500
    min_labels_per_instance: int = 3
    seed: int = 0


class NetEaseCrowdLoader(BaseDataLoader):
    def __init__(self, cache_dir: Optional[str] = None, verbose: bool = True):
        super().__init__()
        self.cache_dir = cache_dir
        self.verbose = verbose
        self._df: Optional[pd.DataFrame] = None
        self._gt: Optional[pd.Series] = None
        self.n_classes: Optional[int] = None

    def load_dataset(self) -> None:
        from crowdkit.datasets import load_dataset as ck_load_dataset

        # Crowd-Kit manages the dataset download and cache.
        df, gt = ck_load_dataset("netease_crowd", data_dir=self.cache_dir)
        self._df = df.copy()
        self._gt = gt.copy()
        if self.verbose:
            print(f"✓ Loaded netease_crowd: df={self._df.shape}, gt={len(self._gt):,}")

    def prepare_data(
        self,
        config: Optional[NetEaseCrowdConfig] = None,
        *,
        max_instances: Optional[int] = None,
        max_workers: Optional[int] = None,
        min_labels_per_instance: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
        if self._df is None or self._gt is None:
            raise ValueError("Call load_dataset() before prepare_data().")

        cfg = config or NetEaseCrowdConfig()
        if max_instances is None:
            max_instances = cfg.max_instances
        if max_workers is None:
            max_workers = cfg.max_workers
        if min_labels_per_instance is None:
            min_labels_per_instance = cfg.min_labels_per_instance
        if seed is None:
            seed = cfg.seed

        rng = np.random.default_rng(int(seed))
        df = self._df
        gt = self._gt

        # Keep only tasks with ground truth
        df = df[df["task"].isin(gt.index)].copy()

        # Normalize labels to contiguous 0..C-1
        all_labels = pd.unique(pd.concat([df["label"], gt], ignore_index=True))
        uniq = sorted([int(x) for x in all_labels if pd.notna(x)])
        mapping = {v: i for i, v in enumerate(uniq)}
        df.loc[:, "label"] = df["label"].map(mapping).astype(int)
        gt_mapped = gt.map(mapping).astype(int)
        n_classes = len(mapping)
        self.n_classes = n_classes

        # Cap workers: keep most active
        if max_workers is not None and int(max_workers) > 0:
            wc = df["worker"].value_counts()
            keep_workers = wc.head(int(max_workers)).index
            df = df[df["worker"].isin(keep_workers)].copy()

        # Filter tasks with enough labels
        tc = df["task"].value_counts()
        eligible = tc[tc >= int(min_labels_per_instance)].index.to_numpy()
        if eligible.size == 0:
            raise ValueError(
                "No tasks left after filtering. "
                f"Try lowering min_labels_per_instance (currently {min_labels_per_instance}) "
                f"or increasing max_workers (currently {max_workers})."
            )

        # Sample tasks weighted by counts
        if max_instances is not None and int(max_instances) > 0 and eligible.size > int(max_instances):
            weights = tc.loc[eligible].to_numpy(dtype=float)
            weights = weights / weights.sum()
            chosen = rng.choice(eligible, size=int(max_instances), replace=False, p=weights)
            keep_tasks = set(chosen.tolist())
        else:
            keep_tasks = set(eligible.tolist())

        df = df[df["task"].isin(keep_tasks)].copy()
        gt_sub = gt_mapped[gt_mapped.index.isin(keep_tasks)].copy()

        # Deduplicate (task, worker) if needed
        df = df.drop_duplicates(subset=["task", "worker"], keep="last")

        task_ids = sorted(gt_sub.index.tolist())
        worker_ids = sorted(df["worker"].unique().tolist())

        task_to_i = {t: i for i, t in enumerate(task_ids)}
        worker_to_j = {w: j for j, w in enumerate(worker_ids)}

        I, J = len(task_ids), len(worker_ids)
        labels = np.full((I, J), -1, dtype=int)
        truth = np.array([int(gt_sub.loc[t]) for t in task_ids], dtype=int)

        i_idx = df["task"].map(task_to_i).to_numpy(dtype=int)
        j_idx = df["worker"].map(worker_to_j).to_numpy(dtype=int)
        l = df["label"].to_numpy(dtype=int)
        labels[i_idx, j_idx] = l

        coverage = float((labels != -1).sum() / (I * J)) if I * J > 0 else 0.0
        avg_anns = float((labels != -1).sum(axis=1).mean()) if I > 0 else 0.0

        meta: Dict[str, Any] = {
            "dataset": "netease_crowd",
            "n_instances": int(I),
            "n_workers": int(J),
            "n_classes": int(n_classes),
            "coverage": coverage,
            "avg_annotations_per_instance": avg_anns,
            "max_instances": max_instances,
            "max_workers": max_workers,
            "min_labels_per_instance": int(min_labels_per_instance),
            "seed": int(seed),
            "task_ids": task_ids,
            "worker_ids": worker_ids,
            "source": {"loader": "crowdkit.datasets.load_dataset('netease_crowd')"},
        }

        if self.verbose:
            print(
                f"Prepared dense matrix: labels shape={labels.shape}, "
                f"coverage={coverage:.2%}, avg_anns/item={avg_anns:.2f}"
            )

        return labels, truth, int(n_classes), meta


def quick_load_netease_crowd(
    cache_dir: Optional[str] = None,
    verbose: bool = True,
    config: Optional[NetEaseCrowdConfig] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
    loader = NetEaseCrowdLoader(cache_dir=cache_dir, verbose=verbose)
    loader.load_dataset()
    return loader.prepare_data(config=config, **kwargs)
