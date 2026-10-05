"""
Bluebirds dataset loader (Welinder et al. / CUBAM demo).

This dataset is commonly used for crowdsourcing label aggregation benchmarks.
We use the official CUBAM demo YAML files as the data source:
  - labels.yaml: worker -> {item_id -> bool}
  - gt.yaml:     item_id -> bool   (gold truth; independent of crowd votes)

Output format follows BaseDataLoader:
  labels: (n_instances, n_workers) with -1 for missing
  truth:  (n_instances,) with class ids {0,1}
  n_classes: 2
  metadata: ids and basic stats
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import urllib.request

import numpy as np
import yaml

from .base_loader import BaseDataLoader


@dataclass(frozen=True)
class BluebirdsSource:
    labels_url: str
    gt_url: str


DEFAULT_SOURCE = BluebirdsSource(
    labels_url="https://raw.githubusercontent.com/welinder/cubam/public/demo/bluebirds/labels.yaml",
    gt_url="https://raw.githubusercontent.com/welinder/cubam/public/demo/bluebirds/gt.yaml",
)


class BluebirdsLoader(BaseDataLoader):
    """
    Bluebirds dataset loader.

    Notes:
    - Binary task: bluebird present (1) vs not present (0)
    - `truth` comes from `gt.yaml` (gold), not derived from worker labels.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        source: BluebirdsSource = DEFAULT_SOURCE,
        verbose: bool = True,
    ):
        super().__init__()
        self.n_classes = 2
        self.cache_dir = cache_dir
        self.source = source
        self.verbose = verbose

        self._raw_labels: Optional[Dict[int, Dict[int, bool]]] = None
        self._raw_gt: Optional[Dict[int, bool]] = None

    def _default_cache_dir(self) -> Path:
        # src/data/bluebirds.py -> parents[2] is repo root
        repo_root = Path(__file__).resolve().parents[2]
        return repo_root / "data" / "bluebirds"

    def _download_if_needed(self, url: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 0:
            return
        if self.verbose:
            print(f"Downloading {url} -> {path}")
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            urllib.request.urlretrieve(url, tmp_path)  # nosec - used for fixed public URLs
            tmp_path.replace(path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

    def load_dataset(self) -> None:
        cache_dir = Path(self.cache_dir) if self.cache_dir else self._default_cache_dir()
        labels_path = cache_dir / "labels.yaml"
        gt_path = cache_dir / "gt.yaml"

        self._download_if_needed(self.source.labels_url, labels_path)
        self._download_if_needed(self.source.gt_url, gt_path)

        with labels_path.open("r", encoding="utf-8") as f:
            raw_labels = yaml.safe_load(f)
        with gt_path.open("r", encoding="utf-8") as f:
            raw_gt = yaml.safe_load(f)

        # YAML keys may be parsed as int already; ensure int for stable sorting/indexing.
        self._raw_labels = {int(w): {int(i): bool(v) for i, v in d.items()} for w, d in raw_labels.items()}
        self._raw_gt = {int(i): bool(v) for i, v in raw_gt.items()}

        if self.verbose:
            n_workers = len(self._raw_labels)
            n_items = len(self._raw_gt)
            print(f"✓ Loaded Bluebirds: {n_items} items, {n_workers} workers")

    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
        if self._raw_labels is None or self._raw_gt is None:
            raise ValueError("Please call load_dataset() before prepare_data().")

        item_ids = sorted(self._raw_gt.keys())
        worker_ids = sorted(self._raw_labels.keys())

        item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
        worker_id_to_idx = {wid: idx for idx, wid in enumerate(worker_ids)}

        n_instances = len(item_ids)
        n_workers = len(worker_ids)

        labels = np.full((n_instances, n_workers), -1, dtype=int)
        truth = np.zeros((n_instances,), dtype=int)

        for iid, is_pos in self._raw_gt.items():
            truth[item_id_to_idx[iid]] = 1 if is_pos else 0

        for wid, ann in self._raw_labels.items():
            j = worker_id_to_idx[wid]
            for iid, is_pos in ann.items():
                if iid not in item_id_to_idx:
                    # Should not happen, but keep robust if labels contain extra ids
                    continue
                i = item_id_to_idx[iid]
                labels[i, j] = 1 if is_pos else 0

        coverage = float((labels != -1).sum() / (n_instances * n_workers))
        avg_annotations = float((labels != -1).sum(axis=1).mean())

        metadata: Dict[str, Any] = {
            "dataset": "bluebirds",
            "n_instances": n_instances,
            "n_workers": n_workers,
            "n_classes": self.n_classes,
            "item_ids": item_ids,
            "worker_ids": worker_ids,
            "item_id_to_idx": item_id_to_idx,
            "worker_id_to_idx": worker_id_to_idx,
            "coverage": coverage,
            "avg_annotations_per_instance": avg_annotations,
            "source": {
                "labels_url": self.source.labels_url,
                "gt_url": self.source.gt_url,
            },
        }

        return labels, truth, self.n_classes, metadata


def quick_load_bluebirds(
    cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
    """Convenience loader for Bluebirds."""
    loader = BluebirdsLoader(cache_dir=cache_dir, verbose=verbose)
    loader.load_dataset()
    return loader.prepare_data()

