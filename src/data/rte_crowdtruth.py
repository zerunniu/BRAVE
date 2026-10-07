"""
CrowdTruth RTE dataset loader (Snow et al., EMNLP 2008).

This dataset is distributed in the CrowdTruth-core tutorial as a single CSV:
  - `rte.standardized.csv`

Each row is an annotation:
  - `orig_id` (unit/task id)
  - `!amt_worker_ids` (worker id)
  - `response` (crowd label: 0/1)
  - `gold` (expert label: 0/1)
  - plus `text` and `hypothesis`

Output:
  labels: (I, J) with -1 for missing
  truth:  (I,) expert gold label
  n_classes: 2
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import urllib.request

import numpy as np
import pandas as pd

from .base_loader import BaseDataLoader


@dataclass(frozen=True)
class CrowdTruthRTESource:
    standardized_csv_url: str


DEFAULT_SOURCE = CrowdTruthRTESource(
    standardized_csv_url="https://raw.githubusercontent.com/CrowdTruth/CrowdTruth-core/master/tutorial/data/rte.standardized.csv"
)


class CrowdTruthRTELoader(BaseDataLoader):
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        source: CrowdTruthRTESource = DEFAULT_SOURCE,
        verbose: bool = True,
    ):
        super().__init__()
        self.n_classes = 2
        self.cache_dir = cache_dir
        self.source = source
        self.verbose = verbose

        self._df: Optional[pd.DataFrame] = None

    def _default_cache_dir(self) -> Path:
        repo_root = Path(__file__).resolve().parents[2]
        return repo_root / "data" / "rte_crowdtruth"

    def _download_if_needed(self, url: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 0:
            return
        if self.verbose:
            print(f"Downloading {url} -> {path}")
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            urllib.request.urlretrieve(url, tmp_path)  # nosec - fixed public URL
            tmp_path.replace(path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

    def load_dataset(self) -> None:
        cache_dir = Path(self.cache_dir) if self.cache_dir else self._default_cache_dir()
        csv_path = cache_dir / "rte.standardized.csv"
        self._download_if_needed(self.source.standardized_csv_url, csv_path)

        df = pd.read_csv(csv_path)
        # Normalize annotation column names.
        required = {"orig_id", "!amt_worker_ids", "response", "gold", "text", "hypothesis"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in RTE CSV: {sorted(missing)}")

        # Ensure numeric labels
        df["orig_id"] = df["orig_id"].astype(int)
        df["response"] = df["response"].astype(int)
        df["gold"] = df["gold"].astype(int)
        df["!amt_worker_ids"] = df["!amt_worker_ids"].astype(str)

        # Filter invalid labels if any
        df = df[df["response"].isin([0, 1]) & df["gold"].isin([0, 1])].copy()

        self._df = df
        if self.verbose:
            n_ann = len(df)
            n_units = df["orig_id"].nunique()
            n_workers = df["!amt_worker_ids"].nunique()
            print(f"✓ Loaded CrowdTruth RTE: {n_units} units, {n_workers} workers, {n_ann} annotations")

    def prepare_data(
        self,
        max_instances: Optional[int] = None,
        min_labels_per_instance: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
        if self._df is None:
            raise ValueError("Call load_dataset() before prepare_data().")

        df = self._df

        # Filter units with enough annotations
        counts = df.groupby("orig_id")["response"].size()
        keep_units = counts[counts >= int(min_labels_per_instance)].index
        df = df[df["orig_id"].isin(keep_units)].copy()

        unit_ids = sorted(df["orig_id"].unique().tolist())
        if max_instances is not None:
            unit_ids = unit_ids[: int(max_instances)]
            df = df[df["orig_id"].isin(unit_ids)].copy()

        worker_ids = sorted(df["!amt_worker_ids"].unique().tolist())

        unit_to_idx = {u: i for i, u in enumerate(unit_ids)}
        worker_to_idx = {w: j for j, w in enumerate(worker_ids)}

        I = len(unit_ids)
        J = len(worker_ids)
        labels = np.full((I, J), -1, dtype=int)

        # Use the first gold label per unit and report conflicting entries.
        gold_by_unit = df.groupby("orig_id")["gold"].agg(["nunique", "first"])
        inconsistent = gold_by_unit[gold_by_unit["nunique"] > 1]
        if len(inconsistent) > 0 and self.verbose:
            print(f"Warning: found {len(inconsistent)} units with inconsistent gold labels; using first.")

        truth = np.zeros((I,), dtype=int)
        for u in unit_ids:
            truth[unit_to_idx[u]] = int(gold_by_unit.loc[u, "first"])

        # Fill annotations
        for _, row in df.iterrows():
            u = int(row["orig_id"])
            if u not in unit_to_idx:
                continue
            w = str(row["!amt_worker_ids"])
            i = unit_to_idx[u]
            j = worker_to_idx[w]
            labels[i, j] = int(row["response"])

        coverage = float((labels != -1).sum() / (I * J))
        avg_annotations = float((labels != -1).sum(axis=1).mean())

        # Keep one text/hypothesis per unit for inspection (optional)
        unit_meta = (
            df.sort_values(["orig_id"])
            .groupby("orig_id")[["text", "hypothesis"]]
            .first()
            .reindex(unit_ids)
        )

        metadata: Dict[str, Any] = {
            "dataset": "rte_crowdtruth",
            "n_instances": I,
            "n_workers": J,
            "n_classes": self.n_classes,
            "unit_ids": unit_ids,
            "worker_ids": worker_ids,
            "unit_to_idx": unit_to_idx,
            "worker_to_idx": worker_to_idx,
            "coverage": coverage,
            "avg_annotations_per_instance": avg_annotations,
            "text": unit_meta["text"].tolist(),
            "hypothesis": unit_meta["hypothesis"].tolist(),
            "source": {"standardized_csv_url": self.source.standardized_csv_url},
        }

        return labels, truth, self.n_classes, metadata


def quick_load_rte_crowdtruth(
    cache_dir: Optional[str] = None,
    verbose: bool = True,
    max_instances: Optional[int] = None,
    min_labels_per_instance: int = 1,
) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
    loader = CrowdTruthRTELoader(cache_dir=cache_dir, verbose=verbose)
    loader.load_dataset()
    return loader.prepare_data(max_instances=max_instances, min_labels_per_instance=min_labels_per_instance)
