"""
CIFAR-10H loader (crowdsourced human labels for CIFAR-10 test set).

Source:
  - https://github.com/jcpeterson/cifar-10h

We download and parse the raw annotator-level data (`cifar10h-raw.zip`),
then expose the dataset in this repo's dense format:
  labels: (I, J) with -1 for missing, labels in {0..9}
  truth:  (I,) CIFAR-10 test-set ground truth label in {0..9}

Notes:
  - Each annotator labels a subset of images, so the matrix is sparse.
  - We drop attention-check trials.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import io
import zipfile
import urllib.request

import numpy as np
import pandas as pd

from .base_loader import BaseDataLoader


@dataclass(frozen=True)
class CIFAR10HSource:
    raw_zip_url: str


DEFAULT_SOURCE = CIFAR10HSource(
    raw_zip_url="https://github.com/jcpeterson/cifar-10h/raw/master/data/cifar10h-raw.zip"
)


class CIFAR10HLoader(BaseDataLoader):
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        source: CIFAR10HSource = DEFAULT_SOURCE,
        verbose: bool = True,
    ):
        super().__init__()
        self.cache_dir = cache_dir
        self.source = source
        self.verbose = verbose
        self.n_classes = 10
        self._df: Optional[pd.DataFrame] = None

    def _default_cache_dir(self) -> Path:
        repo_root = Path(__file__).resolve().parents[2]
        return repo_root / "data" / "cifar10h"

    def _download_if_needed(self, url: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 0:
            return
        if self.verbose:
            print(f"Downloading {url} -> {path}")
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            urllib.request.urlretrieve(url, tmp)  # nosec - fixed public URL
            tmp.replace(path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass

    def load_dataset(self) -> None:
        cache_dir = Path(self.cache_dir) if self.cache_dir else self._default_cache_dir()
        zip_path = cache_dir / "cifar10h-raw.zip"
        self._download_if_needed(self.source.raw_zip_url, zip_path)

        with zip_path.open("rb") as f:
            b = f.read()
        zf = zipfile.ZipFile(io.BytesIO(b))
        member = "cifar10h-raw.csv"
        if member not in zf.namelist():
            raise ValueError(f"Missing {member} in cifar10h-raw.zip. Members={zf.namelist()}")
        with zf.open(member) as f:
            df = pd.read_csv(f)

        # Note: the released file uses `cifar10_test_test_idx` (typo) for the CIFAR-10 test index.
        idx_col = "cifar10_test_set_idx" if "cifar10_test_set_idx" in df.columns else "cifar10_test_test_idx"

        required = {"annotator_id", "is_attn_check", "chosen_label", "true_label", idx_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in CIFAR-10H raw CSV: {sorted(missing)}")

        # keep non-attention-check, valid test-set indices
        df = df[(df["is_attn_check"] == 0) & (df[idx_col] >= 0)].copy()

        # normalize dtypes
        df["annotator_id"] = df["annotator_id"].astype(int)
        df[idx_col] = df[idx_col].astype(int)
        df["chosen_label"] = df["chosen_label"].astype(int)
        df["true_label"] = df["true_label"].astype(int)

        # normalize to a canonical column name used in the rest of this loader
        if idx_col != "cifar10_test_set_idx":
            df = df.rename(columns={idx_col: "cifar10_test_set_idx"})

        self._df = df
        if self.verbose:
            n_ann = len(df)
            n_tasks = df["cifar10_test_set_idx"].nunique()
            n_workers = df["annotator_id"].nunique()
            print(f"✓ Loaded CIFAR-10H raw: tasks={n_tasks:,}, workers={n_workers:,}, annotations={n_ann:,}")

    def prepare_data(
        self,
        max_instances: Optional[int] = 10000,
        max_workers: Optional[int] = None,
        min_labels_per_instance: int = 1,
        seed: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
        if self._df is None:
            raise ValueError("Call load_dataset() before prepare_data().")

        rng = np.random.default_rng(int(seed))
        df = self._df.copy()

        # optionally cap workers (keep most active)
        if max_workers is not None and int(max_workers) > 0:
            wc = df["annotator_id"].value_counts()
            keep = wc.head(int(max_workers)).index
            df = df[df["annotator_id"].isin(keep)].copy()

        # filter tasks with enough labels
        tc = df["cifar10_test_set_idx"].value_counts()
        eligible = tc[tc >= int(min_labels_per_instance)].index.to_numpy()
        if eligible.size == 0:
            raise ValueError("No tasks left after filtering; lower min_labels_per_instance or increase max_workers.")

        # sample tasks
        if max_instances is not None and int(max_instances) > 0 and eligible.size > int(max_instances):
            weights = tc.loc[eligible].to_numpy(dtype=float)
            weights = weights / weights.sum()
            chosen = rng.choice(eligible, size=int(max_instances), replace=False, p=weights)
            keep_tasks = set(map(int, chosen.tolist()))
        else:
            keep_tasks = set(map(int, eligible.tolist()))

        df = df[df["cifar10_test_set_idx"].isin(keep_tasks)].copy()

        # Deduplicate (task, worker) if any; keep last
        df = df.sort_values(["annotator_id"]).drop_duplicates(
            subset=["cifar10_test_set_idx", "annotator_id"], keep="last"
        )

        task_ids = sorted(df["cifar10_test_set_idx"].unique().tolist())
        worker_ids = sorted(df["annotator_id"].unique().tolist())
        task_to_i = {t: i for i, t in enumerate(task_ids)}
        worker_to_j = {w: j for j, w in enumerate(worker_ids)}

        I, J = len(task_ids), len(worker_ids)
        labels = np.full((I, J), -1, dtype=int)

        # truth: true_label is constant per task; take first
        truth_map = df.groupby("cifar10_test_set_idx")["true_label"].first().to_dict()
        truth = np.array([int(truth_map[t]) for t in task_ids], dtype=int)

        i_idx = df["cifar10_test_set_idx"].map(task_to_i).to_numpy(dtype=int)
        j_idx = df["annotator_id"].map(worker_to_j).to_numpy(dtype=int)
        l = df["chosen_label"].to_numpy(dtype=int)
        labels[i_idx, j_idx] = l

        coverage = float((labels != -1).sum() / (I * J)) if I * J > 0 else 0.0
        avg_anns = float((labels != -1).sum(axis=1).mean()) if I > 0 else 0.0

        meta: Dict[str, Any] = {
            "dataset": "cifar10h",
            "n_instances": int(I),
            "n_workers": int(J),
            "n_classes": int(self.n_classes),
            "coverage": coverage,
            "avg_annotations_per_instance": avg_anns,
            "max_instances": max_instances,
            "max_workers": max_workers,
            "min_labels_per_instance": int(min_labels_per_instance),
            "seed": int(seed),
            "task_ids": task_ids,
            "worker_ids": worker_ids,
            "source": {"raw_zip_url": self.source.raw_zip_url},
        }
        return labels, truth, int(self.n_classes), meta


def quick_load_cifar10h(
    cache_dir: Optional[str] = None,
    verbose: bool = True,
    **prepare_kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
    loader = CIFAR10HLoader(cache_dir=cache_dir, verbose=verbose)
    loader.load_dataset()
    return loader.prepare_data(**prepare_kwargs)

