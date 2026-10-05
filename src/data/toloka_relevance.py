"""
Toloka / Crowd-Kit relevance datasets loader.

We support the datasets provided by Toloka Crowd-Kit (Ustalov et al., 2024) via public ZIP URLs:
  - relevance-2: binary relevance (0/1)
  - relevance-5: 1..5 relevance scale

These datasets are large and sparse (task, worker, label tuples). Our algorithms in this repo
expect a dense matrix (n_instances, n_workers) with -1 for missing labels, so we provide
controlled subsampling to keep memory/runtime reasonable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import hashlib
import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from .base_loader import BaseDataLoader


@dataclass(frozen=True)
class TolokaRelevanceSource:
    name: str
    zip_url: str
    md5_url: str
    # Which file names exist inside the zip (Crowd-Kit convention)
    crowd_labels_filename: str = "crowd_labels.csv"
    gt_filename: str = "gt.csv"


RELEVANCE_2 = TolokaRelevanceSource(
    name="relevance-2",
    zip_url="https://tlk.s3.yandex.net/dataset/crowd-kit/relevance-2.zip",
    md5_url="https://tlk.s3.yandex.net/dataset/crowd-kit/relevance-2.md5",
)

RELEVANCE_5 = TolokaRelevanceSource(
    name="relevance-5",
    zip_url="https://tlk.s3.yandex.net/dataset/crowd-kit/relevance-5.zip",
    md5_url="https://tlk.s3.yandex.net/dataset/crowd-kit/relevance-5.md5",
)


class TolokaRelevanceLoader(BaseDataLoader):
    """
    Loader for Toloka/Crowd-Kit Relevance datasets.

    Parameters focus on converting sparse tuples into a manageable dense matrix:
      - max_instances: number of tasks (items) to keep
      - max_workers: number of workers to keep
      - min_labels_per_instance: filter tasks with too few annotations (after worker filtering)
    """

    def __init__(
        self,
        variant: str = "relevance-2",
        cache_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        super().__init__()
        self.verbose = verbose
        self.cache_dir = cache_dir

        if variant == "relevance-2":
            self.source = RELEVANCE_2
            self.n_classes = 2
        elif variant == "relevance-5":
            self.source = RELEVANCE_5
            self.n_classes = 5
        else:
            raise ValueError(f"Unknown variant: {variant}. Use 'relevance-2' or 'relevance-5'.")

        self._labels_df: Optional[pd.DataFrame] = None
        self._gt: Optional[pd.Series] = None

    def _default_cache_dir(self) -> Path:
        repo_root = Path(__file__).resolve().parents[2]
        return repo_root / "data" / self.source.name

    def _download_text(self, url: str) -> str:
        with urllib.request.urlopen(url) as resp:  # nosec - fixed public URL
            return resp.read().decode("utf-8", errors="replace").strip()

    def _download_file(self, url: str, path: Path) -> None:
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

    def _verify_md5(self, file_path: Path, md5_hex: str) -> None:
        h = hashlib.md5()  # nosec - used only for integrity check
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        got = h.hexdigest()
        if got != md5_hex:
            raise ValueError(f"MD5 mismatch for {file_path.name}: expected {md5_hex}, got {got}")

    def load_dataset(self) -> None:
        cache_dir = Path(self.cache_dir) if self.cache_dir else self._default_cache_dir()
        zip_path = cache_dir / f"{self.source.name}.zip"

        self._download_file(self.source.zip_url, zip_path)
        try:
            expected_md5 = self._download_text(self.source.md5_url).split()[0]
            self._verify_md5(zip_path, expected_md5)
        except Exception as e:
            # Integrity check is nice-to-have; keep user unblocked if md5 endpoint fails transiently.
            if self.verbose:
                print(f"Warning: checksum verification skipped/failed: {e}")

        def _open_member(zf: zipfile.ZipFile, filename: str):
            # Some zips contain a top-level folder (e.g. "relevance-2/crowd_labels.csv")
            try:
                return zf.open(filename)
            except KeyError:
                matches = [n for n in zf.namelist() if n.endswith("/" + filename) or n.endswith(filename)]
                if len(matches) == 0:
                    raise
                # Prefer the shortest match (usually the direct path under the root folder)
                member = sorted(matches, key=len)[0]
                return zf.open(member)

        with zipfile.ZipFile(zip_path, "r") as zf:
            with _open_member(zf, self.source.crowd_labels_filename) as f:
                labels_df = pd.read_csv(f)
            with _open_member(zf, self.source.gt_filename) as f:
                gt_df = pd.read_csv(f)

        # Crowd-Kit convention: performer -> worker
        if "performer" in labels_df.columns and "worker" not in labels_df.columns:
            labels_df = labels_df.rename(columns={"performer": "worker"})

        # Required columns: task, worker, label
        for col in ("task", "worker", "label"):
            if col not in labels_df.columns:
                raise ValueError(f"Missing column '{col}' in crowd_labels.csv. Columns={list(labels_df.columns)}")

        if "task" not in gt_df.columns or "label" not in gt_df.columns:
            raise ValueError(f"Unexpected gt.csv columns: {list(gt_df.columns)} (need task,label)")

        gt = gt_df.set_index("task")["label"].rename("true_label")

        self._labels_df = labels_df[["task", "worker", "label"]].copy()
        self._gt = gt

        if self.verbose:
            print(f"✓ Loaded {self.source.name}: labels={len(self._labels_df):,}, tasks_gt={len(self._gt):,}")

    def prepare_data(
        self,
        max_instances: int = 2000,
        max_workers: int = 200,
        min_labels_per_instance: int = 3,
        seed: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
        """
        Convert sparse (task, worker, label) data into dense matrix for this repo.
        """
        if self._labels_df is None or self._gt is None:
            raise ValueError("Please call load_dataset() before prepare_data().")

        rng = np.random.default_rng(seed)

        df = self._labels_df
        gt = self._gt

        # Keep only tasks with ground truth
        df = df[df["task"].isin(gt.index)].copy()

        # Normalize labels into 0..C-1 ints
        # relevance-2 usually 0/1, relevance-5 usually 1..5; we remap if needed.
        y = df["label"]
        if self.source.name == "relevance-2":
            # Ensure {0,1}
            df.loc[:, "label"] = y.astype(int)
            gt_mapped = gt.astype(int)
            # If labels are {1,2} etc, remap by sorted uniques.
            uniq = sorted(pd.unique(pd.concat([df["label"], gt_mapped])))
            if uniq != [0, 1]:
                mapping = {v: i for i, v in enumerate(uniq)}
                df.loc[:, "label"] = df["label"].map(mapping).astype(int)
                gt_mapped = gt_mapped.map(mapping).astype(int)
            gt = gt_mapped
            self.n_classes = 2
        else:
            # 5-point scale
            df.loc[:, "label"] = y.astype(int)
            gt_mapped = gt.astype(int)
            uniq = sorted(pd.unique(pd.concat([df["label"], gt_mapped])))
            mapping = {v: i for i, v in enumerate(uniq)}
            df.loc[:, "label"] = df["label"].map(mapping).astype(int)
            gt = gt_mapped.map(mapping).astype(int)
            self.n_classes = len(mapping)

        # Optionally cap workers: keep most active workers for stability
        if max_workers is not None and max_workers > 0:
            worker_counts = df["worker"].value_counts()
            keep_workers = worker_counts.head(max_workers).index
            df = df[df["worker"].isin(keep_workers)]

        # Filter tasks with enough labels (after worker filtering)
        task_counts = df["task"].value_counts()
        eligible_tasks = task_counts[task_counts >= min_labels_per_instance].index.to_numpy()

        if len(eligible_tasks) == 0:
            raise ValueError(
                "No tasks left after filtering. "
                f"Try lowering min_labels_per_instance (currently {min_labels_per_instance}) "
                f"or increasing max_workers (currently {max_workers})."
            )

        # Cap instances: sample tasks (prefer those with more labels for denser matrix)
        if max_instances is not None and max_instances > 0 and len(eligible_tasks) > max_instances:
            # weighted sampling by annotation count
            weights = task_counts.loc[eligible_tasks].to_numpy(dtype=float)
            weights = weights / weights.sum()
            chosen = rng.choice(eligible_tasks, size=max_instances, replace=False, p=weights)
            keep_tasks = set(chosen.tolist())
        else:
            keep_tasks = set(eligible_tasks.tolist())

        df = df[df["task"].isin(keep_tasks)]
        gt_sub = gt[gt.index.isin(keep_tasks)].copy()

        # Recompute unique ids after filtering
        task_ids = sorted(gt_sub.index.tolist())
        worker_ids = sorted(df["worker"].unique().tolist())

        task_id_to_idx = {t: i for i, t in enumerate(task_ids)}
        worker_id_to_idx = {w: j for j, w in enumerate(worker_ids)}

        I, J = len(task_ids), len(worker_ids)
        labels = np.full((I, J), -1, dtype=int)
        truth = np.zeros((I,), dtype=int)

        for t, y_true in gt_sub.items():
            truth[task_id_to_idx[t]] = int(y_true)

        # Fill dense labels
        # If multiple annotations per (task, worker), keep the last one (rare).
        for row in df.itertuples(index=False):
            i = task_id_to_idx.get(row.task)
            j = worker_id_to_idx.get(row.worker)
            if i is None or j is None:
                continue
            labels[i, j] = int(row.label)

        coverage = float((labels != -1).sum() / (I * J))
        avg_annotations = float((labels != -1).sum(axis=1).mean())

        metadata: Dict[str, Any] = {
            "dataset": self.source.name,
            "n_instances": I,
            "n_workers": J,
            "n_classes": self.n_classes,
            "coverage": coverage,
            "avg_annotations_per_instance": avg_annotations,
            "min_labels_per_instance": min_labels_per_instance,
            "max_instances": max_instances,
            "max_workers": max_workers,
            "seed": seed,
            "task_ids": task_ids,
            "worker_ids": worker_ids,
            "task_id_to_idx": task_id_to_idx,
            "worker_id_to_idx": worker_id_to_idx,
            "source": {"zip_url": self.source.zip_url, "md5_url": self.source.md5_url},
        }

        if self.verbose:
            print(
                f"Prepared dense matrix: labels shape={labels.shape}, "
                f"coverage={coverage:.2%}, avg_anns/item={avg_annotations:.2f}"
            )

        return labels, truth, self.n_classes, metadata


def quick_load_toloka_relevance(
    variant: str = "relevance-2",
    cache_dir: Optional[str] = None,
    verbose: bool = True,
    **prepare_kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
    """Convenience wrapper."""
    loader = TolokaRelevanceLoader(variant=variant, cache_dir=cache_dir, verbose=verbose)
    loader.load_dataset()
    return loader.prepare_data(**prepare_kwargs)

