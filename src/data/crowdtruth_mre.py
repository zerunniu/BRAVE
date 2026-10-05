"""
CrowdTruth: Medical Relation Extraction (MRE) loader.

Source:
  - https://github.com/CrowdTruth/Medical-Relation-Extraction

We load the *raw* crowdsourcing outputs for the RelEx task (CrowdFlower exports),
and build a binary classification task for a chosen target relation:
  - target_relation = "TREATS"  (treat)
  - target_relation = "CAUSES"  (cause)

Each row in the raw RelEx CSV is a worker annotation:
  - task id: `_unit_id`
  - worker id: `_worker_id`
  - worker choice: `step_1_select_the_valid_relations` (string like "[TREATS]" or "[NONE]")
  - expert decision: `expdec` (values 1 / -1) used as gold truth for whether the seed relation holds

Output:
  labels: (I,J) with -1 missing, labels in {0,1}
  truth:  (I,) in {0,1}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import urllib.request

import numpy as np
import pandas as pd

from .base_loader import BaseDataLoader


@dataclass(frozen=True)
class CrowdTruthMRESource:
    base_raw_url: str
    rel_ex_batches: int
    ground_truth_base_url: str


DEFAULT_SOURCE = CrowdTruthMRESource(
    base_raw_url="https://raw.githubusercontent.com/CrowdTruth/Medical-Relation-Extraction/master/raw/RelEx",
    rel_ex_batches=29,  # RelEx_batch_01.csv .. RelEx_batch_29.csv exist
    ground_truth_base_url="https://raw.githubusercontent.com/CrowdTruth/Medical-Relation-Extraction/master",
)


class CrowdTruthMRELoader(BaseDataLoader):
    def __init__(
        self,
        target: str = "treat",
        cache_dir: Optional[str] = None,
        source: CrowdTruthMRESource = DEFAULT_SOURCE,
        verbose: bool = True,
    ):
        super().__init__()
        if target not in {"treat", "cause"}:
            raise ValueError("target must be 'treat' or 'cause'")
        self.target = target
        self.cache_dir = cache_dir
        self.source = source
        self.verbose = verbose
        self.n_classes = 2
        self._df: Optional[pd.DataFrame] = None

    def _default_cache_dir(self) -> Path:
        repo_root = Path(__file__).resolve().parents[2]
        return repo_root / "data" / "crowdtruth_mre"

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

        dfs: List[pd.DataFrame] = []
        for k in range(1, int(self.source.rel_ex_batches) + 1):
            name = f"RelEx_batch_{k:02d}.csv"
            url = f"{self.source.base_raw_url}/{name}"
            path = cache_dir / "RelEx" / name
            self._download_if_needed(url, path)
            df = pd.read_csv(path)

            required = {"_unit_id", "_worker_id", "step_1_select_the_valid_relations", "sentence", "term1", "term2"}
            missing = required - set(df.columns)
            if missing:
                # Unexpected export; skip this batch.
                if self.verbose:
                    print(f"Warning: skipping {name} (missing cols: {sorted(missing)})")
                continue

            df2 = df[["_unit_id", "_worker_id", "step_1_select_the_valid_relations", "sentence", "term1", "term2"]].copy()
            dfs.append(df2)

        df_all = pd.concat(dfs, ignore_index=True)

        # Load processed ground truth for this relation and join by (sentence, term1, term2).
        gt_name = "ground_truth_treat.csv" if self.target == "treat" else "ground_truth_cause.csv"
        gt_url = f"{self.source.ground_truth_base_url}/{gt_name}"
        gt_path = cache_dir / gt_name
        self._download_if_needed(gt_url, gt_path)
        gt = pd.read_csv(gt_path)
        for c in ("sentence", "term1", "term2"):
            gt[c] = gt[c].astype(str).str.strip()
        for c in ("sentence", "term1", "term2"):
            df_all[c] = df_all[c].astype(str).str.strip()

        # `expert` is the expert judgment of whether the baseline label is correct (values 1 / -1, may be NaN).
        gt_small = gt[["sentence", "term1", "term2", "expert"]].copy()
        merged = df_all.merge(gt_small, on=["sentence", "term1", "term2"], how="inner")
        merged = merged[merged["expert"].notna()].copy()
        if len(merged) == 0:
            raise ValueError(f"No rows left after joining expert labels for target={self.target}.")

        # Gold truth per task from expert: 1 -> 1, -1 -> 0
        merged["truth"] = (merged["expert"].astype(int) == 1).astype(int)

        # Worker label: 1 if worker selected the target relation anywhere, else 0.
        # Some rows may contain multiple relations separated by newlines.
        target_token = "TREATS" if self.target == "treat" else "CAUSES"
        sel = merged["step_1_select_the_valid_relations"].astype(str)
        # Match target relation token in strings like "[TREATS]" or multi-line
        # "[TREATS]\n[CAUSES]".
        merged["label_bin"] = sel.str.contains(rf"\[{target_token}\]", regex=True).astype(int)

        self._df = merged[["_unit_id", "_worker_id", "label_bin", "truth"]].copy()
        if self.verbose:
            n_tasks = self._df["_unit_id"].nunique()
            n_workers = self._df["_worker_id"].nunique()
            print(f"✓ Loaded CrowdTruth MRE (target={self.target}): tasks={n_tasks:,}, workers={n_workers:,}, ann={len(self._df):,}")

    def prepare_data(
        self,
        max_instances: Optional[int] = 5000,
        max_workers: Optional[int] = 500,
        min_labels_per_instance: int = 3,
        seed: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
        if self._df is None:
            raise ValueError("Call load_dataset() before prepare_data().")

        rng = np.random.default_rng(int(seed))
        df = self._df.copy()

        # Cap workers by activity
        if max_workers is not None and int(max_workers) > 0:
            wc = df["_worker_id"].value_counts()
            keep_workers = wc.head(int(max_workers)).index
            df = df[df["_worker_id"].isin(keep_workers)].copy()

        # Filter tasks with enough labels
        tc = df["_unit_id"].value_counts()
        eligible = tc[tc >= int(min_labels_per_instance)].index.to_numpy()
        if eligible.size == 0:
            raise ValueError("No tasks left after filtering; lower min_labels_per_instance or increase max_workers.")

        if max_instances is not None and int(max_instances) > 0 and eligible.size > int(max_instances):
            weights = tc.loc[eligible].to_numpy(dtype=float)
            weights = weights / weights.sum()
            chosen = rng.choice(eligible, size=int(max_instances), replace=False, p=weights)
            keep_tasks = set(chosen.tolist())
        else:
            keep_tasks = set(eligible.tolist())

        df = df[df["_unit_id"].isin(keep_tasks)].copy()

        # Deduplicate (task, worker), keep last
        df = df.drop_duplicates(subset=["_unit_id", "_worker_id"], keep="last")

        task_ids = sorted(df["_unit_id"].unique().tolist())
        worker_ids = sorted(df["_worker_id"].unique().tolist())
        task_to_i = {t: i for i, t in enumerate(task_ids)}
        worker_to_j = {w: j for j, w in enumerate(worker_ids)}

        I, J = len(task_ids), len(worker_ids)
        labels = np.full((I, J), -1, dtype=int)

        # Truth is constant per task; take first
        truth_map = df.groupby("_unit_id")["truth"].first().to_dict()
        truth = np.array([int(truth_map[t]) for t in task_ids], dtype=int)

        i_idx = df["_unit_id"].map(task_to_i).to_numpy(dtype=int)
        j_idx = df["_worker_id"].map(worker_to_j).to_numpy(dtype=int)
        l = df["label_bin"].to_numpy(dtype=int)
        labels[i_idx, j_idx] = l

        coverage = float((labels != -1).sum() / (I * J)) if I * J > 0 else 0.0
        avg_anns = float((labels != -1).sum(axis=1).mean()) if I > 0 else 0.0

        meta: Dict[str, Any] = {
            "dataset": "crowdtruth_mre",
            "target": self.target,
            "n_instances": int(I),
            "n_workers": int(J),
            "n_classes": 2,
            "coverage": coverage,
            "avg_annotations_per_instance": avg_anns,
            "max_instances": max_instances,
            "max_workers": max_workers,
            "min_labels_per_instance": int(min_labels_per_instance),
            "seed": int(seed),
            "task_ids": task_ids,
            "worker_ids": worker_ids,
            "source": {"repo": "CrowdTruth/Medical-Relation-Extraction", "raw": "raw/RelEx/*.csv"},
        }

        return labels, truth, 2, meta


def quick_load_crowdtruth_mre(
    target: str = "treat",
    cache_dir: Optional[str] = None,
    verbose: bool = True,
    **prepare_kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
    loader = CrowdTruthMRELoader(target=target, cache_dir=cache_dir, verbose=verbose)
    loader.load_dataset()
    return loader.prepare_data(**prepare_kwargs)

