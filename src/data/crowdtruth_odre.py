"""
CrowdTruth Open Domain Relation Extraction loader.

Repository:
  https://github.com/CrowdTruth/Open-Domain-Relation-Extraction

This dataset is originally multi-label at worker level (`Answer.Q1` may contain
multiple relations split by '|'). To support this repo's single-label
aggregators, we map each worker response to the first selected relation token.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import glob

import numpy as np
import pandas as pd

from .base_loader import BaseDataLoader


class CrowdTruthODRELoader(BaseDataLoader):
    def __init__(self, data_dir: Optional[str | Path] = None, verbose: bool = True) -> None:
        super().__init__()
        self.verbose = verbose
        self.data_dir = (
            Path(data_dir)
            if data_dir is not None
            else Path(__file__).parent.parent.parent / "data" / "crowdtruth_odre" / "Open-Domain-Relation-Extraction"
        )
        self._amt: Optional[pd.DataFrame] = None
        self._agg: Optional[pd.DataFrame] = None
        self.n_classes: Optional[int] = None

    def load_dataset(self) -> None:
        amt_dir = self.data_dir / "data" / "input" / "AMT"
        agg_path = self.data_dir / "data" / "output" / "aggregated_sentences.csv"

        if not amt_dir.exists():
            raise FileNotFoundError(
                f"AMT folder not found: {amt_dir}. "
                "Please clone CrowdTruth/Open-Domain-Relation-Extraction into data/crowdtruth_odre."
            )
        if not agg_path.exists():
            raise FileNotFoundError(f"aggregated_sentences.csv not found: {agg_path}")

        parts: list[pd.DataFrame] = []
        for fp in sorted(glob.glob(str(amt_dir / "*.csv"))):
            df = pd.read_csv(fp, usecols=["WorkerId", "Input.sent_id", "Answer.Q1"])
            parts.append(df)
        if not parts:
            raise ValueError(f"No CSV files found in {amt_dir}")

        self._amt = pd.concat(parts, ignore_index=True)
        self._agg = pd.read_csv(agg_path, usecols=["input.sent_id", "max_rel"])

        if self.verbose:
            print(
                "Loaded CrowdTruth ODRE: "
                f"worker_rows={len(self._amt):,}, "
                f"unique_tasks={self._amt['Input.sent_id'].nunique()}, "
                f"workers={self._amt['WorkerId'].nunique()}"
            )

    @staticmethod
    def _norm_label(raw: str) -> str:
        lab = str(raw).strip()
        if "|" in lab:
            lab = lab.split("|", 1)[0].strip()
        # Normalize known variant in raw AMT data.
        if lab == "per:alternate_names":
            return "org:alternate_names"
        return lab

    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
        if self._amt is None or self._agg is None:
            raise ValueError("Call load_dataset() before prepare_data().")

        amt = self._amt.dropna(subset=["WorkerId", "Input.sent_id", "Answer.Q1"]).copy()
        amt["WorkerId"] = amt["WorkerId"].astype(str)
        amt["Input.sent_id"] = amt["Input.sent_id"].astype(str)
        amt["label_norm"] = amt["Answer.Q1"].astype(str).map(self._norm_label)
        amt = amt[amt["label_norm"] != ""].copy()

        # Keep one latest annotation per (task, worker).
        amt = amt.drop_duplicates(subset=["Input.sent_id", "WorkerId"], keep="last")

        agg = self._agg.dropna(subset=["input.sent_id", "max_rel"]).copy()
        agg["input.sent_id"] = agg["input.sent_id"].astype(str)
        agg["max_rel"] = agg["max_rel"].astype(str).map(self._norm_label)
        agg = agg.drop_duplicates(subset=["input.sent_id"], keep="last")

        truth_map = dict(zip(agg["input.sent_id"], agg["max_rel"]))
        amt = amt[amt["Input.sent_id"].isin(truth_map.keys())].copy()

        task_ids = sorted(amt["Input.sent_id"].unique().tolist())
        worker_ids = sorted(amt["WorkerId"].unique().tolist())

        classes = sorted(set(amt["label_norm"].unique().tolist()) | {truth_map[t] for t in task_ids})
        class_to_idx = {c: i for i, c in enumerate(classes)}

        task_to_i = {t: i for i, t in enumerate(task_ids)}
        worker_to_j = {w: j for j, w in enumerate(worker_ids)}

        I, J, C = len(task_ids), len(worker_ids), len(classes)
        labels = np.full((I, J), -1, dtype=int)
        truth = np.full(I, -1, dtype=int)

        for _, r in amt.iterrows():
            i = task_to_i[r["Input.sent_id"]]
            j = worker_to_j[r["WorkerId"]]
            labels[i, j] = class_to_idx[r["label_norm"]]

        for t, i in task_to_i.items():
            truth[i] = class_to_idx[truth_map[t]]

        self.n_classes = C
        coverage = float((labels != -1).sum() / (I * J)) if I * J > 0 else 0.0
        avg_anns = float((labels != -1).sum(axis=1).mean()) if I > 0 else 0.0

        meta: Dict[str, Any] = {
            "dataset": "crowdtruth_odre",
            "n_instances": int(I),
            "n_workers": int(J),
            "n_classes": int(C),
            "coverage": coverage,
            "avg_annotations_per_instance": avg_anns,
            "task_ids": task_ids,
            "worker_ids": worker_ids,
            "classes": classes,
            "note": "Worker multi-label answers are reduced to first label token for single-label aggregation.",
        }
        return labels, truth, C, meta


def quick_load_crowdtruth_odre(
    data_dir: Optional[str | Path] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
    loader = CrowdTruthODRELoader(data_dir=data_dir, verbose=verbose)
    loader.load_dataset()
    return loader.prepare_data()
