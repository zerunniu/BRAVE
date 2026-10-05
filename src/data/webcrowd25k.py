"""
WebCrowd25K dataset loader.

We treat each (topic, document) pair as an instance and each MTurk worker (wid)
as a worker. The 4-point graded relevance label is mapped to classes:

0 = Definitely Not Relevant
1 = Probably Not Relevant
2 = Probably Relevant
3 = Definitely Relevant

We ignore rows where label == -1 (no response). Gold labels from
`gold_judjements.txt` are loaded for reference but not used as truth by
default, to stay consistent with other crowdsourcing datasets where truth is
defined by majority vote over the crowd.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

from .base_loader import BaseDataLoader


class WebCrowd25KLoader(BaseDataLoader):
    def __init__(self, data_dir: str | Path | None = None) -> None:
        super().__init__()
        self.data_dir = Path(data_dir) if data_dir is not None else Path(__file__).parent.parent.parent / "data" / "webcrowd25k"
        self.n_classes = 4
        self._crowd: pd.DataFrame | None = None
        self._gold: pd.DataFrame | None = None

    def load_dataset(self) -> None:
        """Load crowd and gold judgments into memory."""
        crowd_path = self.data_dir / "crowd_judgements.csv"
        # Dataset releases use inconsistent spellings across mirrors/README.
        gold_path = self.data_dir / "gold_judgements.txt"
        if not gold_path.exists():
            gold_path = self.data_dir / "gold_judjements.txt"

        if not crowd_path.exists():
            raise FileNotFoundError(f"Crowd file not found: {crowd_path}")

        self._crowd = pd.read_csv(crowd_path)
        # Filter out missing labels
        self._crowd = self._crowd[self._crowd["label"] >= 0].copy()
        self._crowd["label"] = self._crowd["label"].astype(int)

        if gold_path.exists():
            self._gold = pd.read_csv(
                gold_path,
                sep=r"\s+",
                header=None,
                names=["tid", "unused", "did", "gold_label"],
            )
        else:
            self._gold = None

        print(f"Loaded WebCrowd25K: {len(self._crowd)} crowd judgments")

    @staticmethod
    def _norm_tid(val) -> str:
        # Crowd csv may store topic ids as float-like strings (e.g., "267.0")
        return str(int(float(val)))

    @staticmethod
    def _norm_did(val) -> str:
        return str(val).strip()

    def prepare_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, int, Dict]:
        """
        Convert WebCrowd25K into (labels, truth, n_classes, metadata).

        - Instances: unique (tid, did) pairs.
        - Workers: unique `wid`.
        - Labels: integer matrix (n_instances, n_workers), -1 for missing.
        - Truth: majority vote over crowd labels per instance (on 4 classes).
        """
        if self._crowd is None:
            raise ValueError("Call load_dataset() first.")

        df = self._crowd.copy()

        # Define instance and worker indices
        df["tid_norm"] = df["tid"].apply(self._norm_tid)
        df["did_norm"] = df["did"].apply(self._norm_did)
        df["instance_id"] = df["tid_norm"] + "::" + df["did_norm"]
        instance_ids = sorted(df["instance_id"].unique())
        worker_ids = sorted(df["wid"].unique())
        inst_to_idx = {iid: i for i, iid in enumerate(instance_ids)}
        wid_to_idx = {wid: j for j, wid in enumerate(worker_ids)}

        I = len(instance_ids)
        J = len(worker_ids)
        C = self.n_classes

        labels = np.full((I, J), -1, dtype=int)

        for _, row in df.iterrows():
            i = inst_to_idx[row["instance_id"]]
            j = wid_to_idx[row["wid"]]
            lab = int(row["label"])
            if 0 <= lab < C:
                labels[i, j] = lab

        # Coverage stats
        coverage = (labels != -1).sum() / float(I * J)
        avg_annotations = (labels != -1).sum(axis=1).mean()

        # Majority vote truth over crowd labels
        truth = np.zeros(I, dtype=int)
        for i in range(I):
            inst_labels = labels[i][labels[i] != -1]
            if inst_labels.size == 0:
                truth[i] = 0
            else:
                counts = np.bincount(inst_labels, minlength=C)
                truth[i] = int(np.argmax(counts))

        print(f"WebCrowd25K stats: I={I}, J={J}, C={C}, coverage={coverage:.3%}, avg_ann_per_item={avg_annotations:.2f}")

        metadata: Dict = {
            "n_instances": I,
            "n_workers": J,
            "n_classes": C,
            "coverage": coverage,
            "avg_annotations_per_instance": avg_annotations,
            "instance_ids": instance_ids,
            "worker_ids": worker_ids,
            "inst_to_idx": inst_to_idx,
            "wid_to_idx": wid_to_idx,
        }
        return labels, truth, C, metadata

    def build_official_gold(
        self,
        instance_ids: list[str],
        mapping: str = "trec6_to_4",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build official gold labels aligned to `instance_ids`.

        Args:
            instance_ids: list of instance keys in the same order as label matrix
                entries. Each key is "tid::did".
            mapping: gold label mapping strategy.
                - "trec6_to_4" (default): {-2,0}->{0}, 1->1, 2->2, {3,4}->{3}
                - "binary_rel": {-2,0,1}->{0}, {2,3,4}->{1}

        Returns:
            official_gold: np.ndarray of mapped labels, -1 if missing/unmapped
            mask: boolean np.ndarray, True where official gold is available
        """
        if self._gold is None:
            raise ValueError("Official gold file not loaded.")

        gold_map = {}
        for _, r in self._gold.iterrows():
            key = f"{self._norm_tid(r['tid'])}::{self._norm_did(r['did'])}"
            raw = int(r["gold_label"])

            if mapping == "trec6_to_4":
                if raw in (-2, 0):
                    val = 0
                elif raw == 1:
                    val = 1
                elif raw == 2:
                    val = 2
                elif raw in (3, 4):
                    val = 3
                else:
                    continue
            elif mapping == "binary_rel":
                if raw in (-2, 0, 1):
                    val = 0
                elif raw in (2, 3, 4):
                    val = 1
                else:
                    continue
            else:
                raise ValueError(f"Unknown mapping: {mapping}")

            gold_map[key] = val

        official_gold = np.full(len(instance_ids), -1, dtype=int)
        for i, key in enumerate(instance_ids):
            if key in gold_map:
                official_gold[i] = gold_map[key]

        mask = official_gold != -1
        return official_gold, mask


def quick_load_webcrowd25k() -> Tuple[np.ndarray, np.ndarray, int, Dict]:
    loader = WebCrowd25KLoader()
    loader.load_dataset()
    return loader.prepare_data()

