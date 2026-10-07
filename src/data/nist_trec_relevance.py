"""
NIST TREC relevance crowdsourcing benchmark loader.

Source package/provenance:
  - Crowd-Kit public dataset mirror: nist-trec-relevance
  - Original benchmark: NIST TREC Relevance Feedback Track 2010

The loader returns the crowd annotation matrix and an evaluation mask marking
tasks with independent NIST gold labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import zipfile

import numpy as np
import pandas as pd
import requests

from .base_loader import BaseDataLoader


@dataclass(frozen=True)
class NistTrecRelevanceSource:
    zip_url: str
    crowd_labels_filename: str = "crowd_labels.csv"
    gt_filename: str = "gt.csv"


DEFAULT_SOURCE = NistTrecRelevanceSource(
    zip_url="https://tlk.s3.yandex.net/dataset/crowd-kit/relevance.zip",
)


class NistTrecRelevanceLoader(BaseDataLoader):
    """
    Load the NIST TREC relevance benchmark into the repo-standard dense matrix.

    Crowd labels and gold labels use the 4-grade relevance scale encoded as
    integer class ids in {0,1,2,3}. The `eval_mask` returned by `prepare_data()`
    marks tasks with available gold labels.
    """

    def __init__(
        self,
        cache_dir: Optional[str | Path] = None,
        source: NistTrecRelevanceSource = DEFAULT_SOURCE,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self._default_cache_dir()
        self.source = source
        self.verbose = verbose
        self.n_classes = 4
        self._labels_df: Optional[pd.DataFrame] = None
        self._gt: Optional[pd.Series] = None

    def _default_cache_dir(self) -> Path:
        return Path(__file__).resolve().parents[2] / "data" / "nist_trec_relevance"

    def _download_file(self, url: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 0:
            return
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        if self.verbose:
            print(f"Downloading {url} -> {path}")
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            with requests.get(url, headers=headers, timeout=120, stream=True) as response:
                response.raise_for_status()
                with tmp_path.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            tmp_path.replace(path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

    @staticmethod
    def _open_member(zf: zipfile.ZipFile, filename: str):
        try:
            return zf.open(filename)
        except KeyError:
            matches = [n for n in zf.namelist() if n.endswith("/" + filename) or n.endswith(filename)]
            if not matches:
                raise
            return zf.open(sorted(matches, key=len)[0])

    def load_dataset(self) -> None:
        zip_path = self.cache_dir / "nist-trec-relevance.zip"
        self._download_file(self.source.zip_url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            with self._open_member(zf, self.source.crowd_labels_filename) as f:
                labels_df = pd.read_csv(f)
            with self._open_member(zf, self.source.gt_filename) as f:
                gt_df = pd.read_csv(f)

        if "performer" in labels_df.columns and "worker" not in labels_df.columns:
            labels_df = labels_df.rename(columns={"performer": "worker"})

        for col in ("task", "worker", "label"):
            if col not in labels_df.columns:
                raise ValueError(
                    f"Missing column '{col}' in crowd labels. Columns={list(labels_df.columns)}"
                )
        if "task" not in gt_df.columns or "label" not in gt_df.columns:
            raise ValueError(f"Unexpected gt.csv columns: {list(gt_df.columns)}")

        labels_df = labels_df[["task", "worker", "label"]].copy()
        labels_df["label"] = labels_df["label"].astype(int)
        gt = gt_df.set_index("task")["label"].astype(int).rename("true_label")

        self._labels_df = labels_df
        self._gt = gt
        if self.verbose:
            print(
                "Loaded NIST TREC relevance: "
                f"{len(labels_df):,} crowd labels, {gt.index.nunique():,} gold tasks"
            )

    def prepare_data(
        self,
        only_gold_tasks: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
        if self._labels_df is None or self._gt is None:
            raise ValueError("Please call load_dataset() before prepare_data().")

        df = self._labels_df.copy()
        gt = self._gt.copy()
        if only_gold_tasks:
            df = df[df["task"].isin(gt.index)].copy()

        task_ids = sorted(df["task"].unique().tolist())
        worker_ids = sorted(df["worker"].unique().tolist())
        task_id_to_idx = {task_id: i for i, task_id in enumerate(task_ids)}
        worker_id_to_idx = {worker_id: j for j, worker_id in enumerate(worker_ids)}

        I, J = len(task_ids), len(worker_ids)
        labels = np.full((I, J), -1, dtype=int)
        for row in df.itertuples(index=False):
            labels[task_id_to_idx[row.task], worker_id_to_idx[row.worker]] = int(row.label)

        truth = np.full(I, -1, dtype=int)
        for task_id, y_true in gt.items():
            idx = task_id_to_idx.get(task_id)
            if idx is not None:
                truth[idx] = int(y_true)

        eval_mask = truth != -1
        n_observed = int((labels != -1).sum())
        coverage = float(n_observed / (I * J))
        avg_annotations = float((labels != -1).sum(axis=1).mean())

        metadata: Dict[str, Any] = {
            "dataset": "nist_trec_relevance",
            "n_instances": I,
            "n_workers": J,
            "n_classes": self.n_classes,
            "n_observed_annotations": n_observed,
            "coverage": coverage,
            "avg_annotations_per_instance": avg_annotations,
            "task_ids": task_ids,
            "worker_ids": worker_ids,
            "task_id_to_idx": task_id_to_idx,
            "worker_id_to_idx": worker_id_to_idx,
            "eval_mask": eval_mask,
            "n_gold_instances": int(eval_mask.sum()),
            "only_gold_tasks": bool(only_gold_tasks),
            "source": {
                "zip_url": self.source.zip_url,
                "gold_origin": "NIST TREC Relevance Feedback Track 2010",
                "crowd_origin": "Crowd-Kit nist-trec-relevance mirror",
            },
        }

        if self.verbose:
            print(
                "Prepared NIST TREC matrix: "
                f"I={I}, J={J}, obs={n_observed}, coverage={coverage:.2%}, "
                f"avg_anns/item={avg_annotations:.2f}, gold={int(eval_mask.sum())}"
            )

        return labels, truth, self.n_classes, metadata


def quick_load_nist_trec_relevance(
    cache_dir: Optional[str | Path] = None,
    verbose: bool = True,
    **prepare_kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
    loader = NistTrecRelevanceLoader(cache_dir=cache_dir, verbose=verbose)
    loader.load_dataset()
    return loader.prepare_data(**prepare_kwargs)
