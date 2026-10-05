"""
Weather Sentiment (AMT) dataset loader.

Source:
  https://eprints.soton.ac.uk/376543/1/WeatherSentiment_amt.csv

Each row format:
  worker_id, task_id, worker_label, gold_label, time_seconds
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .base_loader import BaseDataLoader


class WeatherSentimentAMTLoader(BaseDataLoader):
    DEFAULT_URL = "https://eprints.soton.ac.uk/376543/1/WeatherSentiment_amt.csv"

    def __init__(self, data_dir: Optional[str | Path] = None, verbose: bool = True) -> None:
        super().__init__()
        self.verbose = verbose
        self.data_dir = (
            Path(data_dir)
            if data_dir is not None
            else Path(__file__).parent.parent.parent / "data" / "weather_sentiment_amt"
        )
        self.csv_path = self.data_dir / "WeatherSentiment_amt.csv"
        self._df: Optional[pd.DataFrame] = None
        self.n_classes: int = 5

    def _ensure_downloaded(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.csv_path.exists():
            return
        if self.verbose:
            print(f"Downloading Weather Sentiment-AMT -> {self.csv_path}")
        df = pd.read_csv(
            self.DEFAULT_URL,
            header=None,
            names=["worker", "task", "label", "gold", "seconds"],
        )
        df.to_csv(self.csv_path, index=False)

    def load_dataset(self) -> None:
        self._ensure_downloaded()
        self._df = pd.read_csv(self.csv_path)
        self._df = self._df.dropna(subset=["worker", "task", "label", "gold"]).copy()
        self._df["worker"] = self._df["worker"].astype(str)
        self._df["task"] = self._df["task"].astype(str)
        self._df["label"] = self._df["label"].astype(int)
        self._df["gold"] = self._df["gold"].astype(int)

        if self.verbose:
            print(
                "Loaded Weather Sentiment-AMT: "
                f"rows={len(self._df):,}, tasks={self._df['task'].nunique()}, "
                f"workers={self._df['worker'].nunique()}"
            )

    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
        if self._df is None:
            raise ValueError("Call load_dataset() before prepare_data().")

        df = self._df.copy()
        # Ensure one label per (task, worker)
        df = df.drop_duplicates(subset=["task", "worker"], keep="last")

        task_ids = sorted(df["task"].unique().tolist())
        worker_ids = sorted(df["worker"].unique().tolist())
        task_to_i = {t: i for i, t in enumerate(task_ids)}
        worker_to_j = {w: j for j, w in enumerate(worker_ids)}

        I, J, C = len(task_ids), len(worker_ids), self.n_classes
        labels = np.full((I, J), -1, dtype=int)
        truth = np.full(I, -1, dtype=int)

        for _, r in df.iterrows():
            i = task_to_i[r["task"]]
            j = worker_to_j[r["worker"]]
            labels[i, j] = int(r["label"])

        # Gold is consistent per task in this dataset.
        gold_per_task = (
            df.groupby("task")["gold"]
            .agg(lambda x: int(np.bincount(x.to_numpy(dtype=int), minlength=C).argmax()))
            .to_dict()
        )
        for t, i in task_to_i.items():
            truth[i] = int(gold_per_task[t])

        coverage = float((labels != -1).sum() / (I * J)) if I * J > 0 else 0.0
        avg_anns = float((labels != -1).sum(axis=1).mean()) if I > 0 else 0.0

        meta: Dict[str, Any] = {
            "dataset": "weather_sentiment_amt",
            "n_instances": int(I),
            "n_workers": int(J),
            "n_classes": int(C),
            "coverage": coverage,
            "avg_annotations_per_instance": avg_anns,
            "task_ids": task_ids,
            "worker_ids": worker_ids,
            "source_url": self.DEFAULT_URL,
        }
        return labels, truth, C, meta


def quick_load_weather_sentiment_amt(
    data_dir: Optional[str | Path] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
    loader = WeatherSentimentAMTLoader(data_dir=data_dir, verbose=verbose)
    loader.load_dataset()
    return loader.prepare_data()
