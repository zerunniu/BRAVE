"""
Music Genre crowdsourcing benchmark loader.

Source/provenance:
  - Crowdsourced MTurk annotations and gold metadata:
      https://fprodrigues.com/mturk-datasets.tar.gz
  - Benchmark family used in Filipe Rodrigues' crowdsourcing work and mirrored by
    peerannot's `datasets/music/music.py`.

The loader reads the crowd annotation and gold-label CSV files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import tarfile

import numpy as np
import pandas as pd
import requests

from .base_loader import BaseDataLoader


@dataclass(frozen=True)
class MusicGenreSource:
    archive_url: str


DEFAULT_SOURCE = MusicGenreSource(
    archive_url="https://fprodrigues.com/mturk-datasets.tar.gz",
)


CLASS_ORDER = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]
CLASS_TO_IDX = {label: idx for idx, label in enumerate(CLASS_ORDER)}


class MusicGenreLoader(BaseDataLoader):
    """
    Loader for the MTurk music genre classification benchmark.

    Aggregation uses the 700 training songs that have both worker
    annotations and independent gold labels.
    """

    _NEEDED_MEMBERS = {
        "music_genre_classification/mturk_answers.csv": "mturk_answers.csv",
        "music_genre_classification/music_genre_gold.csv": "music_genre_gold.csv",
        "music_genre_classification/music_genre_test.csv": "music_genre_test.csv",
        "music_genre_classification/music_genre_mturk.csv": "music_genre_mturk.csv",
    }

    def __init__(
        self,
        cache_dir: Optional[str | Path] = None,
        source: MusicGenreSource = DEFAULT_SOURCE,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self._default_cache_dir()
        self.source = source
        self.verbose = verbose
        self.n_classes = len(CLASS_ORDER)
        self._answers_df: Optional[pd.DataFrame] = None
        self._gold_df: Optional[pd.DataFrame] = None
        self._test_df: Optional[pd.DataFrame] = None

    def _default_cache_dir(self) -> Path:
        return Path(__file__).resolve().parents[2] / "data" / "music_genre"

    def _download_archive(self, path: Path) -> None:
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
            print(f"Downloading {self.source.archive_url} -> {path}")
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            with requests.get(self.source.archive_url, headers=headers, timeout=120, stream=True) as response:
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

    def _ensure_extracted_files(self) -> None:
        archive_path = self.cache_dir / "mturk-datasets.tar.gz"
        self._download_archive(archive_path)

        missing = [
            member for member, filename in self._NEEDED_MEMBERS.items() if not (self.cache_dir / filename).exists()
        ]
        if not missing:
            return

        extracted: set[str] = set()
        with tarfile.open(archive_path, "r|gz") as tar:
            for member in tar:
                target_name = self._NEEDED_MEMBERS.get(member.name)
                if target_name is None:
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                data = f.read()
                out_path = self.cache_dir / target_name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(data)
                extracted.add(member.name)
                if self.verbose:
                    print(f"Extracted {member.name} -> {out_path}")
                if extracted == set(self._NEEDED_MEMBERS):
                    break

        still_missing = [
            filename for filename in self._NEEDED_MEMBERS.values() if not (self.cache_dir / filename).exists()
        ]
        if still_missing:
            raise FileNotFoundError(
                "Could not extract the required music benchmark files from "
                f"{archive_path}: missing {still_missing}"
            )

    def load_dataset(self) -> None:
        self._ensure_extracted_files()

        answers_df = pd.read_csv(self.cache_dir / "mturk_answers.csv")
        gold_df = pd.read_csv(self.cache_dir / "music_genre_gold.csv")
        test_df = pd.read_csv(self.cache_dir / "music_genre_test.csv")

        required_answers = {"WorkerID", "Input.song", "Answer.pred_label"}
        if not required_answers.issubset(answers_df.columns):
            raise ValueError(
                f"mturk_answers.csv missing columns {sorted(required_answers - set(answers_df.columns))}"
            )
        if not {"id", "class"}.issubset(gold_df.columns):
            raise ValueError(f"music_genre_gold.csv missing required columns; got {list(gold_df.columns)}")
        if not {"id", "class"}.issubset(test_df.columns):
            raise ValueError(f"music_genre_test.csv missing required columns; got {list(test_df.columns)}")

        self._answers_df = answers_df
        self._gold_df = gold_df[["id", "class"]].copy()
        self._test_df = test_df[["id", "class"]].copy()
        if self.verbose:
            print(
                "Loaded Music Genre benchmark: "
                f"{len(self._answers_df):,} crowd answers, {len(self._gold_df):,} train gold items, "
                f"{len(self._test_df):,} test gold items"
            )

    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
        if self._answers_df is None or self._gold_df is None or self._test_df is None:
            raise ValueError("Please call load_dataset() before prepare_data().")

        answers_df = self._answers_df.copy()
        gold_df = self._gold_df.copy()

        answers_df["item_id"] = answers_df["Input.song"].astype(str)
        answers_df["worker_id"] = answers_df["WorkerID"].astype(str)
        answers_df["label_text"] = answers_df["Answer.pred_label"].astype(str).str.strip().str.lower()
        gold_df["item_id"] = gold_df["id"].astype(str)
        gold_df["label_text"] = gold_df["class"].astype(str).str.strip().str.lower()

        bad_gold = sorted(set(gold_df["label_text"]) - set(CLASS_TO_IDX))
        bad_answers = sorted(set(answers_df["label_text"]) - set(CLASS_TO_IDX))
        if bad_gold:
            raise ValueError(f"Unexpected gold labels in music_genre_gold.csv: {bad_gold}")
        if bad_answers:
            raise ValueError(f"Unexpected crowd labels in mturk_answers.csv: {bad_answers}")

        gold_df = gold_df[gold_df["item_id"].isin(answers_df["item_id"])].copy()
        answers_df = answers_df[answers_df["item_id"].isin(gold_df["item_id"])].copy()

        item_ids = sorted(gold_df["item_id"].unique().tolist())
        worker_ids = sorted(answers_df["worker_id"].unique().tolist())
        item_id_to_idx = {item_id: i for i, item_id in enumerate(item_ids)}
        worker_id_to_idx = {worker_id: j for j, worker_id in enumerate(worker_ids)}

        I, J = len(item_ids), len(worker_ids)
        labels = np.full((I, J), -1, dtype=int)
        truth = np.zeros(I, dtype=int)

        for row in gold_df.itertuples(index=False):
            truth[item_id_to_idx[row.item_id]] = CLASS_TO_IDX[row.label_text]

        for row in answers_df.itertuples(index=False):
            labels[item_id_to_idx[row.item_id], worker_id_to_idx[row.worker_id]] = CLASS_TO_IDX[row.label_text]

        n_observed = int((labels != -1).sum())
        coverage = float(n_observed / (I * J))
        avg_annotations = float((labels != -1).sum(axis=1).mean())

        metadata: Dict[str, Any] = {
            "dataset": "music_genre",
            "n_instances": I,
            "n_workers": J,
            "n_classes": self.n_classes,
            "n_observed_annotations": n_observed,
            "coverage": coverage,
            "avg_annotations_per_instance": avg_annotations,
            "item_ids": item_ids,
            "worker_ids": worker_ids,
            "item_id_to_idx": item_id_to_idx,
            "worker_id_to_idx": worker_id_to_idx,
            "n_test_gold_items": int(len(self._test_df)),
            "source": {
                "archive_url": self.source.archive_url,
                "crowd_annotations": "music_genre_classification/mturk_answers.csv",
                "gold_labels": "music_genre_classification/music_genre_gold.csv",
            },
        }

        if self.verbose:
            print(
                "Prepared Music Genre matrix: "
                f"I={I}, J={J}, obs={n_observed}, coverage={coverage:.2%}, "
                f"avg_anns/item={avg_annotations:.2f}"
            )

        return labels, truth, self.n_classes, metadata


def quick_load_music_genre(
    cache_dir: Optional[str | Path] = None,
    verbose: bool = True,
    **prepare_kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
    loader = MusicGenreLoader(cache_dir=cache_dir, verbose=verbose)
    loader.load_dataset()
    return loader.prepare_data(**prepare_kwargs)
