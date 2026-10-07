"""
CrowdGleason dataset loader (crowdsourced Gleason grading).

Source:
  - Zenodo record: 14178894
  - Annotation archive: `Annotations.zip` (labels and metadata).

Output:
  labels: (I, J) with -1 for missing
  truth:  (I,) test gold labels or train/val majority-vote reference labels

The downloaded `train.csv` / `val.csv` / `test.csv` in `Annotations.zip` contain:
  - marker1..marker7: crowd labels (7 annotators; -1 for missing)
  - Patch filename: task id (string)
  - For train/val: also columns MV/DS/GLAD/MACE (precomputed by authors)
  - For test: column `ground truth`

Each `marker*` column represents a worker.
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
class CrowdGleasonSource:
    annotations_zip_url: str


DEFAULT_SOURCE = CrowdGleasonSource(
    annotations_zip_url="https://zenodo.org/api/records/14178894/files/Annotations.zip/content"
)


class CrowdGleasonLoader(BaseDataLoader):
    """
    CrowdGleason loader.

    Parameters:
      split: one of {"train","val","test"}
    """

    def __init__(
        self,
        split: str = "test",
        cache_dir: Optional[str] = None,
        source: CrowdGleasonSource = DEFAULT_SOURCE,
        verbose: bool = True,
    ):
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of {'train','val','test'}")
        self.split = split
        self.cache_dir = cache_dir
        self.source = source
        self.verbose = verbose

        self._df: Optional[pd.DataFrame] = None
        self.n_classes = 4  # observed labels are 0..3 (Gleason grade groups)

    def _default_cache_dir(self) -> Path:
        repo_root = Path(__file__).resolve().parents[2]
        return repo_root / "data" / "crowdgleason"

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
        zip_path = cache_dir / "Annotations.zip"
        self._download_if_needed(self.source.annotations_zip_url, zip_path)

        with zip_path.open("rb") as f:
            b = f.read()
        zf = zipfile.ZipFile(io.BytesIO(b))
        member = f"{self.split}.csv"
        if member not in zf.namelist():
            raise ValueError(f"Missing {member} in Annotations.zip. Members={zf.namelist()}")
        with zf.open(member) as f:
            df = pd.read_csv(f)

        self._df = df
        if self.verbose:
            print(f"✓ Loaded CrowdGleason split={self.split}: rows={len(df):,}, cols={len(df.columns)}")

    def prepare_data(
        self,
        max_instances: Optional[int] = None,
        seed: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
        if self._df is None:
            raise ValueError("Call load_dataset() before prepare_data().")

        rng = np.random.default_rng(int(seed))
        df = self._df.copy()

        # Select the split's reference-label column.
        if self.split == "test":
            gt_col = "ground truth"
        else:
            # Train/val use the provided majority-vote (`MV`) reference labels.
            gt_col = "MV" if "MV" in df.columns else None
            if gt_col is None:
                raise ValueError("No ground-truth column found for split train/val (expected 'MV').")

        marker_cols = [c for c in df.columns if c.lower().startswith("marker")]
        if len(marker_cols) == 0:
            raise ValueError("No marker* columns found in CrowdGleason annotations.")

        task_col = "Patch filename"
        if task_col not in df.columns:
            raise ValueError(f"Missing task column '{task_col}' in CrowdGleason CSV.")

        # Optional subsample
        if max_instances is not None and len(df) > int(max_instances):
            idx = rng.choice(len(df), size=int(max_instances), replace=False)
            df = df.iloc[idx].copy()

        # Build dense labels: markers are already fixed-size (7 workers)
        labels = df[marker_cols].to_numpy(dtype=int)
        truth = df[gt_col].to_numpy(dtype=int)

        # Map any negative gt (if any) to -1 for "unknown"
        truth = np.where(truth >= 0, truth, -1).astype(int)

        # Convert annotation values to integer dtype.
        labels = labels.astype(int)

        # Derive n_classes from observed truth/labels if needed
        observed = np.unique(np.concatenate([labels[labels >= 0].ravel(), truth[truth >= 0].ravel()]))
        n_classes = int(observed.max() + 1) if observed.size > 0 else self.n_classes

        meta: Dict[str, Any] = {
            "dataset": "crowdgleason",
            "split": self.split,
            "n_instances": int(labels.shape[0]),
            "n_workers": int(labels.shape[1]),
            "n_classes": int(n_classes),
            "task_col": task_col,
            "marker_cols": marker_cols,
            "ground_truth_col": gt_col,
            "seed": int(seed),
            "source": {"annotations_zip_url": self.source.annotations_zip_url},
        }
        return labels, truth, int(n_classes), meta


def quick_load_crowdgleason(
    split: str = "test",
    cache_dir: Optional[str] = None,
    verbose: bool = True,
    max_instances: Optional[int] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
    loader = CrowdGleasonLoader(split=split, cache_dir=cache_dir, verbose=verbose)
    loader.load_dataset()
    return loader.prepare_data(max_instances=max_instances, seed=seed)
