"""
Recognizing Textual Entailment (RTE) as a synthetic crowdsourcing dataset.

The loader samples synthetic worker annotations from SuperGLUE RTE gold labels
using configurable noise and sparsity parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from datasets import load_dataset as hf_load_dataset

from .base_loader import BaseDataLoader


@dataclass(frozen=True)
class SyntheticRTEConfig:
    n_workers: int = 50
    avg_labels_per_instance: int = 5
    # Worker accuracy distribution (Beta): mean = a/(a+b)
    worker_beta_a: float = 8.0
    worker_beta_b: float = 4.0
    # Optional item difficulty (GLAD-like): higher => easier (inverse difficulty)
    use_item_difficulty: bool = True
    item_beta_a: float = 6.0
    item_beta_b: float = 6.0
    seed: int = 0


class SyntheticRTELoader(BaseDataLoader):
    """
    Synthetic crowd-annotated RTE loader.

    Output:
      labels: (I, J) with -1 for missing, labels in {0,1}
      truth:  (I,) in {0,1}
      n_classes: 2
    """

    def __init__(self, cache_dir: Optional[str] = None, verbose: bool = True):
        super().__init__()
        self.cache_dir = cache_dir
        self.verbose = verbose
        self.n_classes = 2
        self.dataset = None

    def load_dataset(self, subset: str = "super_glue", split: str = "train") -> None:
        """
        Load RTE gold dataset.

        Default uses HuggingFace `super_glue` config `rte` split `train`.
        """
        if subset != "super_glue":
            raise ValueError("Only subset='super_glue' is supported currently.")
        if self.verbose:
            print(f"Loading RTE from HuggingFace: dataset={subset}, config=rte, split={split}")
        self.dataset = hf_load_dataset(subset, "rte", split=split, cache_dir=self.cache_dir)

    def prepare_data(
        self,
        max_instances: Optional[int] = 5000,
        config: Optional[SyntheticRTEConfig] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
        if self.dataset is None:
            raise ValueError("Call load_dataset() first.")

        cfg = config or SyntheticRTEConfig()
        rng = np.random.default_rng(cfg.seed)

        # Extract gold labels; filter out unlabeled (-1) if any.
        labels_gold = []
        for ex in self.dataset:
            y = int(ex["label"])
            if y != -1:
                labels_gold.append(y)

        truth = np.asarray(labels_gold, dtype=int)
        if max_instances is not None:
            truth = truth[: int(max_instances)]

        I = int(truth.shape[0])
        J = int(cfg.n_workers)

        # Sample worker "competence" in (0,1)
        worker_acc = rng.beta(cfg.worker_beta_a, cfg.worker_beta_b, size=J)

        # Optional item "ease" in (0,1)
        if cfg.use_item_difficulty:
            item_ease = rng.beta(cfg.item_beta_a, cfg.item_beta_b, size=I)
        else:
            item_ease = np.ones(I, dtype=float)

        # Build sparse annotation matrix
        labels = np.full((I, J), -1, dtype=int)
        k = int(max(1, cfg.avg_labels_per_instance))

        for i in range(I):
            # choose k distinct workers
            w = rng.choice(J, size=min(k, J), replace=False)
            for j in w:
                # Effective accuracy combines worker competence and item ease
                p_correct = float(worker_acc[j] * item_ease[i] + (1 - item_ease[i]) * 0.5)
                if rng.random() < p_correct:
                    labels[i, j] = truth[i]
                else:
                    labels[i, j] = 1 - truth[i]

        coverage = float((labels != -1).sum() / (I * J))
        avg_annotations = float((labels != -1).sum(axis=1).mean())

        metadata: Dict[str, Any] = {
            "dataset": "RTE(super_glue) + synthetic crowd",
            "n_instances": I,
            "n_workers": J,
            "n_classes": self.n_classes,
            "coverage": coverage,
            "avg_annotations_per_instance": avg_annotations,
            "seed": cfg.seed,
            "worker_acc": worker_acc,
            "item_ease": item_ease,
            "config": cfg.__dict__,
        }

        if self.verbose:
            print(f"Prepared synthetic crowd labels: labels shape={labels.shape}, coverage={coverage:.2%}")

        return labels, truth, self.n_classes, metadata


def quick_load_synthetic_rte(
    split: str = "train",
    max_instances: Optional[int] = 5000,
    config: Optional[SyntheticRTEConfig] = None,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
    loader = SyntheticRTELoader(cache_dir=cache_dir, verbose=verbose)
    loader.load_dataset(split=split)
    return loader.prepare_data(max_instances=max_instances, config=config)
