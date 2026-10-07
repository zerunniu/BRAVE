"""
BRAVE global posterior synchronization and parameter updates.

BRAVE = Block-wise Structural Regularization via Controlled Evidence Feedback

Blocks provide local log-evidence and sufficient statistics. This module
synchronizes the global label posterior and updates shared reliability
components, component weights, and class priors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class BRAVEUpdate:
    """Block-local evidence and statistics used by BRAVE global updates."""

    block_id: str
    instance_log_probs: np.ndarray  # (I, C): log p(labels_block | y) + log p(y) as used by the block
    class_prior_contrib: np.ndarray  # (C,): block-normalized observed-label frequencies
    n_annotations: np.ndarray  # (I,): number of annotations per instance in this block
    class_prior_count_contrib: Optional[np.ndarray] = None  # (C,): raw block-level class counts
    # BRAVE (mixture reliability) specific
    component_reliability_contrib: Optional[np.ndarray] = None  # (K, C, C)
    component_weight_contrib: Optional[np.ndarray] = None  # (K,)


class BRAVEGlobalUpdater:
    """
    BRAVE global posterior and parameter updater.

    Maintains shared parameters and aggregates block contributions.
    """

    def __init__(
        self,
        n_instances: int,
        n_classes: int,
        n_components: int = 3,
        alpha: float = 0.0,
        beta: float = 0.0,
        verbose: bool = True,
    ):
        self.n_instances = n_instances
        self.n_classes = n_classes
        self.n_components = n_components
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.verbose = verbose

        # Global parameters
        self.global_class_priors: np.ndarray = np.ones(n_classes) / n_classes  # (C,)
        self.global_instance_probs: np.ndarray = np.ones((n_instances, n_classes)) / n_classes  # (I, C)

        # Mixture/reliability parameters
        self.global_component_weights: np.ndarray = np.ones(n_components) / n_components  # (K,)
        self.global_component_reliability: np.ndarray = np.zeros((n_components, n_classes, n_classes))  # (K,C,C)
        for k in range(n_components):
            base_acc = 0.5 + 0.4 * (k / max(1, n_components - 1))
            off_diag = (1 - base_acc) / max(1, (n_classes - 1))
            self.global_component_reliability[k] = off_diag
            np.fill_diagonal(self.global_component_reliability[k], base_acc)

        self.block_ids: List[str] = []

    def register_block(self, block_id: str) -> None:
        if block_id not in self.block_ids:
            self.block_ids.append(block_id)
            if self.verbose:
                print(f"BRAVE Global Update: registered block '{block_id}'")

    def get_global_params(self) -> Dict[str, np.ndarray]:
        """Shared parameters supplied to blocks."""
        return {
            "class_priors": self.global_class_priors.copy(),
            "instance_probs": self.global_instance_probs.copy(),
            "component_weights": self.global_component_weights.copy(),
            "component_reliability": self.global_component_reliability.copy(),
        }

    def aggregate_updates(self, updates: List[BRAVEUpdate]) -> Dict[str, np.ndarray]:
        """
        Aggregate additive contributions across blocks.

        Key identity:
          log p(y | all blocks) ∝ log p(y) + Σ_b log p(labels_b | y)

        Each block's `instance_log_probs` includes `log p(y)`. Subtracting
        (B-1) * log p(y) after summation leaves one copy of the class prior.
        """
        if len(updates) == 0:
            return self.get_global_params()

        eps = 1e-10
        n_blocks = len(updates)

        all_log_probs = np.stack([u.instance_log_probs for u in updates], axis=0)  # (B,I,C)
        summed_log_probs = all_log_probs.sum(axis=0)  # (I,C)

        log_prior = np.log(self.global_class_priors + eps)  # (C,)
        summed_log_probs = summed_log_probs - (n_blocks - 1) * log_prior

        # Normalize to probabilities
        max_log = summed_log_probs.max(axis=1, keepdims=True)
        exp_probs = np.exp(summed_log_probs - max_log)
        self.global_instance_probs = exp_probs / (exp_probs.sum(axis=1, keepdims=True) + eps)

        # With positive alpha, pool raw counts and add class-prior smoothing.
        # With zero alpha, average block-normalized label frequencies.
        all_class_priors = np.stack([u.class_prior_contrib for u in updates], axis=0)  # (B,C)
        if self.alpha > 0.0 and updates[0].class_prior_count_contrib is not None:
            all_class_counts = np.stack([u.class_prior_count_contrib for u in updates], axis=0)  # (B,C)
            summed_counts = all_class_counts.sum(axis=0) + self.alpha
            self.global_class_priors = summed_counts
        else:
            self.global_class_priors = all_class_priors.mean(axis=0)
        self.global_class_priors = self.global_class_priors / (self.global_class_priors.sum() + eps)

        # Update reliability/mixture parameters if provided.
        # Reliability statistics are computed from block-local posteriors.
        if updates[0].component_reliability_contrib is not None:
            all_rel = np.stack([u.component_reliability_contrib for u in updates], axis=0)  # (B,K,C,C)
            summed_rel = all_rel.sum(axis=0)  # (K,C,C)
            if self.beta > 0.0:
                summed_rel = summed_rel + self.beta
            row_sums = summed_rel.sum(axis=2, keepdims=True)
            self.global_component_reliability = summed_rel / (row_sums + eps)

            all_w = np.stack([u.component_weight_contrib for u in updates], axis=0)  # (B,K)
            summed_w = all_w.sum(axis=0)  # (K,)
            self.global_component_weights = summed_w / (summed_w.sum() + eps)

        if self.verbose:
            print(f"BRAVE Global Update: aggregated {n_blocks} blocks")

        return self.get_global_params()

    def get_aggregated_labels(self) -> np.ndarray:
        return self.global_instance_probs.argmax(axis=1)

    def get_label_probabilities(self) -> np.ndarray:
        return self.global_instance_probs.copy()
