"""
BRAVE block-local evidence construction and worker-profile refinement.

BRAVE = Block-wise Structural Regularization via Controlled Evidence Feedback

Worker-specific mixture weights are maintained within each worker block.
The block computes additive evidence and statistical contributions for the
global posterior and shared parameter updates.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .global_update import BRAVEUpdate


class BRAVEBlock:
    """
    A BRAVE block holds a subset of workers' labels over the (shared) set of instances.

    labels: (I, J_block), with values in {0..C-1} and -1 indicating missing.
    """

    def __init__(
        self,
        block_id: str,
        labels: np.ndarray,
        n_classes: int,
        n_components: int = 3,
        verbose: bool = True,
        dp_epsilon: Optional[float] = None,
    ):
        self.block_id = block_id
        self.n_classes = n_classes
        self.n_components = n_components
        self.verbose = verbose
        self.dp_epsilon = dp_epsilon

        self.n_instances, self.n_workers = labels.shape
        self.labels = labels.copy()

        # Precompute
        self._valid_mask = self.labels != -1  # (I,J)
        self._n_annotations = self._valid_mask.sum(axis=1).astype(float)  # (I,)

        self._labels_safe = self.labels.copy()
        self._labels_safe[~self._valid_mask] = 0

        # One-hot of observed labels
        self._labels_onehot = np.zeros((self.n_instances, self.n_workers, self.n_classes), dtype=float)
        valid_i, valid_j = np.where(self._valid_mask)
        self._edge_i = valid_i
        self._edge_j = valid_j
        self._edge_labels = self.labels[valid_i, valid_j] if valid_i.size > 0 else np.array([], dtype=int)
        if valid_i.size > 0:
            self._labels_onehot[valid_i, valid_j, self.labels[valid_i, valid_j]] = 1.0

        # Local state (kept inside block)
        self.worker_confusion_matrices: Optional[np.ndarray] = None  # (J,C,C)
        self.worker_component_probs: Optional[np.ndarray] = None  # (J,K)

        if self.verbose:
            total_annotations = int(self._valid_mask.sum())
            print(
                f"BRAVE block '{block_id}': workers={self.n_workers}, instances={self.n_instances}, "
                f"annotations={total_annotations}"
            )

    def initialize_local_params(self) -> None:
        """Initialize local worker confusion matrices and worker-component responsibilities."""
        C, J, K = self.n_classes, self.n_workers, self.n_components
        I = self.n_instances

        # Majority vote per instance (within block)
        initial_labels = np.zeros(I, dtype=int)
        for i in range(I):
            valid_labels = self.labels[i][self._valid_mask[i]]
            if len(valid_labels) > 0:
                counts = np.bincount(valid_labels, minlength=C)
                initial_labels[i] = int(np.argmax(counts))
            else:
                initial_labels[i] = 0

        # Initialize confusion matrices based on MV labels
        self.worker_confusion_matrices = np.zeros((J, C, C), dtype=float)
        for j in range(J):
            for c in range(C):
                mask = (initial_labels == c) & self._valid_mask[:, j]
                if mask.sum() > 0:
                    worker_labels = self.labels[mask, j]
                    counts = np.bincount(worker_labels, minlength=C)
                    self.worker_confusion_matrices[j, c, :] = counts / max(1, counts.sum())
                else:
                    self.worker_confusion_matrices[j, c, :] = 1.0 / C

        # Light smoothing
        self.worker_confusion_matrices = (self.worker_confusion_matrices + 0.01) / (1 + 0.01 * C)

        # Random init for worker-to-component probabilities (softmax(randn * 0.1))
        random_logits = np.random.randn(J, K) * 0.1
        exp_logits = np.exp(random_logits - random_logits.max(axis=1, keepdims=True))
        self.worker_component_probs = exp_logits / (exp_logits.sum(axis=1, keepdims=True) + 1e-10)

    def refine_worker_profiles(
        self,
        global_instance_probs: np.ndarray,
        global_component_reliability: np.ndarray,
    ) -> None:
        """
        Refine worker profiles using the synchronized global posterior:
          - update worker confusion matrices (for diagnostics / quality proxy)
          - update worker-specific mixture weights (kept local)
        """
        eps = 1e-10
        I, J, C, K = self.n_instances, self.n_workers, self.n_classes, self.n_components

        # Update local confusion matrices using current global q(y)
        expected_counts = np.einsum("ic,ijp->jcp", global_instance_probs, self._labels_onehot)  # (J,C,C')
        expected_counts = expected_counts + eps
        row_sums = expected_counts.sum(axis=2, keepdims=True)
        self.worker_confusion_matrices = expected_counts / row_sums

        # Update worker->component responsibilities:
        # log p(z_j=k | ...) ∝ Σ_i valid_ij * Σ_c q(y_i=c) * log R_k[c, l_ij]
        worker_log_probs = np.zeros((J, K), dtype=float)
        for obs_label in range(C):
            edge_mask = self._edge_labels == obs_label
            if not np.any(edge_mask):
                continue
            edge_i = self._edge_i[edge_mask]
            edge_j = self._edge_j[edge_mask]
            log_rel = np.log(global_component_reliability[:, :, obs_label] + eps)  # (K,C)
            item_component_scores = global_instance_probs @ log_rel.T  # (I,K)
            np.add.at(worker_log_probs, edge_j, item_component_scores[edge_i])

        max_log = worker_log_probs.max(axis=1, keepdims=True)
        exp_probs = np.exp(worker_log_probs - max_log)
        self.worker_component_probs = exp_probs / (exp_probs.sum(axis=1, keepdims=True) + eps)

    def compute_block_contributions(
        self,
        global_class_priors: np.ndarray,
        global_component_weights: np.ndarray,
        global_component_reliability: np.ndarray,
        include_diagnostics: bool = False,
    ) -> BRAVEUpdate:
        """
        Construct block-local evidence and shared-statistic contributions.

        instance_log_probs:
          log p(y_i=c | block labels) ∝ log π_c + Σ_j valid_ij * log( Σ_k w_jk * R_k[c, l_ij] )
        """
        eps = 1e-10
        I, J, C, K = self.n_instances, self.n_workers, self.n_classes, self.n_components

        if self.worker_component_probs is None:
            raise ValueError("Block not initialized. Call initialize_local_params() first.")

        instance_log_probs = np.zeros((I, C), dtype=float)
        for obs_label in range(C):
            edge_mask = self._edge_labels == obs_label
            if not np.any(edge_mask):
                continue
            edge_i = self._edge_i[edge_mask]
            edge_j = self._edge_j[edge_mask]
            weighted_reliability = self.worker_component_probs @ global_component_reliability[:, :, obs_label]
            np.add.at(instance_log_probs, edge_i, np.log(weighted_reliability[edge_j] + eps))

        instance_log_probs += np.log(global_class_priors + eps)

        # Optional DP noise (kept identical to existing repo behavior)
        if self.dp_epsilon is not None:
            sensitivity = 2 * np.log(1 / eps)
            noise_scale = sensitivity / self.dp_epsilon
            noise = np.random.laplace(0, noise_scale, instance_log_probs.shape)
            instance_log_probs += noise

        # Local posterior q(y) derived from instance_log_probs (local softmax)
        max_log = instance_log_probs.max(axis=1, keepdims=True)
        exp_probs = np.exp(instance_log_probs - max_log)
        local_instance_probs = exp_probs / (exp_probs.sum(axis=1, keepdims=True) + eps)  # (I,C)

        # Expected counts for component reliability (do NOT normalize here)
        reliability_contrib = np.zeros((K, C, C), dtype=float)
        for obs_label in range(C):
            edge_mask = self._edge_labels == obs_label
            if not np.any(edge_mask):
                continue
            edge_i = self._edge_i[edge_mask]
            edge_j = self._edge_j[edge_mask]
            reliability_contrib[:, :, obs_label] = (
                self.worker_component_probs[edge_j].T @ local_instance_probs[edge_i]
            )

        # Component weight contribution: expected counts (do NOT normalize)
        weight_contrib = self.worker_component_probs.sum(axis=0)  # (K,)

        # Preserve the legacy normalized prior contribution while also exposing raw counts.
        local_class_counts = self._labels_onehot.sum(axis=(0, 1))  # (C,)
        local_class_priors = local_class_counts / (local_class_counts.sum() + eps)

        return BRAVEUpdate(
            block_id=self.block_id,
            instance_log_probs=instance_log_probs,
            class_prior_contrib=local_class_priors,
            class_prior_count_contrib=local_class_counts,
            n_annotations=self._n_annotations,
            component_reliability_contrib=reliability_contrib,
            component_weight_contrib=weight_contrib,
            labels=self.labels.copy() if include_diagnostics else None,
            valid_mask=self._valid_mask.copy() if include_diagnostics else None,
            labels_onehot=self._labels_onehot.copy() if include_diagnostics else None,
            worker_component_probs=self.worker_component_probs.copy() if include_diagnostics else None,
            local_instance_probs=local_instance_probs.copy() if include_diagnostics else None,
        )

    def get_worker_quality(self) -> np.ndarray:
        """
        Simple worker-quality proxy from local confusion matrices.
        """
        if self.worker_confusion_matrices is None:
            return np.ones(self.n_workers) * 0.5
        return np.array([np.diag(self.worker_confusion_matrices[j]).mean() for j in range(self.n_workers)])

