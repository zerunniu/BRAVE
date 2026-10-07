"""
BRAVE: Block-wise Structural Regularization via Controlled Evidence Feedback

Blockwise label aggregation with worker-side blocks and synchronized posterior
and reliability updates.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .block import BRAVEBlock
from .global_update import BRAVEGlobalUpdater, BRAVEUpdate


class BRAVE:
    """
    BRAVE = Block-wise Structural Regularization via Controlled Evidence Feedback

    Parameters:
      n_classes: number of label classes
      n_components: mixture components used for reliability modeling
      max_rounds: maximum block-wise inference rounds
      local_epochs: number of worker-profile refinement steps per round (after round 1)
      alpha: optional class-prior smoothing strength
      beta: optional reliability smoothing strength
      tol: stopping threshold for the mean item-wise L1 posterior shift
    """

    def __init__(
        self,
        n_classes: int,
        n_components: int = 3,
        max_rounds: int = 20,
        local_epochs: int = 2,
        alpha: float = 0.0,
        beta: float = 0.0,
        tol: float = 1e-4,
        disable_early_stopping: bool = False,
        verbose: bool = True,
    ):
        self.n_classes = n_classes
        self.n_components = n_components
        self.max_rounds = max_rounds
        self.local_epochs = local_epochs
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tol = tol
        self.disable_early_stopping = bool(disable_early_stopping)
        self.verbose = verbose

        self.global_updater: Optional[BRAVEGlobalUpdater] = None
        self.blocks: List[BRAVEBlock] = []

        self.is_fitted = False
        self.history: List[Dict] = []
        self.n_rounds_: int = 0
        self.stop_reason_: Optional[str] = None

    # -------------------- block management --------------------
    def add_block(
        self,
        block_id: str,
        labels: np.ndarray,
    ) -> BRAVEBlock:
        block = BRAVEBlock(
            block_id=block_id,
            labels=labels,
            n_classes=self.n_classes,
            n_components=self.n_components,
            verbose=self.verbose,
        )
        block.initialize_local_params()
        self.blocks.append(block)
        return block

    # -------------------- training --------------------
    def fit(
        self,
        n_instances: Optional[int] = None,
        disable_early_stopping: Optional[bool] = None,
    ) -> "BRAVE":
        if len(self.blocks) == 0:
            raise ValueError("At least one block is required.")

        no_early_stop = (
            self.disable_early_stopping
            if disable_early_stopping is None
            else bool(disable_early_stopping)
        )

        if n_instances is None:
            n_instances = self.blocks[0].n_instances

        self.global_updater = BRAVEGlobalUpdater(
            n_instances=n_instances,
            n_classes=self.n_classes,
            n_components=self.n_components,
            alpha=self.alpha,
            beta=self.beta,
            verbose=self.verbose,
        )

        for b in self.blocks:
            self.global_updater.register_block(b.block_id)

        if self.verbose:
            print("\n" + "=" * 60)
            print("BRAVE Training (Block-wise Structural Regularization via Controlled Evidence Feedback)")
            print(f"  Blocks:     {len(self.blocks)}")
            print(f"  Instances:  {n_instances}")
            print(f"  Classes:    {self.n_classes}")
            print(f"  Components: {self.n_components}")
            print(f"  Alpha:      {self.alpha}")
            print(f"  Beta:       {self.beta}")
            print(f"  Max rounds: {self.max_rounds}")
            print(f"  Disable early stopping: {no_early_stop}")
            print("=" * 60)

        self.n_rounds_ = 0
        self.stop_reason_ = "max_rounds"

        for round_idx in range(self.max_rounds):
            if self.verbose:
                print(f"\n--- Round {round_idx + 1}/{self.max_rounds} ---")

            global_params = self.global_updater.get_global_params()
            q_before = global_params["instance_probs"].copy()

            updates: List[BRAVEUpdate] = []
            for block in self.blocks:
                # Refine worker profiles from the previous round's posterior and reliability.
                if round_idx > 0:
                    for _ in range(self.local_epochs):
                        block.refine_worker_profiles(
                            global_instance_probs=global_params["instance_probs"],
                            global_component_reliability=global_params["component_reliability"],
                        )

                upd = block.compute_block_contributions(
                    global_class_priors=global_params["class_priors"],
                    global_component_weights=global_params["component_weights"],
                    global_component_reliability=global_params["component_reliability"],
                )
                updates.append(upd)

                if self.verbose:
                    print(f"  Block '{block.block_id}': done")

            self.global_updater.aggregate_updates(updates)

            current_probs = self.global_updater.global_instance_probs
            # Item-wise L1 posterior shift, averaged over items.
            delta_q = float(np.abs(current_probs - q_before).sum(axis=1).mean())
            self.n_rounds_ = round_idx + 1
            self.history.append(
                {
                    "round": round_idx + 1,
                    "delta_q": delta_q,
                }
            )

            if self.verbose:
                print(f"  Posterior shift Delta_q: {delta_q:.6f}")

            if (not no_early_stop) and delta_q < self.tol:
                self.stop_reason_ = "tolerance"
                if self.verbose:
                    print(f"\nStopped at round {round_idx + 1}: tolerance reached")
                break

        self.is_fitted = True

        if self.verbose:
            print("\n" + "=" * 60)
            print("BRAVE Training Complete")
            print("=" * 60)

        return self

    # -------------------- inference --------------------
    def predict(self) -> np.ndarray:
        if not self.is_fitted or self.global_updater is None:
            raise ValueError("Model not trained.")
        return self.global_updater.get_aggregated_labels()

    def predict_proba(self) -> np.ndarray:
        if not self.is_fitted or self.global_updater is None:
            raise ValueError("Model not trained.")
        return self.global_updater.get_label_probabilities()

    @property
    def instance_label_probs(self) -> np.ndarray:
        """Global instance-label posterior probabilities."""
        if self.global_updater is None:
            return None
        return self.global_updater.global_instance_probs
