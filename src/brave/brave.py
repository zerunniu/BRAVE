"""
BRAVE: Block-wise Structural Regularization via Controlled Evidence Feedback

Reference implementation for the BRAVE blockwise inference procedure.

API design goals:
  - Expose a clean `BRAVE` class name for paper/code.
  - Use worker-side blocks as the unit of local computation.
  - Keep the public API small for experiments and direct reuse.
"""

from __future__ import annotations

from pathlib import Path
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
      tol: convergence threshold (mean absolute change in instance probabilities)
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
        reliability_update_mode: str = "local",
        collect_diagnostics: bool = False,
        diagnostics_output_dir: Optional[str] = None,
        disable_early_stopping: bool = False,
        verbose: bool = True,
    ):
        if reliability_update_mode not in {"local", "global"}:
            raise ValueError("reliability_update_mode must be 'local' or 'global'.")

        self.n_classes = n_classes
        self.n_components = n_components
        self.max_rounds = max_rounds
        self.local_epochs = local_epochs
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tol = tol
        self.reliability_update_mode = reliability_update_mode
        self.collect_diagnostics = bool(collect_diagnostics)
        self.diagnostics_output_dir = diagnostics_output_dir
        self.disable_early_stopping = bool(disable_early_stopping)
        self.verbose = verbose

        self.global_updater: Optional[BRAVEGlobalUpdater] = None
        self.blocks: List[BRAVEBlock] = []

        self.is_fitted = False
        self.history: List[Dict] = []
        self.diagnostic_trace: List[Dict] = []

    # -------------------- block management --------------------
    def add_block(
        self,
        block_id: str,
        labels: np.ndarray,
        dp_epsilon: Optional[float] = None,
    ) -> BRAVEBlock:
        block = BRAVEBlock(
            block_id=block_id,
            labels=labels,
            n_classes=self.n_classes,
            n_components=self.n_components,
            verbose=self.verbose,
            dp_epsilon=dp_epsilon,
        )
        block.initialize_local_params()
        self.blocks.append(block)
        return block

    # -------------------- training --------------------
    def fit(
        self,
        n_instances: Optional[int] = None,
        collect_diagnostics: Optional[bool] = None,
        diagnostics_output_dir: Optional[str] = None,
        disable_early_stopping: Optional[bool] = None,
    ) -> "BRAVE":
        if len(self.blocks) == 0:
            raise ValueError("At least one block is required.")

        do_diagnostics = self.collect_diagnostics if collect_diagnostics is None else bool(collect_diagnostics)
        output_dir = diagnostics_output_dir if diagnostics_output_dir is not None else self.diagnostics_output_dir
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
            reliability_update_mode=self.reliability_update_mode,
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
            print(f"  Reliability update: {self.reliability_update_mode}")
            print(f"  Max rounds: {self.max_rounds}")
            print(f"  Disable early stopping: {no_early_stop}")
            print("=" * 60)

        self.diagnostic_trace = []
        trace_dir = Path(output_dir) if output_dir else None
        if do_diagnostics and trace_dir is not None:
            trace_dir.mkdir(parents=True, exist_ok=True)

        for round_idx in range(self.max_rounds):
            if self.verbose:
                print(f"\n--- Round {round_idx + 1}/{self.max_rounds} ---")

            global_params = self.global_updater.get_global_params()
            q_before = global_params["instance_probs"].copy()
            reliability_before = global_params["component_reliability"].copy()
            priors_before = global_params["class_priors"].copy()

            updates: List[BRAVEUpdate] = []
            for block in self.blocks:
                # Worker-profile refinement (skip at round 1, as before).
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
                    include_diagnostics=do_diagnostics or self.reliability_update_mode == "global",
                )
                updates.append(upd)

                if self.verbose:
                    print(f"  Block '{block.block_id}': done")

            self.global_updater.aggregate_updates(updates)

            current_probs = self.global_updater.global_instance_probs
            reliability_after = self.global_updater.global_component_reliability.copy()
            priors_after = self.global_updater.global_class_priors.copy()

            # Theory-facing posterior shift: item-wise L1 averaged over items.
            delta_q = float(np.abs(current_probs - q_before).sum(axis=1).mean())
            # Keep the legacy element-wise mean under `diff` for compatibility.
            diff = float(np.abs(current_probs - q_before).mean())
            delta_R = float(np.abs(reliability_after - reliability_before).mean())
            delta_pi = float(np.abs(priors_after - priors_before).sum())
            self.history.append(
                {
                    "round": round_idx + 1,
                    "diff": diff,
                    "delta_q": delta_q,
                    "delta_R": delta_R,
                    "delta_pi": delta_pi,
                }
            )

            if do_diagnostics:
                trace_t = self._build_diagnostic_trace(
                    round_idx=round_idx + 1,
                    q_before=q_before,
                    q_after=current_probs.copy(),
                    reliability_before=reliability_before,
                    reliability_after=reliability_after,
                    priors_before=priors_before,
                    priors_after=priors_after,
                    updates=updates,
                    delta_q=delta_q,
                    delta_R=delta_R,
                    delta_pi=delta_pi,
                )
                self.diagnostic_trace.append(trace_t)
                if trace_dir is not None:
                    self._save_diagnostic_trace(trace_t, trace_dir / f"round_{round_idx + 1:04d}.npz")

            if self.verbose:
                print(f"  Posterior shift Delta_q: {delta_q:.6f}")

            if (not no_early_stop) and delta_q < self.tol:
                if self.verbose:
                    print(f"\n✓ Converged at round {round_idx + 1}")
                break

        self.is_fitted = True

        if self.verbose:
            print("\n" + "=" * 60)
            print("BRAVE Training Complete")
            print("=" * 60)

        return self

    def _build_diagnostic_trace(
        self,
        *,
        round_idx: int,
        q_before: np.ndarray,
        q_after: np.ndarray,
        reliability_before: np.ndarray,
        reliability_after: np.ndarray,
        priors_before: np.ndarray,
        priors_after: np.ndarray,
        updates: List[BRAVEUpdate],
        delta_q: float,
        delta_R: float,
        delta_pi: float,
    ) -> Dict:
        blocks = []
        for upd in updates:
            blocks.append(
                {
                    "block_id": upd.block_id,
                    "instance_log_probs": upd.instance_log_probs.copy(),
                    "worker_component_probs": None
                    if upd.worker_component_probs is None
                    else upd.worker_component_probs.copy(),
                    "labels": None if upd.labels is None else upd.labels.copy(),
                    "valid_mask": None if upd.valid_mask is None else upd.valid_mask.copy(),
                    "local_instance_probs": None
                    if upd.local_instance_probs is None
                    else upd.local_instance_probs.copy(),
                }
            )
        return {
            "round": int(round_idx),
            "reliability_update_mode": self.reliability_update_mode,
            "q_before": q_before,
            "q_after": q_after,
            "component_reliability_before": reliability_before,
            "component_reliability_after": reliability_after,
            "class_priors_before": priors_before,
            "class_priors_after": priors_after,
            "delta_q": float(delta_q),
            "delta_R": float(delta_R),
            "delta_pi": float(delta_pi),
            "blocks": blocks,
        }

    @staticmethod
    def _save_diagnostic_trace(trace_t: Dict, path: Path) -> None:
        payload = {
            "round": np.array(trace_t["round"], dtype=int),
            "reliability_update_mode": np.array(trace_t["reliability_update_mode"]),
            "q_before": trace_t["q_before"],
            "q_after": trace_t["q_after"],
            "component_reliability_before": trace_t["component_reliability_before"],
            "component_reliability_after": trace_t["component_reliability_after"],
            "class_priors_before": trace_t["class_priors_before"],
            "class_priors_after": trace_t["class_priors_after"],
            "delta_q": np.array(trace_t["delta_q"], dtype=float),
            "delta_R": np.array(trace_t["delta_R"], dtype=float),
            "delta_pi": np.array(trace_t["delta_pi"], dtype=float),
        }
        for idx, block in enumerate(trace_t["blocks"]):
            prefix = f"block_{idx}"
            payload[f"{prefix}_id"] = np.array(block["block_id"])
            payload[f"{prefix}_instance_log_probs"] = block["instance_log_probs"]
            if block["worker_component_probs"] is not None:
                payload[f"{prefix}_worker_component_probs"] = block["worker_component_probs"]
            if block["labels"] is not None:
                payload[f"{prefix}_labels"] = block["labels"]
            if block["valid_mask"] is not None:
                payload[f"{prefix}_valid_mask"] = block["valid_mask"]
            if block["local_instance_probs"] is not None:
                payload[f"{prefix}_local_instance_probs"] = block["local_instance_probs"]
        np.savez_compressed(path, **payload)

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
        """Compatibility property for the evaluator."""
        if self.global_updater is None:
            return None
        return self.global_updater.global_instance_probs

    # -------------------- diagnostics --------------------
    def get_block_worker_quality(self, block_id: str) -> np.ndarray:
        for b in self.blocks:
            if b.block_id == block_id:
                return b.get_worker_quality()
        raise ValueError(f"Block not found: {block_id}")

    def get_all_worker_quality(self) -> Dict[str, np.ndarray]:
        return {b.block_id: b.get_worker_quality() for b in self.blocks}

    def get_component_info(self) -> Dict:
        if not self.is_fitted or self.global_updater is None:
            raise ValueError("Model not trained.")

        component_quality = []
        for k in range(self.n_components):
            diag_mean = np.diag(self.global_updater.global_component_reliability[k]).mean()
            component_quality.append(diag_mean)

        return {
            "weights": self.global_updater.global_component_weights,
            "quality": np.array(component_quality),
            "reliability": self.global_updater.global_component_reliability,
        }

