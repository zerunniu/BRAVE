"""Diagnostics for theory-aligned BRAVE experiments.

These utilities intentionally avoid ELBO diagnostics. They measure fixed-point
posterior shifts and the one-round feedback gap from the BRAVE analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np


WARNING_TEXT = (
    "These diagnostics verify fixed-point behavior and theorem-aligned one-round "
    "feedback control. They do not establish monotone ELBO ascent or global convergence."
)


def compute_delta_q(q_after: np.ndarray, q_before: np.ndarray) -> float:
    """Mean item-wise L1 posterior shift."""
    return float(np.abs(np.asarray(q_after) - np.asarray(q_before)).sum(axis=1).mean())


def compute_delta_R(R_after: np.ndarray, R_before: np.ndarray) -> float:
    """Mean absolute component-reliability shift."""
    return float(np.abs(np.asarray(R_after) - np.asarray(R_before)).mean())


def compute_delta_pi(pi_after: np.ndarray, pi_before: np.ndarray) -> float:
    """L1 class-prior shift."""
    return float(np.abs(np.asarray(pi_after) - np.asarray(pi_before)).sum())


def t_epsilon(history: Iterable[Mapping], eps: float) -> Optional[int]:
    """First round whose theory-facing Delta_q is below eps."""
    for row in history:
        value = row.get("delta_q", row.get("Delta_q"))
        if value is not None and float(value) < float(eps):
            return int(row["round"])
    return None


def t_epsilon_display(history: Iterable[Mapping], eps: float, max_rounds: int) -> int | str:
    """First threshold-crossing round, or a clear censoring label."""
    value = t_epsilon(history, eps)
    return value if value is not None else f"> {int(max_rounds)}"


def wrong_confidence(probabilities: np.ndarray, truth: np.ndarray) -> float:
    """Mean confidence on wrong hard predictions."""
    probabilities = np.asarray(probabilities)
    truth = np.asarray(truth)
    valid = truth != -1
    if probabilities.ndim != 2 or valid.sum() == 0:
        return float("nan")
    probs = probabilities[valid]
    y = truth[valid].astype(int)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    wrong = pred != y
    if not np.any(wrong):
        return 0.0
    return float(conf[wrong].mean())


def load_npz_trace(trace_dir: str | Path) -> List[Dict]:
    """Load round traces saved by BRAVE.fit(..., collect_diagnostics=True)."""
    out: List[Dict] = []
    for path in sorted(Path(trace_dir).glob("round_*.npz")):
        data = np.load(path, allow_pickle=True)
        blocks = []
        block_indices = sorted(
            {
                int(k.split("_")[1])
                for k in data.files
                if k.startswith("block_") and k.endswith("_id")
            }
        )
        for idx in block_indices:
            prefix = f"block_{idx}"
            blocks.append(
                {
                    "block_id": str(data[f"{prefix}_id"].item()),
                    "instance_log_probs": data[f"{prefix}_instance_log_probs"],
                    "worker_component_probs": _npz_get(data, f"{prefix}_worker_component_probs"),
                    "labels": _npz_get(data, f"{prefix}_labels"),
                    "valid_mask": _npz_get(data, f"{prefix}_valid_mask"),
                    "local_instance_probs": _npz_get(data, f"{prefix}_local_instance_probs"),
                }
            )
        out.append(
            {
                "round": int(data["round"]),
                "reliability_update_mode": str(data["reliability_update_mode"].item()),
                "q_before": data["q_before"],
                "q_after": data["q_after"],
                "component_reliability_before": data["component_reliability_before"],
                "component_reliability_after": data["component_reliability_after"],
                "class_priors_before": data["class_priors_before"],
                "class_priors_after": data["class_priors_after"],
                "delta_q": float(data["delta_q"]),
                "delta_R": float(data["delta_R"]),
                "delta_pi": float(data["delta_pi"]),
                "blocks": blocks,
            }
        )
    return out


def compute_feedback_gap(
    trace_t: Mapping,
    labels_true: np.ndarray,
    eps: float = 1e-12,
    tol: float = 1e-10,
) -> Dict[str, float]:
    """Compute theorem-aligned feedback-gap diagnostics for one round trace.

    Only item-rival pairs satisfying the aligned-error condition are used for
    theorem validation. Mixed-evidence pairs are counted separately.
    """
    labels_true = np.asarray(labels_true)
    pi = np.asarray(trace_t["class_priors_before"], dtype=float)
    C = int(pi.shape[0])
    blocks = list(trace_t.get("blocks", []))

    num_pairs_total = 0
    num_pairs_eligible = 0
    num_mixed_pairs = 0
    G_sum = 0.0
    U_sum = 0.0
    lower_violations = 0
    upper_violations = 0
    term_count = 0

    for i, c_star_raw in enumerate(labels_true):
        if c_star_raw == -1:
            continue
        c_star = int(c_star_raw)
        if c_star < 0 or c_star >= C:
            continue

        touching = []
        for block in blocks:
            valid_mask = block.get("valid_mask")
            if valid_mask is None:
                labels = block.get("labels")
                valid_mask = None if labels is None else labels != -1
            if valid_mask is None or i >= valid_mask.shape[0] or not np.any(valid_mask[i]):
                continue
            touching.append(block)
        if len(touching) < 2:
            continue

        for a in range(C):
            if a == c_star:
                continue
            num_pairs_total += 1
            etas = []
            for block in touching:
                S = np.asarray(block["instance_log_probs"], dtype=float)
                eta = (S[i, a] - np.log(pi[a] + eps)) - (S[i, c_star] - np.log(pi[c_star] + eps))
                etas.append(float(eta))
            etas_arr = np.asarray(etas, dtype=float)

            if not np.all(etas_arr >= -tol):
                num_mixed_pairs += 1
                continue
            if np.count_nonzero(etas_arr > tol) < 2:
                continue

            num_pairs_eligible += 1
            lam = float(np.log((pi[a] + eps) / (pi[c_star] + eps)))
            qbar_global = _sigmoid(lam + float(etas_arr.sum()))

            for block_idx, block in enumerate(touching):
                labels = block.get("labels")
                worker_component_probs = block.get("worker_component_probs")
                if labels is None or worker_component_probs is None:
                    continue
                labels_i = np.asarray(labels)[i]
                w = np.asarray(worker_component_probs, dtype=float)
                qbar_local = _sigmoid(lam + float(etas_arr[block_idx]))
                posterior_gap = qbar_global - qbar_local
                upper_gap = 0.25 * float(etas_arr.sum() - etas_arr[block_idx])

                for k in range(w.shape[1]):
                    for ell in range(C):
                        mask = labels_i == ell
                        if not np.any(mask):
                            continue
                        mass = float(w[mask, k].sum())
                        if mass <= eps:
                            continue
                        g_term = mass * posterior_gap
                        u_term = mass * upper_gap
                        G_sum += g_term
                        U_sum += u_term
                        term_count += 1
                        if g_term < -tol:
                            lower_violations += 1
                        if g_term - u_term > tol:
                            upper_violations += 1

    eligible_ratio = num_pairs_eligible / max(1, num_pairs_total)
    mixed_ratio = num_mixed_pairs / max(1, num_pairs_total)
    return {
        "round": int(trace_t.get("round", -1)),
        "num_pairs_total": int(num_pairs_total),
        "num_pairs_eligible": int(num_pairs_eligible),
        "eligible_ratio": float(eligible_ratio),
        "G_sum": float(G_sum),
        "U_sum": float(U_sum),
        "G_over_U": float(G_sum / (U_sum + eps)),
        "violation_lower_rate": float(lower_violations / max(1, term_count)),
        "violation_upper_rate": float(upper_violations / max(1, term_count)),
        "mean_gap_per_pair": float(G_sum / max(1, num_pairs_eligible)),
        "mean_upper_per_pair": float(U_sum / max(1, num_pairs_eligible)),
        "num_mixed_pairs": int(num_mixed_pairs),
        "mixed_ratio": float(mixed_ratio),
        "num_terms": int(term_count),
    }


def aggregate_feedback_gap(trace: Iterable[Mapping], labels_true: np.ndarray) -> List[Dict[str, float]]:
    return [compute_feedback_gap(round_trace, labels_true) for round_trace in trace]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


def _npz_get(data: np.lib.npyio.NpzFile, key: str) -> Optional[np.ndarray]:
    return data[key] if key in data.files else None
