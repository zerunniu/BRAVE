"""Validation-only B/K selection followed by transductive test evaluation."""

from __future__ import annotations

import hashlib

import numpy as np

from .evaluator import Evaluator


def make_split(truth, eval_mask, dataset_key, split_seed):
    """Create a deterministic class-stratified 60/20/20 split.

    Non-reference items remain context. Singleton classes go to test; classes
    with two items place one in validation and one in test.
    """
    truth = np.asarray(truth)
    if truth.ndim != 1:
        raise ValueError("truth must be a one-dimensional array")
    gold = truth != -1
    if eval_mask is not None:
        supplied = np.asarray(eval_mask, dtype=bool)
        if supplied.shape != truth.shape:
            raise ValueError("eval_mask and truth must have the same shape")
        gold &= supplied
    train = np.zeros(len(truth), dtype=bool)
    validation = np.zeros(len(truth), dtype=bool)
    test = np.zeros(len(truth), dtype=bool)
    payload = f"{int(split_seed)}:{dataset_key}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    rng = np.random.default_rng(derived_seed)

    for class_id in sorted(np.unique(truth[gold]).tolist()):
        indices = rng.permutation(np.flatnonzero(gold & (truth == class_id)))
        n_class = len(indices)
        if n_class == 1:
            test[indices[0]] = True
            continue
        if n_class == 2:
            validation[indices[0]] = True
            test[indices[1]] = True
            continue
        n_validation = max(1, int(round(0.20 * n_class)))
        n_test = max(1, int(round(0.20 * n_class)))
        overflow = n_validation + n_test - (n_class - 1)
        while overflow > 0 and (n_validation > 1 or n_test > 1):
            if n_validation >= n_test and n_validation > 1:
                n_validation -= 1
            elif n_test > 1:
                n_test -= 1
            overflow -= 1
        validation[indices[:n_validation]] = True
        test[indices[n_validation:n_validation + n_test]] = True
        train[indices[n_validation + n_test:]] = True

    if not validation.any() or not test.any():
        raise ValueError("dataset must have nonempty validation and test splits")
    assignment = np.full(len(truth), "context", dtype="U10")
    assignment[train] = "train"
    assignment[validation] = "validation"
    assignment[test] = "test"
    return dict(gold=gold, train=train, validation=validation, test=test,
                context=~gold, assignment=assignment)


def summarize_runs(runs):
    """Report means and sample standard deviations across model seeds."""
    summary = {"n_seeds": len(runs)}
    for metric in ("accuracy", "nll", "ece"):
        values = np.asarray([row[metric] for row in runs], dtype=float)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f"cannot summarize empty or non-finite {metric} results")
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return summary


def select_configuration(runs, blocks_grid, components_grid, model_seeds):
    """Select only from a complete validation grid, with B/K tie breaking."""
    expected = {(b, k, seed) for b in blocks_grid
                for k in components_grid for seed in model_seeds}
    indexed = {}
    for row in runs:
        if row["phase"] != "validation":
            continue
        key = (row["B"], row["K"], row["model_seed"])
        if key not in expected or key in indexed:
            raise ValueError("validation grid has unexpected or duplicate run keys")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("validation grid is incomplete")
    summaries = []
    for b in blocks_grid:
        for k in components_grid:
            rows = [indexed[b, k, seed] for seed in model_seeds]
            summaries.append({"B": b, "K": k, **summarize_runs(rows)})
    selected = min(summaries, key=lambda row: (row["nll_mean"], row["B"], row["K"]))
    return dict(selected), summaries


def run_validation_selected_test(
    *, labels, truth, n_classes, eval_mask, dataset_key, split_seed,
    blocks_grid, components_grid, model_seeds, fit_predict,
    on_selection=None, on_run=None,
):
    """Fit callbacks receive annotation matrices, configurations and model seeds.

    Validation fits exclude test rows and keep non-gold context. Once selection
    is frozen, fresh test fits use the full annotation matrix. Only the selected
    B/K cell is tested. Each phase removes workers with no observed annotations.
    """
    labels, truth = np.asarray(labels), np.asarray(truth)
    if labels.ndim != 2 or not min(labels.shape):
        raise ValueError("labels must be a nonempty two-dimensional matrix")
    if truth.shape != (labels.shape[0],):
        raise ValueError("truth must have one label per annotation-matrix row")
    for name, values, lower in (("blocks grid", blocks_grid, 1),
                                ("components grid", components_grid, 1),
                                ("model seeds", model_seeds, 0)):
        if not values or len(values) != len(set(values)) or any(v < lower for v in values):
            raise ValueError(f"{name} must contain unique valid integers")
    split = make_split(truth, eval_mask, dataset_key, split_seed)
    if np.any(truth[split["gold"]] < 0) or np.any(truth[split["gold"]] >= n_classes):
        raise ValueError("reference labels must be in [0, n_classes)")
    keep = ~split["test"]
    validation_labels = labels[keep]
    validation_workers = np.flatnonzero(np.any(validation_labels != -1, axis=0))
    test_workers = np.flatnonzero(np.any(labels != -1, axis=0))
    if not len(validation_workers) or not len(test_workers):
        raise ValueError("each fit phase must have at least one observed worker")
    validation_labels = validation_labels[:, validation_workers]
    test_labels = labels[:, test_workers]
    split["validation_worker_indices"] = validation_workers
    split["test_worker_indices"] = test_workers
    evaluator = Evaluator()

    def run_one(phase, fit_labels, fit_truth, mask, b, k, seed):
        effective_b = min(b, fit_labels.shape[1])
        result = fit_predict(fit_labels, n_classes, effective_b, k, seed, phase)
        metrics = evaluator.evaluate(
            predictions=result["predictions"][mask],
            probabilities=result["probabilities"][mask],
            ground_truth=fit_truth[mask],
        )
        row = dict(phase=phase, B=b, K=k, effective_B=effective_b, model_seed=seed,
                   n_fit_items=fit_labels.shape[0], n_fit_workers=fit_labels.shape[1],
                   n_eval_items=int(mask.sum()), rounds=result["rounds"],
                   time_s=result["time_s"])
        for name in ("history", "stop_reason"):
            if name in result:
                row[name] = result[name]
        row.update({name: float(metrics[name]) for name in ("accuracy", "nll", "ece")})
        if not all(np.isfinite(row[name]) for name in ("accuracy", "nll", "ece")):
            raise ValueError(f"non-finite metrics for {phase} B={b} K={k} seed={seed}")
        if on_run is not None:
            on_run(row)
        return row, result

    validation_runs = []
    for b in blocks_grid:
        for k in components_grid:
            for seed in model_seeds:
                row, _ = run_one("validation", validation_labels, truth[keep],
                                 split["validation"][keep], b, k, seed)
                validation_runs.append(row)
    selection, validation_summary = select_configuration(
        validation_runs, blocks_grid, components_grid, model_seeds,
    )
    selection.update(metric="mean validation NLL", tie_break="smaller B, then smaller K")
    if on_selection is not None:
        on_selection(selection, split, validation_runs, validation_summary)

    test_runs, predictions, probabilities = [], [], []
    for seed in model_seeds:
        row, result = run_one("test", test_labels, truth, split["test"],
                              selection["B"], selection["K"], seed)
        test_runs.append(row)
        predictions.append(result["predictions"])
        probabilities.append(result["probabilities"])
    return dict(split=split, selection=selection, validation_runs=validation_runs,
                validation_summary=validation_summary, test_runs=test_runs,
                test_summary=summarize_runs(test_runs),
                predictions=np.stack(predictions), probabilities=np.stack(probabilities))
