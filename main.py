"""Run BRAVE directly or select B/K on validation before test evaluation."""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import time


# Per-dataset defaults: components K, worker blocks B, maximum rounds, local epochs.
DEFAULTS = {
    "synthetic": (3, 4, 50, 2),
    "multipref_expert": (3, 1, 100, 2),
    "toloka_rel2": (5, 1, 50, 2),
    "toloka_rel5": (5, 2, 50, 2),
    "rte_ct": (5, 2, 50, 2),
    "bluebirds": (3, 1, 80, 2),
    "cifar10h": (5, 2, 30, 2),
    "netease": (5, 4, 30, 2),
    "webcrowd25k": (5, 8, 50, 1),
    "weather": (3, 4, 20, 1),
    "odre": (3, 4, 20, 1),
    "mre_treat": (5, 4, 50, 2),
    "mre_cause": (2, 1, 20, 1),
    "nist_trec": (5, 8, 30, 1),
    "music_genre": (3, 2, 30, 1),
}

# Dataset module, loader class and constructor options.
LOADERS = {
    "multipref_expert": ("multipref", "MultiPrefLoader", {}),
    "toloka_rel2": ("toloka_relevance", "TolokaRelevanceLoader", {"variant": "relevance-2"}),
    "toloka_rel5": ("toloka_relevance", "TolokaRelevanceLoader", {"variant": "relevance-5"}),
    "rte_ct": ("rte_crowdtruth", "CrowdTruthRTELoader", {}),
    "bluebirds": ("bluebirds", "BluebirdsLoader", {}),
    "cifar10h": ("cifar10h", "CIFAR10HLoader", {}),
    "netease": ("netease_crowd", "NetEaseCrowdLoader", {}),
    "webcrowd25k": ("webcrowd25k", "WebCrowd25KLoader", {}),
    "weather": ("weather_sentiment_amt", "WeatherSentimentAMTLoader", {}),
    "odre": ("crowdtruth_odre", "CrowdTruthODRELoader", {}),
    "mre_treat": ("crowdtruth_mre", "CrowdTruthMRELoader", {"target": "treat"}),
    "mre_cause": ("crowdtruth_mre", "CrowdTruthMRELoader", {"target": "cause"}),
    "nist_trec": ("nist_trec_relevance", "NistTrecRelevanceLoader", {}),
    "music_genre": ("music_genre", "MusicGenreLoader", {}),
}


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite, nonnegative number")
    return number


def integer_list(value: str):
    try:
        values = tuple(int(item) for item in value.replace(",", " ").split())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be comma-separated integers") from exc
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("must contain at least one integer, without duplicates")
    return values


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run BRAVE on one dataset with command-line parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example: python main.py --dataset bluebirds --blocks 2 --n-components 3",
    )
    parser.add_argument("--dataset", choices=list(DEFAULTS), default="synthetic", help="benchmark to load")
    parser.add_argument("--seed", type=int, default=0, help="single-run model seed; also the single-run data seed when --data-seed is omitted")
    parser.add_argument("--data-seed", type=int, help="data sampling seed; default: --seed for single runs, 0 for validation selection")

    selection = parser.add_argument_group("validation-selected test")
    selection.add_argument("--validation-selected-test", action="store_true",
                           help="select B/K by mean validation NLL, then evaluate the selected cell on test")
    selection.add_argument("--blocks-grid", type=integer_list, help="candidate B values; default: 1,2,4,8")
    selection.add_argument("--components-grid", type=integer_list, help="candidate K values; default: 2,3,4,5")
    selection.add_argument("--model-seeds", type=integer_list, help="model seeds used in both phases; default: 0,1,2,3,4")
    selection.add_argument("--split-seed", type=int, help="fixed stratified split seed; default: 20260814")

    model = parser.add_argument_group("BRAVE parameters")
    model.add_argument("--n-components", type=positive_int, help="mixture components K; omitted: dataset default")
    model.add_argument("--blocks", type=positive_int, help="worker blocks B; omitted: dataset default")
    model.add_argument("--max-rounds", type=positive_int, help="maximum inference rounds; omitted: dataset default")
    model.add_argument("--local-epochs", type=positive_int, help="worker-profile refinement steps; omitted: dataset default")
    model.add_argument("--alpha", type=nonnegative_float, default=0.0, help="class-prior smoothing")
    model.add_argument("--beta", type=nonnegative_float, default=0.0, help="reliability smoothing")
    model.add_argument("--tol", type=nonnegative_float, default=1e-5, help="posterior-shift stopping threshold")
    model.add_argument("--disable-early-stopping", action="store_true", help="run all requested rounds")

    synthetic = parser.add_argument_group("synthetic data parameters (only for --dataset synthetic)")
    synthetic.add_argument("--n-instances", type=positive_int, default=200, help="number of items")
    synthetic.add_argument("--n-workers", type=positive_int, default=40, help="number of workers")
    synthetic.add_argument("--n-classes", type=positive_int, default=3, help="number of label classes (at least 2)")
    synthetic.add_argument("--n-groups", type=positive_int, default=3, help="worker groups with aligned errors")
    synthetic.add_argument("--labels-per-item", type=positive_int, default=4, help="annotations per item")
    synthetic.add_argument("--gamma", type=nonnegative_float, default=0.4, help="aligned-error strength in [0, 1]")

    output = parser.add_argument_group("output")
    output.add_argument("--output-dir", type=Path, help="optional new/empty directory for results.json and predictions.npz")
    output.add_argument("--verbose", action="store_true", help="print data-loading and training details")
    return parser


def load_data(args):
    """Prepare one annotation matrix, independently of the model seeds."""
    import numpy as np

    data_seed = args.data_seed if getattr(args, "data_seed", None) is not None else args.seed
    if args.dataset == "synthetic":
        from src.data.synthetic_aligned_error import (
            SyntheticAlignedErrorConfig,
            generate_synthetic_aligned_error,
        )

        return generate_synthetic_aligned_error(SyntheticAlignedErrorConfig(
            n_instances=args.n_instances,
            n_workers=args.n_workers,
            n_classes=args.n_classes,
            n_groups=args.n_groups,
            labels_per_item=args.labels_per_item,
            gamma=args.gamma,
            seed=data_seed,
        ))

    module, class_name, options = LOADERS[args.dataset]
    options = dict(options)
    if args.dataset not in {"multipref_expert", "webcrowd25k"}:
        options["verbose"] = args.verbose
    loader = getattr(importlib.import_module(f"src.data.{module}"), class_name)(**options)
    loader.load_dataset()

    preparation = {}
    if args.dataset == "multipref_expert":
        preparation = dict(preference_type="overall", include_experts=True, include_normals=True, min_annotations=1)
    elif args.dataset in {"toloka_rel2", "toloka_rel5"}:
        preparation = dict(max_instances=2000, max_workers=200, min_labels_per_instance=3, seed=data_seed)
    elif args.dataset == "rte_ct":
        preparation = dict(min_labels_per_instance=1)
    elif args.dataset == "cifar10h":
        preparation = dict(max_instances=10000, max_workers=None, min_labels_per_instance=20, seed=data_seed)
    elif args.dataset == "netease":
        preparation = dict(max_instances=3000, max_workers=300, min_labels_per_instance=3, seed=data_seed)
    elif args.dataset in {"mre_treat", "mre_cause"}:
        preparation = dict(max_instances=5000, max_workers=500, min_labels_per_instance=3, seed=data_seed)
    elif args.dataset == "nist_trec":
        preparation = dict(only_gold_tasks=False)

    labels, truth, n_classes, metadata = loader.prepare_data(**preparation)
    if args.dataset == "multipref_expert":
        experts = [idx for worker, idx in metadata["worker_id_to_idx"].items() if worker.startswith("expert_")]
        truth = np.full(labels.shape[0], -1, dtype=int)
        for i, row in enumerate(labels[:, experts]):
            votes = row[row != -1]
            if votes.size:
                truth[i] = int(np.argmax(np.bincount(votes, minlength=n_classes)))
        metadata["eval_mask"] = truth != -1
    elif args.dataset == "webcrowd25k":
        truth, metadata["eval_mask"] = loader.build_official_gold(
            instance_ids=metadata["instance_ids"], mapping="trec6_to_4",
        )
    return labels, truth, n_classes, metadata


def json_value(value):
    """Convert evaluator NumPy values and undefined metrics to strict JSON."""
    import numpy as np

    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path, value):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_value(value), handle, indent=2, allow_nan=False)
        handle.write("\n")


def fit_brave(labels, n_classes, args, settings, seed):
    """Fit a fresh BRAVE model from the annotation matrix."""
    import numpy as np
    from src.brave import BRAVE, partition_annotations

    np.random.seed(seed)
    block_labels, block_ids = partition_annotations(labels, n_blocks=settings["blocks"])
    model = BRAVE(
        n_classes=n_classes,
        n_components=settings["n_components"],
        max_rounds=settings["max_rounds"],
        local_epochs=settings["local_epochs"],
        alpha=args.alpha, beta=args.beta, tol=args.tol,
        disable_early_stopping=args.disable_early_stopping,
        verbose=args.verbose,
    )
    for block_id, block_data in zip(block_ids, block_labels):
        model.add_block(block_id, block_data)
    started = time.perf_counter()
    model.fit(n_instances=labels.shape[0])
    return model, time.perf_counter() - started


def validation_selected_test(args, labels, truth, n_classes, eval_mask, settings):
    import hashlib
    import numpy as np
    from src.evaluation.validation import run_validation_selected_test

    config = {**vars(args), "max_rounds": settings["max_rounds"],
              "local_epochs": settings["local_epochs"],
              "dataset_n_classes": int(n_classes),
              "dataset_n_instances": int(labels.shape[0]),
              "dataset_n_workers": int(labels.shape[1])}
    # Record the exact prepared matrices and reference eligibility for this run.
    digest = hashlib.sha256()
    for values in (labels, truth, eval_mask):
        values = np.asarray(values, dtype="<i8")
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())
    config["data_sha256"] = digest.hexdigest()
    protocol = {
        "split": "approximately 60/20/20 per reference class; other items are context",
        "rare_classes": "singletons go to test; two-item classes go to validation and test",
        "validation_fit": "train + validation + context annotations; test rows excluded",
        "test_fit": "fresh fit on all annotation rows; only test reference labels scored",
        "worker_filter": "remove workers with no observed annotation in each fit phase",
        "block_count": "cap requested B at the number of active workers in each phase",
        "std_ddof": 1,
    }
    def fit_predict(fit_labels, classes, b, k, seed, phase):
        candidate_settings = {**settings, "blocks": b, "n_components": k}
        model, elapsed = fit_brave(fit_labels, classes, args, candidate_settings, seed)
        return dict(predictions=model.predict(), probabilities=model.predict_proba(),
                    rounds=model.n_rounds_, history=model.history,
                    stop_reason=model.stop_reason_, time_s=elapsed)

    def freeze_selection(selection, split, runs, summary):
        canonical_split = "\n".join(f"{i}:{value}" for i, value in enumerate(split["assignment"]))
        config["split_sha256"] = hashlib.sha256(canonical_split.encode("utf-8")).hexdigest()
        print(f"Selected B={selection['B']} K={selection['K']} "
              f"validation NLL={selection['nll_mean']:.4f} +/- {selection['nll_std']:.4f}", flush=True)
        if args.output_dir is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            # Freeze the decision and split before the first test fit begins.
            save_json(args.output_dir / "selection.json",
                      dict(config=config, protocol=protocol, selection=selection))
            save_json(args.output_dir / "validation_results.json", dict(runs=runs, summary=summary))
            np.savez_compressed(args.output_dir / "splits.npz", **split)

    def show_run(row):
        print(f"{row['phase']} B={row['B']} K={row['K']} seed={row['model_seed']} "
              f"NLL={row['nll']:.4f}", flush=True)

    result = run_validation_selected_test(
        labels=labels, truth=truth, n_classes=n_classes, eval_mask=eval_mask,
        dataset_key=args.dataset, split_seed=args.split_seed,
        blocks_grid=args.blocks_grid, components_grid=args.components_grid,
        model_seeds=args.model_seeds, fit_predict=fit_predict,
        on_selection=freeze_selection, on_run=show_run,
    )
    summary = result["test_summary"]
    print(f"Test ({summary['n_seeds']} seeds): "
          f"accuracy={summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f} "
          f"NLL={summary['nll_mean']:.4f} +/- {summary['nll_std']:.4f} "
          f"ECE={summary['ece_mean']:.4f} +/- {summary['ece_std']:.4f}")
    if args.output_dir is not None:
        split_counts = {name: int(result["split"][name].sum())
                        for name in ("gold", "train", "validation", "test", "context")}
        save_json(args.output_dir / "results.json",
                  dict(config=config, protocol=protocol, split_counts=split_counts,
                       selection=result["selection"], validation_runs=result["validation_runs"],
                       validation_summary=result["validation_summary"],
                       test_runs=result["test_runs"], test_summary=summary))
        np.savez_compressed(args.output_dir / "predictions.npz",
                            model_seeds=np.asarray(args.model_seeds),
                            predictions=result["predictions"], probabilities=result["probabilities"],
                            truth=truth, eval_mask=result["split"]["test"])
        print(f"Saved results to {args.output_dir.resolve()}")
    return 0


def main(argv=None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    if args.data_seed is None:
        args.data_seed = 0 if args.validation_selected_test else args.seed
    if not 0 <= args.data_seed < 2**32:
        parser.error("--data-seed must be in [0, 2**32)")
    selection_options = (args.blocks_grid, args.components_grid, args.model_seeds, args.split_seed)
    if args.validation_selected_test:
        if args.blocks is not None or args.n_components is not None:
            parser.error("use --blocks-grid and --components-grid in validation-selected test mode")
        args.blocks_grid = args.blocks_grid if args.blocks_grid is not None else (1, 2, 4, 8)
        args.components_grid = args.components_grid if args.components_grid is not None else (2, 3, 4, 5)
        args.model_seeds = args.model_seeds if args.model_seeds is not None else (0, 1, 2, 3, 4)
        args.split_seed = args.split_seed if args.split_seed is not None else 20260814
        if any(value < 1 for value in (*args.blocks_grid, *args.components_grid)):
            parser.error("B and K grid values must be positive")
        if any(not 0 <= seed < 2**32 for seed in (*args.model_seeds, args.split_seed)):
            parser.error("model seeds and --split-seed must be in [0, 2**32)")
    elif any(value is not None for value in selection_options):
        parser.error("grid, model-seeds and split-seed options require --validation-selected-test")
    if args.dataset == "synthetic":
        if args.n_classes < 2:
            parser.error("--n-classes must be at least 2")
        if args.labels_per_item > args.n_workers:
            parser.error("--labels-per-item cannot exceed --n-workers")
        if args.gamma > 1:
            parser.error("--gamma must be in [0, 1]")
    if args.output_dir is not None and args.output_dir.exists():
        if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):
            parser.error("--output-dir must be a new or empty directory; choose another path")

    import numpy as np
    from src.evaluation.evaluator import Evaluator

    labels, truth, n_classes, metadata = load_data(args)
    if labels.ndim != 2 or min(labels.shape) == 0:
        parser.error("the dataset must contain at least one instance and worker")
    defaults = DEFAULTS[args.dataset]
    names = ("n_components", "blocks", "max_rounds", "local_epochs")
    settings = {name: getattr(args, name) if getattr(args, name) is not None else default
                for name, default in zip(names, defaults)}
    eval_mask = np.asarray(metadata.get("eval_mask", truth != -1), dtype=bool) & (truth != -1)
    if not np.any(eval_mask):
        parser.error("the selected dataset has no valid evaluation labels")

    print(f"dataset={args.dataset} instances={labels.shape[0]} workers={labels.shape[1]} classes={n_classes}")
    if args.validation_selected_test:
        print(f"data_seed={args.data_seed} model_seeds={list(args.model_seeds)} split_seed={args.split_seed}")
        try:
            return validation_selected_test(args, labels, truth, n_classes, eval_mask, settings)
        except ValueError as exc:
            parser.error(str(exc))
    if settings["blocks"] > labels.shape[1]:
        parser.error(f"--blocks cannot exceed the number of workers ({labels.shape[1]})")
    print(f"K={settings['n_components']} blocks={settings['blocks']} max_rounds={settings['max_rounds']} "
          f"local_epochs={settings['local_epochs']} seed={args.seed}")
    model, elapsed = fit_brave(labels, n_classes, args, settings, args.seed)
    predictions, probabilities = model.predict(), model.predict_proba()
    metrics = Evaluator().evaluate(
        predictions=predictions[eval_mask], ground_truth=truth[eval_mask],
        probabilities=probabilities[eval_mask],
        algorithm_name="BRAVE",
        dataset_name=args.dataset, training_time=elapsed,
    )
    print(f"accuracy={metrics['accuracy']:.4f} NLL={metrics['nll']:.4f} ECE={metrics['ece']:.4f} "
          f"rounds={model.n_rounds_} stop_reason={model.stop_reason_} time={elapsed:.3f}s")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        config = {**vars(args), **settings, "dataset_n_classes": int(n_classes),
                  "dataset_n_instances": int(labels.shape[0]), "dataset_n_workers": int(labels.shape[1])}
        save_json(args.output_dir / "results.json", {
            "config": config, "metrics": metrics, "history": model.history,
            "rounds": model.n_rounds_, "stop_reason": model.stop_reason_,
        })
        np.savez_compressed(args.output_dir / "predictions.npz", predictions=predictions,
                            probabilities=probabilities, truth=truth, eval_mask=eval_mask)
        print(f"Saved results to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
