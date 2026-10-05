"""Unified BRAVE CLI; independent of the optional experiments directory."""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import time


# K, worker blocks, maximum rounds, local epochs: existing benchmark settings.
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

# Module, loader class, constructor options. Imports are lazy, so --help needs
# only Python's standard library and unused datasets need no optional imports.
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


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run BRAVE on one dataset with command-line parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example: python main.py --dataset bluebirds --blocks 2 --n-components 3",
    )
    parser.add_argument("--dataset", choices=list(DEFAULTS), default="synthetic", help="benchmark to load")
    parser.add_argument("--seed", type=int, default=0, help="data sampling and model initialization seed")

    model = parser.add_argument_group("BRAVE parameters")
    model.add_argument("--n-components", type=positive_int, help="mixture components K; omitted: dataset default")
    model.add_argument("--blocks", type=positive_int, help="worker blocks B; omitted: dataset default")
    model.add_argument("--max-rounds", type=positive_int, help="maximum EM rounds; omitted: dataset default")
    model.add_argument("--local-epochs", type=positive_int, help="local E-step repeats; omitted: dataset default")
    model.add_argument("--alpha", type=nonnegative_float, default=0.0, help="class-prior smoothing")
    model.add_argument("--beta", type=nonnegative_float, default=0.0, help="reliability smoothing")
    model.add_argument("--tol", type=nonnegative_float, default=1e-5, help="posterior-shift stopping threshold")
    model.add_argument("--reliability-update-mode", choices=["local", "global"], default="local",
                       help="local: BRAVE; global: BRAVE-GC")
    model.add_argument("--dp-epsilon", type=nonnegative_float, help="optional worker-block privacy budget; must be > 0")
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
    output.add_argument("--collect-diagnostics", action="store_true", help="save per-round traces; requires --output-dir")
    output.add_argument("--verbose", action="store_true", help="print data-loading and training details")
    return parser


def load_data(args):
    """Use the same preparation and evaluation policies as existing runners."""
    import numpy as np

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
            seed=args.seed,
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
        preparation = dict(max_instances=2000, max_workers=200, min_labels_per_instance=3, seed=args.seed)
    elif args.dataset == "rte_ct":
        preparation = dict(min_labels_per_instance=1)
    elif args.dataset == "cifar10h":
        preparation = dict(max_instances=10000, max_workers=None, min_labels_per_instance=20, seed=args.seed)
    elif args.dataset == "netease":
        preparation = dict(max_instances=3000, max_workers=300, min_labels_per_instance=3, seed=args.seed)
    elif args.dataset in {"mre_treat", "mre_cause"}:
        preparation = dict(max_instances=5000, max_workers=500, min_labels_per_instance=3, seed=args.seed)
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


def main(argv=None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    if args.dp_epsilon is not None and args.dp_epsilon <= 0:
        parser.error("--dp-epsilon must be > 0")
    if args.dataset == "synthetic":
        if args.n_classes < 2:
            parser.error("--n-classes must be at least 2")
        if args.labels_per_item > args.n_workers:
            parser.error("--labels-per-item cannot exceed --n-workers")
        if args.gamma > 1:
            parser.error("--gamma must be in [0, 1]")
    if args.collect_diagnostics and args.output_dir is None:
        parser.error("--collect-diagnostics requires --output-dir")
    if args.output_dir is not None and args.output_dir.exists():
        if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):
            parser.error("--output-dir must be a new or empty directory; choose another path")

    import numpy as np
    from src.brave import BRAVE, simulate_federated_scenario
    from src.evaluation.evaluator import Evaluator

    labels, truth, n_classes, metadata = load_data(args)
    if labels.ndim != 2 or min(labels.shape) == 0:
        parser.error("the dataset must contain at least one instance and worker")
    defaults = DEFAULTS[args.dataset]
    names = ("n_components", "blocks", "max_rounds", "local_epochs")
    settings = {name: getattr(args, name) if getattr(args, name) is not None else default
                for name, default in zip(names, defaults)}
    if settings["blocks"] > labels.shape[1]:
        parser.error(f"--blocks cannot exceed the number of workers ({labels.shape[1]})")
    eval_mask = np.asarray(metadata.get("eval_mask", truth != -1), dtype=bool) & (truth != -1)
    if not np.any(eval_mask):
        parser.error("the selected dataset has no valid evaluation labels")

    # Reset immediately before initializing blocks, matching the experiment runners.
    np.random.seed(args.seed)
    block_labels, block_ids = simulate_federated_scenario(labels, n_clients=settings["blocks"])
    model = BRAVE(
        n_classes=n_classes,
        n_components=settings["n_components"],
        max_rounds=settings["max_rounds"],
        local_epochs=settings["local_epochs"],
        alpha=args.alpha,
        beta=args.beta,
        tol=args.tol,
        reliability_update_mode=args.reliability_update_mode,
        collect_diagnostics=args.collect_diagnostics,
        diagnostics_output_dir=str(args.output_dir / "traces") if args.collect_diagnostics else None,
        disable_early_stopping=args.disable_early_stopping,
        verbose=args.verbose,
    )
    for block_id, block_data in zip(block_ids, block_labels):
        model.add_block(block_id, block_data, dp_epsilon=args.dp_epsilon)

    print(f"dataset={args.dataset} instances={labels.shape[0]} workers={labels.shape[1]} classes={n_classes}")
    print(f"K={settings['n_components']} blocks={settings['blocks']} max_rounds={settings['max_rounds']} "
          f"local_epochs={settings['local_epochs']} mode={args.reliability_update_mode} seed={args.seed}")
    started = time.perf_counter()
    model.fit(n_instances=labels.shape[0])
    elapsed = time.perf_counter() - started
    predictions, probabilities = model.predict(), model.predict_proba()
    metrics = Evaluator().evaluate(
        predictions=predictions[eval_mask], ground_truth=truth[eval_mask],
        probabilities=probabilities[eval_mask],
        algorithm_name="BRAVE" if args.reliability_update_mode == "local" else "BRAVE-GC",
        dataset_name=args.dataset, training_time=elapsed,
    )
    print(f"accuracy={metrics['accuracy']:.4f} NLL={metrics['nll']:.4f} ECE={metrics['ece']:.4f} "
          f"rounds={len(model.history)} time={elapsed:.3f}s")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        config = {**vars(args), **settings, "dataset_n_classes": int(n_classes),
                  "dataset_n_instances": int(labels.shape[0]), "dataset_n_workers": int(labels.shape[1])}
        results = json_value({"config": config, "metrics": metrics, "history": model.history})
        with (args.output_dir / "results.json").open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, allow_nan=False)
            handle.write("\n")
        np.savez_compressed(args.output_dir / "predictions.npz", predictions=predictions,
                            probabilities=probabilities, truth=truth, eval_mask=eval_mask)
        print(f"Saved results to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
