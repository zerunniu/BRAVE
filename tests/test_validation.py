"""Offline checks for validation selection, phase isolation and CLI outputs."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import main
from src.evaluation.validation import (
    make_split, run_validation_selected_test, select_configuration, summarize_runs,
)


def validation_row(b, k, seed, nll, phase="validation"):
    return dict(phase=phase, B=b, K=k, model_seed=seed, accuracy=0.5, nll=nll, ece=0.1)


class SplitTests(unittest.TestCase):
    def test_matches_existing_runner_and_preserves_context_and_rare_classes(self):
        truth = np.array([0] * 5 + [1] * 4 + [2] * 2 + [3] + [-1] * 2)
        eligible = truth != -1
        eligible[5] = False
        split = make_split(truth, eligible, "fixture", 20260814)
        # Expected class-stratified assignments, including rare classes.
        expected = ["test", "train", "validation", "train", "train", "context",
                    "train", "test", "validation", "test", "validation", "test",
                    "context", "context"]
        self.assertEqual(split["assignment"].tolist(), expected)
        self.assertTrue(np.array_equal(split["gold"], eligible))
        self.assertTrue(np.array_equal(split["train"] | split["validation"] | split["test"], eligible))
        self.assertFalse(np.any(split["train"] & split["validation"]))
        self.assertFalse(np.any(split["validation"] & split["test"]))
        self.assertFalse(np.any(split["train"] & split["test"]))
        repeated = make_split(truth, eligible, "fixture", 20260814)
        self.assertTrue(np.array_equal(split["assignment"], repeated["assignment"]))

    def test_rejects_empty_validation_and_malformed_mask(self):
        with self.assertRaisesRegex(ValueError, "nonempty validation"):
            make_split(np.array([0, 1]), None, "fixture", 0)
        with self.assertRaisesRegex(ValueError, "same shape"):
            make_split(np.arange(10), np.ones(9), "fixture", 0)


class SelectionTests(unittest.TestCase):
    def test_test_scores_are_ignored_and_ties_choose_smaller_b_then_k(self):
        runs = [validation_row(b, k, seed, 1.0)
                for b in (2, 1) for k in (3, 2) for seed in (0, 1)]
        runs += [validation_row(2, 3, 0, -100.0, phase="test")]
        selected, _ = select_configuration(runs, (2, 1), (3, 2), (0, 1))
        self.assertEqual((selected["B"], selected["K"]), (1, 2))

    def test_missing_duplicate_or_wrong_seed_cannot_count_as_complete(self):
        valid = [validation_row(1, 2, seed, 1.0) for seed in (0, 1)]
        for runs in (valid[:1], valid + valid[:1], [valid[0], validation_row(1, 2, 4, 1.0)]):
            with self.subTest(runs=runs), self.assertRaises(ValueError):
                select_configuration(runs, (1,), (2,), (0, 1))

    def test_nonfinite_nll_rejected_and_sample_std_reported(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            select_configuration([validation_row(1, 2, 0, float("nan"))], (1,), (2,), (0,))
        summary = summarize_runs([validation_row(1, 2, 0, 1.0), validation_row(1, 2, 1, 3.0)])
        self.assertEqual(summary["nll_mean"], 2.0)
        self.assertAlmostEqual(summary["nll_std"], np.sqrt(2.0))


class WorkflowTests(unittest.TestCase):
    def test_phase_rows_worker_filter_and_fresh_selected_test_fits(self):
        truth = np.array([0] * 5 + [1] * 5 + [-1, -1])
        split = make_split(truth, None, "fixture", 20260814)
        labels = np.full((len(truth), 3), -1, dtype=int)
        labels[:, 0] = np.arange(len(truth)) % 2
        labels[split["test"], 1] = 1  # This worker exists only in test rows.
        events = []
        frozen = []

        def fit_predict(fit_labels, classes, b, k, seed, phase):
            self.assertEqual(classes, 2)
            if phase == "validation":
                self.assertFalse(frozen)
                self.assertTrue(np.array_equal(fit_labels, labels[~split["test"], :1]))
            else:
                self.assertTrue(frozen)
                self.assertTrue(np.array_equal(fit_labels, labels[:, :2]))
            events.append((phase, b, k, seed))
            probs = np.full((len(fit_labels), 2), 0.5)
            return dict(predictions=probs.argmax(axis=1), probabilities=probs, rounds=1, time_s=0.0)

        def on_selection(selection, masks, runs, summary):
            self.assertEqual(len(events), 8)
            self.assertEqual((selection["B"], selection["K"]), (1, 2))
            self.assertTrue(masks["context"][-2:].all())
            frozen.append(True)

        result = run_validation_selected_test(
            labels=labels, truth=truth, n_classes=2, eval_mask=None,
            dataset_key="fixture", split_seed=20260814, blocks_grid=(1, 2),
            components_grid=(2, 3), model_seeds=(0, 1),
            fit_predict=fit_predict, on_selection=on_selection,
        )
        self.assertEqual(events[-2:], [("test", 1, 2, 0), ("test", 1, 2, 1)])
        self.assertEqual(result["predictions"].shape, (2, len(truth)))
        self.assertEqual(result["probabilities"].shape, (2, len(truth), 2))
        self.assertEqual(result["split"]["validation_worker_indices"].tolist(), [0])
        self.assertEqual(result["split"]["test_worker_indices"].tolist(), [0, 1])
        self.assertTrue(all(row["effective_B"] == 1 for row in result["validation_runs"]))


class CLITests(unittest.TestCase):
    def test_cli_saves_frozen_selection_and_seeded_test_outputs(self):
        truth = np.tile(np.arange(3), 20)
        labels = np.column_stack([truth, truth, (truth + 1) % 3])
        base_fit = main.fit_brave
        test_fits = []
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"

            def checked_fit(*args, **kwargs):
                if args[0].shape[0] == labels.shape[0]:
                    self.assertTrue((output / "selection.json").exists())
                    self.assertTrue((output / "splits.npz").exists())
                    test_fits.append(args[4])
                return base_fit(*args, **kwargs)

            argv = ["--dataset", "synthetic", "--validation-selected-test",
                    "--blocks-grid", "1,2", "--components-grid", "2,3",
                    "--model-seeds", "0,1", "--data-seed", "17", "--max-rounds", "3",
                    "--output-dir", str(output)]
            with patch.object(main, "load_data", return_value=(labels, truth, 3, {})) as load:
                with patch.object(main, "fit_brave", side_effect=checked_fit):
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(main.main(argv), 0)
                self.assertEqual(load.call_count, 1)
                self.assertEqual(load.call_args.args[0].data_seed, 17)
            saved = json.loads((output / "results.json").read_text())
            selection = json.loads((output / "selection.json").read_text())
            self.assertEqual(selection["selection"], saved["selection"])
            self.assertEqual(saved["config"]["model_seeds"], [0, 1])
            self.assertEqual(len(saved["validation_runs"]), 8)
            self.assertEqual(len(saved["test_runs"]), 2)
            self.assertEqual(test_fits, [0, 1])
            arrays = np.load(output / "predictions.npz", allow_pickle=False)
            masks = np.load(output / "splits.npz", allow_pickle=False)
            self.assertEqual(arrays["probabilities"].shape, (2, 60, 3))
            self.assertTrue(np.array_equal(arrays["eval_mask"], masks["test"]))
            for row in saved["validation_runs"] + saved["test_runs"]:
                self.assertEqual(row["rounds"], len(row["history"]))
                self.assertIn(row["stop_reason"], ("tolerance", "max_rounds"))
            self.assertFalse((output / "traces").exists())

    def test_selection_only_options_rejected_before_loading(self):
        cases = [["--blocks-grid", "1,2"],
                 ["--validation-selected-test", "--blocks", "2"],
                 ["--validation-selected-test", "--model-seeds", "0,0"],
                 ["--validation-selected-test", "--components-grid", "0,2"],
                 ["--validation-selected-test", "--model-seeds", "-1,0"],
                 ["--validation-selected-test", "--data-seed", "-1"]]
        for argv in cases:
            with self.subTest(argv=argv), patch.object(main, "load_data") as load:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                    main.main(argv)
                self.assertEqual(exc.exception.code, 2)
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
