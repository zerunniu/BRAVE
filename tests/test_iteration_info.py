"""Check posterior-shift records and stopping reasons."""

import unittest

import numpy as np

from src.brave import BRAVE


class IterationInfoTests(unittest.TestCase):
    def fit_model(self, **settings):
        np.random.seed(17)
        model = BRAVE(n_classes=3, n_components=2, verbose=False, **settings)
        model.add_block("workers", np.array([[0, 0], [1, -1], [2, 1], [-1, -1]]))
        model.fit()
        return model

    def test_tolerance_stop_reports_the_actual_itemwise_l1_shift(self):
        model = self.fit_model(max_rounds=5, tol=2.0)
        self.assertEqual(model.n_rounds_, 1)
        self.assertEqual(model.stop_reason_, "tolerance")
        self.assertEqual(len(model.history), 1)
        expected = np.abs(model.predict_proba() - 1.0 / 3).sum(axis=1).mean()
        self.assertEqual(model.history, [{"round": 1, "delta_q": float(expected)}])

    def test_budget_exhaustion_with_and_without_early_stopping(self):
        for disabled, tolerance in ((False, 0.0), (True, 2.0)):
            with self.subTest(disabled=disabled):
                model = self.fit_model(max_rounds=3, tol=tolerance,
                                       disable_early_stopping=disabled)
                self.assertEqual(model.n_rounds_, 3)
                self.assertEqual(model.stop_reason_, "max_rounds")
                self.assertEqual([row["round"] for row in model.history], [1, 2, 3])
                self.assertTrue(all(set(row) == {"round", "delta_q"} for row in model.history))


if __name__ == "__main__":
    unittest.main()
