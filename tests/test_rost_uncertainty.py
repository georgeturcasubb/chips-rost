from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("rost_uncertainty", ROOT / "scripts" / "rost_uncertainty.py")
rost_uncertainty = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = rost_uncertainty
SPEC.loader.exec_module(rost_uncertainty)


class RostUncertaintyTests(unittest.TestCase):
    def test_wilson_interval_for_perfect_score_is_not_degenerate(self):
        low, high = rost_uncertainty.wilson_interval(58, 58)

        self.assertGreater(low, 0.9)
        self.assertLess(low, 1.0)
        self.assertEqual(high, 1.0)

    def test_exact_sign_test_counts_discordant_pairs(self):
        self.assertEqual(rost_uncertainty.exact_two_sided_sign_pvalue(1, 0), 1.0)
        self.assertEqual(rost_uncertainty.exact_two_sided_sign_pvalue(4, 0), 0.125)

    def test_stratified_macro_f1_bootstrap_preserves_perfect_prediction(self):
        y_true = np.asarray(["A", "A", "B", "B"], dtype=object)
        y_pred = np.asarray(["A", "A", "B", "B"], dtype=object)

        low, high = rost_uncertainty.stratified_bootstrap_macro_f1(
            y_true,
            y_pred,
            ["A", "B"],
            n_samples=100,
            seed=42,
        )

        self.assertEqual(low, 1.0)
        self.assertEqual(high, 1.0)

    def test_metric_row_has_release_table_fields(self):
        rows = [
            {"pred": "A"},
            {"pred": "B"},
        ]
        y_true = np.asarray(["A", "B"], dtype=object)

        row = rost_uncertainty.metric_row(
            "toy",
            "Toy model",
            "toy.csv",
            "predicted_author",
            "pred",
            y_true,
            rows,
            ["A", "B"],
            bootstrap_samples=50,
            seed=42,
        )

        self.assertEqual(row["display_model"], "Toy model")
        self.assertEqual(row["prediction_file"], "toy.csv")
        self.assertEqual(row["prediction_column"], "predicted_author")
        self.assertEqual(row["n_groups"], 2)

    def test_chips_f_cross_check_rejects_prediction_mismatch(self):
        rows = [
            {
                "work_group": "A__one",
                "true_author": "A",
                "doc_count": "1",
                "doc_ids": "A_one.txt",
                "chips_f_pred": "A",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.csv"
            path.write_text(
                "work_group,doc_count,doc_ids,true_author,predicted_author\n"
                "A__one,1,A_one.txt,A,B\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                rost_uncertainty.validate_chips_f_predictions(rows, path)


if __name__ == "__main__":
    unittest.main()
