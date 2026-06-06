from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("chips_rerank", ROOT / "scripts" / "chips_rerank.py")
chips_rerank = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = chips_rerank
SPEC.loader.exec_module(chips_rerank)


class ChipsRerankSmokeTests(unittest.TestCase):
    def make_base(self):
        class_order = ["A", "B", "C", "D", "E", "F"]
        return chips_rerank.GroupBasePredictions(
            group_ids=["A__one", "F__two"],
            y_true=["B", "F"],
            class_order=class_order,
            probabilities={
                "CH_SVM": np.asarray(
                    [
                        [0.50, 0.40, 0.03, 0.025, 0.02, 0.025],
                        [0.60, 0.15, 0.10, 0.08, 0.05, 0.02],
                    ],
                    dtype=float,
                ),
                "FFT12_LR": np.asarray(
                    [
                        [0.30, 0.45, 0.10, 0.08, 0.04, 0.03],
                        [0.30, 0.20, 0.18, 0.15, 0.12, 0.05],
                    ],
                    dtype=float,
                ),
                "CHiPS_F": np.asarray(
                    [
                        [0.42, 0.41, 0.06, 0.05, 0.03, 0.03],
                        [0.48, 0.17, 0.13, 0.10, 0.08, 0.04],
                    ],
                    dtype=float,
                ),
            },
            doc_counts={"A__one": 1, "F__two": 2},
            doc_ids={"A__one": ["A_one.txt"], "F__two": ["F_two-1.txt", "F_two-2.txt"]},
            char_counts={"A__one": 1200, "F__two": 2400},
        )

    def test_candidate_indices_keep_anchor_first(self):
        base = self.make_base()
        probs = {key: value[0] for key, value in base.probabilities.items()}

        candidates = chips_rerank._candidate_indices(probs, "CH_SVM_TOP5", top_k=5)

        self.assertEqual(candidates[0], 0)
        self.assertEqual(len(candidates), 5)
        self.assertEqual(len(set(candidates)), 5)

    def test_reranker_dataset_tracks_candidate_coverage(self):
        dataset = chips_rerank._build_reranker_dataset(self.make_base(), "CH_SVM_TOP5", top_k=5)

        self.assertEqual(dataset.X.shape[0], 2)
        self.assertEqual(dataset.candidates.shape, (2, 5))
        self.assertTrue(dataset.true_in_candidates[0])
        self.assertEqual(dataset.y_position[0], 1)
        self.assertFalse(dataset.true_in_candidates[1])
        self.assertEqual(dataset.y_position[1], 0)
        self.assertGreater(len(dataset.feature_names), 0)

    def test_anchor_margin_gate_only_reranks_low_margin_cases(self):
        base = self.make_base()
        dataset = chips_rerank._build_reranker_dataset(base, "CH_SVM_TOP5", top_k=5)
        pos_probs = np.asarray(
            [
                [0.10, 0.70, 0.05, 0.10, 0.05],
                [0.05, 0.80, 0.05, 0.05, 0.05],
            ],
            dtype=float,
        )

        pred = chips_rerank._apply_mode(
            dataset,
            pos_probs,
            {"mode": "anchor_margin_gate", "anchor_margin_max": 0.20, "meta_margin_min": 0.10},
        )

        self.assertEqual(pred[0], dataset.candidates[0, 1])
        self.assertEqual(pred[1], dataset.candidates[1, 0])

    def test_transition_summary_counts_fixed_and_unchanged_errors(self):
        base = self.make_base()
        dataset = chips_rerank._build_reranker_dataset(base, "CH_SVM_TOP5", top_k=5)
        pred = np.asarray([dataset.y_author_index[0], dataset.candidates[1, 0]], dtype=int)

        summary = chips_rerank._transition_summary_vs_ch_svm(base, dataset, pred)

        self.assertEqual(summary["fixed_ch_svm_errors"], 1)
        self.assertEqual(summary["unchanged_ch_svm_errors"], 1)
        self.assertEqual(summary["broken_ch_svm_correct"], 0)

    def test_transition_summary_can_compare_against_chips_f(self):
        base = self.make_base()
        dataset = chips_rerank._build_reranker_dataset(base, "CHIPS_F_TOP5", top_k=5)
        pred = np.asarray([dataset.y_author_index[0], dataset.candidates[1, 0]], dtype=int)

        summary = chips_rerank._transition_summary_vs_chips_f(base, dataset, pred)

        self.assertIn("fixed_chips_f_errors", summary)
        self.assertIn("broken_chips_f_correct", summary)


if __name__ == "__main__":
    unittest.main()
