from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.chips_r_detail import (  # noqa: E402
    feature_family_counts,
    feature_family_phrase,
    transition_summary,
    write_latex_table,
)


class ChipsRDetailTests(unittest.TestCase):
    def toy_rows(self) -> list[dict[str, str]]:
        return [
            {
                "true_author": "A",
                "ch_svm_pred": "B",
                "chips_f_pred": "A",
                "chips_r_pred": "A",
            },
            {
                "true_author": "B",
                "ch_svm_pred": "B",
                "chips_f_pred": "B",
                "chips_r_pred": "A",
            },
            {
                "true_author": "C",
                "ch_svm_pred": "A",
                "chips_f_pred": "A",
                "chips_r_pred": "B",
            },
            {
                "true_author": "D",
                "ch_svm_pred": "A",
                "chips_f_pred": "A",
                "chips_r_pred": "A",
            },
        ]

    def test_transition_summary_counts_fixed_broken_and_wrong_to_wrong(self):
        summary = transition_summary(self.toy_rows(), "CHiPS_F", "chips_f_pred")

        self.assertEqual(summary["test_groups"], 4)
        self.assertEqual(summary["comparator_correct"], 2)
        self.assertEqual(summary["chips_r_correct"], 1)
        self.assertEqual(summary["fixed_errors"], 0)
        self.assertEqual(summary["broken_correct"], 1)
        self.assertEqual(summary["wrong_to_wrong_changes"], 1)
        self.assertEqual(summary["unchanged_comparator_errors"], 1)

    def test_feature_family_phrase_includes_expected_groups(self):
        counts = feature_family_counts(
            [
                "log_doc_count",
                "CH_SVM_top1_prob",
                "CH_SVM_eq_CHiPS_F_top1",
                "pos1_author_A",
                "pos1_CH_SVM_prob",
                "pos1_CH_SVM_rank",
                "pos1_CH_SVM_gap_from_top1",
                "pos1_top1_vote_count",
            ]
        )
        phrase = feature_family_phrase(8, counts)

        self.assertIn("8:", phrase)
        self.assertIn("document size", phrase)
        self.assertIn("candidate probabilities", phrase)
        self.assertIn("author-position one-hot", phrase)

    def test_latex_table_contains_transition_and_optional_framing(self):
        row = {
            "corpus": "Locked ROST",
            "C": 0.03,
            "class_weight": "none",
            "non_anchor_weight": 4.0,
            "policy_display": "CH-SVM top-5",
            "top_k": 5,
            "gate_display": "anchor margin <= 0.50; meta margin >= 0.20",
            "feature_families": "215: document size; base confidence",
            "chips_r_correct": 53,
            "chips_f_correct": 54,
            "test_work_groups": 58,
            "fixes_vs_chips_f": 0,
            "breaks_vs_chips_f": 1,
            "wrong_to_wrong_vs_chips_f": 0,
            "conclusion": "does not improve CHiPS-F (-1 source-text groups); primary ablation",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "table.tex"
            write_latex_table(path, [row])
            text = path.read_text(encoding="utf-8")

        self.assertIn("Selected CHiPS-R reranker details", text)
        self.assertIn("53/58 vs 54/58", text)
        self.assertIn("primary ablation", text)


if __name__ == "__main__":
    unittest.main()
