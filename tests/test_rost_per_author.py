from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rost_per_author import (  # noqa: E402
    PredictionRecord,
    chips_f_confusion_text,
    confusion_matrix,
    per_author_metrics,
    write_latex_table,
)


class RostPerAuthorTests(unittest.TestCase):
    def toy_records(self) -> list[PredictionRecord]:
        return [
            PredictionRecord(
                "A__one",
                "A",
                {
                    "CHAR_NGRAM_2_5_SVM": "A",
                    "CH_SVM": "A",
                    "FFT12_LR": "B",
                    "CHiPS_F": "A",
                    "CHiPS_R": "A",
                },
            ),
            PredictionRecord(
                "A__two",
                "A",
                {
                    "CHAR_NGRAM_2_5_SVM": "A",
                    "CH_SVM": "B",
                    "FFT12_LR": "A",
                    "CHiPS_F": "B",
                    "CHiPS_R": "B",
                },
            ),
            PredictionRecord(
                "B__one",
                "B",
                {
                    "CHAR_NGRAM_2_5_SVM": "B",
                    "CH_SVM": "B",
                    "FFT12_LR": "B",
                    "CHiPS_F": "B",
                    "CHiPS_R": "B",
                },
            ),
        ]

    def test_per_author_metrics_count_support_and_errors(self):
        rows = per_author_metrics(self.toy_records(), ["A", "B"])
        chips_f_a = next(row for row in rows if row["model"] == "CHiPS_F" and row["author"] == "A")

        self.assertEqual(chips_f_a["support"], 2)
        self.assertEqual(chips_f_a["correct"], 1)
        self.assertEqual(chips_f_a["errors"], 1)
        self.assertAlmostEqual(chips_f_a["recall"], 0.5)

    def test_confusion_matrix_uses_true_rows_and_predicted_columns(self):
        matrix = confusion_matrix(self.toy_records(), ["A", "B"], "CHiPS_F")

        self.assertEqual(matrix[0]["true_author"], "A")
        self.assertEqual(matrix[0]["A"], 1)
        self.assertEqual(matrix[0]["B"], 1)
        self.assertEqual(matrix[1]["B"], 1)

    def test_confusion_text_lists_nonzero_chips_f_errors(self):
        text = chips_f_confusion_text(self.toy_records(), "A")

        self.assertEqual(text, "1 as B")

    def test_latex_table_contains_author_support_and_confusion_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "table.tex"
            write_latex_table(path, self.toy_records(), ["A", "B"])
            text = path.read_text(encoding="utf-8")

        self.assertIn("Per-author locked ROST", text)
        self.assertIn("CHiPS-F confusions", text)
        self.assertIn("A & 2", text)


if __name__ == "__main__":
    unittest.main()
