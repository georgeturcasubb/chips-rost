from __future__ import annotations

import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chips.data import DocumentRecord
from chips.modeling import (
    CharNGramSVMConfig,
    aggregate_probabilities_by_work_group,
    default_char_ngram_grid,
    fit_char_ngram_svm,
    labels_from_proba,
    predict_char_ngram_svm_proba,
)


def make_record(author: str, title: str, text: str) -> DocumentRecord:
    doc_id = f"{author}_{title}.txt"
    return DocumentRecord(
        doc_id=doc_id,
        source_path=doc_id,
        author=author,
        title=title,
        work_group=f"{author}__{title}",
        text=text,
    )


class CharNGramBaselineTests(unittest.TestCase):
    def test_default_grid_is_fixed_to_character_2_5_grams(self):
        grid = default_char_ngram_grid(quick=True)

        self.assertEqual(len(grid), 1)
        self.assertEqual(grid[0].ngram_min, 2)
        self.assertEqual(grid[0].ngram_max, 5)

    def test_fit_predict_uses_case_preserving_character_tfidf(self):
        records = [
            make_record("A", "one", "AAAA bbbb AAAA bbbb"),
            make_record("A", "two", "AAAA cccc AAAA cccc"),
            make_record("B", "one", "zzzz yyyy zzzz yyyy"),
            make_record("B", "two", "zzzz xxxx zzzz xxxx"),
        ]
        model = fit_char_ngram_svm(records, CharNGramSVMConfig(c=1.0, class_weight=None, min_df=1))

        self.assertEqual(model.vectorizer.analyzer, "char")
        self.assertFalse(model.vectorizer.lowercase)
        self.assertEqual(model.vectorizer.ngram_range, (2, 5))

        probs = predict_char_ngram_svm_proba(model, records)
        _, y_group, group_probs = aggregate_probabilities_by_work_group(records, probs)
        pred = labels_from_proba(group_probs, model.class_order)

        self.assertEqual(pred, y_group)


if __name__ == "__main__":
    unittest.main()
