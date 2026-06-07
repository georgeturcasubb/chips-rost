from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from chips.data import DocumentRecord
from chips.features import CharHistogramVectorizer
from scripts.rost_interpretability import (
    aligned_coefficients,
    character_category,
    feature_metadata,
    selected_indices_for_variant,
)
from sklearn.svm import LinearSVC


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


class RostInterpretabilityTests(unittest.TestCase):
    def test_character_categories_cover_surface_cues(self):
        self.assertEqual(character_category(","), "punctuation")
        self.assertEqual(character_category(" "), "whitespace")
        self.assertEqual(character_category("5"), "digit")
        self.assertEqual(character_category("ă"), "diacritic")
        self.assertEqual(character_category("a"), "letter")

    def test_variant_selection_removes_expected_feature_groups(self):
        vectorizer = CharHistogramVectorizer(min_char_count=1).fit(
            ["Aa 11, șș text", "Bb 22! ăă text"]
        )
        metadata = feature_metadata(vectorizer)

        no_punctuation = selected_indices_for_variant(metadata, "no_punctuation")
        no_digits = selected_indices_for_variant(metadata, "no_digits")
        letters_only = selected_indices_for_variant(metadata, "letters_only")

        self.assertTrue(all(metadata[index].category != "punctuation" for index in no_punctuation))
        self.assertTrue(all(metadata[index].category != "digit" for index in no_digits))
        self.assertTrue(all(metadata[index].kind == "char_frequency" for index in letters_only))
        self.assertTrue(all(metadata[index].is_alpha for index in letters_only))

    def test_binary_linear_svc_coefficients_are_expanded_to_class_order(self):
        X = np.asarray([[0.0], [0.1], [1.0], [1.1]])
        y = np.asarray(["A", "A", "B", "B"], dtype=object)
        classifier = LinearSVC(C=1.0, dual=False, max_iter=20000).fit(X, y)

        coef = aligned_coefficients(classifier, ["A", "B"])

        self.assertEqual(coef.shape, (2, 1))
        self.assertAlmostEqual(float(coef[0, 0]), -float(coef[1, 0]))


if __name__ == "__main__":
    unittest.main()
