from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from chips.data import DocumentRecord
from scripts.rost_ngram_audit import (
    mask_author_title_strings,
    normalize_for_matching,
    normalized_phrase_intervals,
    strip_first_last_nonempty_lines,
)


def make_record(author: str, title: str, text: str) -> DocumentRecord:
    return DocumentRecord(
        doc_id=f"{author}_{title}.txt",
        source_path=f"{author}_{title}.txt",
        author=author,
        title=title,
        work_group=f"{author}__{title.split('-', 1)[0]}",
        text=text,
    )


class RostNgramAuditTests(unittest.TestCase):
    def test_normalized_matching_ignores_case_diacritics_and_punctuation(self):
        text = "Povestea lui Ionică, cel prost"
        intervals = normalized_phrase_intervals(text, ["PovesteaLuiIonicaCelProst"])

        self.assertEqual(len(intervals), 1)
        self.assertEqual(text[intervals[0][0] : intervals[0][1]], text)

    def test_author_title_masking_preserves_length_and_removes_normalized_cues(self):
        record = make_record(
            "Creanga",
            "PovesteaLuiIonicaCelProst",
            "Ion Creangă\nPovestea lui Ionică, cel prost\ntext body",
        )

        masked, counts = mask_author_title_strings(record, ["Creanga", "Eminescu"])

        self.assertEqual(len(masked), len(record.text))
        self.assertGreaterEqual(counts["author_interval_count"], 1)
        self.assertGreaterEqual(counts["title_interval_count"], 1)
        self.assertNotIn("creanga", normalize_for_matching(masked))
        self.assertNotIn("povestealuiionicacelprost", normalize_for_matching(masked))

    def test_strip_first_last_nonempty_lines_preserves_middle_blank_lines(self):
        text = "\nHeader\n\nbody one\nbody two\n\nFooter\n"

        stripped = strip_first_last_nonempty_lines(text)

        self.assertNotIn("Header", stripped)
        self.assertNotIn("Footer", stripped)
        self.assertIn("body one", stripped)
        self.assertIn("body two", stripped)


if __name__ == "__main__":
    unittest.main()
