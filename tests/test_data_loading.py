from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chips.data import load_documents


class DataLoadingTests(unittest.TestCase):
    def test_zip_loader_skips_macos_resource_fork_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "sample.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("corpus/Author_Title.txt", "real text")
                archive.writestr("__MACOSX/corpus/._Author_Title.txt", "resource fork")

            records = load_documents(archive_path, dataset="rostories")

        self.assertEqual([record.doc_id for record in records], ["Author_Title.txt"])
        self.assertEqual(records[0].text, "real text")


if __name__ == "__main__":
    unittest.main()
