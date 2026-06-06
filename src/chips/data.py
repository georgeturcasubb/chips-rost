"""Dataset loading utilities for ROST and the extended Romanian stories corpus."""
from __future__ import annotations

import csv
import re
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class DocumentRecord:
    """A source text used for grouped authorship attribution."""

    doc_id: str
    source_path: str
    author: str
    title: str
    work_group: str
    text: str

    def short(self) -> dict:
        row = asdict(self)
        row.pop("text", None)
        row["char_count"] = len(self.text)
        return row


def _strip_ro_stories_suffix(stem: str) -> str:
    # Extended-corpus files often look like Name_Text-pNumber-wc.txt.
    return re.sub(r"-p\d+(?:-wc)?$", "", stem)


def parse_author_title(filename: str, dataset: str = "auto") -> tuple[str, str, str]:
    """Return (author, title, work_group) from a dataset filename.

    ROST files normally use Author_Title.txt. For the extended stories corpus,
    paragraph/chunk suffixes such as -p17-wc are removed before grouping.
    """
    stem = Path(filename).stem
    clean_stem = _strip_ro_stories_suffix(stem) if dataset in {"auto", "rostories", "extended"} else stem
    if "_" not in clean_stem:
        raise ValueError(f"Filename does not contain an author/title underscore: {filename}")
    author, title = clean_stem.split("_", 1)
    # Some older ROST files have title variants after a hyphen. Group them by base title.
    group_title = title.split("-", 1)[0]
    return author, title, f"{author}__{group_title}"


def build_record(filename: str, source_path: str, text: str, dataset: str = "auto") -> DocumentRecord:
    author, title, work_group = parse_author_title(filename, dataset=dataset)
    return DocumentRecord(
        doc_id=filename,
        source_path=source_path,
        author=author,
        title=title,
        work_group=work_group,
        text=text.replace("\r\n", "\n"),
    )


def load_documents(input_path: Path, dataset: str = "auto") -> list[DocumentRecord]:
    """Load .txt files from a directory or zip archive."""
    input_path = Path(input_path)
    if input_path.is_dir():
        return load_documents_from_directory(input_path, dataset=dataset)
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        return load_documents_from_zip(input_path, dataset=dataset)
    raise ValueError(f"Unsupported input path: {input_path}")


def _is_ignored_text_artifact(path: Path) -> bool:
    """Return true for macOS/resource-fork files that are not corpus texts."""
    return path.name.startswith("._") or path.name == ".DS_Store" or "__MACOSX" in path.parts


def load_documents_from_directory(root: Path, dataset: str = "auto") -> list[DocumentRecord]:
    records: list[DocumentRecord] = []
    for path in sorted(root.rglob("*.txt")):
        if not path.is_file() or _is_ignored_text_artifact(path):
            continue
        text = path.read_text(encoding="utf-8")
        records.append(build_record(path.name, path.as_posix(), text, dataset=dataset))
    if not records:
        raise ValueError(f"No .txt files found in {root}")
    return records


def load_documents_from_zip(archive_path: Path, dataset: str = "auto") -> list[DocumentRecord]:
    records: list[DocumentRecord] = []
    with zipfile.ZipFile(archive_path) as archive:
        infos = [
            info
            for info in archive.infolist()
            if not info.is_dir()
            and Path(info.filename).suffix.lower() == ".txt"
            and not _is_ignored_text_artifact(Path(info.filename))
        ]
        for info in sorted(infos, key=lambda item: Path(item.filename).name):
            text = archive.read(info).decode("utf-8")
            records.append(build_record(Path(info.filename).name, info.filename, text, dataset=dataset))
    if not records:
        raise ValueError(f"No .txt files found in {archive_path}")
    return records


def dataset_summary(records: Sequence[DocumentRecord]) -> dict:
    authors = sorted({r.author for r in records})
    groups = sorted({r.work_group for r in records})
    per_author: dict[str, dict[str, int]] = {}
    for author in authors:
        subset = [r for r in records if r.author == author]
        per_author[author] = {
            "documents": len(subset),
            "work_groups": len({r.work_group for r in subset}),
            "characters": sum(len(r.text) for r in subset),
        }
    return {
        "documents": len(records),
        "authors": len(authors),
        "work_groups": len(groups),
        "characters": sum(len(r.text) for r in records),
        "per_author": per_author,
    }


def write_records_csv(records: Sequence[DocumentRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.short() for r in records]
    fieldnames = ["doc_id", "source_path", "author", "title", "work_group", "char_count"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
