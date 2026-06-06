#!/usr/bin/env python3
"""Export a normalized flat text corpus from a CHiPS-supported input."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chips.data import load_documents
from chips.normalize import normalize_text


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export normalized .txt files from a dataset directory or zip archive.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="auto", choices=["auto", "rost", "rostories", "extended"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        existing_texts = sorted(args.output.glob("*.txt"))
        if existing_texts and not args.overwrite:
            raise SystemExit(f"{args.output} already contains .txt files; pass --overwrite to replace them.")
        if args.overwrite:
            for path in existing_texts:
                path.unlink()
    args.output.mkdir(parents=True, exist_ok=True)

    records = load_documents(args.input, dataset=args.dataset)
    seen_doc_ids = set()
    rows = []
    for record in sorted(records, key=lambda r: r.doc_id):
        if record.doc_id in seen_doc_ids:
            raise ValueError(f"Duplicate document id after flattening: {record.doc_id}")
        seen_doc_ids.add(record.doc_id)
        text = normalize_text(record.text)
        (args.output / record.doc_id).write_text(text, encoding="utf-8")
        rows.append(
            {
                "doc_id": record.doc_id,
                "source_path": record.source_path,
                "author": record.author,
                "title": record.title,
                "work_group": record.work_group,
                "input_char_count": len(record.text),
                "normalized_char_count": len(text),
            }
        )

    write_csv(args.output / "normalization_manifest.csv", rows)
    write_json(
        args.output / "normalization_summary.json",
        {
            "input": str(args.input),
            "output": str(args.output),
            "dataset": args.dataset,
            "documents": len(rows),
            "normalization": (
                "Unicode NFC; Romanian cedilla-to-comma conversion; quote, apostrophe, dash, "
                "ellipsis, space/tab, trailing-space, and repeated-newline normalization; "
                "case, punctuation, digits, and diacritics preserved."
            ),
        },
    )
    print(f"Exported {len(rows)} normalized documents to {args.output}")


if __name__ == "__main__":
    main()
