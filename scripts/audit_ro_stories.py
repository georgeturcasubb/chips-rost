#!/usr/bin/env python3
"""Lightweight audit for public ro-stories / extended corpus consistency issues."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chips.data import load_documents


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit duplicate and grouping issues in an extended Romanian stories corpus.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="rostories")
    args = parser.parse_args()

    records = load_documents(args.input, dataset=args.dataset)
    by_group = defaultdict(list)
    by_hash = defaultdict(list)
    rows = []
    for r in records:
        digest = sha256_text(r.text)
        by_group[r.work_group].append(r)
        by_hash[digest].append(r)
        rows.append({
            "doc_id": r.doc_id,
            "author": r.author,
            "title": r.title,
            "work_group": r.work_group,
            "char_count": len(r.text),
            "sha256": digest,
            "author_name_in_text_casefold": str(r.author.casefold() in r.text.casefold()),
        })

    duplicate_groups = [group for group, docs in by_group.items() if len(docs) > 1]
    exact_duplicate_hashes = [digest for digest, docs in by_hash.items() if len(docs) > 1]
    different_size_duplicate_groups = []
    same_size_duplicate_groups = []
    for group in duplicate_groups:
        sizes = {len(r.text) for r in by_group[group]}
        if len(sizes) == 1:
            same_size_duplicate_groups.append(group)
        else:
            different_size_duplicate_groups.append(group)

    summary = {
        "documents": len(records),
        "authors": len({r.author for r in records}),
        "work_groups": len({r.work_group for r in records}),
        "duplicate_author_title_groups": len(duplicate_groups),
        "same_size_duplicate_groups": len(same_size_duplicate_groups),
        "different_size_duplicate_groups": len(different_size_duplicate_groups),
        "exact_duplicate_hashes": len(exact_duplicate_hashes),
        "documents_with_author_name_in_text_casefold": sum(row["author_name_in_text_casefold"] == "True" for row in rows),
        "note": "This audit is diagnostic; it does not define exclusions without a documented corpus rule.",
    }

    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "audit_summary.json", summary)
    write_csv(args.output / "audit_documents.csv", rows)
    write_json(args.output / "duplicate_groups.json", {
        "same_size_duplicate_groups": same_size_duplicate_groups,
        "different_size_duplicate_groups": different_size_duplicate_groups,
        "exact_duplicate_hashes": exact_duplicate_hashes,
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
