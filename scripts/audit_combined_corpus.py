#!/usr/bin/env python3
"""Audit a corpus made by taking a primary corpus over a secondary corpus.

For CHiPS this is used to verify ROSTories-cleaned: all source-text groups from
ROST should keep the ROST files, and remaining groups should come from the
cleaned ro-stories export.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chips.data import DocumentRecord, load_documents
from chips.splitting import load_split


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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def by_group(records: Sequence[DocumentRecord]) -> dict[str, list[DocumentRecord]]:
    grouped: dict[str, list[DocumentRecord]] = defaultdict(list)
    for record in records:
        grouped[record.work_group].append(record)
    return {group: sorted(items, key=lambda r: r.doc_id) for group, items in grouped.items()}


def record_signature(record: DocumentRecord) -> dict:
    return {
        "doc_id": record.doc_id,
        "author": record.author,
        "title": record.title,
        "work_group": record.work_group,
        "char_count": len(record.text),
        "sha256": sha256_text(record.text),
    }


def group_signature(records: Sequence[DocumentRecord]) -> list[dict]:
    return [record_signature(record) for record in sorted(records, key=lambda r: r.doc_id)]


def signature_key(records: Sequence[DocumentRecord]) -> list[tuple[str, str]]:
    return [(row["doc_id"], row["sha256"]) for row in group_signature(records)]


def split_locations(split_payload: dict | None) -> dict[str, str]:
    if split_payload is None:
        return {}
    locations = {}
    for split_name, doc_ids in split_payload["doc_ids"].items():
        for doc_id in doc_ids:
            locations[doc_id] = split_name
    return locations


def duplicate_hash_rows(records: Sequence[DocumentRecord], locations: dict[str, str]) -> tuple[list[dict], list[dict]]:
    by_hash: dict[str, list[DocumentRecord]] = defaultdict(list)
    for record in records:
        by_hash[sha256_text(record.text)].append(record)

    duplicate_hashes = []
    rows = []
    for digest, items in sorted(by_hash.items()):
        if len(items) < 2:
            continue
        split_names = sorted({locations.get(item.doc_id, "") for item in items})
        duplicate_hashes.append(
            {
                "sha256": digest,
                "documents": len(items),
                "splits": [split for split in split_names if split],
                "touches_test": "test" in split_names,
                "doc_ids": [item.doc_id for item in sorted(items, key=lambda r: r.doc_id)],
            }
        )
        for item in sorted(items, key=lambda r: r.doc_id):
            rows.append(
                {
                    "sha256": digest,
                    "doc_id": item.doc_id,
                    "work_group": item.work_group,
                    "author": item.author,
                    "char_count": len(item.text),
                    "split": locations.get(item.doc_id, ""),
                }
            )
    return duplicate_hashes, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a primary-over-secondary combined text corpus.")
    parser.add_argument("--primary", type=Path, required=True, help="Corpus whose files win on source-text group overlap.")
    parser.add_argument("--secondary", type=Path, required=True, help="Corpus used for groups absent from the primary corpus.")
    parser.add_argument("--combined", type=Path, required=True, help="Combined corpus to verify.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-name", default="primary")
    parser.add_argument("--secondary-name", default="secondary")
    parser.add_argument("--combined-name", default="combined")
    parser.add_argument("--dataset", default="rostories")
    parser.add_argument("--split-json", type=Path)
    parser.add_argument("--normalize-text", action="store_true", help="Apply CHiPS text normalization before hashing/comparison.")
    args = parser.parse_args()

    primary = load_documents(args.primary, dataset=args.dataset)
    secondary = load_documents(args.secondary, dataset=args.dataset)
    combined = load_documents(args.combined, dataset=args.dataset)
    if args.normalize_text:
        from chips.normalize import normalize_text

        primary = [record.__class__(**{**record.__dict__, "text": normalize_text(record.text)}) for record in primary]
        secondary = [record.__class__(**{**record.__dict__, "text": normalize_text(record.text)}) for record in secondary]
        combined = [record.__class__(**{**record.__dict__, "text": normalize_text(record.text)}) for record in combined]

    primary_groups = by_group(primary)
    secondary_groups = by_group(secondary)
    combined_groups = by_group(combined)
    expected_groups = set(primary_groups) | set(secondary_groups)

    group_rows = []
    mismatches = []
    for group in sorted(expected_groups | set(combined_groups)):
        in_primary = group in primary_groups
        in_secondary = group in secondary_groups
        in_combined = group in combined_groups
        expected_source = args.primary_name if in_primary else args.secondary_name
        expected_records = primary_groups.get(group) if in_primary else secondary_groups.get(group, [])
        expected_key = signature_key(expected_records)
        combined_key = signature_key(combined_groups.get(group, []))
        status = "ok" if in_combined and expected_key == combined_key else "mismatch"
        if status != "ok":
            mismatches.append(
                {
                    "work_group": group,
                    "expected_source": expected_source,
                    "in_primary": in_primary,
                    "in_secondary": in_secondary,
                    "in_combined": in_combined,
                    "expected_doc_ids": [doc_id for doc_id, _ in expected_key],
                    "combined_doc_ids": [doc_id for doc_id, _ in combined_key],
                }
            )
        group_rows.append(
            {
                "work_group": group,
                "expected_source": expected_source,
                "status": status,
                "in_primary": in_primary,
                "in_secondary": in_secondary,
                "in_combined": in_combined,
                "expected_documents": len(expected_key),
                "combined_documents": len(combined_key),
                "expected_doc_ids": "|".join(doc_id for doc_id, _ in expected_key),
                "combined_doc_ids": "|".join(doc_id for doc_id, _ in combined_key),
            }
        )

    primary_doc_ids = {record.doc_id for record in primary}
    secondary_doc_ids = {record.doc_id for record in secondary}
    combined_doc_ids = {record.doc_id for record in combined}
    shared_doc_ids = primary_doc_ids & secondary_doc_ids
    primary_hash_by_doc = {record.doc_id: sha256_text(record.text) for record in primary}
    combined_hash_by_doc = {record.doc_id: sha256_text(record.text) for record in combined}
    primary_kept_on_shared_doc_id = [
        doc_id
        for doc_id in sorted(shared_doc_ids & combined_doc_ids)
        if combined_hash_by_doc.get(doc_id) == primary_hash_by_doc.get(doc_id)
    ]
    shared_doc_id_mismatches = [
        doc_id
        for doc_id in sorted(shared_doc_ids & combined_doc_ids)
        if combined_hash_by_doc.get(doc_id) != primary_hash_by_doc.get(doc_id)
    ]

    split_payload = load_split(args.split_json) if args.split_json else None
    duplicate_hashes, duplicate_rows = duplicate_hash_rows(combined, split_locations(split_payload))

    overlap_groups = set(primary_groups) & set(secondary_groups)
    overlap_in_combined = overlap_groups & set(combined_groups)
    overlap_match_primary = 0
    overlap_match_secondary = 0
    overlap_match_neither = []
    for group in sorted(overlap_in_combined):
        if signature_key(combined_groups[group]) == signature_key(primary_groups[group]):
            overlap_match_primary += 1
        elif signature_key(combined_groups[group]) == signature_key(secondary_groups[group]):
            overlap_match_secondary += 1
        else:
            overlap_match_neither.append(group)

    summary = {
        "combined_name": args.combined_name,
        "construction_rule": f"{args.primary_name} wins for overlapping source-text groups; {args.secondary_name} supplies remaining groups.",
        "normalized_text_comparison": args.normalize_text,
        "primary": {
            "name": args.primary_name,
            "documents": len(primary),
            "work_groups": len(primary_groups),
            "authors": len({record.author for record in primary}),
        },
        "secondary": {
            "name": args.secondary_name,
            "documents": len(secondary),
            "work_groups": len(secondary_groups),
            "authors": len({record.author for record in secondary}),
        },
        "combined": {
            "documents": len(combined),
            "work_groups": len(combined_groups),
            "authors": len({record.author for record in combined}),
            "characters": sum(len(record.text) for record in combined),
        },
        "overlap": {
            "work_groups_in_both_components": len(set(primary_groups) & set(secondary_groups)),
            "work_groups_primary_only": len(set(primary_groups) - set(secondary_groups)),
            "work_groups_secondary_only": len(set(secondary_groups) - set(primary_groups)),
            "exact_doc_ids_in_both_components": len(shared_doc_ids),
            "shared_doc_ids_kept_from_primary": len(primary_kept_on_shared_doc_id),
            "shared_doc_id_hash_mismatches": len(shared_doc_id_mismatches),
        },
        "verification": {
            "expected_work_groups": len(expected_groups),
            "actual_work_groups": len(combined_groups),
            "missing_expected_work_groups": len(expected_groups - set(combined_groups)),
            "extra_combined_work_groups": len(set(combined_groups) - expected_groups),
            "group_rule_mismatches": len(mismatches),
            "shared_doc_id_hash_mismatches": shared_doc_id_mismatches,
            "overlap_groups_matching_primary": overlap_match_primary,
            "overlap_groups_matching_secondary": overlap_match_secondary,
            "overlap_groups_matching_neither": len(overlap_match_neither),
            "overlap_groups_matching_neither_items": overlap_match_neither,
            "mismatches": mismatches[:50],
        },
        "duplicate_hashes": {
            "count": len(duplicate_hashes),
            "touching_test_count": sum(1 for row in duplicate_hashes if row["touches_test"]),
            "items": duplicate_hashes,
        },
    }

    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "combined_source_audit.json", summary)
    write_csv(args.output / "combined_group_sources.csv", group_rows)
    write_csv(args.output / "duplicate_hash_documents.csv", duplicate_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
