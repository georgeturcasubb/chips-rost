"""Leakage-safe grouped splitting."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from .data import DocumentRecord


def _allocate_group_counts(n_groups: int, validation_ratio: float, test_ratio: float) -> tuple[int, int]:
    if n_groups <= 1:
        return 0, 0
    n_test = max(1, int(round(n_groups * test_ratio)))
    n_validation = max(1, int(round(n_groups * validation_ratio)))
    while n_test + n_validation > n_groups - 1:
        if n_validation >= n_test and n_validation > 1:
            n_validation -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
    return n_validation, n_test


def assert_no_group_leakage(split_docs: dict[str, Sequence[DocumentRecord]]) -> None:
    split_groups = {name: {r.work_group for r in docs} for name, docs in split_docs.items()}
    names = sorted(split_groups)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = split_groups[left] & split_groups[right]
            if overlap:
                raise AssertionError(f"Group leakage between {left} and {right}: {sorted(overlap)}")


def make_grouped_split(
    records: Sequence[DocumentRecord],
    seed: int = 42,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[dict[str, list[DocumentRecord]], dict]:
    """Create a per-author source-text grouped train/validation/test split."""
    if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.")

    by_author_group: dict[str, dict[str, list[DocumentRecord]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_author_group[record.author][record.work_group].append(record)

    rng = random.Random(seed)
    split_docs: dict[str, list[DocumentRecord]] = {"train": [], "validation": [], "test": []}
    author_summary: dict[str, dict[str, dict[str, object]]] = {}

    for author in sorted(by_author_group):
        groups = sorted(by_author_group[author])
        rng.shuffle(groups)
        n_validation, n_test = _allocate_group_counts(len(groups), validation_ratio, test_ratio)
        test_groups = groups[:n_test]
        validation_groups = groups[n_test : n_test + n_validation]
        train_groups = groups[n_test + n_validation :]
        plan = {"train": train_groups, "validation": validation_groups, "test": test_groups}
        author_summary[author] = {}
        for split_name, assigned_groups in plan.items():
            docs_for_split: list[DocumentRecord] = []
            for group in assigned_groups:
                docs_for_split.extend(sorted(by_author_group[author][group], key=lambda r: r.doc_id))
            split_docs[split_name].extend(docs_for_split)
            author_summary[author][split_name] = {
                "document_count": len(docs_for_split),
                "work_group_count": len(assigned_groups),
                "work_groups": assigned_groups,
                "documents": [r.doc_id for r in docs_for_split],
            }

    assert_no_group_leakage(split_docs)
    summary = {
        "seed": seed,
        "ratios": {"train": train_ratio, "validation": validation_ratio, "test": test_ratio},
        "totals": {
            "documents": len(records),
            "work_groups": len({r.work_group for r in records}),
            "train_documents": len(split_docs["train"]),
            "validation_documents": len(split_docs["validation"]),
            "test_documents": len(split_docs["test"]),
            "train_work_groups": len({r.work_group for r in split_docs["train"]}),
            "validation_work_groups": len({r.work_group for r in split_docs["validation"]}),
            "test_work_groups": len({r.work_group for r in split_docs["test"]}),
        },
        "authors": author_summary,
        "global_work_groups": {
            name: sorted({r.work_group for r in docs}) for name, docs in split_docs.items()
        },
    }
    return split_docs, summary


def split_to_doc_ids(split_docs: dict[str, Sequence[DocumentRecord]]) -> dict[str, list[str]]:
    return {name: [r.doc_id for r in docs] for name, docs in split_docs.items()}


def save_split(summary: dict, split_docs: dict[str, Sequence[DocumentRecord]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**summary, "doc_ids": split_to_doc_ids(split_docs)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def apply_split(records: Sequence[DocumentRecord], split_payload: dict) -> dict[str, list[DocumentRecord]]:
    by_doc = {r.doc_id: r for r in records}
    result: dict[str, list[DocumentRecord]] = {}
    for name, doc_ids in split_payload["doc_ids"].items():
        missing = [doc_id for doc_id in doc_ids if doc_id not in by_doc]
        if missing:
            raise ValueError(f"Split references missing documents in {name}: {missing[:5]}")
        result[name] = [by_doc[doc_id] for doc_id in doc_ids]
    assert_no_group_leakage(result)
    return result


def load_split(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
