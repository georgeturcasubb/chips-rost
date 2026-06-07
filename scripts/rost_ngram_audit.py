#!/usr/bin/env python3
"""Audit the perfect locked ROST character n-gram baseline."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chips.data import DocumentRecord, dataset_summary, load_documents
from chips.modeling import (
    CharNGramSVMConfig,
    aggregate_probabilities_by_work_group,
    evaluate_predictions,
    fit_char_ngram_svm,
    labels_from_proba,
    predict_char_ngram_svm_proba,
    software_versions,
)
from chips.splitting import apply_split, load_split


MASK_CHAR = "#"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strip_diacritics(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def normalize_for_matching(value: str) -> str:
    """Case/diacritic/punctuation-insensitive alphanumeric form."""
    value = _strip_diacritics(value).casefold()
    return "".join(ch for ch in value if ch.isalnum())


def normalized_text_with_index(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    indices: list[int] = []
    for index, ch in enumerate(text):
        normalized = normalize_for_matching(ch)
        for out_ch in normalized:
            chars.append(out_ch)
            indices.append(index)
    return "".join(chars), indices


def normalized_phrase_intervals(text: str, phrases: Sequence[str]) -> list[tuple[int, int, str]]:
    """Find phrase intervals in original text, ignoring case, diacritics, and punctuation."""
    normalized_text, index_map = normalized_text_with_index(text)
    if not normalized_text:
        return []
    intervals: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int, str]] = set()
    for phrase in phrases:
        normalized_phrase = normalize_for_matching(phrase)
        if len(normalized_phrase) < 4:
            continue
        start = 0
        while True:
            found = normalized_text.find(normalized_phrase, start)
            if found < 0:
                break
            end_pos = found + len(normalized_phrase) - 1
            original_start = index_map[found]
            original_end = index_map[end_pos] + 1
            item = (original_start, original_end, normalized_phrase)
            if item not in seen:
                seen.add(item)
                intervals.append(item)
            start = found + 1
    intervals.sort(key=lambda item: (item[0], item[1], item[2]))
    return intervals


def mask_intervals(text: str, intervals: Sequence[tuple[int, int, str]], mask_char: str = MASK_CHAR) -> str:
    if not intervals:
        return text
    mask = [False] * len(text)
    for start, end, _ in intervals:
        for index in range(max(0, start), min(len(text), end)):
            mask[index] = True
    return "".join(mask_char if flag else ch for ch, flag in zip(text, mask))


def candidate_title_phrases(record: DocumentRecord) -> list[str]:
    phrases = [record.title]
    group_title = record.work_group.split("__", 1)[1]
    if group_title not in phrases:
        phrases.append(group_title)
    return phrases


def mask_author_title_strings(
    record: DocumentRecord,
    candidate_authors: Sequence[str],
    *,
    mask_char: str = MASK_CHAR,
) -> tuple[str, dict[str, int]]:
    author_intervals = normalized_phrase_intervals(record.text, candidate_authors)
    title_intervals = normalized_phrase_intervals(record.text, candidate_title_phrases(record))
    masked = mask_intervals(record.text, author_intervals + title_intervals, mask_char=mask_char)
    return masked, {
        "author_interval_count": len(author_intervals),
        "title_interval_count": len(title_intervals),
        "masked_character_count": sum(1 for before, after in zip(record.text, masked) if before != after),
    }


def first_last_nonempty_lines(text: str) -> tuple[str, str]:
    lines = [line.strip() for line in text.splitlines()]
    nonempty = [line for line in lines if line]
    if not nonempty:
        return "", ""
    return nonempty[0], nonempty[-1]


def strip_first_last_nonempty_lines(text: str) -> str:
    lines = text.splitlines()
    nonempty_indices = [index for index, line in enumerate(lines) if line.strip()]
    if not nonempty_indices:
        return text
    drop = {nonempty_indices[0], nonempty_indices[-1]}
    return "\n".join(line for index, line in enumerate(lines) if index not in drop)


def text_records_with_transform(
    records: Sequence[DocumentRecord],
    transform: Callable[[DocumentRecord], str],
) -> list[DocumentRecord]:
    return [replace(record, text=transform(record)) for record in records]


def load_selected_ngram_config(path: Path) -> CharNGramSVMConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CharNGramSVMConfig(**payload["char_ngram_2_5_svm"])


def group_docs(records: Sequence[DocumentRecord]) -> dict[str, list[DocumentRecord]]:
    by_group: dict[str, list[DocumentRecord]] = defaultdict(list)
    for record in records:
        by_group[record.work_group].append(record)
    return dict(by_group)


def group_text(records: Sequence[DocumentRecord]) -> str:
    return "\n".join(record.text for record in sorted(records, key=lambda item: item.doc_id))


def doc_partition_map(split_docs: dict[str, Sequence[DocumentRecord]]) -> dict[str, str]:
    return {record.doc_id: split_name for split_name, docs in split_docs.items() for record in docs}


def group_partition_map(split_docs: dict[str, Sequence[DocumentRecord]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for split_name, docs in split_docs.items():
        for record in docs:
            previous = mapping.get(record.work_group)
            if previous is not None and previous != split_name:
                raise ValueError(f"Work group crosses partitions: {record.work_group}")
            mapping[record.work_group] = split_name
    return mapping


def exact_duplicate_summary(split_docs: dict[str, Sequence[DocumentRecord]]) -> tuple[list[dict[str, object]], dict]:
    by_hash: dict[str, list[tuple[str, DocumentRecord]]] = defaultdict(list)
    for split_name, docs in split_docs.items():
        for record in docs:
            by_hash[sha256_text(record.text)].append((split_name, record))

    rows: list[dict[str, object]] = []
    crossing_hashes = 0
    test_touching_hashes = 0
    for digest, items in sorted(by_hash.items()):
        partitions = sorted({split_name for split_name, _ in items})
        if len(items) <= 1 or len(partitions) <= 1:
            continue
        crossing_hashes += 1
        if "test" in partitions:
            test_touching_hashes += 1
        rows.append(
            {
                "sha256": digest,
                "partitions": ";".join(partitions),
                "touches_test": "yes" if "test" in partitions else "no",
                "doc_count": len(items),
                "doc_ids": ";".join(record.doc_id for _, record in items),
                "work_groups": ";".join(sorted({record.work_group for _, record in items})),
            }
        )
    summary = {
        "exact_duplicate_hashes_crossing_partitions": crossing_hashes,
        "exact_duplicate_hashes_touching_test": test_touching_hashes,
        "exact_duplicate_documents_crossing_partitions": sum(int(row["doc_count"]) for row in rows),
    }
    return rows, summary


def heldout_surface_rows(
    test_records: Sequence[DocumentRecord],
    candidate_authors: Sequence[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in test_records:
        self_author_intervals = normalized_phrase_intervals(record.text, [record.author])
        any_author_intervals = normalized_phrase_intervals(record.text, candidate_authors)
        title_intervals = normalized_phrase_intervals(record.text, candidate_title_phrases(record))
        first_line, last_line = first_last_nonempty_lines(record.text)
        rows.append(
            {
                "doc_id": record.doc_id,
                "work_group": record.work_group,
                "author": record.author,
                "char_count": len(record.text),
                "self_author_hit": bool(self_author_intervals),
                "any_candidate_author_hit": bool(any_author_intervals),
                "self_title_hit": bool(title_intervals),
                "first_line_self_author_hit": bool(normalized_phrase_intervals(first_line, [record.author])),
                "first_line_self_title_hit": bool(normalized_phrase_intervals(first_line, candidate_title_phrases(record))),
                "last_line_self_author_hit": bool(normalized_phrase_intervals(last_line, [record.author])),
                "last_line_self_title_hit": bool(normalized_phrase_intervals(last_line, candidate_title_phrases(record))),
                "self_author_hit_count": len(self_author_intervals),
                "candidate_author_hit_count": len(any_author_intervals),
                "self_title_hit_count": len(title_intervals),
            }
        )
    return rows


def group_count_with_flag(rows: Sequence[dict[str, object]], field: str) -> int:
    return len({str(row["work_group"]) for row in rows if row[field]})


def near_duplicate_pairs(
    split_docs: dict[str, Sequence[DocumentRecord]],
    *,
    top_n: int,
    high_similarity_threshold: float,
) -> tuple[list[dict[str, object]], dict]:
    trainval_records = list(split_docs["train"]) + list(split_docs["validation"])
    test_records = list(split_docs["test"])
    trainval_groups = group_docs(trainval_records)
    test_groups = group_docs(test_records)
    group_partitions = group_partition_map(split_docs)

    trainval_keys = sorted(trainval_groups)
    test_keys = sorted(test_groups)
    trainval_texts = [group_text(trainval_groups[key]) for key in trainval_keys]
    test_texts = [group_text(test_groups[key]) for key in test_keys]

    vectorizer = TfidfVectorizer(
        analyzer="char",
        lowercase=False,
        ngram_range=(5, 5),
        min_df=1,
        use_idf=True,
        norm="l2",
        dtype=np.float64,
    )
    matrix = vectorizer.fit_transform(trainval_texts + test_texts)
    train_matrix = matrix[: len(trainval_texts)]
    test_matrix = matrix[len(trainval_texts) :]
    similarities = cosine_similarity(test_matrix, train_matrix)

    candidate_rows: list[dict[str, object]] = []
    for test_index, test_key in enumerate(test_keys):
        best_order = np.argsort(similarities[test_index])[::-1]
        for train_index in best_order[: max(1, min(top_n, len(best_order)))]:
            test_docs = test_groups[test_key]
            train_docs = trainval_groups[trainval_keys[int(train_index)]]
            score = float(similarities[test_index, int(train_index)])
            candidate_rows.append(
                {
                    "test_work_group": test_key,
                    "test_author": test_docs[0].author,
                    "test_doc_ids": ";".join(record.doc_id for record in sorted(test_docs, key=lambda item: item.doc_id)),
                    "trainval_work_group": trainval_keys[int(train_index)],
                    "trainval_author": train_docs[0].author,
                    "trainval_partition": group_partitions[trainval_keys[int(train_index)]],
                    "trainval_doc_ids": ";".join(record.doc_id for record in sorted(train_docs, key=lambda item: item.doc_id)),
                    "same_author": test_docs[0].author == train_docs[0].author,
                    "cosine_char5_tfidf": score,
                    "test_char_count": len(group_text(test_docs)),
                    "trainval_char_count": len(group_text(train_docs)),
                }
            )
    candidate_rows.sort(key=lambda row: float(row["cosine_char5_tfidf"]), reverse=True)
    top_rows = []
    for rank, row in enumerate(candidate_rows[:top_n], start=1):
        top_rows.append({"rank": rank, **row})

    per_test_max = np.max(similarities, axis=1) if similarities.size else np.array([])
    summary = {
        "comparison_unit": "held-out source-text group versus train+validation source-text group",
        "similarity": "character 5-gram TF-IDF cosine",
        "test_groups": len(test_keys),
        "trainval_groups": len(trainval_keys),
        "max_similarity": float(np.max(per_test_max)) if per_test_max.size else math.nan,
        "median_of_test_group_max_similarity": float(np.median(per_test_max)) if per_test_max.size else math.nan,
        "pairs_at_or_above_threshold": int(np.sum(similarities >= high_similarity_threshold)),
        "high_similarity_threshold": high_similarity_threshold,
    }
    return top_rows, summary


def prediction_rows_for_variant(
    variant_name: str,
    test_records: Sequence[DocumentRecord],
    group_ids: Sequence[str],
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> list[dict[str, object]]:
    docs_by_group = group_docs(test_records)
    rows = []
    for group_id, true_author, predicted_author in zip(group_ids, y_true, y_pred):
        docs = docs_by_group[group_id]
        rows.append(
            {
                "variant": variant_name,
                "work_group": group_id,
                "doc_ids": ";".join(record.doc_id for record in sorted(docs, key=lambda item: item.doc_id)),
                "true_author": true_author,
                "predicted_author": predicted_author,
                "correct": true_author == predicted_author,
            }
        )
    return rows


def evaluate_fixed_variant(
    name: str,
    trainval_records: Sequence[DocumentRecord],
    test_records: Sequence[DocumentRecord],
    config: CharNGramSVMConfig,
    class_order: Sequence[str],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    model = fit_char_ngram_svm(trainval_records, config)
    probs = predict_char_ngram_svm_proba(model, test_records)
    group_ids, y_true, group_probs = aggregate_probabilities_by_work_group(test_records, probs)
    y_pred = labels_from_proba(group_probs, model.class_order)
    metrics = evaluate_predictions(y_true, y_pred, class_order)
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    row = {
        "variant": name,
        "correct": correct,
        "n_groups": len(y_true),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
    }
    return row, prediction_rows_for_variant(name, test_records, group_ids, y_true, y_pred)


def run_sensitivity(
    split_docs: dict[str, Sequence[DocumentRecord]],
    config: CharNGramSVMConfig,
    candidate_authors: Sequence[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict]:
    trainval = list(split_docs["train"]) + list(split_docs["validation"])
    test = list(split_docs["test"])
    class_order = sorted({record.author for record in trainval})

    def mask_transform(record: DocumentRecord) -> str:
        masked, counts = mask_author_title_strings(record, candidate_authors)
        return masked

    variants: list[tuple[str, Callable[[DocumentRecord], str]]] = [
        ("original_fixed_config", lambda record: record.text),
        ("mask_author_title_strings", mask_transform),
        ("strip_first_last_nonempty_lines", lambda record: strip_first_last_nonempty_lines(record.text)),
        (
            "mask_author_title_and_strip_lines",
            lambda record: strip_first_last_nonempty_lines(mask_transform(record)),
        ),
    ]

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    original_predictions: dict[str, str] | None = None
    for name, transform in variants:
        transformed_trainval = text_records_with_transform(trainval, transform)
        transformed_test = text_records_with_transform(test, transform)
        metrics, rows = evaluate_fixed_variant(name, transformed_trainval, transformed_test, config, class_order)
        if original_predictions is None:
            original_predictions = {str(row["work_group"]): str(row["predicted_author"]) for row in rows}
            metrics["changed_predictions_from_original"] = 0
        else:
            changed = [
                row
                for row in rows
                if original_predictions.get(str(row["work_group"])) != str(row["predicted_author"])
            ]
            metrics["changed_predictions_from_original"] = len(changed)
            metrics["changed_work_groups_from_original"] = ";".join(str(row["work_group"]) for row in changed)
        metric_rows.append(metrics)
        prediction_rows.extend(rows)

    masking_counts = [mask_author_title_strings(record, candidate_authors)[1] for record in trainval + test]
    masking_summary = {
        "documents_with_author_masks": sum(1 for row in masking_counts if row["author_interval_count"] > 0),
        "documents_with_title_masks": sum(1 for row in masking_counts if row["title_interval_count"] > 0),
        "total_author_intervals": sum(row["author_interval_count"] for row in masking_counts),
        "total_title_intervals": sum(row["title_interval_count"] for row in masking_counts),
        "total_masked_characters": sum(row["masked_character_count"] for row in masking_counts),
    }
    return metric_rows, prediction_rows, masking_summary


def make_diagnostic_rows(
    duplicate_summary: dict,
    near_duplicate_summary: dict,
    surface_rows: Sequence[dict[str, object]],
    sensitivity_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    test_docs = len(surface_rows)
    test_groups = len({row["work_group"] for row in surface_rows})
    sensitivity_by_name = {str(row["variant"]): row for row in sensitivity_rows}

    rows = [
        {
            "diagnostic": "Exact duplicate hashes crossing split partitions",
            "value": duplicate_summary["exact_duplicate_hashes_crossing_partitions"],
            "unit": "text hashes",
            "note": "Computed from full normalized document text.",
        },
        {
            "diagnostic": "Exact duplicate hashes touching held-out test",
            "value": duplicate_summary["exact_duplicate_hashes_touching_test"],
            "unit": "text hashes",
            "note": "Any train/validation/test crossing duplicate involving test.",
        },
        {
            "diagnostic": "Maximum held-out to train+validation similarity",
            "value": f"{near_duplicate_summary['max_similarity']:.4f}",
            "unit": "char 5-gram TF-IDF cosine",
            "note": "Maximum over held-out source-text groups.",
        },
        {
            "diagnostic": "Near-duplicate pairs above threshold",
            "value": near_duplicate_summary["pairs_at_or_above_threshold"],
            "unit": f"pairs >= {near_duplicate_summary['high_similarity_threshold']:.2f}",
            "note": "Held-out source-text group versus train+validation source-text group.",
        },
        {
            "diagnostic": "Held-out documents with self-author string",
            "value": sum(1 for row in surface_rows if row["self_author_hit"]),
            "unit": f"of {test_docs} documents",
            "note": "Case/diacritic/punctuation-insensitive normalized match.",
        },
        {
            "diagnostic": "Held-out groups with self-author string",
            "value": group_count_with_flag(surface_rows, "self_author_hit"),
            "unit": f"of {test_groups} groups",
            "note": "A group is flagged if any held-out document in it is flagged.",
        },
        {
            "diagnostic": "Held-out documents with self-title string",
            "value": sum(1 for row in surface_rows if row["self_title_hit"]),
            "unit": f"of {test_docs} documents",
            "note": "Title is detected after ignoring spaces, punctuation, case, and diacritics.",
        },
        {
            "diagnostic": "Held-out documents with first-line author/title hit",
            "value": sum(1 for row in surface_rows if row["first_line_self_author_hit"] or row["first_line_self_title_hit"]),
            "unit": f"of {test_docs} documents",
            "note": "Detectable source-header cue in first non-empty line.",
        },
        {
            "diagnostic": "Held-out documents with last-line author/title hit",
            "value": sum(1 for row in surface_rows if row["last_line_self_author_hit"] or row["last_line_self_title_hit"]),
            "unit": f"of {test_docs} documents",
            "note": "Detectable source-footer cue in last non-empty line.",
        },
    ]
    for variant_name in (
        "original_fixed_config",
        "mask_author_title_strings",
        "strip_first_last_nonempty_lines",
        "mask_author_title_and_strip_lines",
    ):
        row = sensitivity_by_name[variant_name]
        rows.append(
            {
                "diagnostic": f"Fixed selected n-gram SVM after {variant_name}",
                "value": f"{row['correct']}/{row['n_groups']}",
                "unit": "held-out groups correct",
                "note": f"accuracy={float(row['accuracy']):.4f}; no new test-based model selection.",
            }
        )
    return rows


def make_latex_table(path: Path, diagnostic_rows: Sequence[dict[str, object]]) -> None:
    by_name = {str(row["diagnostic"]): row for row in diagnostic_rows}
    first_line = by_name["Held-out documents with first-line author/title hit"]
    last_line = by_name["Held-out documents with last-line author/title hit"]
    rows = [
        (
            "Exact duplicate hashes touching test",
            by_name["Exact duplicate hashes touching held-out test"]["value"],
            "text hashes",
        ),
        (
            "Max held-out/train+validation similarity",
            by_name["Maximum held-out to train+validation similarity"]["value"],
            "char 5-gram TF--IDF cosine",
        ),
        (
            "Near-duplicate pairs ($\\geq 0.95$)",
            by_name["Near-duplicate pairs above threshold"]["value"],
            "pairs",
        ),
        (
            "Held-out docs with self-author string",
            by_name["Held-out documents with self-author string"]["value"],
            by_name["Held-out documents with self-author string"]["unit"],
        ),
        (
            "Held-out docs with self-title string",
            by_name["Held-out documents with self-title string"]["value"],
            by_name["Held-out documents with self-title string"]["unit"],
        ),
        (
            "First/last-line author-title hits",
            f"{first_line['value']} / {last_line['value']}",
            first_line["unit"],
        ),
        (
            "Original fixed selected n-gram SVM",
            by_name["Fixed selected n-gram SVM after original_fixed_config"]["value"],
            "groups correct",
        ),
        (
            "After masking author/title strings",
            by_name["Fixed selected n-gram SVM after mask_author_title_strings"]["value"],
            "groups correct",
        ),
        (
            "After stripping first/last non-empty lines",
            by_name["Fixed selected n-gram SVM after strip_first_last_nonempty_lines"]["value"],
            "groups correct",
        ),
        (
            "After both masking and line stripping",
            by_name["Fixed selected n-gram SVM after mask_author_title_and_strip_lines"]["value"],
            "groups correct",
        ),
    ]
    lines = [
        "\\begin{table}",
        "  \\tbl{\\caption{ROST audit for the perfect matched character 2--5-gram TF--IDF SVM result. Similarity is character 5-gram TF--IDF cosine between each held-out source-text group and train+validation source-text groups. Sensitivity rows keep the originally selected n-gram SVM configuration fixed and do not tune on held-out labels.}\\label{tab:rost-ngram-audit}}",
        "  {\\tablefont\\begin{tabular}{p{0.46\\textwidth}p{0.14\\textwidth}p{0.25\\textwidth}}",
        "    \\hline",
        "    Diagnostic & Value & Unit\\\\",
        "    \\hline",
    ]
    for label, value, unit in rows:
        lines.append(f"    {label} & {value} & {unit}\\\\")
    lines.extend(
        [
            "    \\hline",
            "  \\end{tabular}}",
            "  {}",
            "\\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the locked ROST character 2-5 gram baseline.")
    parser.add_argument("--input", type=Path, default=Path("data/ROST-NormElip"))
    parser.add_argument("--dataset", default="rost")
    parser.add_argument("--split-json", type=Path, default=Path("experiments/configs/rost_source_split_seed42.json"))
    parser.add_argument(
        "--selected-config",
        type=Path,
        default=Path("experiments/runs/rost_char_ngram_2_5_cv5/selected_config.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("experiments/results/rost_ngram_audit"))
    parser.add_argument("--top-near-duplicates", type=int, default=25)
    parser.add_argument("--high-similarity-threshold", type=float, default=0.95)
    args = parser.parse_args()

    records = load_documents(args.input, dataset=args.dataset)
    split_docs = apply_split(records, load_split(args.split_json))
    config = load_selected_ngram_config(args.selected_config)
    candidate_authors = sorted({record.author for record in records})

    duplicate_rows, duplicate_summary = exact_duplicate_summary(split_docs)
    surface_rows = heldout_surface_rows(split_docs["test"], candidate_authors)
    near_rows, near_summary = near_duplicate_pairs(
        split_docs,
        top_n=args.top_near_duplicates,
        high_similarity_threshold=args.high_similarity_threshold,
    )
    sensitivity_rows, sensitivity_prediction_rows, masking_summary = run_sensitivity(
        split_docs,
        config,
        candidate_authors,
    )
    diagnostic_rows = make_diagnostic_rows(
        duplicate_summary,
        near_summary,
        surface_rows,
        sensitivity_rows,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "diagnostic_table.csv", diagnostic_rows)
    write_csv(args.output / "near_duplicate_pairs.csv", near_rows)
    write_csv(args.output / "exact_duplicate_cross_split_hashes.csv", duplicate_rows)
    write_csv(args.output / "heldout_surface_cues.csv", surface_rows)
    write_csv(args.output / "sensitivity_predictions.csv", sensitivity_prediction_rows)
    write_json(
        args.output / "sensitivity_metrics.json",
        {
            "dataset_summary": dataset_summary(records),
            "split_json": str(args.split_json),
            "selected_config_json": str(args.selected_config),
            "fixed_selected_config": config.__dict__,
            "duplicate_summary": duplicate_summary,
            "near_duplicate_summary": near_summary,
            "masking_summary": masking_summary,
            "sensitivity": sensitivity_rows,
            "software_versions": software_versions(),
            "note": "Sensitivity variants keep the original grouped-CV-selected n-gram config fixed and do not use held-out labels for selection.",
        },
    )
    make_latex_table(args.output / "table_rost_ngram_audit.tex", diagnostic_rows)
    print(json.dumps({row["diagnostic"]: row["value"] for row in diagnostic_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
