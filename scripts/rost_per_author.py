#!/usr/bin/env python3
"""Export locked ROST per-author support and confusion diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]

MODEL_COLUMNS = [
    ("CHAR_NGRAM_2_5_SVM", "Char 2--5 TF-IDF SVM", "predicted_author"),
    ("CH_SVM", "CH-SVM", "ch_svm_pred"),
    ("FFT12_LR", "FFT12-LR", "fft12_lr_pred"),
    ("CHiPS_F", "CHiPS-F", "chips_f_pred"),
    ("CHiPS_R", "CHiPS-R", "chips_r_pred"),
]


@dataclass(frozen=True)
class PredictionRecord:
    work_group: str
    true_author: str
    predictions: dict[str, str]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_records(chips_r_predictions: Path, char_ngram_predictions: Path) -> list[PredictionRecord]:
    chips_rows = read_csv(chips_r_predictions)
    ngram_rows = read_csv(char_ngram_predictions)

    ngram_by_group = {row["work_group"]: row for row in ngram_rows}
    if len(ngram_by_group) != len(ngram_rows):
        raise ValueError(f"duplicate work_group in {char_ngram_predictions}")

    records: list[PredictionRecord] = []
    missing_groups: list[str] = []
    for row in chips_rows:
        work_group = row["work_group"]
        ngram_row = ngram_by_group.get(work_group)
        if ngram_row is None:
            missing_groups.append(work_group)
            continue
        if row["true_author"] != ngram_row["true_author"]:
            raise ValueError(
                f"true_author mismatch for {work_group}: "
                f"{row['true_author']} != {ngram_row['true_author']}"
            )
        records.append(
            PredictionRecord(
                work_group=work_group,
                true_author=row["true_author"],
                predictions={
                    "CHAR_NGRAM_2_5_SVM": ngram_row["predicted_author"],
                    "CH_SVM": row["ch_svm_pred"],
                    "FFT12_LR": row["fft12_lr_pred"],
                    "CHiPS_F": row["chips_f_pred"],
                    "CHiPS_R": row["chips_r_pred"],
                },
            )
        )

    extra_groups = sorted(set(ngram_by_group) - {row["work_group"] for row in chips_rows})
    if missing_groups or extra_groups:
        raise ValueError(
            "prediction files do not describe the same held-out work groups: "
            f"missing in n-gram={missing_groups}; extra in n-gram={extra_groups}"
        )
    return records


def sorted_authors(records: Iterable[PredictionRecord]) -> list[str]:
    return sorted({record.true_author for record in records})


def safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def per_author_metrics(records: list[PredictionRecord], authors: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    support = Counter(record.true_author for record in records)

    for model_key, display_model, _ in MODEL_COLUMNS:
        predictions = {record.work_group: record.predictions[model_key] for record in records}
        for author in authors:
            tp = sum(
                1
                for record in records
                if record.true_author == author and predictions[record.work_group] == author
            )
            fp = sum(
                1
                for record in records
                if record.true_author != author and predictions[record.work_group] == author
            )
            fn = support[author] - tp
            precision = safe_divide(tp, tp + fp)
            recall = safe_divide(tp, support[author])
            f1 = safe_divide(2 * precision * recall, precision + recall)
            rows.append(
                {
                    "model": model_key,
                    "display_model": display_model,
                    "author": author,
                    "support": support[author],
                    "correct": tp,
                    "errors": fn,
                    "false_positives": fp,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
    return rows


def confusion_matrix(records: list[PredictionRecord], authors: list[str], model_key: str) -> list[dict[str, object]]:
    counts: dict[str, Counter[str]] = {author: Counter() for author in authors}
    for record in records:
        counts[record.true_author][record.predictions[model_key]] += 1
    return [
        {"true_author": author, **{predicted: counts[author][predicted] for predicted in authors}}
        for author in authors
    ]


def confusion_pairs(records: list[PredictionRecord]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for record in records:
        for model_key, _, _ in MODEL_COLUMNS:
            predicted = record.predictions[model_key]
            if predicted != record.true_author:
                grouped[(model_key, record.true_author, predicted)].append(record.work_group)

    rows: list[dict[str, object]] = []
    display_by_key = {model_key: display_model for model_key, display_model, _ in MODEL_COLUMNS}
    for (model_key, true_author, predicted_author), groups in sorted(grouped.items()):
        rows.append(
            {
                "model": model_key,
                "display_model": display_by_key[model_key],
                "true_author": true_author,
                "predicted_author": predicted_author,
                "count": len(groups),
                "work_groups": ";".join(sorted(groups)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def chips_f_confusion_text(records: list[PredictionRecord], author: str) -> str:
    counts = Counter(
        record.predictions["CHiPS_F"]
        for record in records
        if record.true_author == author and record.predictions["CHiPS_F"] != author
    )
    if not counts:
        return "--"
    return "; ".join(f"{count} as {predicted}" for predicted, count in sorted(counts.items()))


def correct_count(records: list[PredictionRecord], author: str, model_key: str) -> int:
    return sum(
        1
        for record in records
        if record.true_author == author and record.predictions[model_key] == author
    )


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def write_latex_table(path: Path, records: list[PredictionRecord], authors: list[str]) -> None:
    lines = [
        r"\begin{table}",
        r"  \tbl{\caption{Per-author locked ROST held-out support and correct source-text groups. All columns use the same 58 held-out source-text groups. The last column lists nonzero CHiPS-F confusions by predicted author.}\label{tab:rost-per-author}}",
        r"  {\tablefont\begin{tabular}{lccccccp{0.20\textwidth}}",
        r"    \hline",
        r"    Author & Support & Char 2--5 & CH-SVM & FFT12-LR & CHiPS-F & CHiPS-R & CHiPS-F confusions\\",
        r"    \hline",
    ]
    support = Counter(record.true_author for record in records)
    for author in authors:
        values = [
            latex_escape(author),
            str(support[author]),
            str(correct_count(records, author, "CHAR_NGRAM_2_5_SVM")),
            str(correct_count(records, author, "CH_SVM")),
            str(correct_count(records, author, "FFT12_LR")),
            str(correct_count(records, author, "CHiPS_F")),
            str(correct_count(records, author, "CHiPS_R")),
            latex_escape(chips_f_confusion_text(records, author)),
        ]
        lines.append("    " + " & ".join(values) + r"\\")
    lines.extend(
        [
            r"    \hline",
            r"  \end{tabular}}",
            r"  {\begin{tabnote}Protocol: same locked ROST grouped split, seed 42, and held-out source-text predictions as Table~\ref{tab:rost-results}; generated from the released prediction CSV files.\end{tabnote}}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_metadata(path: Path, args: argparse.Namespace, records: list[PredictionRecord]) -> None:
    payload = {
        "chips_r_predictions": display_path(args.chips_r_predictions),
        "char_ngram_predictions": display_path(args.char_ngram_predictions),
        "n_groups": len(records),
        "models": [
            {"model": model_key, "display_model": display_model, "prediction_column": column}
            for model_key, display_model, column in MODEL_COLUMNS
        ],
        "main_confusion_model": "CHiPS_F",
        "unit": "held-out ROST source-text group",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chips-r-predictions",
        type=Path,
        default=ROOT / "experiments" / "runs" / "rost_cv5_chips_r" / "chips_r_predictions_test.csv",
    )
    parser.add_argument(
        "--char-ngram-predictions",
        type=Path,
        default=ROOT / "experiments" / "runs" / "rost_char_ngram_2_5_cv5" / "predictions_test.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "results" / "per_author",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.chips_r_predictions, args.char_ngram_predictions)
    authors = sorted_authors(records)
    metrics = per_author_metrics(records, authors)
    matrix = confusion_matrix(records, authors, "CHiPS_F")
    pairs = confusion_pairs(records)

    write_csv(
        args.output / "rost_per_author_metrics.csv",
        metrics,
        [
            "model",
            "display_model",
            "author",
            "support",
            "correct",
            "errors",
            "false_positives",
            "precision",
            "recall",
            "f1",
        ],
    )
    write_csv(args.output / "rost_confusion_matrix.csv", matrix, ["true_author", *authors])
    write_csv(
        args.output / "rost_confusion_pairs.csv",
        pairs,
        ["model", "display_model", "true_author", "predicted_author", "count", "work_groups"],
    )
    write_latex_table(args.output / "table_rost_per_author.tex", records, authors)
    write_metadata(args.output / "per_author_metadata.json", args, records)
    print(args.output)


if __name__ == "__main__":
    main()
