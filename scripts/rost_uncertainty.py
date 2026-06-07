#!/usr/bin/env python3
"""Compute uncertainty summaries for the locked ROST test predictions."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


MODEL_ROWS = [
    (
        "CHAR_NGRAM_2_5_SVM",
        "Char 2--5 TF--IDF SVM (matched)",
        "experiments/runs/rost_char_ngram_2_5_cv5/predictions_test.csv",
        "predicted_author",
        "char_ngram_pred",
    ),
    (
        "CH_SVM",
        "CH-SVM",
        "experiments/runs/rost_cv5_chips_r/chips_r_predictions_test.csv",
        "ch_svm_pred",
        "ch_svm_pred",
    ),
    (
        "FFT12_LR",
        "FFT12-LR",
        "experiments/runs/rost_cv5_chips_r/chips_r_predictions_test.csv",
        "fft12_lr_pred",
        "fft12_lr_pred",
    ),
    (
        "CHiPS_F",
        "CHiPS-F",
        "experiments/runs/rost_cv5_chips_r/chips_r_predictions_test.csv",
        "chips_f_pred",
        "chips_f_pred",
    ),
    (
        "CHiPS_R",
        "CHiPS-R (OOF ablation)",
        "experiments/runs/rost_cv5_chips_r/chips_r_predictions_test.csv",
        "chips_r_pred",
        "chips_r_pred",
    ),
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def wilson_interval(correct: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be positive")
    phat = correct / n
    denominator = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denominator
    half_width = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def exact_two_sided_sign_pvalue(a_better: int, b_better: int) -> float:
    """Exact two-sided binomial sign-test p-value over paired discordances."""
    n_discordant = a_better + b_better
    if n_discordant == 0:
        return 1.0
    tail = min(a_better, b_better)
    probability = 2.0 * sum(math.comb(n_discordant, i) for i in range(tail + 1)) / (2**n_discordant)
    return min(1.0, probability)


def stratified_bootstrap_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    *,
    n_samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    groups_by_author = [np.flatnonzero(y_true == label) for label in labels]
    if any(len(indices) == 0 for indices in groups_by_author):
        raise ValueError("Each label must have at least one held-out example for stratified bootstrap")

    values = np.empty(n_samples, dtype=float)
    for sample_index in range(n_samples):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in groups_by_author]
        )
        values[sample_index] = f1_score(
            y_true[sampled],
            y_pred[sampled],
            labels=labels,
            average="macro",
            zero_division=0,
        )
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def load_aligned_predictions(chips_r_predictions: Path, ngram_predictions: Path) -> list[dict[str, str]]:
    chips_rows = read_csv_rows(chips_r_predictions)
    ngram_by_group = {row["work_group"]: row for row in read_csv_rows(ngram_predictions)}

    missing = sorted({row["work_group"] for row in chips_rows} - set(ngram_by_group))
    extra = sorted(set(ngram_by_group) - {row["work_group"] for row in chips_rows})
    if missing or extra:
        raise ValueError(f"Prediction work_group mismatch. Missing n-gram={missing}; extra n-gram={extra}")

    aligned: list[dict[str, str]] = []
    for row in chips_rows:
        ngram = ngram_by_group[row["work_group"]]
        if row["true_author"] != ngram["true_author"]:
            raise ValueError(
                f"true_author mismatch for {row['work_group']}: "
                f"{row['true_author']} vs {ngram['true_author']}"
            )
        if row["doc_count"] != ngram["doc_count"]:
            raise ValueError(
                f"doc_count mismatch for {row['work_group']}: {row['doc_count']} vs {ngram['doc_count']}"
            )
        if row["doc_ids"] != ngram["doc_ids"]:
            raise ValueError(f"doc_ids mismatch for {row['work_group']}")
        merged = dict(row)
        merged["char_ngram_pred"] = ngram["predicted_author"]
        aligned.append(merged)
    return aligned


def validate_chips_f_predictions(rows: list[dict[str, str]], chips_f_predictions: Path) -> None:
    chips_f_by_group = {row["work_group"]: row for row in read_csv_rows(chips_f_predictions)}
    row_groups = {row["work_group"] for row in rows}
    missing = sorted(row_groups - set(chips_f_by_group))
    extra = sorted(set(chips_f_by_group) - row_groups)
    if missing or extra:
        raise ValueError(f"CHiPS-F work_group mismatch. Missing={missing}; extra={extra}")
    for row in rows:
        chips_f = chips_f_by_group[row["work_group"]]
        for field in ("true_author", "doc_count", "doc_ids"):
            if row[field] != chips_f[field]:
                raise ValueError(f"CHiPS-F {field} mismatch for {row['work_group']}")
        if row["chips_f_pred"] != chips_f["predicted_author"]:
            raise ValueError(
                f"CHiPS-F prediction mismatch for {row['work_group']}: "
                f"{row['chips_f_pred']} vs {chips_f['predicted_author']}"
            )


def metric_row(
    model_key: str,
    model_label: str,
    prediction_file: str,
    prediction_column_original: str,
    prediction_column: str,
    y_true: np.ndarray,
    rows: list[dict[str, str]],
    labels: list[str],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    y_pred = np.asarray([row[prediction_column] for row in rows], dtype=object)
    correct = int(np.sum(y_pred == y_true))
    n = int(len(y_true))
    accuracy_low, accuracy_high = wilson_interval(correct, n)
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    macro_f1_low, macro_f1_high = stratified_bootstrap_macro_f1(
        y_true,
        y_pred,
        labels,
        n_samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "model_key": model_key,
        "display_model": model_label,
        "prediction_file": prediction_file,
        "prediction_column": prediction_column_original,
        "n_groups": n,
        "correct": correct,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "accuracy_ci_method": "95% Wilson score interval",
        "accuracy_ci_low": accuracy_low,
        "accuracy_ci_high": accuracy_high,
        "macro_f1": macro_f1,
        "macro_f1_ci_method": "95% stratified bootstrap percentile interval",
        "macro_f1_ci_low": macro_f1_low,
        "macro_f1_ci_high": macro_f1_high,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def paired_comparison(
    name: str,
    model_a_key: str,
    model_a_label: str,
    model_a_column: str,
    model_b_key: str,
    model_b_label: str,
    model_b_column: str,
    y_true: np.ndarray,
    rows: list[dict[str, str]],
) -> dict[str, object]:
    pred_a = np.asarray([row[model_a_column] for row in rows], dtype=object)
    pred_b = np.asarray([row[model_b_column] for row in rows], dtype=object)
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    a_better = int(np.sum(correct_a & ~correct_b))
    b_better = int(np.sum(~correct_a & correct_b))
    both_correct = int(np.sum(correct_a & correct_b))
    both_wrong = int(np.sum(~correct_a & ~correct_b))
    return {
        "name": name,
        "model_a_key": model_a_key,
        "model_a_label": model_a_label,
        "model_b_key": model_b_key,
        "model_b_label": model_b_label,
        "model_a_correct_model_b_wrong": a_better,
        "model_a_wrong_model_b_correct": b_better,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "discordant": a_better + b_better,
        "model_a_accuracy": float(np.mean(correct_a)),
        "model_b_accuracy": float(np.mean(correct_b)),
        "accuracy_delta_model_a_minus_model_b": float(np.mean(correct_a) - np.mean(correct_b)),
        "two_sided_exact_binomial_p": exact_two_sided_sign_pvalue(a_better, b_better),
    }


def fmt4(value: object) -> str:
    return f"{float(value):.4f}"


def write_latex_table(path: Path, rows: list[dict[str, object]], paired_tests: dict) -> None:
    lines = [
        r"\begin{table}",
        r"  \tbl{\caption{Locked ROST uncertainty over the 58 held-out source-text groups. Accuracy intervals are 95\% Wilson score intervals; macro-F1 intervals are 95\% stratified bootstrap percentile intervals over source-text groups within author.}\label{tab:rost-uncertainty}}",
        r"  {\tablefont\begin{tabular}{lccc}",
        r"    \hline",
        r"    Model & Correct / 58 & Accuracy (95\% CI) & Macro-F1 (95\% CI)\\",
        r"    \hline",
    ]
    for row in rows:
        accuracy = f"{fmt4(row['accuracy'])} [{fmt4(row['accuracy_ci_low'])}, {fmt4(row['accuracy_ci_high'])}]"
        macro_f1 = f"{fmt4(row['macro_f1'])} [{fmt4(row['macro_f1_ci_low'])}, {fmt4(row['macro_f1_ci_high'])}]"
        lines.append(
            f"    {row['display_model']} & {row['correct']}/{row['n_groups']} & {accuracy} & {macro_f1}\\\\"
        )
    lines += [
        r"    \hline",
        r"  \end{tabular}}",
        r"  {}",
        r"\end{table}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_outputs(args: argparse.Namespace) -> None:
    rows = load_aligned_predictions(args.chips_r_predictions, args.ngram_predictions)
    validate_chips_f_predictions(rows, args.chips_f_predictions)
    y_true = np.asarray([row["true_author"] for row in rows], dtype=object)
    labels = sorted(set(y_true.tolist()))

    summary_rows = [
        metric_row(
            model_key,
            model_label,
            prediction_file,
            prediction_column_original,
            prediction_column,
            y_true,
            rows,
            labels,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        for model_key, model_label, prediction_file, prediction_column_original, prediction_column in MODEL_ROWS
    ]

    paired_tests = {
        "method": (
            "Exact two-sided binomial sign test over paired discordant "
            "source-text correctness outcomes; equivalent to exact McNemar "
            "for the paired binary correct/incorrect comparison."
        ),
        "unit": "locked ROST held-out source-text group",
        "n": int(len(rows)),
        "comparisons": [
            paired_comparison(
                "CHiPS-F vs CH-SVM",
                "CHiPS_F",
                "CHiPS-F",
                "chips_f_pred",
                "CH_SVM",
                "CH-SVM",
                "ch_svm_pred",
                y_true,
                rows,
            ),
            paired_comparison(
                "Char 2--5 TF--IDF SVM vs CHiPS-F",
                "CHAR_NGRAM_2_5_SVM",
                "Char 2--5 TF--IDF SVM",
                "char_ngram_pred",
                "CHiPS_F",
                "CHiPS-F",
                "chips_f_pred",
                y_true,
                rows,
            ),
        ],
    }

    metadata = {
        "chips_f_predictions": str(args.chips_f_predictions),
        "chips_r_predictions": str(args.chips_r_predictions),
        "ngram_predictions": str(args.ngram_predictions),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "accuracy_ci": "Wilson score interval, 95 percent",
        "macro_f1_ci": "Stratified nonparametric bootstrap percentile interval, 95 percent",
        "labels": labels,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv_rows(args.output / "uncertainty_table.csv", summary_rows)
    write_json(args.output / "paired_tests.json", paired_tests)
    write_json(args.output / "uncertainty_metadata.json", metadata)
    write_latex_table(args.output / "table_rost_uncertainty.tex", summary_rows, paired_tests)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chips-f-predictions",
        type=Path,
        default=Path("experiments/runs/rost_cv5_full/predictions_test.csv"),
        help="Full CHiPS-F source-group predictions used to cross-check the reranker master table.",
    )
    parser.add_argument(
        "--chips-r-predictions",
        type=Path,
        default=Path("experiments/runs/rost_cv5_chips_r/chips_r_predictions_test.csv"),
        help="Aligned CH-SVM, FFT12-LR, CHiPS-F, and CHiPS-R source-group predictions.",
    )
    parser.add_argument(
        "--ngram-predictions",
        type=Path,
        default=Path("experiments/runs/rost_char_ngram_2_5_cv5/predictions_test.csv"),
        help="Matched character 2--5 gram TF-IDF SVM source-group predictions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/rost_uncertainty"),
        help="Output directory for uncertainty artifacts.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    build_outputs(parse_args())
