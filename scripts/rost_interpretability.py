#!/usr/bin/env python3
"""Generate locked-ROST CH-SVM interpretability and surface-cue ablations."""
from __future__ import annotations

import argparse
import csv
import json
import string
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chips.data import DocumentRecord, dataset_summary, load_documents
from chips.features import CharHistogramVectorizer, ROMANIAN_DIACRITICS, chunk_text, display_char, is_punctuation
from chips.modeling import CHSVMConfig, aggregate_probabilities_by_work_group, labels_from_proba, software_versions
from chips.splitting import apply_split, load_split


SCALAR_CATEGORIES = {
    "char_entropy": "distribution",
    "space_ratio": "whitespace",
    "newline_ratio": "whitespace",
    "uppercase_ratio": "case",
    "digit_ratio": "digit",
    "punctuation_ratio": "punctuation",
    "romanian_diacritic_ratio": "diacritic",
    "log1p_length": "length",
}
SCALAR_FEATURES = list(SCALAR_CATEGORIES)
TOP_FEATURES_PER_AUTHOR = 5


@dataclass(frozen=True)
class FeatureMeta:
    index: int
    name: str
    label: str
    latex_label: str
    category: str
    kind: str
    character: str
    is_alpha: bool
    is_uppercase: bool
    is_diacritic: bool


@dataclass
class MaskedCHSVMModel:
    config: CHSVMConfig
    vectorizer: CharHistogramVectorizer
    selected_indices: np.ndarray
    scaler: StandardScaler
    classifier: LinearSVC
    class_order: list[str]


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


def load_selected_ch_svm_config(path: Path) -> CHSVMConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CHSVMConfig(**payload["ch_svm"])


def character_category(ch: str) -> str:
    if ch.isspace():
        return "whitespace"
    if ch.isdigit():
        return "digit"
    if ch in ROMANIAN_DIACRITICS:
        return "diacritic"
    if is_punctuation(ch):
        return "punctuation"
    if ch.isalpha():
        return "letter"
    return "symbol"


def latex_char_label(ch: str) -> str:
    replacements = {
        " ": "space",
        "\n": "newline",
        "\t": "tab",
        '"': "quote",
        "'": "apostrophe",
        "\\": "backslash",
        ",": "comma",
        ".": "period",
        ":": "colon",
        ";": "semicolon",
        "!": "exclamation mark",
        "?": "question mark",
        "-": "dash",
        "…": "ellipsis",
        "{": "left brace",
        "}": "right brace",
        "$": "dollar sign",
        "%": "percent sign",
        "&": "ampersand",
        "#": "hash sign",
        "_": "underscore",
        "^": "caret",
        "~": "tilde",
        "ă": "a-breve",
        "Ă": "A-breve",
        "â": "a-circumflex",
        "Â": "A-circumflex",
        "î": "i-circumflex",
        "Î": "I-circumflex",
        "ș": "s-comma",
        "Ș": "S-comma",
        "ţ": "t-cedilla",
        "Ţ": "T-cedilla",
        "ț": "t-comma",
        "Ț": "T-comma",
    }
    if ch in replacements:
        return replacements[ch]
    if ch in string.punctuation:
        return f"punctuation {ch}"
    if ch.isprintable():
        return ch
    return f"U+{ord(ch):04X}"


def latex_feature_label(meta: FeatureMeta) -> str:
    return meta.latex_label


def feature_metadata(vectorizer: CharHistogramVectorizer) -> list[FeatureMeta]:
    if vectorizer.vocabulary is None:
        raise RuntimeError("Vectorizer must be fitted before feature metadata can be built.")
    rows: list[FeatureMeta] = []
    for index, ch in enumerate(vectorizer.vocabulary):
        shown = display_char(ch)
        category = character_category(ch)
        rows.append(
            FeatureMeta(
                index=index,
                name=f"char_freq:{shown}",
                label=f"freq({shown})",
                latex_label=f"freq({latex_char_label(ch)})",
                category=category,
                kind="char_frequency",
                character=ch,
                is_alpha=ch.isalpha(),
                is_uppercase=ch.isalpha() and ch.isupper(),
                is_diacritic=ch in ROMANIAN_DIACRITICS,
            )
        )
    offset = len(vectorizer.vocabulary)
    for local_index, name in enumerate(SCALAR_FEATURES):
        label = name.replace("_", " ")
        rows.append(
            FeatureMeta(
                index=offset + local_index,
                name=name,
                label=label,
                latex_label=label,
                category=SCALAR_CATEGORIES[name],
                kind="scalar",
                character="",
                is_alpha=False,
                is_uppercase=(name == "uppercase_ratio"),
                is_diacritic=(name == "romanian_diacritic_ratio"),
            )
        )
    return rows


def selected_indices_for_variant(metadata: Sequence[FeatureMeta], variant_key: str) -> np.ndarray:
    selectors: dict[str, Callable[[FeatureMeta], bool]] = {
        "full": lambda meta: True,
        "no_punctuation": lambda meta: meta.category != "punctuation",
        "no_digits": lambda meta: meta.category != "digit",
        "no_diacritics": lambda meta: meta.category != "diacritic",
        "no_uppercase": lambda meta: not meta.is_uppercase,
        "no_length": lambda meta: meta.category != "length",
        "letters_only": lambda meta: meta.kind == "char_frequency" and meta.is_alpha,
    }
    if variant_key not in selectors:
        raise ValueError(f"Unknown ablation variant: {variant_key}")
    selected = [meta.index for meta in metadata if selectors[variant_key](meta)]
    if not selected:
        raise ValueError(f"Ablation variant selected no features: {variant_key}")
    return np.asarray(selected, dtype=int)


def _chunk_matrix(
    records: Sequence[DocumentRecord],
    vectorizer: CharHistogramVectorizer,
    config: CHSVMConfig,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for record in records:
        for chunk in chunk_text(record.text, config.chunk_size, config.overlap):
            rows.append(vectorizer.transform_chunk(chunk))
            labels.append(record.author)
    if not rows:
        raise ValueError("No chunks produced for CH-SVM matrix construction.")
    return np.vstack(rows), np.asarray(labels, dtype=object)


def fit_masked_ch_svm(
    records: Sequence[DocumentRecord],
    config: CHSVMConfig,
    selected_indices: np.ndarray,
    vectorizer: CharHistogramVectorizer | None = None,
) -> MaskedCHSVMModel:
    if vectorizer is None:
        vectorizer = CharHistogramVectorizer(min_char_count=config.min_char_count).fit([record.text for record in records])
    X, y = _chunk_matrix(records, vectorizer, config)
    X_selected = X[:, selected_indices]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    classifier = LinearSVC(C=config.c, class_weight=config.class_weight, max_iter=20000, dual=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        classifier.fit(X_scaled, y)
    return MaskedCHSVMModel(
        config=config,
        vectorizer=vectorizer,
        selected_indices=selected_indices,
        scaler=scaler,
        classifier=classifier,
        class_order=sorted(set(y.tolist())),
    )


def aligned_decision_scores(classifier: LinearSVC, class_order: Sequence[str], X_scaled: np.ndarray) -> np.ndarray:
    scores = classifier.decision_function(X_scaled)
    if scores.ndim == 1:
        scores = np.column_stack((-scores, scores))
    order = [list(classifier.classes_).index(label) for label in class_order]
    return np.asarray(scores)[:, order]


def aligned_coefficients(classifier: LinearSVC, class_order: Sequence[str]) -> np.ndarray:
    coef = np.asarray(classifier.coef_, dtype=np.float64)
    classes = list(classifier.classes_)
    if coef.shape[0] == 1 and len(classes) == 2:
        coef = np.vstack([-coef[0], coef[0]])
    order = [classes.index(label) for label in class_order]
    return coef[order]


def predict_masked_ch_svm_scores(model: MaskedCHSVMModel, records: Sequence[DocumentRecord]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for record in records:
        chunks = chunk_text(record.text, model.config.chunk_size, model.config.overlap)
        X = model.vectorizer.transform_chunks(chunks)[:, model.selected_indices]
        scores = aligned_decision_scores(model.classifier, model.class_order, model.scaler.transform(X))
        rows.append(scores.mean(axis=0))
    return np.vstack(rows) if rows else np.empty((0, len(model.class_order)))


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / (exp_scores.sum(axis=1, keepdims=True) + 1e-12)


def evaluate_variant(
    variant_key: str,
    variant_label: str,
    variant_description: str,
    trainval: Sequence[DocumentRecord],
    test: Sequence[DocumentRecord],
    config: CHSVMConfig,
    class_order: Sequence[str],
) -> tuple[dict[str, object], list[dict[str, object]], MaskedCHSVMModel, list[FeatureMeta]]:
    base_vectorizer = CharHistogramVectorizer(min_char_count=config.min_char_count).fit([record.text for record in trainval])
    metadata = feature_metadata(base_vectorizer)
    selected_indices = selected_indices_for_variant(metadata, variant_key)
    model = fit_masked_ch_svm(trainval, config, selected_indices, vectorizer=base_vectorizer)
    scores = predict_masked_ch_svm_scores(model, test)
    probs = softmax(scores)
    group_ids, y_group, group_probs = aggregate_probabilities_by_work_group(test, probs)
    pred = labels_from_proba(group_probs, model.class_order)
    correct = int(sum(true == predicted for true, predicted in zip(y_group, pred)))
    row = {
        "variant_key": variant_key,
        "variant_label": variant_label,
        "description": variant_description,
        "selected_features": int(len(selected_indices)),
        "total_features": int(len(metadata)),
        "removed_features": int(len(metadata) - len(selected_indices)),
        "n_groups": len(y_group),
        "correct": correct,
        "accuracy": float(accuracy_score(y_group, pred)),
        "macro_f1": float(f1_score(y_group, pred, labels=list(class_order), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_group, pred)),
    }
    prediction_rows = [
        {
            "variant_key": variant_key,
            "work_group": group_id,
            "true_author": true,
            "predicted_author": predicted,
            "correct": int(true == predicted),
        }
        for group_id, true, predicted in zip(group_ids, y_group, pred)
    ]
    return row, prediction_rows, model, metadata


def coefficient_rows(model: MaskedCHSVMModel, metadata: Sequence[FeatureMeta], top_k: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selected_metadata = [metadata[index] for index in model.selected_indices]
    coefficients = aligned_coefficients(model.classifier, model.class_order)
    top_rows: list[dict[str, object]] = []
    category_rows: list[dict[str, object]] = []

    for class_index, author in enumerate(model.class_order):
        weights = coefficients[class_index]
        positive_order = np.argsort(weights)[::-1][:top_k]
        negative_order = np.argsort(weights)[:top_k]
        for rank, selected_position in enumerate(positive_order, start=1):
            meta = selected_metadata[int(selected_position)]
            top_rows.append(
                {
                    "author": author,
                    "direction": "positive",
                    "rank": rank,
                    "feature": meta.name,
                    "display_feature": meta.label,
                    "latex_feature": latex_feature_label(meta),
                    "category": meta.category,
                    "standardized_weight": float(weights[int(selected_position)]),
                }
            )
        for rank, selected_position in enumerate(negative_order, start=1):
            meta = selected_metadata[int(selected_position)]
            top_rows.append(
                {
                    "author": author,
                    "direction": "negative",
                    "rank": rank,
                    "feature": meta.name,
                    "display_feature": meta.label,
                    "latex_feature": latex_feature_label(meta),
                    "category": meta.category,
                    "standardized_weight": float(weights[int(selected_position)]),
                }
            )

        by_category: dict[str, list[tuple[FeatureMeta, float]]] = defaultdict(list)
        for meta, weight in zip(selected_metadata, weights):
            by_category[meta.category].append((meta, float(weight)))
        total_abs = float(np.abs(weights).sum())
        for category, items in sorted(by_category.items()):
            category_weights = np.asarray([weight for _, weight in items], dtype=np.float64)
            best_meta, best_weight = max(items, key=lambda item: item[1])
            category_rows.append(
                {
                    "author": author,
                    "category": category,
                    "feature_count": len(items),
                    "positive_weight_l1": float(np.maximum(category_weights, 0.0).sum()),
                    "negative_weight_l1": float(np.maximum(-category_weights, 0.0).sum()),
                    "absolute_weight_l1": float(np.abs(category_weights).sum()),
                    "absolute_weight_share": float(np.abs(category_weights).sum() / total_abs) if total_abs else 0.0,
                    "strongest_positive_feature": best_meta.name,
                    "strongest_positive_latex": latex_feature_label(best_meta),
                    "strongest_positive_weight": float(best_weight),
                }
            )
    return top_rows, category_rows


def top_feature_summary(top_rows: Sequence[dict[str, object]], top_k: int = 3) -> dict[str, str]:
    by_author: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in top_rows:
        if row["direction"] == "positive":
            by_author[str(row["author"])].append(row)
    summary: dict[str, str] = {}
    for author in sorted(by_author):
        rows = sorted(by_author[author], key=lambda row: int(row["rank"]))[:top_k]
        parts = [
            f"{row['latex_feature']} ({row['category']}, {float(row['standardized_weight']):.3f})"
            for row in rows
        ]
        summary[author] = "; ".join(parts)
    return summary


def write_top_feature_table(path: Path, top_rows: Sequence[dict[str, object]]) -> None:
    summary = top_feature_summary(top_rows, top_k=3)
    lines = [
        r"\begin{table}",
        r"  \tbl{\caption{Strongest positive standardized CH-SVM coefficients by author on the locked ROST train+validation partition. The model is the grouped-CV-selected CH-SVM refit before the held-out test evaluation; weights are inspected after feature standardization.}\label{tab:chsvm-interpretability}}",
        r"  {\tablefont\begin{tabular}{lp{0.68\textwidth}}",
        r"    \hline",
        r"    Author & Strongest positive CH-SVM features\\",
        r"    \hline",
    ]
    for author, features in summary.items():
        lines.append(f"    {author} & {features}\\\\")
    lines.extend(
        [
            r"    \hline",
            r"  \end{tabular}}",
            r"  {}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_ablation_table(path: Path, rows: Sequence[dict[str, object]]) -> None:
    order = ["full", "letters_only", "no_punctuation", "no_digits", "no_diacritics", "no_uppercase", "no_length"]
    by_key = {str(row["variant_key"]): row for row in rows}
    lines = [
        r"\begin{table}",
        r"  \tbl{\caption{Fixed-config CH-SVM surface-cue ablations on the locked ROST held-out source-text groups. Each variant reuses the grouped-CV-selected CH-SVM hyperparameters, refits on train+validation only, and is evaluated once on the same held-out groups. Variants are diagnostics, not model-selection candidates.}\label{tab:chsvm-ablation}}",
        r"  {\tablefont\begin{tabular}{lcccc}",
        r"    \hline",
        r"    Variant & Features used & Correct / 58 & Accuracy & Macro-F1\\",
        r"    \hline",
    ]
    for key in order:
        row = by_key[key]
        lines.append(
            f"    {row['variant_label']} & {int(row['selected_features'])} & "
            f"{int(row['correct'])}/58 & {float(row['accuracy']):.4f} & {float(row['macro_f1']):.4f}\\\\"
        )
    lines.extend(
        [
            r"    \hline",
            r"  \end{tabular}}",
            r"  {}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def variant_plan() -> list[tuple[str, str, str]]:
    return [
        ("full", "Full CH-SVM", "all one-character frequency and scalar CH-SVM features"),
        ("letters_only", "Letters only", "only alphabetic one-character frequency features; no scalar statistics"),
        ("no_punctuation", "No punctuation", "punctuation character frequencies and punctuation_ratio removed"),
        ("no_digits", "No digits", "digit character frequencies and digit_ratio removed"),
        ("no_diacritics", "No diacritics", "Romanian diacritic character frequencies and romanian_diacritic_ratio removed"),
        ("no_uppercase", "No uppercase", "uppercase character frequencies and uppercase_ratio removed"),
        ("no_length", "No length", "log1p_length removed"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset", default="rost", choices=["auto", "rost", "rostories", "extended"])
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--selected-config", type=Path, required=True)
    parser.add_argument("--interpretability-output", type=Path, required=True)
    parser.add_argument("--ablation-output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=TOP_FEATURES_PER_AUTHOR)
    args = parser.parse_args()

    records = load_documents(args.input, dataset=args.dataset)
    split_payload = load_split(args.split_json)
    split_docs = apply_split(records, split_payload)
    trainval = split_docs["train"] + split_docs["validation"]
    test = split_docs["test"]
    class_order = sorted({record.author for record in trainval})
    config = load_selected_ch_svm_config(args.selected_config)

    full_row, full_predictions, full_model, metadata = evaluate_variant(
        "full",
        "Full CH-SVM",
        "all one-character frequency and scalar CH-SVM features",
        trainval,
        test,
        config,
        class_order,
    )
    top_rows, category_rows = coefficient_rows(full_model, metadata, args.top_k)

    variant_rows = [full_row]
    prediction_rows = full_predictions
    for key, label, description in variant_plan()[1:]:
        row, predictions, _, _ = evaluate_variant(
            key,
            label,
            description,
            trainval,
            test,
            config,
            class_order,
        )
        variant_rows.append(row)
        prediction_rows.extend(predictions)

    args.interpretability_output.mkdir(parents=True, exist_ok=True)
    args.ablation_output.mkdir(parents=True, exist_ok=True)
    write_csv(args.interpretability_output / "ch_svm_top_features.csv", top_rows)
    write_csv(args.interpretability_output / "ch_svm_feature_category_weights.csv", category_rows)
    write_top_feature_table(args.interpretability_output / "table_ch_svm_top_features.tex", top_rows)
    write_json(
        args.interpretability_output / "interpretability_metadata.json",
        {
            "artifact": "CH-SVM standardized linear coefficients",
            "input": str(args.input),
            "dataset": args.dataset,
            "normalization": "ROST-NormElip normalized input: NFC, cedilla-to-comma Romanian diacritics, quote/dash/ellipsis normalization, whitespace collapse; case, punctuation, digits, diacritics, and line breaks preserved",
            "split_json": str(args.split_json),
            "selected_config": str(args.selected_config),
            "training_partition": "locked train+validation source-text groups",
            "test_usage": "held-out test labels are not used for coefficient inspection",
            "ch_svm_config": config.__dict__,
            "class_order": class_order,
            "trainval_documents": len(trainval),
            "trainval_work_groups": len({record.work_group for record in trainval}),
            "test_documents": len(test),
            "test_work_groups": len({record.work_group for record in test}),
            "dataset_summary": dataset_summary(records),
            "software_versions": software_versions(),
        },
    )

    write_csv(args.ablation_output / "ch_svm_surface_cue_ablation.csv", variant_rows)
    write_csv(args.ablation_output / "ch_svm_surface_cue_predictions.csv", prediction_rows)
    write_ablation_table(args.ablation_output / "table_ch_svm_surface_cue_ablation.tex", variant_rows)
    write_json(
        args.ablation_output / "surface_cue_ablation_metadata.json",
        {
            "artifact": "fixed-config CH-SVM surface-cue ablation",
            "input": str(args.input),
            "dataset": args.dataset,
            "normalization": "ROST-NormElip normalized input: NFC, cedilla-to-comma Romanian diacritics, quote/dash/ellipsis normalization, whitespace collapse; case, punctuation, digits, diacritics, and line breaks preserved",
            "split_json": str(args.split_json),
            "selected_config": str(args.selected_config),
            "protocol": "all variants reuse the grouped-CV-selected CH-SVM hyperparameters; each variant refits on locked train+validation groups and is evaluated once on held-out test groups",
            "test_usage": "held-out labels are used only for the fixed diagnostic evaluation, not for selecting variants or hyperparameters",
            "ch_svm_config": config.__dict__,
            "variants": [
                {"variant_key": key, "variant_label": label, "description": description}
                for key, label, description in variant_plan()
            ],
            "class_order": class_order,
            "trainval_documents": len(trainval),
            "trainval_work_groups": len({record.work_group for record in trainval}),
            "test_documents": len(test),
            "test_work_groups": len({record.work_group for record in test}),
            "software_versions": software_versions(),
        },
    )

    print(
        json.dumps(
            {
                "interpretability": str(args.interpretability_output),
                "ablation": str(args.ablation_output),
                "full_ch_svm": {
                    "correct": full_row["correct"],
                    "accuracy": full_row["accuracy"],
                    "macro_f1": full_row["macro_f1"],
                },
                "variants": {
                    row["variant_key"]: {
                        "correct": row["correct"],
                        "accuracy": row["accuracy"],
                        "macro_f1": row["macro_f1"],
                    }
                    for row in variant_rows
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
