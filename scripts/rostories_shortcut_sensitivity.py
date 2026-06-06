#!/usr/bin/env python3
"""Shortcut-risk sensitivity checks for the ROSTories-cleaned secondary run.

The main check masks each document's own author label when it appears inside
the text, keeps the locked ROSTories-cleaned split and already selected base
configs, then refits/evaluates without using held-out test labels for any
selection.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from chips.data import DocumentRecord, dataset_summary, load_documents, write_records_csv
from chips.modeling import (
    CHiPSBundle,
    FusionConfig,
    aggregate_probabilities_by_work_group,
    evaluate_predictions,
    fit_ch_svm,
    fit_fft12_lr,
    labels_from_proba,
    predict_bundle_proba,
    predict_ch_svm_proba,
    predict_fft12_proba,
    save_bundle,
    software_versions,
    topk_group_rows,
    write_csv,
    write_json,
)
from chips.splitting import apply_split, assert_no_group_leakage, load_split
from chips_rerank import (
    POLICIES,
    _apply_mode,
    _base_metric_block,
    _build_reranker_dataset,
    _candidate_oracle,
    _fit_final_reranker,
    _load_selected_configs,
    _make_oof_predictions,
    _predict_group_base,
    _predict_positions,
    _select_reranker,
    _test_prediction_rows,
    _transition_summary_vs_ch_svm,
    _transition_summary_vs_chips_f,
    _write_csv_any,
)


MASK_CHARACTER = "x"


def _mask_literal_case_insensitive(text: str, needle: str) -> tuple[str, int]:
    pattern = re.compile(re.escape(needle), flags=re.IGNORECASE)
    return pattern.subn(lambda match: MASK_CHARACTER * len(match.group(0)), text)


def mask_self_author_names(records: Sequence[DocumentRecord]) -> tuple[list[DocumentRecord], list[dict]]:
    masked: list[DocumentRecord] = []
    rows: list[dict] = []
    for record in records:
        masked_text, count = _mask_literal_case_insensitive(record.text, record.author)
        masked.append(replace(record, text=masked_text))
        rows.append(
            {
                "doc_id": record.doc_id,
                "author": record.author,
                "work_group": record.work_group,
                "masked_occurrences": count,
                "original_char_count": len(record.text),
                "masked_char_count": len(masked_text),
            }
        )
    return masked, rows


def _split_lookup(split_payload: dict) -> dict[str, str]:
    return {doc_id: split_name for split_name, doc_ids in split_payload["doc_ids"].items() for doc_id in doc_ids}


def _flagged_groups(mask_rows: Sequence[dict], split_payload: dict) -> dict[str, set[str]]:
    doc_to_split = _split_lookup(split_payload)
    out: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    for row in mask_rows:
        if int(row["masked_occurrences"]) <= 0:
            continue
        split_name = doc_to_split.get(str(row["doc_id"]))
        if split_name in out:
            out[split_name].add(str(row["work_group"]))
    return out


def _subset_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict:
    if not y_true:
        return {"total": 0, "correct": 0, "accuracy": None, "macro_f1": None, "balanced_accuracy": None}
    labels = sorted(set(y_true) | set(y_pred))
    correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))
    return {
        "total": len(y_true),
        "correct": int(correct),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def _diagnostic_subsets_from_rows(rows: Sequence[dict], prediction_columns: Sequence[str], flagged_test_groups: set[str]) -> dict:
    out: dict[str, dict] = {}
    for column in prediction_columns:
        for subset_name, include_flagged in (("flagged_test_groups", True), ("unflagged_test_groups", False)):
            subset = [row for row in rows if (row["work_group"] in flagged_test_groups) == include_flagged]
            out[f"{column}__{subset_name}"] = _subset_metrics(
                [str(row["true_author"]) for row in subset],
                [str(row[column]) for row in subset],
            )
    return out


def _base_prediction_rows(
    test: Sequence[DocumentRecord],
    ch_probs: np.ndarray,
    fft_probs: np.ndarray,
    fused_probs: np.ndarray,
    class_order: Sequence[str],
    flagged_test_groups: set[str],
) -> list[dict]:
    groups, y_true, ch_group_probs = aggregate_probabilities_by_work_group(test, ch_probs)
    fft_groups, _, fft_group_probs = aggregate_probabilities_by_work_group(test, fft_probs)
    fused_groups, _, fused_group_probs = aggregate_probabilities_by_work_group(test, fused_probs)
    if not (groups == fft_groups == fused_groups):
        raise RuntimeError("Test group order mismatch between model predictions.")
    ch_pred = labels_from_proba(ch_group_probs, class_order)
    fft_pred = labels_from_proba(fft_group_probs, class_order)
    fused_pred = labels_from_proba(fused_group_probs, class_order)
    return [
        {
            "work_group": group,
            "true_author": true_author,
            "CH_SVM": ch_label,
            "FFT12_LR": fft_label,
            "CHiPS_F": fused_label,
            "author_name_flagged_test_group": group in flagged_test_groups,
        }
        for group, true_author, ch_label, fft_label, fused_label in zip(groups, y_true, ch_pred, fft_pred, fused_pred)
    ]


def _run_masked_fixed_base(
    output: Path,
    records: Sequence[DocumentRecord],
    split_docs: dict[str, list[DocumentRecord]],
    split_payload: dict,
    base_run: Path,
    dataset: str,
    input_path: Path,
    flagged_test_groups: set[str],
) -> tuple[CHiPSBundle, dict, list[dict]]:
    base_output = output / "base_masked_fixed"
    base_output.mkdir(parents=True, exist_ok=True)
    ch_config, fft_config, alpha, selected_base = _load_selected_configs(base_run)
    train = split_docs["train"]
    validation = split_docs["validation"]
    test = split_docs["test"]
    trainval = train + validation
    class_order = sorted({record.author for record in trainval})

    write_json(base_output / "split_summary.json", split_payload)
    write_json(base_output / "dataset_summary.json", dataset_summary(records))
    write_json(base_output / "software_versions.json", software_versions())
    write_records_csv(records, base_output / "documents.csv")
    write_json(
        base_output / "selected_config.json",
        {
            **selected_base,
            "sensitivity_note": (
                "Fixed-config shortcut-risk sensitivity: self-author strings were masked in text before "
                "feature extraction. Base configs and fusion alpha are inherited from the original "
                "ROSTories-cleaned grouped-CV run; no held-out test labels are used for selection."
            ),
            "input": str(input_path),
            "dataset": dataset,
        },
    )

    ch_model = fit_ch_svm(trainval, ch_config)
    fft_model = fit_fft12_lr(trainval, fft_config)
    bundle = CHiPSBundle(
        ch_svm=ch_model,
        fft12_lr=fft_model,
        fusion=FusionConfig(alpha_ch_svm=alpha),
        metadata={
            "input": str(input_path),
            "dataset": dataset,
            "evaluation_unit": "work_group/source_text",
            "selection": "fixed selected configs inherited from original ROSTories-cleaned grouped-CV run",
            "sensitivity": "self-author strings masked in-memory before fitting and prediction",
            "ch_svm_config": ch_config.__dict__,
            "fft12_lr_config": fft_config.__dict__,
            "fusion_alpha_ch_svm": alpha,
            "class_order": class_order,
        },
    )
    save_bundle(base_output / "chips_bundle.joblib", bundle)

    ch_probs = predict_ch_svm_proba(ch_model, test)
    fft_probs = predict_fft12_proba(fft_model, test)
    fused_probs = predict_bundle_proba(bundle, test)
    test_groups, y_test, ch_group_probs = aggregate_probabilities_by_work_group(test, ch_probs)
    fft_test_groups, _, fft_group_probs = aggregate_probabilities_by_work_group(test, fft_probs)
    fused_test_groups, _, fused_group_probs = aggregate_probabilities_by_work_group(test, fused_probs)
    if not (test_groups == fft_test_groups == fused_test_groups):
        raise RuntimeError("Test group order mismatch between model predictions.")

    evaluations = {
        "CH_SVM": evaluate_predictions(y_test, labels_from_proba(ch_group_probs, class_order), class_order),
        "FFT12_LR": evaluate_predictions(y_test, labels_from_proba(fft_group_probs, class_order), class_order),
        "CHiPS_F": evaluate_predictions(y_test, labels_from_proba(fused_group_probs, class_order), class_order),
    }
    detailed_rows = _base_prediction_rows(test, ch_probs, fft_probs, fused_probs, class_order, flagged_test_groups)
    evaluations["diagnostic_subsets"] = _diagnostic_subsets_from_rows(
        detailed_rows,
        ["CH_SVM", "FFT12_LR", "CHiPS_F"],
        flagged_test_groups,
    )
    evaluations["_metadata"] = {
        "evaluation_unit": "work_group/source_text",
        "test_work_groups": len(test_groups),
        "test_documents": len(test),
        "selection_method": "fixed_original_grouped_cv_selection",
        "masking": "case-insensitive self-author label replaced by same-length x characters",
    }
    write_json(base_output / "metrics.json", evaluations)
    write_csv(base_output / "base_predictions_test_detailed.csv", detailed_rows)
    write_csv(base_output / "predictions_test.csv", topk_group_rows(test, fused_probs, class_order, top_k=5))
    return bundle, evaluations, detailed_rows


def _run_masked_reranker(
    output: Path,
    split_docs: dict[str, list[DocumentRecord]],
    base_run: Path,
    input_path: Path,
    dataset: str,
    split_json: Path,
    top_k: int,
    oof_folds: int,
    meta_folds: int,
    seed: int,
    flagged_test_groups: set[str],
) -> dict:
    rerank_output = output / "chips_r_masked"
    rerank_output.mkdir(parents=True, exist_ok=True)
    ch_config, fft_config, alpha, selected_base = _load_selected_configs(base_run)
    train = split_docs["train"]
    validation = split_docs["validation"]
    test = split_docs["test"]
    trainval = train + validation
    class_order = sorted({record.author for record in trainval})

    write_json(rerank_output / "dataset_summary.json", dataset_summary(trainval + test))
    write_json(rerank_output / "software_versions.json", software_versions())
    write_json(
        rerank_output / "reranker_protocol.json",
        {
            "method": "CHiPS-R top-5 listwise reranker",
            "input": str(input_path),
            "dataset": dataset,
            "split_json": str(split_json),
            "base_run": str(base_run),
            "base_selection": selected_base,
            "oof_folds_requested": oof_folds,
            "meta_folds_requested": meta_folds,
            "seed": seed,
            "top_k": top_k,
            "sensitivity": "self-author strings masked in-memory before base and reranker feature extraction",
            "leakage_note": "Base reranker training features are grouped OOF predictions on masked train+validation. The held-out test split is used once after selecting the reranker by meta-CV.",
        },
    )

    oof_base, fold_rows = _make_oof_predictions(
        trainval=trainval,
        ch_config=ch_config,
        fft_config=fft_config,
        alpha=alpha,
        class_order=class_order,
        n_folds=oof_folds,
        seed=seed,
    )
    write_csv(rerank_output / "oof_folds.csv", fold_rows)

    ch_model = fit_ch_svm(trainval, ch_config)
    fft_model = fit_fft12_lr(trainval, fft_config)
    test_base = _predict_group_base(test, ch_model, fft_model, alpha, class_order)

    oracle_rows = []
    for policy in POLICIES:
        oracle_rows.append({"split": "trainval_oof", **_candidate_oracle(oof_base, policy, top_k)})
        oracle_rows.append({"split": "test", **_candidate_oracle(test_base, policy, top_k)})
    write_csv(rerank_output / "candidate_policy_oracle_diagnostics.csv", oracle_rows)

    selected_reranker, cv_rows, train_datasets = _select_reranker(
        oof_base=oof_base,
        top_k=top_k,
        meta_folds=meta_folds,
        seed=seed,
    )
    write_csv(rerank_output / "chips_r_cv_results.csv", cv_rows)

    train_dataset = train_datasets[selected_reranker["policy"]]
    final_model = _fit_final_reranker(train_dataset, selected_reranker, seed)
    test_dataset = _build_reranker_dataset(test_base, selected_reranker["policy"], top_k)
    test_pos_probs = _predict_positions(final_model, test_dataset.X, top_k)
    test_pred_indices = _apply_mode(test_dataset, test_pos_probs, selected_reranker["mode"])
    chips_r_test_authors = [test_base.class_order[int(i)] for i in test_pred_indices]
    chips_r_metrics = evaluate_predictions(test_base.y_true, chips_r_test_authors, test_base.class_order)

    base_metrics = _base_metric_block(test_base)
    selected_clean = {**selected_reranker, "class_weight": selected_reranker["class_weight"] or ""}
    prediction_rows = _test_prediction_rows(test_base, test_dataset, test_pos_probs, test_pred_indices, selected_reranker)
    for row in prediction_rows:
        row["author_name_flagged_test_group"] = row["work_group"] in flagged_test_groups

    metrics = {
        "base_models": base_metrics,
        "CHiPS_R": chips_r_metrics,
        "selected_reranker": selected_clean,
        "transition_summary_vs_ch_svm": _transition_summary_vs_ch_svm(test_base, test_dataset, test_pred_indices),
        "transition_summary_vs_chips_f": _transition_summary_vs_chips_f(test_base, test_dataset, test_pred_indices),
        "candidate_oracle": oracle_rows,
        "diagnostic_subsets": _diagnostic_subsets_from_rows(
            prediction_rows,
            ["ch_svm_pred", "fft12_lr_pred", "chips_f_pred", "chips_r_pred"],
            flagged_test_groups,
        ),
        "_metadata": {
            "evaluation_unit": "work_group/source_text",
            "train_work_groups": len({record.work_group for record in train}),
            "validation_work_groups": len({record.work_group for record in validation}),
            "trainval_work_groups": len(oof_base.group_ids),
            "test_work_groups": len(test_base.group_ids),
            "test_documents": len(test),
            "top_k": top_k,
            "masking": "case-insensitive self-author label replaced by same-length x characters",
        },
    }
    write_json(rerank_output / "chips_r_metrics.json", metrics)
    _write_csv_any(rerank_output / "chips_r_predictions_test.csv", prediction_rows)
    write_json(
        rerank_output / "chips_r_feature_names.json",
        {"policy": selected_reranker["policy"], "feature_count": len(train_dataset.feature_names), "feature_names": train_dataset.feature_names},
    )
    joblib.dump(final_model, rerank_output / "chips_r_meta_model.joblib")
    return metrics


def _compact_model_metrics(metrics: dict, keys: Sequence[str]) -> dict:
    return {
        key: {
            metric: metrics[key][metric]
            for metric in ("accuracy", "macro_f1", "balanced_accuracy")
            if metric in metrics[key]
        }
        for key in keys
    }


def cmd_run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    records = load_documents(args.input, dataset=args.dataset)
    split_payload = load_split(args.split_json)
    masked_records, mask_rows = mask_self_author_names(records)
    doc_to_split = _split_lookup(split_payload)
    for row in mask_rows:
        row["split"] = doc_to_split.get(str(row["doc_id"]), "")
    masked_split_docs = apply_split(masked_records, split_payload)
    assert_no_group_leakage(masked_split_docs)

    flagged = _flagged_groups(mask_rows, split_payload)
    flagged_test_groups = flagged["test"]
    write_csv(args.output / "masking_documents.csv", mask_rows)
    mask_summary = {
        "input": str(args.input),
        "dataset": args.dataset,
        "split_json": str(args.split_json),
        "base_run": str(args.base_run),
        "masking": "case-insensitive self-author label replaced by same-length x characters",
        "documents": len(mask_rows),
        "documents_with_masked_occurrences": sum(int(row["masked_occurrences"]) > 0 for row in mask_rows),
        "masked_occurrences": sum(int(row["masked_occurrences"]) for row in mask_rows),
        "documents_with_masked_occurrences_by_split": dict(
            Counter(row["split"] for row in mask_rows if int(row["masked_occurrences"]) > 0)
        ),
        "flagged_work_groups_by_split": {split_name: sorted(groups) for split_name, groups in flagged.items()},
        "test_flagged_work_groups": sorted(flagged_test_groups),
        "test_usage": "held-out test labels are used only for final evaluation and diagnostic subset reporting",
    }
    write_json(args.output / "masking_summary.json", mask_summary)

    _, base_metrics, _ = _run_masked_fixed_base(
        output=args.output,
        records=masked_records,
        split_docs=masked_split_docs,
        split_payload=split_payload,
        base_run=args.base_run,
        dataset=args.dataset,
        input_path=args.input,
        flagged_test_groups=flagged_test_groups,
    )
    reranker_metrics = _run_masked_reranker(
        output=args.output,
        split_docs=masked_split_docs,
        base_run=args.base_run,
        input_path=args.input,
        dataset=args.dataset,
        split_json=args.split_json,
        top_k=args.top_k,
        oof_folds=args.oof_folds,
        meta_folds=args.meta_folds,
        seed=args.seed,
        flagged_test_groups=flagged_test_groups,
    )

    summary = {
        "masking_summary": mask_summary,
        "base_masked_fixed": _compact_model_metrics(base_metrics, ["CH_SVM", "FFT12_LR", "CHiPS_F"]),
        "chips_r_masked": {
            "base_models": _compact_model_metrics(reranker_metrics["base_models"], ["CH_SVM", "FFT12_LR", "CHiPS_F"]),
            "CHiPS_R": {
                metric: reranker_metrics["CHiPS_R"][metric]
                for metric in ("accuracy", "macro_f1", "balanced_accuracy")
            },
            "selected_reranker": reranker_metrics["selected_reranker"],
            "transition_summary_vs_chips_f": reranker_metrics["transition_summary_vs_chips_f"],
        },
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ROSTories-cleaned author-name shortcut-risk sensitivity.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset", default="rostories", choices=["auto", "rost", "rostories", "extended"])
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--oof-folds", type=int, default=5)
    parser.add_argument("--meta-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.top_k < 2:
        raise ValueError("--top-k must be at least 2")
    cmd_run(args)


if __name__ == "__main__":
    main()
