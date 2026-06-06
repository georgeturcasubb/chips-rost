#!/usr/bin/env python3
"""Command-line interface for CHiPS experiments.

Examples:
  python scripts/chips.py describe --input data/ROST-NormElip
  python scripts/chips.py normalize --input raw/ROST --output data/ROST-NormElip
  python scripts/chips.py train-all --input data/ROST-NormElip --output experiments/runs/rost_quick --quick
  python scripts/chips.py predict --model experiments/runs/rost_quick/chips_bundle.joblib --input data/ROST-NormElip/Creanga_PopaDuhu.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chips.data import build_record, dataset_summary, load_documents, write_records_csv
from chips.modeling import (
    CHiPSBundle,
    FusionConfig,
    aggregate_probabilities_by_work_group,
    default_char_ngram_grid,
    default_ch_grid,
    default_fft_grid,
    evaluate_predictions,
    fit_char_ngram_svm,
    fit_ch_svm,
    fit_fft12_lr,
    labels_from_proba,
    load_bundle,
    majority_baseline_by_work_group,
    predict_bundle_proba,
    predict_char_ngram_svm_proba,
    predict_ch_svm_proba,
    predict_fft12_proba,
    save_bundle,
    save_char_ngram_svm,
    select_ch_svm_config,
    select_fft12_config,
    select_fusion_alpha,
    software_versions,
    topk_group_rows,
    topk_rows,
    write_csv,
    write_json,
)
from chips.normalize import normalize_directory, normalize_file
from chips.splitting import apply_split, load_split, make_grouped_split, save_split



def _limit_per_author(records, limit: int | None):
    if not limit:
        return records
    kept = []
    counts = {}
    for record in sorted(records, key=lambda r: (r.author, r.doc_id)):
        count = counts.get(record.author, 0)
        if count < limit:
            kept.append(record)
            counts[record.author] = count + 1
    return kept

def cmd_describe(args: argparse.Namespace) -> None:
    records = load_documents(args.input, dataset=args.dataset)
    summary = dataset_summary(records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        write_json(args.output / "dataset_summary.json", summary)
        write_records_csv(records, args.output / "documents.csv")


def cmd_normalize(args: argparse.Namespace) -> None:
    if args.input.is_dir():
        count = normalize_directory(args.input, args.output)
        print(f"Normalized {count} files into {args.output}")
    else:
        normalize_file(args.input, args.output)
        print(f"Normalized {args.input} -> {args.output}")


def _load_or_make_split(records, args):
    if args.split_json:
        payload = load_split(args.split_json)
        split_docs = apply_split(records, payload)
        return split_docs, payload
    split_docs, summary = make_grouped_split(records, seed=args.seed)
    return split_docs, summary


def cmd_make_split(args: argparse.Namespace) -> None:
    records = load_documents(args.input, dataset=args.dataset)
    split_docs, summary = make_grouped_split(records, seed=args.seed)
    save_split(summary, split_docs, args.output)
    print(f"Wrote leakage-safe grouped split to {args.output}")


def _records_by_work_group(records):
    by_group = defaultdict(list)
    group_author = {}
    for record in records:
        by_group[record.work_group].append(record)
        previous = group_author.setdefault(record.work_group, record.author)
        if previous != record.author:
            raise ValueError(f"Work group has multiple authors: {record.work_group}")
    group_ids = sorted(by_group)
    y_group = [group_author[group_id] for group_id in group_ids]
    return group_ids, y_group, by_group


def _make_stratified_group_folds(records, requested_folds: int, seed: int):
    group_ids, y_group, by_group = _records_by_work_group(records)
    label_counts = Counter(y_group)
    usable_folds = min(requested_folds, min(label_counts.values()))
    if usable_folds < 2:
        raise ValueError(f"Need at least two source texts per author for grouped CV; counts={dict(label_counts)}")

    y_array = np.asarray(y_group)
    splitter = StratifiedKFold(n_splits=usable_folds, shuffle=True, random_state=seed)
    folds = []
    for fold_id, (fit_idx, holdout_idx) in enumerate(splitter.split(np.zeros(len(group_ids)), y_array), start=1):
        fit_groups = {group_ids[i] for i in fit_idx}
        holdout_groups = {group_ids[i] for i in holdout_idx}
        fit_records = [record for group_id in group_ids if group_id in fit_groups for record in by_group[group_id]]
        holdout_records = [record for group_id in group_ids if group_id in holdout_groups for record in by_group[group_id]]
        folds.append(
            {
                "fold": fold_id,
                "fit_groups": fit_groups,
                "holdout_groups": holdout_groups,
                "fit_records": fit_records,
                "holdout_records": holdout_records,
            }
        )
    return folds, {"requested_folds": requested_folds, "used_folds": usable_folds, "label_counts": dict(label_counts)}


def _score_key_from_row(row: dict) -> tuple[float, float]:
    return (float(row["mean_macro_f1"]), float(row["mean_accuracy"]))


def _summarize_fold_metrics(config_order: int, config, fold_rows: list[dict]) -> dict:
    accuracies = np.asarray([row["accuracy"] for row in fold_rows], dtype=np.float64)
    macro_f1 = np.asarray([row["macro_f1"] for row in fold_rows], dtype=np.float64)
    balanced = np.asarray([row["balanced_accuracy"] for row in fold_rows], dtype=np.float64)
    return {
        "order": config_order,
        **config.__dict__,
        "mean_accuracy": float(accuracies.mean()),
        "std_accuracy": float(accuracies.std(ddof=0)),
        "mean_macro_f1": float(macro_f1.mean()),
        "std_macro_f1": float(macro_f1.std(ddof=0)),
        "mean_balanced_accuracy": float(balanced.mean()),
        "std_balanced_accuracy": float(balanced.std(ddof=0)),
    }


def _select_ch_svm_config_cv(configs, folds, class_order):
    summary_rows = []
    fold_rows_all = []
    best_config = configs[0]
    best_key = (-1.0, -1.0)
    for order, config in enumerate(configs):
        print(f"  CH-SVM CV config {order + 1}/{len(configs)}: {config}", flush=True)
        fold_rows = []
        for fold in folds:
            model = fit_ch_svm(fold["fit_records"], config)
            probs = predict_ch_svm_proba(model, fold["holdout_records"])
            _, y_group, group_probs = aggregate_probabilities_by_work_group(fold["holdout_records"], probs)
            pred = labels_from_proba(group_probs, model.class_order)
            metrics = evaluate_predictions(y_group, pred, class_order)
            row = {
                "order": order,
                "fold": fold["fold"],
                **config.__dict__,
                "fit_work_groups": len(fold["fit_groups"]),
                "holdout_work_groups": len(fold["holdout_groups"]),
                "fit_documents": len(fold["fit_records"]),
                "holdout_documents": len(fold["holdout_records"]),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
            }
            fold_rows.append(row)
            fold_rows_all.append(row)
        summary = _summarize_fold_metrics(order, config, fold_rows)
        summary_rows.append(summary)
        key = _score_key_from_row(summary)
        if key > best_key:
            best_key = key
            best_config = config
    return best_config, summary_rows, fold_rows_all


def _select_fft12_config_cv(configs, folds, class_order):
    summary_rows = []
    fold_rows_all = []
    best_config = configs[0]
    best_key = (-1.0, -1.0)
    for order, config in enumerate(configs):
        print(f"  FFT12-LR CV config {order + 1}/{len(configs)}: {config}", flush=True)
        fold_rows = []
        for fold in folds:
            model = fit_fft12_lr(fold["fit_records"], config)
            probs = predict_fft12_proba(model, fold["holdout_records"])
            _, y_group, group_probs = aggregate_probabilities_by_work_group(fold["holdout_records"], probs)
            pred = labels_from_proba(group_probs, model.class_order)
            metrics = evaluate_predictions(y_group, pred, class_order)
            row = {
                "order": order,
                "fold": fold["fold"],
                **config.__dict__,
                "fit_work_groups": len(fold["fit_groups"]),
                "holdout_work_groups": len(fold["holdout_groups"]),
                "fit_documents": len(fold["fit_records"]),
                "holdout_documents": len(fold["holdout_records"]),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
            }
            fold_rows.append(row)
            fold_rows_all.append(row)
        summary = _summarize_fold_metrics(order, config, fold_rows)
        summary_rows.append(summary)
        key = _score_key_from_row(summary)
        if key > best_key:
            best_key = key
            best_config = config
    return best_config, summary_rows, fold_rows_all


def _select_char_ngram_config_cv(configs, folds, class_order):
    summary_rows = []
    fold_rows_all = []
    best_config = configs[0]
    best_key = (-1.0, -1.0)
    for order, config in enumerate(configs):
        print(f"  Char 2-5 TF-IDF SVM CV config {order + 1}/{len(configs)}: {config}", flush=True)
        fold_rows = []
        for fold in folds:
            model = fit_char_ngram_svm(fold["fit_records"], config)
            probs = predict_char_ngram_svm_proba(model, fold["holdout_records"])
            _, y_group, group_probs = aggregate_probabilities_by_work_group(fold["holdout_records"], probs)
            pred = labels_from_proba(group_probs, model.class_order)
            metrics = evaluate_predictions(y_group, pred, class_order)
            row = {
                "order": order,
                "fold": fold["fold"],
                **config.__dict__,
                "fit_work_groups": len(fold["fit_groups"]),
                "holdout_work_groups": len(fold["holdout_groups"]),
                "fit_documents": len(fold["fit_records"]),
                "holdout_documents": len(fold["holdout_records"]),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
            }
            fold_rows.append(row)
            fold_rows_all.append(row)
        summary = _summarize_fold_metrics(order, config, fold_rows)
        summary_rows.append(summary)
        key = _score_key_from_row(summary)
        if key > best_key:
            best_key = key
            best_config = config
    return best_config, summary_rows, fold_rows_all


def _align_probs(probs: np.ndarray, model_class_order: list[str], class_order: list[str]) -> np.ndarray:
    if set(model_class_order) != set(class_order):
        raise ValueError(f"Class mismatch. Model={model_class_order} expected={class_order}")
    order = [model_class_order.index(label) for label in class_order]
    return probs[:, order]


def _selected_config_oof_predictions(trainval, ch_config, fft_config, folds, class_order):
    group_ids, y_group, _ = _records_by_work_group(trainval)
    group_to_index = {group_id: index for index, group_id in enumerate(group_ids)}
    ch_oof = np.full((len(group_ids), len(class_order)), np.nan, dtype=np.float64)
    fft_oof = np.full((len(group_ids), len(class_order)), np.nan, dtype=np.float64)
    fold_rows = []

    for fold in folds:
        print(
            f"  OOF fold {fold['fold']}/{len(folds)}: fit {len(fold['fit_groups'])} groups, "
            f"predict {len(fold['holdout_groups'])} groups",
            flush=True,
        )
        ch_model = fit_ch_svm(fold["fit_records"], ch_config)
        fft_model = fit_fft12_lr(fold["fit_records"], fft_config)
        ch_probs = predict_ch_svm_proba(ch_model, fold["holdout_records"])
        fft_probs = predict_fft12_proba(fft_model, fold["holdout_records"])
        ch_groups, _, ch_group_probs = aggregate_probabilities_by_work_group(fold["holdout_records"], ch_probs)
        fft_groups, _, fft_group_probs = aggregate_probabilities_by_work_group(fold["holdout_records"], fft_probs)
        if ch_groups != fft_groups:
            raise RuntimeError("OOF group order mismatch between CH-SVM and FFT12-LR predictions.")
        ch_group_probs = _align_probs(ch_group_probs, ch_model.class_order, class_order)
        fft_group_probs = _align_probs(fft_group_probs, fft_model.class_order, class_order)
        for row_index, group_id in enumerate(ch_groups):
            target_index = group_to_index[group_id]
            ch_oof[target_index] = ch_group_probs[row_index]
            fft_oof[target_index] = fft_group_probs[row_index]
        fold_rows.append(
            {
                "fold": fold["fold"],
                "fit_work_groups": len(fold["fit_groups"]),
                "holdout_work_groups": len(fold["holdout_groups"]),
                "fit_documents": len(fold["fit_records"]),
                "holdout_documents": len(fold["holdout_records"]),
            }
        )

    if np.isnan(ch_oof).any() or np.isnan(fft_oof).any():
        raise RuntimeError("Missing selected-config OOF predictions.")
    return group_ids, y_group, ch_oof, fft_oof, fold_rows


def cmd_train_all_cv(args: argparse.Namespace) -> None:
    records = _limit_per_author(load_documents(args.input, dataset=args.dataset), args.limit_per_author)
    split_docs, split_summary = _load_or_make_split(records, args)
    train = split_docs["train"]
    validation = split_docs["validation"]
    test = split_docs["test"]
    trainval = train + validation
    class_order = sorted({r.author for r in trainval})
    train_group_count = len({r.work_group for r in train})
    validation_group_count = len({r.work_group for r in validation})
    trainval_group_count = len({r.work_group for r in trainval})
    test_group_count = len({r.work_group for r in test})
    args.output.mkdir(parents=True, exist_ok=True)

    write_json(args.output / "split_summary.json", split_summary)
    write_json(args.output / "dataset_summary.json", dataset_summary(records))
    write_json(args.output / "software_versions.json", software_versions())
    write_records_csv(records, args.output / "documents.csv")

    folds, cv_summary = _make_stratified_group_folds(trainval, args.cv_folds, args.seed)
    write_json(
        args.output / "cv_protocol.json",
        {
            "selection_method": "stratified grouped CV on locked train+validation source-text groups",
            "fusion_alpha_selection": "selected from OOF probabilities generated by the CV-selected base configs",
            "test_usage": "held-out test source-text groups are used once after refitting selected models on train+validation",
            "seed": args.seed,
            "train_work_groups": train_group_count,
            "validation_work_groups": validation_group_count,
            "trainval_work_groups": trainval_group_count,
            "test_work_groups": test_group_count,
            **cv_summary,
        },
    )

    print("Selecting CH-SVM configuration by grouped CV on train+validation...", flush=True)
    ch_config, ch_summary_rows, ch_fold_rows = _select_ch_svm_config_cv(
        default_ch_grid(quick=args.quick),
        folds,
        class_order,
    )
    write_csv(args.output / "ch_svm_cv5_grid.csv", ch_summary_rows)
    write_csv(args.output / "ch_svm_cv5_folds.csv", ch_fold_rows)

    print("Selecting FFT12-LR configuration by grouped CV on train+validation...", flush=True)
    fft_config, fft_summary_rows, fft_fold_rows = _select_fft12_config_cv(
        default_fft_grid(quick=args.quick),
        folds,
        class_order,
    )
    write_csv(args.output / "fft12_lr_cv5_grid.csv", fft_summary_rows)
    write_csv(args.output / "fft12_lr_cv5_folds.csv", fft_fold_rows)

    print("Selecting fusion alpha from grouped OOF predictions for the selected base configs...", flush=True)
    oof_groups, y_oof, ch_oof, fft_oof, oof_fold_rows = _selected_config_oof_predictions(
        trainval,
        ch_config,
        fft_config,
        folds,
        class_order,
    )
    write_csv(args.output / "selected_config_oof_folds.csv", oof_fold_rows)
    alpha, alpha_rows = select_fusion_alpha(
        y_oof,
        class_order,
        ch_oof,
        fft_oof,
        [i / 20 for i in range(21)],
    )
    write_csv(args.output / "fusion_alpha_cv5_oof_grid.csv", alpha_rows)
    write_json(
        args.output / "selected_config.json",
        {
            "selection_method": "CH-SVM and FFT12-LR selected by stratified grouped CV on train+validation work groups; fusion alpha selected on grouped OOF train+validation predictions; held-out test used once after refit.",
            "evaluation_unit": "work_group/source_text",
            "cv_folds_requested": args.cv_folds,
            "cv_folds_used": cv_summary["used_folds"],
            "train_work_groups": train_group_count,
            "validation_work_groups": validation_group_count,
            "trainval_work_groups": trainval_group_count,
            "test_work_groups": test_group_count,
            "oof_work_groups": len(oof_groups),
            "ch_svm": ch_config.__dict__,
            "fft12_lr": fft_config.__dict__,
            "fusion": {"alpha_ch_svm": alpha},
        },
    )

    print("Refitting CV-selected models on train+validation and evaluating on test...", flush=True)
    ch_model = fit_ch_svm(trainval, ch_config)
    fft_model = fit_fft12_lr(trainval, fft_config)
    bundle = CHiPSBundle(
        ch_svm=ch_model,
        fft12_lr=fft_model,
        fusion=FusionConfig(alpha_ch_svm=alpha),
        metadata={
            "input": str(args.input),
            "dataset": args.dataset,
            "quick": args.quick,
            "evaluation_unit": "work_group/source_text",
            "selection": "configs selected by grouped CV on train+validation; fusion alpha selected on selected-config OOF predictions; final test used after refit",
            "cv_folds_requested": args.cv_folds,
            "cv_folds_used": cv_summary["used_folds"],
            "ch_svm_config": ch_config.__dict__,
            "fft12_lr_config": fft_config.__dict__,
            "fusion_alpha_ch_svm": alpha,
            "class_order": class_order,
        },
    )
    save_bundle(args.output / "chips_bundle.joblib", bundle)

    ch_probs = predict_ch_svm_proba(ch_model, test)
    fft_probs = predict_fft12_proba(fft_model, test)
    fused_probs = predict_bundle_proba(bundle, test)
    test_groups, y_test, ch_group_probs = aggregate_probabilities_by_work_group(test, ch_probs)
    fft_test_groups, _, fft_group_probs = aggregate_probabilities_by_work_group(test, fft_probs)
    fused_test_groups, _, fused_group_probs = aggregate_probabilities_by_work_group(test, fused_probs)
    if not (test_groups == fft_test_groups == fused_test_groups):
        raise RuntimeError("Test group order mismatch between model predictions.")
    evaluations = {
        "majority_baseline": majority_baseline_by_work_group(trainval, test, class_order),
        "CH_SVM": evaluate_predictions(y_test, labels_from_proba(ch_group_probs, class_order), class_order),
        "FFT12_LR": evaluate_predictions(y_test, labels_from_proba(fft_group_probs, class_order), class_order),
        "CHiPS_F": evaluate_predictions(y_test, labels_from_proba(fused_group_probs, class_order), class_order),
    }
    evaluations["_metadata"] = {
        "evaluation_unit": "work_group/source_text",
        "test_work_groups": test_group_count,
        "test_documents": len(test),
        "selection_method": "grouped_cv_trainval",
    }
    write_json(args.output / "metrics.json", evaluations)
    write_csv(args.output / "predictions_test.csv", topk_group_rows(test, fused_probs, class_order, top_k=5))
    print(json.dumps({k: {m: v[m] for m in ["accuracy", "macro_f1", "balanced_accuracy"]} for k, v in evaluations.items() if not k.startswith("_")}, indent=2))


def cmd_train_all(args: argparse.Namespace) -> None:
    records = _limit_per_author(load_documents(args.input, dataset=args.dataset), args.limit_per_author)
    split_docs, split_summary = _load_or_make_split(records, args)
    train = split_docs["train"]
    validation = split_docs["validation"]
    test = split_docs["test"]
    trainval = train + validation
    class_order = sorted({r.author for r in trainval})
    validation_group_count = len({r.work_group for r in validation})
    test_group_count = len({r.work_group for r in test})
    args.output.mkdir(parents=True, exist_ok=True)

    write_json(args.output / "split_summary.json", split_summary)
    write_json(args.output / "dataset_summary.json", dataset_summary(records))
    write_json(args.output / "software_versions.json", software_versions())
    write_records_csv(records, args.output / "documents.csv")

    print("Selecting CH-SVM configuration on validation data...", flush=True)
    ch_config, ch_rows = select_ch_svm_config(train, validation, default_ch_grid(quick=args.quick))
    write_csv(args.output / "ch_svm_validation_grid.csv", ch_rows)

    print("Selecting FFT12-LR configuration on validation data...", flush=True)
    fft_config, fft_rows = select_fft12_config(train, validation, default_fft_grid(quick=args.quick))
    write_csv(args.output / "fft12_lr_validation_grid.csv", fft_rows)

    # Select alpha using models trained only on train, then retrain on train+validation.
    ch_train_model = fit_ch_svm(train, ch_config)
    fft_train_model = fit_fft12_lr(train, fft_config)
    ch_val_probs = predict_ch_svm_proba(ch_train_model, validation)
    fft_val_probs = predict_fft12_proba(fft_train_model, validation)
    val_groups, y_val, ch_val_group_probs = aggregate_probabilities_by_work_group(validation, ch_val_probs)
    fft_val_groups, _, fft_val_group_probs = aggregate_probabilities_by_work_group(validation, fft_val_probs)
    if val_groups != fft_val_groups:
        raise RuntimeError("Validation group order mismatch between CH-SVM and FFT12-LR predictions.")
    alpha, alpha_rows = select_fusion_alpha(
        y_val,
        ch_train_model.class_order,
        ch_val_group_probs,
        fft_val_group_probs,
        [i / 20 for i in range(21)],
    )
    write_csv(args.output / "fusion_alpha_validation_grid.csv", alpha_rows)
    write_json(
        args.output / "selected_config.json",
        {
            "selection_method": "CH-SVM config, FFT12-LR config, and fusion alpha selected on validation work groups only; held-out test used once after refit.",
            "evaluation_unit": "work_group/source_text",
            "validation_work_groups": validation_group_count,
            "test_work_groups": test_group_count,
            "ch_svm": ch_config.__dict__,
            "fft12_lr": fft_config.__dict__,
            "fusion": {"alpha_ch_svm": alpha},
        },
    )

    print("Refitting selected models on train+validation and evaluating on test...", flush=True)
    ch_model = fit_ch_svm(trainval, ch_config)
    fft_model = fit_fft12_lr(trainval, fft_config)
    bundle = CHiPSBundle(
        ch_svm=ch_model,
        fft12_lr=fft_model,
        fusion=FusionConfig(alpha_ch_svm=alpha),
        metadata={
            "input": str(args.input),
            "dataset": args.dataset,
            "quick": args.quick,
            "evaluation_unit": "work_group/source_text",
            "selection": "configs and fusion alpha selected on validation work groups only; final test used after refit",
            "ch_svm_config": ch_config.__dict__,
            "fft12_lr_config": fft_config.__dict__,
            "fusion_alpha_ch_svm": alpha,
            "class_order": class_order,
        },
    )
    save_bundle(args.output / "chips_bundle.joblib", bundle)

    ch_probs = predict_ch_svm_proba(ch_model, test)
    fft_probs = predict_fft12_proba(fft_model, test)
    fused_probs = predict_bundle_proba(bundle, test)
    test_groups, y_test, ch_group_probs = aggregate_probabilities_by_work_group(test, ch_probs)
    fft_test_groups, _, fft_group_probs = aggregate_probabilities_by_work_group(test, fft_probs)
    fused_test_groups, _, fused_group_probs = aggregate_probabilities_by_work_group(test, fused_probs)
    if not (test_groups == fft_test_groups == fused_test_groups):
        raise RuntimeError("Test group order mismatch between model predictions.")
    evaluations = {
        "majority_baseline": majority_baseline_by_work_group(trainval, test, class_order),
        "CH_SVM": evaluate_predictions(y_test, labels_from_proba(ch_group_probs, class_order), class_order),
        "FFT12_LR": evaluate_predictions(y_test, labels_from_proba(fft_group_probs, class_order), class_order),
        "CHiPS_F": evaluate_predictions(y_test, labels_from_proba(fused_group_probs, class_order), class_order),
    }
    evaluations["_metadata"] = {
        "evaluation_unit": "work_group/source_text",
        "test_work_groups": test_group_count,
        "test_documents": len(test),
    }
    write_json(args.output / "metrics.json", evaluations)
    write_csv(args.output / "predictions_test.csv", topk_group_rows(test, fused_probs, class_order, top_k=5))
    print(json.dumps({k: {m: v[m] for m in ["accuracy", "macro_f1", "balanced_accuracy"]} for k, v in evaluations.items() if not k.startswith("_")}, indent=2))


def cmd_train_char_ngram_cv(args: argparse.Namespace) -> None:
    records = _limit_per_author(load_documents(args.input, dataset=args.dataset), args.limit_per_author)
    split_docs, split_summary = _load_or_make_split(records, args)
    train = split_docs["train"]
    validation = split_docs["validation"]
    test = split_docs["test"]
    trainval = train + validation
    class_order = sorted({r.author for r in trainval})
    train_group_count = len({r.work_group for r in train})
    validation_group_count = len({r.work_group for r in validation})
    trainval_group_count = len({r.work_group for r in trainval})
    test_group_count = len({r.work_group for r in test})
    args.output.mkdir(parents=True, exist_ok=True)

    write_json(args.output / "split_summary.json", split_summary)
    write_json(args.output / "dataset_summary.json", dataset_summary(records))
    write_json(args.output / "software_versions.json", software_versions())
    write_records_csv(records, args.output / "documents.csv")

    folds, cv_summary = _make_stratified_group_folds(trainval, args.cv_folds, args.seed)
    write_json(
        args.output / "cv_protocol.json",
        {
            "baseline": "character 2-5 gram TF-IDF plus LinearSVC",
            "selection_method": "stratified grouped CV on locked train+validation source-text groups",
            "test_usage": "held-out test source-text groups are used once after refitting the selected model on train+validation",
            "evaluation_unit": "work_group/source_text",
            "seed": args.seed,
            "train_work_groups": train_group_count,
            "validation_work_groups": validation_group_count,
            "trainval_work_groups": trainval_group_count,
            "test_work_groups": test_group_count,
            **cv_summary,
        },
    )

    print("Selecting character 2-5 gram TF-IDF + LinearSVC baseline by grouped CV on train+validation...", flush=True)
    config, summary_rows, fold_rows = _select_char_ngram_config_cv(
        default_char_ngram_grid(quick=args.quick),
        folds,
        class_order,
    )
    write_csv(args.output / "char_ngram_2_5_svm_cv5_grid.csv", summary_rows)
    write_csv(args.output / "char_ngram_2_5_svm_cv5_folds.csv", fold_rows)

    print("Refitting CV-selected character n-gram baseline on train+validation and evaluating on test...", flush=True)
    model = fit_char_ngram_svm(trainval, config)
    save_char_ngram_svm(args.output / "char_ngram_2_5_svm.joblib", model)
    probs = predict_char_ngram_svm_proba(model, test)
    test_groups, y_test, group_probs = aggregate_probabilities_by_work_group(test, probs)
    pred = labels_from_proba(group_probs, class_order)
    evaluations = {
        "majority_baseline": majority_baseline_by_work_group(trainval, test, class_order),
        "CHAR_NGRAM_2_5_SVM": evaluate_predictions(y_test, pred, class_order),
    }
    evaluations["_metadata"] = {
        "evaluation_unit": "work_group/source_text",
        "test_work_groups": test_group_count,
        "test_documents": len(test),
        "prediction_work_groups": len(test_groups),
        "selection_method": "grouped_cv_trainval",
        "feature_family": "character 2-5 gram TF-IDF",
        "classifier": "LinearSVC",
        "score_interpretation": "softmax-normalized LinearSVC decision scores are used for source-text aggregation and top-k ranking; they are not calibrated probabilities",
    }
    write_json(args.output / "metrics.json", evaluations)
    write_json(
        args.output / "selected_config.json",
        {
            "selection_method": "character n-gram baseline selected by stratified grouped CV on train+validation work groups; held-out test used once after refit.",
            "evaluation_unit": "work_group/source_text",
            "cv_folds_requested": args.cv_folds,
            "cv_folds_used": cv_summary["used_folds"],
            "train_work_groups": train_group_count,
            "validation_work_groups": validation_group_count,
            "trainval_work_groups": trainval_group_count,
            "test_work_groups": test_group_count,
            "char_ngram_2_5_svm": config.__dict__,
            "class_order": class_order,
            "vocabulary_size": len(model.vectorizer.vocabulary_),
            "vectorizer": {
                "analyzer": "char",
                "lowercase": False,
                "ngram_range": [config.ngram_min, config.ngram_max],
                "use_idf": True,
                "norm": "l2",
            },
            "score_interpretation": "softmax-normalized LinearSVC decision scores are used for source-text aggregation and top-k ranking; they are not calibrated probabilities",
        },
    )
    write_csv(args.output / "predictions_test.csv", topk_group_rows(test, probs, class_order, top_k=5))
    print(json.dumps({k: {m: v[m] for m in ["accuracy", "macro_f1", "balanced_accuracy"]} for k, v in evaluations.items() if not k.startswith("_")}, indent=2))


def cmd_predict(args: argparse.Namespace) -> None:
    bundle = load_bundle(args.model)
    if args.input.is_dir() or args.input.suffix.lower() == ".zip":
        records = load_documents(args.input, dataset=args.dataset)
    else:
        text = args.input.read_text(encoding="utf-8")
        records = [build_record(args.input.name if "_" in args.input.stem else f"UNKNOWN_{args.input.stem}.txt", args.input.as_posix(), text, dataset=args.dataset)]
    probs = predict_bundle_proba(bundle, records)
    rows = topk_rows(records, probs, bundle.metadata.get("class_order", bundle.ch_svm.class_order), top_k=args.top_k)
    if args.output:
        write_csv(args.output, rows)
        print(f"Wrote predictions to {args.output}")
    else:
        print(json.dumps(rows, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CHiPS workspace CLI")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("describe", help="Describe a dataset directory or zip")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--dataset", default="auto", choices=["auto", "rost", "rostories", "extended"])
    p.add_argument("--output", type=Path)
    p.set_defaults(func=cmd_describe)

    p = sub.add_parser("normalize", help="Normalize Romanian text files")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=cmd_normalize)

    p = sub.add_parser("make-split", help="Create a grouped split JSON")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", default="auto", choices=["auto", "rost", "rostories", "extended"])
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_make_split)

    p = sub.add_parser("train-all", help="Train CH-SVM, FFT12-LR, and CHiPS-F")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", default="auto", choices=["auto", "rost", "rostories", "extended"])
    p.add_argument("--split-json", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="Use one small config per model for smoke testing")
    p.add_argument("--limit-per-author", type=int, help="Optional smoke-test limit; not for final reported results")
    p.set_defaults(func=cmd_train_all)

    p = sub.add_parser("train-all-cv", help="Train CH-SVM, FFT12-LR, and CHiPS-F with grouped CV model selection")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", default="auto", choices=["auto", "rost", "rostories", "extended"])
    p.add_argument("--split-json", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--quick", action="store_true", help="Use one small config per model for smoke testing")
    p.add_argument("--limit-per-author", type=int, help="Optional smoke-test limit; not for final reported results")
    p.set_defaults(func=cmd_train_all_cv)

    p = sub.add_parser("train-char-ngram-cv", help="Train a matched character 2-5 gram TF-IDF + LinearSVC baseline with grouped CV model selection")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", default="auto", choices=["auto", "rost", "rostories", "extended"])
    p.add_argument("--split-json", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--quick", action="store_true", help="Use one small config for smoke testing")
    p.add_argument("--limit-per-author", type=int, help="Optional smoke-test limit; not for final reported results")
    p.set_defaults(func=cmd_train_char_ngram_cv)

    p = sub.add_parser("predict", help="Predict authors using a trained CHiPS bundle")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--dataset", default="auto", choices=["auto", "rost", "rostories", "extended"])
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--output", type=Path)
    p.set_defaults(func=cmd_predict)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
