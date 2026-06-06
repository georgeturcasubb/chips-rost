#!/usr/bin/env python3
"""Leakage-safe top-5 listwise reranker for the ROST CHiPS benchmark."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chips.data import DocumentRecord, dataset_summary, load_documents
from chips.modeling import (
    CHSVMConfig,
    FFT12LRConfig,
    aggregate_probabilities_by_work_group,
    blend_probabilities,
    evaluate_predictions,
    fit_ch_svm,
    fit_fft12_lr,
    labels_from_proba,
    load_bundle,
    predict_ch_svm_proba,
    predict_fft12_proba,
    software_versions,
    write_csv,
    write_json,
)
from chips.splitting import apply_split, assert_no_group_leakage, load_split


MODEL_KEYS = ("CH_SVM", "FFT12_LR", "CHiPS_F")
POLICIES = {
    "CH_SVM_TOP5": {"source": "CH_SVM", "anchor": "CH_SVM"},
    "CHIPS_F_TOP5": {"source": "CHiPS_F", "anchor": "CHiPS_F"},
    "UNION_RRF_TOP5": {"source": "UNION_RRF", "anchor": "CH_SVM"},
}
EPSILON = 1e-12


@dataclass
class GroupBasePredictions:
    group_ids: list[str]
    y_true: list[str]
    class_order: list[str]
    probabilities: dict[str, np.ndarray]
    doc_counts: dict[str, int]
    doc_ids: dict[str, list[str]]
    char_counts: dict[str, int]


@dataclass
class RerankerDataset:
    X: np.ndarray
    y_position: np.ndarray
    y_author_index: np.ndarray
    candidates: np.ndarray
    feature_names: list[str]
    true_in_candidates: np.ndarray
    anchor_margins: np.ndarray


def _load_selected_configs(base_run: Path) -> tuple[CHSVMConfig, FFT12LRConfig, float, dict]:
    payload = json.loads((base_run / "selected_config.json").read_text(encoding="utf-8"))
    ch_config = CHSVMConfig(**payload["ch_svm"])
    fft_config = FFT12LRConfig(**payload["fft12_lr"])
    alpha = float(payload["fusion"]["alpha_ch_svm"])
    return ch_config, fft_config, alpha, payload


def _group_metadata(records: Sequence[DocumentRecord]) -> tuple[list[str], list[str], dict[str, int], dict[str, list[str]], dict[str, int]]:
    group_order: list[str] = []
    authors: dict[str, str] = {}
    doc_counts: dict[str, int] = defaultdict(int)
    doc_ids: dict[str, list[str]] = defaultdict(list)
    char_counts: dict[str, int] = defaultdict(int)
    for record in records:
        if record.work_group not in authors:
            group_order.append(record.work_group)
            authors[record.work_group] = record.author
        elif authors[record.work_group] != record.author:
            raise ValueError(f"Work group has multiple authors: {record.work_group}")
        doc_counts[record.work_group] += 1
        doc_ids[record.work_group].append(record.doc_id)
        char_counts[record.work_group] += len(record.text)
    return group_order, [authors[group] for group in group_order], dict(doc_counts), dict(doc_ids), dict(char_counts)


def _predict_group_base(
    records: Sequence[DocumentRecord],
    ch_model,
    fft_model,
    alpha: float,
    class_order: Sequence[str],
) -> GroupBasePredictions:
    ch_probs = predict_ch_svm_proba(ch_model, records)
    fft_probs = predict_fft12_proba(fft_model, records)
    ch_groups, y_true, ch_group_probs = aggregate_probabilities_by_work_group(records, ch_probs)
    fft_groups, fft_y_true, fft_group_probs = aggregate_probabilities_by_work_group(records, fft_probs)
    if ch_groups != fft_groups or y_true != fft_y_true:
        raise RuntimeError("Group order mismatch between CH-SVM and FFT12-LR predictions.")
    if list(ch_model.class_order) != list(class_order) or list(fft_model.class_order) != list(class_order):
        raise RuntimeError("Base model class order does not match the requested class order.")
    fused_group_probs = blend_probabilities(ch_group_probs, fft_group_probs, alpha)
    meta_group_order, _, doc_counts, doc_ids, char_counts = _group_metadata(records)
    if ch_groups != meta_group_order:
        raise RuntimeError("Prediction group order does not match record metadata order.")
    return GroupBasePredictions(
        group_ids=ch_groups,
        y_true=y_true,
        class_order=list(class_order),
        probabilities={"CH_SVM": ch_group_probs, "FFT12_LR": fft_group_probs, "CHiPS_F": fused_group_probs},
        doc_counts=doc_counts,
        doc_ids=doc_ids,
        char_counts=char_counts,
    )


def _make_oof_predictions(
    trainval: Sequence[DocumentRecord],
    ch_config: CHSVMConfig,
    fft_config: FFT12LRConfig,
    alpha: float,
    class_order: Sequence[str],
    n_folds: int,
    seed: int,
) -> tuple[GroupBasePredictions, list[dict]]:
    group_ids, y_true, doc_counts, doc_ids, char_counts = _group_metadata(trainval)
    y_array = np.asarray(y_true)
    label_counts = Counter(y_true)
    usable_folds = min(n_folds, min(label_counts.values()))
    if usable_folds < 2:
        raise ValueError(f"Need at least two source texts per author for OOF folds; counts={dict(label_counts)}")

    oof_probs = {key: np.full((len(group_ids), len(class_order)), np.nan, dtype=np.float64) for key in MODEL_KEYS}
    group_to_index = {group: i for i, group in enumerate(group_ids)}
    folds: list[dict] = []
    splitter = StratifiedKFold(n_splits=usable_folds, shuffle=True, random_state=seed)
    for fold_id, (fit_idx, holdout_idx) in enumerate(splitter.split(np.zeros(len(group_ids)), y_array), start=1):
        fit_groups = {group_ids[i] for i in fit_idx}
        holdout_groups = {group_ids[i] for i in holdout_idx}
        fit_records = [record for record in trainval if record.work_group in fit_groups]
        holdout_records = [record for record in trainval if record.work_group in holdout_groups]
        print(
            f"OOF fold {fold_id}/{usable_folds}: fitting on {len(fit_groups)} groups, "
            f"predicting {len(holdout_groups)} groups...",
            flush=True,
        )
        ch_model = fit_ch_svm(fit_records, ch_config)
        fft_model = fit_fft12_lr(fit_records, fft_config)
        fold_pred = _predict_group_base(holdout_records, ch_model, fft_model, alpha, class_order)
        for row_index, group_id in enumerate(fold_pred.group_ids):
            target_index = group_to_index[group_id]
            for key in MODEL_KEYS:
                oof_probs[key][target_index] = fold_pred.probabilities[key][row_index]
        folds.append(
            {
                "fold": fold_id,
                "fit_work_groups": len(fit_groups),
                "holdout_work_groups": len(holdout_groups),
                "fit_documents": len(fit_records),
                "holdout_documents": len(holdout_records),
            }
        )

    for key, matrix in oof_probs.items():
        if np.isnan(matrix).any():
            missing = [group_ids[i] for i in np.where(np.isnan(matrix).any(axis=1))[0]]
            raise RuntimeError(f"Missing OOF predictions for {key}: {missing[:10]}")
    return (
        GroupBasePredictions(
            group_ids=group_ids,
            y_true=y_true,
            class_order=list(class_order),
            probabilities=oof_probs,
            doc_counts=doc_counts,
            doc_ids=doc_ids,
            char_counts=char_counts,
        ),
        folds,
    )


def _rank_vector(probs: np.ndarray) -> np.ndarray:
    ranks = np.empty(len(probs), dtype=np.int64)
    ranks[np.argsort(-probs)] = np.arange(1, len(probs) + 1)
    return ranks


def _model_summary_features(probs: np.ndarray) -> tuple[list[float], list[str]]:
    values: list[float] = []
    names: list[str] = []
    n_classes = len(probs)
    for key in MODEL_KEYS:
        row = probs[key]
        order = np.argsort(-row)
        top1 = float(row[order[0]])
        top2 = float(row[order[1]]) if n_classes > 1 else 0.0
        entropy = float(-(row * np.log(row + EPSILON)).sum() / math.log(max(n_classes, 2)))
        values.extend([top1, top1 - top2, entropy])
        names.extend([f"{key}_top1_prob", f"{key}_top1_margin", f"{key}_entropy"])
    top1s = {key: int(np.argmax(probs[key])) for key in MODEL_KEYS}
    values.extend(
        [
            float(top1s["CH_SVM"] == top1s["FFT12_LR"]),
            float(top1s["CH_SVM"] == top1s["CHiPS_F"]),
            float(top1s["FFT12_LR"] == top1s["CHiPS_F"]),
            float(len(set(top1s.values())) == 1),
        ]
    )
    names.extend(["CH_SVM_eq_FFT12_top1", "CH_SVM_eq_CHiPS_F_top1", "FFT12_eq_CHiPS_F_top1", "all_models_agree_top1"])
    return values, names


def _candidate_indices(probs: dict[str, np.ndarray], policy: str, top_k: int) -> list[int]:
    policy_info = POLICIES[policy]
    anchor_key = policy_info["anchor"]
    anchor_index = int(np.argmax(probs[anchor_key]))

    if policy_info["source"] in MODEL_KEYS:
        ranked = [int(i) for i in np.argsort(-probs[policy_info["source"]])]
    elif policy_info["source"] == "UNION_RRF":
        scores = np.zeros_like(probs["CH_SVM"], dtype=np.float64)
        for key in MODEL_KEYS:
            for rank, idx in enumerate(np.argsort(-probs[key]), start=1):
                scores[int(idx)] += 1.0 / (60.0 + rank)
        ranked = [int(i) for i in np.argsort(-scores)]
    else:
        raise ValueError(f"Unknown candidate policy source: {policy_info['source']}")

    candidates = [anchor_index]
    for idx in ranked:
        if idx not in candidates:
            candidates.append(idx)
        if len(candidates) == top_k:
            break
    return candidates


def _build_reranker_dataset(base: GroupBasePredictions, policy: str, top_k: int) -> RerankerDataset:
    class_order = base.class_order
    class_to_index = {label: i for i, label in enumerate(class_order)}
    n_classes = len(class_order)
    rows: list[list[float]] = []
    candidate_rows: list[list[int]] = []
    y_position: list[int] = []
    y_author_index: list[int] = []
    true_in_candidates: list[bool] = []
    anchor_margins: list[float] = []
    feature_names: list[str] | None = None
    anchor_key = POLICIES[policy]["anchor"]

    for row_index, group_id in enumerate(base.group_ids):
        probs = {key: base.probabilities[key][row_index] for key in MODEL_KEYS}
        candidates = _candidate_indices(probs, policy, top_k)
        if len(candidates) != top_k:
            raise RuntimeError(f"Expected {top_k} candidates for {group_id}, got {len(candidates)}")
        true_index = class_to_index[base.y_true[row_index]]
        true_position = candidates.index(true_index) if true_index in candidates else 0
        ranks = {key: _rank_vector(probs[key]) for key in MODEL_KEYS}
        model_values, model_names = _model_summary_features(probs)
        values: list[float] = [
            math.log1p(float(base.doc_counts[group_id])),
            math.log1p(float(base.char_counts[group_id])),
            *model_values,
        ]
        names: list[str] = ["log_doc_count", "log_char_count", *model_names]

        anchor_order = np.argsort(-probs[anchor_key])
        anchor_margin = float(probs[anchor_key][anchor_order[0]] - probs[anchor_key][anchor_order[1]])
        anchor_margins.append(anchor_margin)

        for pos, candidate in enumerate(candidates):
            prefix = f"pos{pos + 1}"
            candidate_author_onehot = [1.0 if candidate == class_idx else 0.0 for class_idx in range(n_classes)]
            values.extend(candidate_author_onehot)
            names.extend([f"{prefix}_author_{author}" for author in class_order])
            values.append(float(pos == 0))
            names.append(f"{prefix}_is_anchor")
            top1_vote = 0
            top3_vote = 0
            for key in MODEL_KEYS:
                row = probs[key]
                rank = int(ranks[key][candidate])
                order = np.argsort(-row)
                anchor_prob = float(row[candidates[0]])
                top1_vote += int(rank == 1)
                top3_vote += int(rank <= 3)
                values.extend(
                    [
                        float(row[candidate]),
                        float(math.log(float(row[candidate]) + EPSILON)),
                        float(rank),
                        float(1.0 / rank),
                        float(rank / n_classes),
                        float(rank == 1),
                        float(rank <= 3),
                        float(row[order[0]] - row[candidate]),
                        float(row[candidate] - anchor_prob),
                    ]
                )
                names.extend(
                    [
                        f"{prefix}_{key}_prob",
                        f"{prefix}_{key}_log_prob",
                        f"{prefix}_{key}_rank",
                        f"{prefix}_{key}_inverse_rank",
                        f"{prefix}_{key}_rank_scaled",
                        f"{prefix}_{key}_is_top1",
                        f"{prefix}_{key}_is_top3",
                        f"{prefix}_{key}_gap_from_top1",
                        f"{prefix}_{key}_prob_minus_anchor",
                    ]
                )
            values.extend([float(top1_vote), float(top3_vote)])
            names.extend([f"{prefix}_top1_vote_count", f"{prefix}_top3_vote_count"])

        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise RuntimeError("Feature name mismatch while building reranker matrix.")
        rows.append(values)
        candidate_rows.append(candidates)
        y_position.append(true_position)
        y_author_index.append(true_index)
        true_in_candidates.append(true_index in candidates)

    if feature_names is None:
        raise ValueError("No rows available for reranker dataset.")
    return RerankerDataset(
        X=np.asarray(rows, dtype=np.float64),
        y_position=np.asarray(y_position, dtype=np.int64),
        y_author_index=np.asarray(y_author_index, dtype=np.int64),
        candidates=np.asarray(candidate_rows, dtype=np.int64),
        feature_names=feature_names,
        true_in_candidates=np.asarray(true_in_candidates, dtype=bool),
        anchor_margins=np.asarray(anchor_margins, dtype=np.float64),
    )


def _make_meta_model(c: float, class_weight: str | None, seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=c,
                    class_weight=class_weight,
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )


def _sample_weights(y_position: np.ndarray, non_anchor_weight: float) -> np.ndarray:
    return np.where(y_position == 0, 1.0, non_anchor_weight).astype(np.float64)


def _mode_grid() -> list[dict]:
    modes = [{"mode": "always", "anchor_margin_max": None, "meta_margin_min": 0.0}]
    for anchor_margin_max in (0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50):
        for meta_margin_min in (0.0, 0.05, 0.10, 0.20):
            modes.append(
                {
                    "mode": "anchor_margin_gate",
                    "anchor_margin_max": anchor_margin_max,
                    "meta_margin_min": meta_margin_min,
                }
            )
    return modes


def _position_margin(pos_probs: np.ndarray) -> np.ndarray:
    if pos_probs.shape[1] == 1:
        return np.ones(pos_probs.shape[0], dtype=np.float64)
    sorted_probs = np.sort(pos_probs, axis=1)[:, ::-1]
    return sorted_probs[:, 0] - sorted_probs[:, 1]


def _apply_mode(dataset: RerankerDataset, pos_probs: np.ndarray, mode: dict) -> np.ndarray:
    raw_positions = np.argmax(pos_probs, axis=1)
    final_positions = raw_positions.copy()
    if mode["mode"] == "anchor_margin_gate":
        meta_margins = _position_margin(pos_probs)
        use_reranker = dataset.anchor_margins <= float(mode["anchor_margin_max"])
        use_reranker &= meta_margins >= float(mode["meta_margin_min"])
        final_positions = np.where(use_reranker, raw_positions, 0)
    elif mode["mode"] != "always":
        raise ValueError(f"Unknown deployment mode: {mode}")
    return dataset.candidates[np.arange(len(final_positions)), final_positions]


def _score_author_indices(y_true: np.ndarray, y_pred: np.ndarray, class_order: Sequence[str]) -> dict:
    labels = list(range(len(class_order)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }


def _candidate_oracle(base: GroupBasePredictions, policy: str, top_k: int) -> dict:
    dataset = _build_reranker_dataset(base, policy, top_k)
    anchor_pred = dataset.candidates[:, 0]
    y_true = dataset.y_author_index
    anchor_correct = anchor_pred == y_true
    true_in = dataset.true_in_candidates
    return {
        "policy": policy,
        "top_k": top_k,
        "groups": int(len(y_true)),
        "anchor_accuracy": float(anchor_correct.mean()),
        "true_author_in_candidates": int(true_in.sum()),
        "candidate_oracle_accuracy": float(true_in.mean()),
        "fixable_anchor_errors": int((~anchor_correct & true_in).sum()),
        "unfixable_anchor_errors": int((~anchor_correct & ~true_in).sum()),
    }


def _select_reranker(
    oof_base: GroupBasePredictions,
    top_k: int,
    meta_folds: int,
    seed: int,
) -> tuple[dict, list[dict], dict[str, RerankerDataset]]:
    datasets = {policy: _build_reranker_dataset(oof_base, policy, top_k) for policy in POLICIES}
    y_author = np.asarray([oof_base.class_order.index(author) for author in oof_base.y_true], dtype=np.int64)
    label_counts = Counter(y_author.tolist())
    usable_folds = min(meta_folds, min(label_counts.values()))
    if usable_folds < 2:
        raise ValueError(f"Need at least two trainval source texts per author for meta-CV; counts={dict(label_counts)}")
    splitter = StratifiedKFold(n_splits=usable_folds, shuffle=True, random_state=seed + 17)
    splits = list(splitter.split(np.zeros(len(y_author)), y_author))

    rows: list[dict] = []
    best: dict | None = None
    best_key = (-1.0, -1.0, -10_000, 10_000)
    c_values = (0.03, 0.1, 0.3, 1.0, 3.0)
    class_weights: tuple[str | None, ...] = (None, "balanced")
    non_anchor_weights = (1.0, 2.0, 4.0)
    modes = _mode_grid()

    for policy, dataset in datasets.items():
        anchor_metrics = _score_author_indices(dataset.y_author_index, dataset.candidates[:, 0], oof_base.class_order)
        for c in c_values:
            for class_weight in class_weights:
                for non_anchor_weight in non_anchor_weights:
                    fold_pos_probs = np.zeros((len(dataset.y_position), top_k), dtype=np.float64)
                    for fold_id, (fit_idx, valid_idx) in enumerate(splits, start=1):
                        model = _make_meta_model(c, class_weight, seed + fold_id)
                        weights = _sample_weights(dataset.y_position[fit_idx], non_anchor_weight)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", ConvergenceWarning)
                            model.fit(dataset.X[fit_idx], dataset.y_position[fit_idx], clf__sample_weight=weights)
                        probs = model.predict_proba(dataset.X[valid_idx])
                        aligned = np.zeros((len(valid_idx), top_k), dtype=np.float64)
                        for class_index, label in enumerate(model.named_steps["clf"].classes_):
                            aligned[:, int(label)] = probs[:, class_index]
                        fold_pos_probs[valid_idx] = aligned

                    for mode in modes:
                        pred = _apply_mode(dataset, fold_pos_probs, mode)
                        metrics = _score_author_indices(dataset.y_author_index, pred, oof_base.class_order)
                        overrides = int((pred != dataset.candidates[:, 0]).sum())
                        row = {
                            "policy": policy,
                            "C": c,
                            "class_weight": class_weight or "",
                            "non_anchor_weight": non_anchor_weight,
                            "mode": mode["mode"],
                            "anchor_margin_max": "" if mode["anchor_margin_max"] is None else mode["anchor_margin_max"],
                            "meta_margin_min": mode["meta_margin_min"],
                            "cv_accuracy": metrics["accuracy"],
                            "cv_macro_f1": metrics["macro_f1"],
                            "anchor_cv_accuracy": anchor_metrics["accuracy"],
                            "anchor_cv_macro_f1": anchor_metrics["macro_f1"],
                            "net_accuracy_gain": metrics["accuracy"] - anchor_metrics["accuracy"],
                            "overrides": overrides,
                        }
                        rows.append(row)
                        key = (
                            metrics["macro_f1"],
                            metrics["accuracy"],
                            metrics["accuracy"] - anchor_metrics["accuracy"],
                            -overrides,
                        )
                        if key > best_key:
                            best_key = key
                            best = {
                                "policy": policy,
                                "C": c,
                                "class_weight": class_weight,
                                "non_anchor_weight": non_anchor_weight,
                                "mode": mode,
                                "cv_metrics": metrics,
                                "anchor_cv_metrics": anchor_metrics,
                                "net_accuracy_gain": metrics["accuracy"] - anchor_metrics["accuracy"],
                                "overrides": overrides,
                            }

    if best is None:
        raise RuntimeError("No reranker configuration was selected.")
    return best, rows, datasets


def _fit_final_reranker(dataset: RerankerDataset, selected: dict, seed: int) -> Pipeline:
    model = _make_meta_model(selected["C"], selected["class_weight"], seed + 101)
    weights = _sample_weights(dataset.y_position, selected["non_anchor_weight"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(dataset.X, dataset.y_position, clf__sample_weight=weights)
    return model


def _predict_positions(model: Pipeline, X: np.ndarray, top_k: int) -> np.ndarray:
    probs = model.predict_proba(X)
    aligned = np.zeros((len(X), top_k), dtype=np.float64)
    for class_index, label in enumerate(model.named_steps["clf"].classes_):
        aligned[:, int(label)] = probs[:, class_index]
    return aligned


def _base_metric_block(base: GroupBasePredictions) -> dict:
    out = {}
    for key in MODEL_KEYS:
        pred = labels_from_proba(base.probabilities[key], base.class_order)
        out[key] = evaluate_predictions(base.y_true, pred, base.class_order)
    return out


def _test_prediction_rows(
    base: GroupBasePredictions,
    dataset: RerankerDataset,
    pos_probs: np.ndarray,
    pred_indices: np.ndarray,
    selected: dict,
) -> list[dict]:
    rows: list[dict] = []
    raw_positions = np.argmax(pos_probs, axis=1)
    meta_margins = _position_margin(pos_probs)
    ch_pred = np.argmax(base.probabilities["CH_SVM"], axis=1)
    chips_f_pred = np.argmax(base.probabilities["CHiPS_F"], axis=1)
    fft_pred = np.argmax(base.probabilities["FFT12_LR"], axis=1)
    for i, group_id in enumerate(base.group_ids):
        true_idx = dataset.y_author_index[i]
        anchor_idx = int(dataset.candidates[i, 0])
        pred_idx = int(pred_indices[i])
        row = {
            "work_group": group_id,
            "doc_count": base.doc_counts[group_id],
            "doc_ids": "|".join(base.doc_ids[group_id]),
            "true_author": base.class_order[true_idx],
            "ch_svm_pred": base.class_order[int(ch_pred[i])],
            "fft12_lr_pred": base.class_order[int(fft_pred[i])],
            "chips_f_pred": base.class_order[int(chips_f_pred[i])],
            "chips_r_pred": base.class_order[pred_idx],
            "anchor_author": base.class_order[anchor_idx],
            "override_anchor": pred_idx != anchor_idx,
            "ch_svm_correct": int(ch_pred[i]) == true_idx,
            "chips_r_correct": pred_idx == true_idx,
            "true_in_candidates": bool(dataset.true_in_candidates[i]),
            "selected_policy": selected["policy"],
            "raw_meta_position": int(raw_positions[i]) + 1,
            "raw_meta_author": base.class_order[int(dataset.candidates[i, raw_positions[i]])],
            "raw_meta_probability": float(pos_probs[i, raw_positions[i]]),
            "raw_meta_margin": float(meta_margins[i]),
            "anchor_margin": float(dataset.anchor_margins[i]),
        }
        for rank, candidate in enumerate(dataset.candidates[i], start=1):
            row[f"candidate{rank}_author"] = base.class_order[int(candidate)]
            row[f"candidate{rank}_meta_probability"] = float(pos_probs[i, rank - 1])
            for key in MODEL_KEYS:
                row[f"candidate{rank}_{key}_probability"] = float(base.probabilities[key][i, candidate])
        rows.append(row)
    return rows


def _transition_summary_vs_model(
    base: GroupBasePredictions,
    dataset: RerankerDataset,
    pred_indices: np.ndarray,
    model_key: str,
    label: str,
) -> dict:
    model_pred = np.argmax(base.probabilities[model_key], axis=1)
    y_true = dataset.y_author_index
    anchor_pred = dataset.candidates[:, 0]
    pred_indices = np.asarray(pred_indices, dtype=np.int64)
    return {
        "test_overrides_vs_selected_policy_anchor": int((pred_indices != anchor_pred).sum()),
        f"test_overrides_vs_{label}": int((pred_indices != model_pred).sum()),
        f"fixed_{label}_errors": int(((model_pred != y_true) & (pred_indices == y_true)).sum()),
        f"broken_{label}_correct": int(((model_pred == y_true) & (pred_indices != y_true)).sum()),
        "wrong_to_wrong_changes": int(((model_pred != y_true) & (pred_indices != y_true) & (pred_indices != model_pred)).sum()),
        f"unchanged_{label}_errors": int(((model_pred != y_true) & (pred_indices == model_pred)).sum()),
    }


def _transition_summary_vs_ch_svm(base: GroupBasePredictions, dataset: RerankerDataset, pred_indices: np.ndarray) -> dict:
    return _transition_summary_vs_model(base, dataset, pred_indices, "CH_SVM", "ch_svm")


def _transition_summary_vs_chips_f(base: GroupBasePredictions, dataset: RerankerDataset, pred_indices: np.ndarray) -> dict:
    return _transition_summary_vs_model(base, dataset, pred_indices, "CHiPS_F", "chips_f")


def _write_csv_any(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def cmd_run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    ch_config, fft_config, alpha, selected_base = _load_selected_configs(args.base_run)
    records = load_documents(args.input, dataset=args.dataset)
    split_payload = load_split(args.split_json)
    split_docs = apply_split(records, split_payload)
    assert_no_group_leakage(split_docs)
    train = split_docs["train"]
    validation = split_docs["validation"]
    test = split_docs["test"]
    trainval = train + validation
    class_order = sorted({record.author for record in trainval})

    write_json(args.output / "dataset_summary.json", dataset_summary(records))
    write_json(args.output / "software_versions.json", software_versions())
    write_json(
        args.output / "reranker_protocol.json",
        {
            "method": "CHiPS-R top-5 listwise reranker",
            "input": str(args.input),
            "dataset": args.dataset,
            "split_json": str(args.split_json),
            "base_run": str(args.base_run),
            "base_selection": selected_base,
            "oof_folds_requested": args.oof_folds,
            "meta_folds_requested": args.meta_folds,
            "seed": args.seed,
            "top_k": args.top_k,
            "leakage_note": "Base reranker training features are grouped OOF predictions on train+validation. The held-out test split is used once after selecting the reranker by meta-CV.",
        },
    )

    print("Building grouped OOF base predictions for train+validation...", flush=True)
    oof_base, fold_rows = _make_oof_predictions(
        trainval=trainval,
        ch_config=ch_config,
        fft_config=fft_config,
        alpha=alpha,
        class_order=class_order,
        n_folds=args.oof_folds,
        seed=args.seed,
    )
    write_csv(args.output / "oof_folds.csv", fold_rows)

    print("Loading final train+validation base bundle and predicting locked test split...", flush=True)
    bundle = load_bundle(args.base_run / "chips_bundle.joblib")
    if list(bundle.ch_svm.class_order) != class_order or list(bundle.fft12_lr.class_order) != class_order:
        raise RuntimeError("Loaded base bundle class order does not match train+validation class order.")
    test_base = _predict_group_base(test, bundle.ch_svm, bundle.fft12_lr, alpha, class_order)

    oracle_rows = []
    for policy in POLICIES:
        oracle_rows.append({"split": "trainval_oof", **_candidate_oracle(oof_base, policy, args.top_k)})
        oracle_rows.append({"split": "test", **_candidate_oracle(test_base, policy, args.top_k)})
    write_csv(args.output / "candidate_policy_oracle_diagnostics.csv", oracle_rows)

    print("Selecting listwise reranker by train+validation OOF meta-CV...", flush=True)
    selected_reranker, cv_rows, train_datasets = _select_reranker(
        oof_base=oof_base,
        top_k=args.top_k,
        meta_folds=args.meta_folds,
        seed=args.seed,
    )
    write_csv(args.output / "chips_r_cv_results.csv", cv_rows)

    train_dataset = train_datasets[selected_reranker["policy"]]
    final_model = _fit_final_reranker(train_dataset, selected_reranker, args.seed)
    test_dataset = _build_reranker_dataset(test_base, selected_reranker["policy"], args.top_k)
    test_pos_probs = _predict_positions(final_model, test_dataset.X, args.top_k)
    test_pred_indices = _apply_mode(test_dataset, test_pos_probs, selected_reranker["mode"])
    chips_r_test_authors = [test_base.class_order[int(i)] for i in test_pred_indices]
    chips_r_metrics = evaluate_predictions(test_base.y_true, chips_r_test_authors, test_base.class_order)

    base_metrics = _base_metric_block(test_base)
    anchor_test_indices = test_dataset.candidates[:, 0]
    anchor_test_authors = [test_base.class_order[int(i)] for i in anchor_test_indices]
    anchor_test_metrics = evaluate_predictions(test_base.y_true, anchor_test_authors, test_base.class_order)
    selected_clean = {
        **selected_reranker,
        "class_weight": selected_reranker["class_weight"] or "",
    }
    metrics = {
        "base_models": base_metrics,
        "CHiPS_R": chips_r_metrics,
        "selected_reranker": selected_clean,
        "selected_policy_anchor_test": anchor_test_metrics,
        "transition_summary_vs_ch_svm": _transition_summary_vs_ch_svm(test_base, test_dataset, test_pred_indices),
        "transition_summary_vs_chips_f": _transition_summary_vs_chips_f(test_base, test_dataset, test_pred_indices),
        "candidate_oracle": oracle_rows,
        "_metadata": {
            "evaluation_unit": "work_group/source_text",
            "train_work_groups": len({record.work_group for record in train}),
            "validation_work_groups": len({record.work_group for record in validation}),
            "trainval_work_groups": len(oof_base.group_ids),
            "test_work_groups": len(test_base.group_ids),
            "test_documents": len(test),
            "top_k": args.top_k,
        },
    }
    write_json(args.output / "chips_r_metrics.json", metrics)
    _write_csv_any(
        args.output / "chips_r_predictions_test.csv",
        _test_prediction_rows(test_base, test_dataset, test_pos_probs, test_pred_indices, selected_reranker),
    )
    write_json(
        args.output / "chips_r_feature_names.json",
        {"policy": selected_reranker["policy"], "feature_count": len(train_dataset.feature_names), "feature_names": train_dataset.feature_names},
    )
    joblib.dump(final_model, args.output / "chips_r_meta_model.joblib")

    summary = {
        "CH_SVM": {key: base_metrics["CH_SVM"][key] for key in ("accuracy", "macro_f1", "balanced_accuracy")},
        "CHiPS_F": {key: base_metrics["CHiPS_F"][key] for key in ("accuracy", "macro_f1", "balanced_accuracy")},
        "CHiPS_R": {key: chips_r_metrics[key] for key in ("accuracy", "macro_f1", "balanced_accuracy")},
        "selected_policy": selected_reranker["policy"],
        "selected_mode": selected_reranker["mode"],
        "cv_metrics": selected_reranker["cv_metrics"],
    }
    print(json.dumps(summary, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate a leakage-safe CHiPS-R top-5 reranker.")
    parser.add_argument("--input", type=Path, required=True, help="Normalized ROST dataset directory or zip.")
    parser.add_argument("--dataset", default="rost", choices=["auto", "rost", "rostories", "extended"])
    parser.add_argument("--split-json", type=Path, required=True, help="Locked grouped split JSON.")
    parser.add_argument("--base-run", type=Path, required=True, help="Run directory with selected_config.json and chips_bundle.joblib.")
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
