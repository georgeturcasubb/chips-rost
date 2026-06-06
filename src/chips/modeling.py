"""Model training and evaluation for the CHiPS code release."""
from __future__ import annotations

import csv
import importlib.metadata as importlib_metadata
import json
import math
import platform
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import DocumentRecord
from .features import CharHistogramVectorizer, FFT12Vectorizer, chunk_text
from . import __version__

EPSILON = 1e-12
PACKAGE_DISTRIBUTIONS = ("numpy", "pandas", "scipy", "scikit-learn", "joblib")


def software_versions() -> dict:
    """Return the local software versions needed to reproduce a run."""
    packages: dict[str, str] = {}
    for package in PACKAGE_DISTRIBUTIONS:
        try:
            packages[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    return {
        "chips": __version__,
        "python": sys.version.split()[0],
        "python_full": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


@dataclass(frozen=True)
class CHSVMConfig:
    chunk_size: int = 1024
    overlap: int = 0
    c: float = 0.1
    class_weight: str | None = "balanced"
    min_char_count: int = 20


@dataclass(frozen=True)
class FFT12LRConfig:
    c: float = 0.3
    class_weight: str | None = "balanced"
    nfft: int = 2048


@dataclass(frozen=True)
class CharNGramSVMConfig:
    ngram_min: int = 2
    ngram_max: int = 5
    c: float = 1.0
    class_weight: str | None = "balanced"
    min_df: int = 1
    sublinear_tf: bool = True


@dataclass(frozen=True)
class FusionConfig:
    alpha_ch_svm: float = 0.8


@dataclass
class CHSVMModel:
    config: CHSVMConfig
    vectorizer: CharHistogramVectorizer
    scaler: StandardScaler
    classifier: LinearSVC
    class_order: list[str]


@dataclass
class FFT12LRModel:
    config: FFT12LRConfig
    vectorizer: FFT12Vectorizer
    scaler: StandardScaler
    classifier: LogisticRegression
    class_order: list[str]


@dataclass
class CharNGramSVMModel:
    config: CharNGramSVMConfig
    vectorizer: TfidfVectorizer
    classifier: LinearSVC
    class_order: list[str]


@dataclass
class CHiPSBundle:
    ch_svm: CHSVMModel
    fft12_lr: FFT12LRModel
    fusion: FusionConfig
    metadata: dict


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / (exp_scores.sum(axis=1, keepdims=True) + EPSILON)


def _align_matrix(matrix: np.ndarray, model_classes: Sequence[str], class_order: Sequence[str]) -> np.ndarray:
    if set(model_classes) != set(class_order):
        raise ValueError(f"Class mismatch. Model={list(model_classes)} expected={list(class_order)}")
    order = [list(model_classes).index(label) for label in class_order]
    return np.asarray(matrix)[:, order]


def _align_vector(vector: np.ndarray, model_classes: Sequence[str], class_order: Sequence[str]) -> np.ndarray:
    if set(model_classes) != set(class_order):
        raise ValueError(f"Class mismatch. Model={list(model_classes)} expected={list(class_order)}")
    order = [list(model_classes).index(label) for label in class_order]
    return np.asarray(vector)[order]


def fit_ch_svm(records: Sequence[DocumentRecord], config: CHSVMConfig) -> CHSVMModel:
    vectorizer = CharHistogramVectorizer(min_char_count=config.min_char_count).fit([r.text for r in records])
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for record in records:
        for chunk in chunk_text(record.text, config.chunk_size, config.overlap):
            rows.append(vectorizer.transform_chunk(chunk))
            labels.append(record.author)
    if not rows:
        raise ValueError("No chunks produced for CH-SVM training.")
    X = np.vstack(rows)
    y = np.asarray(labels)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    classifier = LinearSVC(C=config.c, class_weight=config.class_weight, max_iter=20000, dual=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        classifier.fit(X_scaled, y)
    return CHSVMModel(config=config, vectorizer=vectorizer, scaler=scaler, classifier=classifier, class_order=sorted(set(y.tolist())))


def predict_ch_svm_scores(model: CHSVMModel, records: Sequence[DocumentRecord]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for record in records:
        chunks = chunk_text(record.text, model.config.chunk_size, model.config.overlap)
        X = model.vectorizer.transform_chunks(chunks)
        scores = model.classifier.decision_function(model.scaler.transform(X))
        if scores.ndim == 1:
            scores = np.column_stack((-scores, scores))
        avg_scores = scores.mean(axis=0)
        rows.append(_align_vector(avg_scores, model.classifier.classes_, model.class_order))
    return np.vstack(rows) if rows else np.empty((0, len(model.class_order)))


def predict_ch_svm_proba(model: CHSVMModel, records: Sequence[DocumentRecord]) -> np.ndarray:
    return softmax(predict_ch_svm_scores(model, records))


def fit_fft12_lr(records: Sequence[DocumentRecord], config: FFT12LRConfig) -> FFT12LRModel:
    vectorizer = FFT12Vectorizer(nfft=config.nfft)
    X = vectorizer.transform([r.text for r in records])
    y = np.asarray([r.author for r in records])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    classifier = LogisticRegression(
        C=config.c,
        class_weight=config.class_weight,
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        classifier.fit(X_scaled, y)
    return FFT12LRModel(config=config, vectorizer=vectorizer, scaler=scaler, classifier=classifier, class_order=sorted(set(y.tolist())))


def predict_fft12_proba(model: FFT12LRModel, records: Sequence[DocumentRecord]) -> np.ndarray:
    X = model.vectorizer.transform([r.text for r in records])
    probs = model.classifier.predict_proba(model.scaler.transform(X))
    return _align_matrix(probs, model.classifier.classes_, model.class_order)


def fit_char_ngram_svm(records: Sequence[DocumentRecord], config: CharNGramSVMConfig) -> CharNGramSVMModel:
    if config.ngram_min < 1 or config.ngram_max < config.ngram_min:
        raise ValueError(f"Invalid n-gram range: {config.ngram_min}--{config.ngram_max}")
    if config.min_df < 1:
        raise ValueError("min_df must be a positive integer")
    texts = [r.text for r in records]
    y = np.asarray([r.author for r in records])
    vectorizer = TfidfVectorizer(
        analyzer="char",
        lowercase=False,
        ngram_range=(config.ngram_min, config.ngram_max),
        min_df=config.min_df,
        sublinear_tf=config.sublinear_tf,
        use_idf=True,
        norm="l2",
        dtype=np.float64,
    )
    X = vectorizer.fit_transform(texts)
    classifier = LinearSVC(C=config.c, class_weight=config.class_weight, max_iter=20000, dual="auto")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        classifier.fit(X, y)
    return CharNGramSVMModel(
        config=config,
        vectorizer=vectorizer,
        classifier=classifier,
        class_order=sorted(set(y.tolist())),
    )


def predict_char_ngram_svm_scores(model: CharNGramSVMModel, records: Sequence[DocumentRecord]) -> np.ndarray:
    X = model.vectorizer.transform([r.text for r in records])
    scores = model.classifier.decision_function(X)
    if scores.ndim == 1:
        scores = np.column_stack((-scores, scores))
    return _align_matrix(scores, model.classifier.classes_, model.class_order)


def predict_char_ngram_svm_proba(model: CharNGramSVMModel, records: Sequence[DocumentRecord]) -> np.ndarray:
    return softmax(predict_char_ngram_svm_scores(model, records))


def blend_probabilities(ch_probs: np.ndarray, fft_probs: np.ndarray, alpha: float) -> np.ndarray:
    if ch_probs.shape != fft_probs.shape:
        raise ValueError(f"Probability shape mismatch: {ch_probs.shape} vs {fft_probs.shape}")
    return alpha * ch_probs + (1.0 - alpha) * fft_probs


def labels_from_proba(probs: np.ndarray, class_order: Sequence[str]) -> list[str]:
    return [class_order[int(i)] for i in np.argmax(probs, axis=1)]


def aggregate_probabilities_by_work_group(
    records: Sequence[DocumentRecord],
    probs: np.ndarray,
) -> tuple[list[str], list[str], np.ndarray]:
    """Average record-level probabilities to source-text/work-group level."""
    if len(records) != len(probs):
        raise ValueError(f"Record/probability length mismatch: {len(records)} vs {len(probs)}")
    group_order: list[str] = []
    group_authors: dict[str, str] = {}
    group_probs: dict[str, list[np.ndarray]] = defaultdict(list)
    for record, row in zip(records, probs):
        if record.work_group not in group_authors:
            group_order.append(record.work_group)
            group_authors[record.work_group] = record.author
        elif group_authors[record.work_group] != record.author:
            raise ValueError(f"Work group has multiple authors: {record.work_group}")
        group_probs[record.work_group].append(np.asarray(row, dtype=np.float64))
    if not group_order:
        return [], [], np.empty((0, probs.shape[1] if probs.ndim == 2 else 0), dtype=np.float64)
    averaged = np.vstack([np.vstack(group_probs[group]).mean(axis=0) for group in group_order])
    labels = [group_authors[group] for group in group_order]
    return group_order, labels, averaged


def evaluate_predictions(y_true: Sequence[str], y_pred: Sequence[str], class_order: Sequence[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(class_order), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "classification_report": classification_report(y_true, y_pred, labels=list(class_order), output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(class_order)).tolist(),
    }


def majority_baseline(records_train: Sequence[DocumentRecord], records_eval: Sequence[DocumentRecord], class_order: Sequence[str]) -> dict:
    counts: dict[str, int] = {}
    for r in records_train:
        counts[r.author] = counts.get(r.author, 0) + 1
    majority = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    y_true = [r.author for r in records_eval]
    y_pred = [majority] * len(records_eval)
    out = evaluate_predictions(y_true, y_pred, class_order)
    out["majority_author"] = majority
    return out


def majority_baseline_by_work_group(records_train: Sequence[DocumentRecord], records_eval: Sequence[DocumentRecord], class_order: Sequence[str]) -> dict:
    counts: dict[str, int] = {}
    seen_groups: set[str] = set()
    for r in records_train:
        if r.work_group in seen_groups:
            continue
        seen_groups.add(r.work_group)
        counts[r.author] = counts.get(r.author, 0) + 1
    majority = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    group_ids, y_true, _ = aggregate_probabilities_by_work_group(
        records_eval,
        np.zeros((len(records_eval), len(class_order)), dtype=np.float64),
    )
    y_pred = [majority] * len(group_ids)
    out = evaluate_predictions(y_true, y_pred, class_order)
    out["majority_author"] = majority
    return out


def _score_key(metrics: dict) -> tuple[float, float]:
    return (float(metrics["macro_f1"]), float(metrics["accuracy"]))


def select_ch_svm_config(train: Sequence[DocumentRecord], validation: Sequence[DocumentRecord], configs: Sequence[CHSVMConfig]) -> tuple[CHSVMConfig, list[dict]]:
    class_order = sorted({r.author for r in train})
    rows: list[dict] = []
    best_config = configs[0]
    best_key = (-1.0, -1.0)
    for order, config in enumerate(configs):
        model = fit_ch_svm(train, config)
        probs = predict_ch_svm_proba(model, validation)
        _, y_group, group_probs = aggregate_probabilities_by_work_group(validation, probs)
        pred = labels_from_proba(group_probs, model.class_order)
        metrics = evaluate_predictions(y_group, pred, class_order)
        row = {"order": order, **asdict(config), "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "balanced_accuracy": metrics["balanced_accuracy"]}
        rows.append(row)
        key = _score_key(metrics)
        if key > best_key:
            best_key = key
            best_config = config
    return best_config, rows


def select_fft12_config(train: Sequence[DocumentRecord], validation: Sequence[DocumentRecord], configs: Sequence[FFT12LRConfig]) -> tuple[FFT12LRConfig, list[dict]]:
    class_order = sorted({r.author for r in train})
    rows: list[dict] = []
    best_config = configs[0]
    best_key = (-1.0, -1.0)
    for order, config in enumerate(configs):
        model = fit_fft12_lr(train, config)
        probs = predict_fft12_proba(model, validation)
        _, y_group, group_probs = aggregate_probabilities_by_work_group(validation, probs)
        pred = labels_from_proba(group_probs, model.class_order)
        metrics = evaluate_predictions(y_group, pred, class_order)
        row = {"order": order, **asdict(config), "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "balanced_accuracy": metrics["balanced_accuracy"]}
        rows.append(row)
        key = _score_key(metrics)
        if key > best_key:
            best_key = key
            best_config = config
    return best_config, rows


def select_fusion_alpha(y_true: Sequence[str], class_order: Sequence[str], ch_probs: np.ndarray, fft_probs: np.ndarray, alphas: Sequence[float]) -> tuple[float, list[dict]]:
    rows: list[dict] = []
    best_alpha = float(alphas[0])
    best_key = (-1.0, -1.0)
    for alpha in alphas:
        fused = blend_probabilities(ch_probs, fft_probs, alpha)
        pred = labels_from_proba(fused, class_order)
        metrics = evaluate_predictions(y_true, pred, class_order)
        row = {"alpha_ch_svm": float(alpha), "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "balanced_accuracy": metrics["balanced_accuracy"]}
        rows.append(row)
        key = _score_key(metrics)
        if key > best_key:
            best_key = key
            best_alpha = float(alpha)
    return best_alpha, rows


def default_ch_grid(quick: bool = False) -> list[CHSVMConfig]:
    if quick:
        return [CHSVMConfig(chunk_size=1024, overlap=0, c=0.1, class_weight="balanced")]
    return [
        CHSVMConfig(chunk_size=chunk, overlap=overlap, c=c, class_weight=weight)
        for chunk in (768, 1024, 1280)
        for overlap in (0, chunk // 2)
        for c in (0.03, 0.1, 0.3, 1.0, 3.0)
        for weight in (None, "balanced")
    ]


def default_fft_grid(quick: bool = False) -> list[FFT12LRConfig]:
    if quick:
        return [FFT12LRConfig(c=0.3, class_weight="balanced")]
    return [FFT12LRConfig(c=c, class_weight=weight) for c in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0) for weight in (None, "balanced")]


def default_char_ngram_grid(quick: bool = False) -> list[CharNGramSVMConfig]:
    if quick:
        return [CharNGramSVMConfig(c=1.0, class_weight="balanced", min_df=1, sublinear_tf=True)]
    return [
        CharNGramSVMConfig(c=c, class_weight=weight, min_df=min_df, sublinear_tf=sublinear_tf)
        for c in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
        for weight in (None, "balanced")
        for min_df in (1, 2)
        for sublinear_tf in (False, True)
    ]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict]) -> None:
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


def save_bundle(path: Path, bundle: CHiPSBundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def save_char_ngram_svm(path: Path, model: CharNGramSVMModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_bundle(path: Path) -> CHiPSBundle:
    obj = joblib.load(path)
    if not isinstance(obj, CHiPSBundle):
        raise TypeError(f"Expected CHiPSBundle in {path}, got {type(obj)!r}")
    return obj


def predict_bundle_proba(bundle: CHiPSBundle, records: Sequence[DocumentRecord]) -> np.ndarray:
    ch_probs = predict_ch_svm_proba(bundle.ch_svm, records)
    fft_probs = predict_fft12_proba(bundle.fft12_lr, records)
    return blend_probabilities(ch_probs, fft_probs, bundle.fusion.alpha_ch_svm)


def topk_rows(records: Sequence[DocumentRecord], probs: np.ndarray, class_order: Sequence[str], top_k: int = 5) -> list[dict]:
    rows: list[dict] = []
    for record, prob in zip(records, probs):
        order = list(np.argsort(prob)[::-1][:top_k])
        row = {
            "doc_id": record.doc_id,
            "true_author": record.author,
            "predicted_author": class_order[order[0]],
            "confidence": float(prob[order[0]]),
            "margin": float(prob[order[0]] - prob[order[1]]) if len(order) > 1 else 1.0,
        }
        for rank, idx in enumerate(order, start=1):
            row[f"top{rank}_author"] = class_order[idx]
            row[f"top{rank}_probability"] = float(prob[idx])
        rows.append(row)
    return rows


def topk_group_rows(records: Sequence[DocumentRecord], probs: np.ndarray, class_order: Sequence[str], top_k: int = 5) -> list[dict]:
    group_ids, labels, group_probs = aggregate_probabilities_by_work_group(records, probs)
    group_doc_ids: dict[str, list[str]] = defaultdict(list)
    for record in records:
        group_doc_ids[record.work_group].append(record.doc_id)
    rows: list[dict] = []
    for group_id, true_author, prob in zip(group_ids, labels, group_probs):
        order = list(np.argsort(prob)[::-1][:top_k])
        row = {
            "work_group": group_id,
            "doc_count": len(group_doc_ids[group_id]),
            "doc_ids": "|".join(group_doc_ids[group_id]),
            "true_author": true_author,
            "predicted_author": class_order[order[0]],
            "confidence": float(prob[order[0]]),
            "margin": float(prob[order[0]] - prob[order[1]]) if len(order) > 1 else 1.0,
        }
        for rank, idx in enumerate(order, start=1):
            row[f"top{rank}_author"] = class_order[idx]
            row[f"top{rank}_probability"] = float(prob[idx])
        rows.append(row)
    return rows
