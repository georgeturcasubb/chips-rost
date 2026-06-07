#!/usr/bin/env python3
"""Export compact CHiPS-R reranker selection and transition details."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    display_name: str
    role: str
    run_dir: Path


DEFAULT_RUNS = [
    RunSpec(
        "rost",
        "Locked ROST",
        "primary ablation",
        ROOT / "experiments" / "runs" / "rost_cv5_chips_r",
    ),
    RunSpec(
        "ro_stories_original",
        "Public ro-stories original",
        "contextual public corpus",
        ROOT / "experiments" / "runs" / "ro_stories_original_cv5_chips_r",
    ),
    RunSpec(
        "rostories_cleaned",
        "ROSTories-cleaned",
        "secondary ROST-overlapping corpus",
        ROOT / "experiments" / "runs" / "rostories_cleaned_cv5_chips_r",
    ),
]

DETAIL_FIELDS = [
    "run_id",
    "corpus",
    "role",
    "run_dir",
    "dataset",
    "input",
    "split_json",
    "base_run",
    "train_work_groups",
    "validation_work_groups",
    "trainval_work_groups",
    "test_work_groups",
    "test_documents",
    "top_k",
    "oof_folds_requested",
    "meta_folds_requested",
    "seed",
    "model_class",
    "scaler",
    "policy",
    "policy_display",
    "C",
    "class_weight",
    "non_anchor_weight",
    "gate_mode",
    "anchor_margin_max",
    "meta_margin_min",
    "gate_display",
    "feature_count",
    "feature_families",
    "feature_family_counts_json",
    "cv_accuracy",
    "cv_macro_f1",
    "anchor_cv_accuracy",
    "anchor_cv_macro_f1",
    "net_accuracy_gain",
    "cv_overrides",
    "ch_svm_correct",
    "chips_f_correct",
    "chips_r_correct",
    "chips_r_accuracy",
    "chips_r_macro_f1",
    "chips_r_balanced_accuracy",
    "delta_correct_vs_ch_svm",
    "delta_correct_vs_chips_f",
    "fixes_vs_ch_svm",
    "breaks_vs_ch_svm",
    "wrong_to_wrong_vs_ch_svm",
    "fixes_vs_chips_f",
    "breaks_vs_chips_f",
    "wrong_to_wrong_vs_chips_f",
    "selected_policy_test_oracle_accuracy",
    "selected_policy_test_true_author_in_candidates",
    "selected_policy_test_fixable_anchor_errors",
    "selected_policy_test_unfixable_anchor_errors",
    "improves_chips_f",
    "conclusion",
]

TRANSITION_FIELDS = [
    "run_id",
    "corpus",
    "comparator",
    "comparator_column",
    "test_groups",
    "comparator_correct",
    "chips_r_correct",
    "delta_correct",
    "overrides",
    "fixed_errors",
    "broken_correct",
    "wrong_to_wrong_changes",
    "unchanged_comparator_errors",
]

FEATURE_FAMILY_LABELS = {
    "document_size": "document size",
    "base_confidence": "base confidence",
    "base_agreement": "base-model agreement",
    "candidate_author_position": "author-position one-hot",
    "candidate_anchor": "anchor flags/deltas",
    "candidate_probability": "candidate probabilities",
    "candidate_rank": "candidate ranks",
    "candidate_vote": "top-k vote counts",
    "other": "other",
}

FEATURE_FAMILY_ORDER = [
    "document_size",
    "base_confidence",
    "base_agreement",
    "candidate_probability",
    "candidate_rank",
    "candidate_anchor",
    "candidate_vote",
    "candidate_author_position",
    "other",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


def format_float(value: object, digits: int = 4) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.{digits}f}"


def latex_escape(text: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    out = str(text)
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def policy_display(policy: str) -> str:
    labels = {
        "CH_SVM_TOP5": "CH-SVM top-5",
        "CHIPS_F_TOP5": "CHiPS-F top-5",
        "UNION_RRF_TOP5": "RRF union top-5",
    }
    return labels.get(policy, policy.replace("_", " "))


def format_gate(mode: dict[str, object]) -> str:
    if mode.get("mode") != "anchor_margin_gate":
        return str(mode.get("mode", ""))
    return (
        f"anchor margin <= {float(mode['anchor_margin_max']):.2f}; "
        f"meta margin >= {float(mode['meta_margin_min']):.2f}"
    )


def feature_family(name: str) -> str:
    if name in {"log_doc_count", "log_char_count"}:
        return "document_size"
    if "_author_" in name:
        return "candidate_author_position"
    if name.endswith(("_top1_prob", "_top1_margin", "_entropy")) and not name.startswith("pos"):
        return "base_confidence"
    if "_eq_" in name or name == "all_models_agree_top1":
        return "base_agreement"
    if name.endswith("_is_anchor") or name.endswith("_gap_from_top1") or name.endswith("_prob_minus_anchor"):
        return "candidate_anchor"
    if name.endswith("_vote_count"):
        return "candidate_vote"
    if (
        name.endswith("_rank")
        or name.endswith("_inverse_rank")
        or name.endswith("_rank_scaled")
        or name.endswith("_is_top1")
        or name.endswith("_is_top3")
    ):
        return "candidate_rank"
    if name.endswith("_prob") or name.endswith("_log_prob"):
        return "candidate_probability"
    return "other"


def feature_family_counts(feature_names: Iterable[str]) -> dict[str, int]:
    counts = {key: 0 for key in FEATURE_FAMILY_ORDER}
    for name in feature_names:
        counts[feature_family(name)] += 1
    return {key: counts[key] for key in FEATURE_FAMILY_ORDER if counts[key]}


def feature_family_phrase(feature_count: int, counts: dict[str, int]) -> str:
    labels = [FEATURE_FAMILY_LABELS[key] for key in FEATURE_FAMILY_ORDER if counts.get(key)]
    return f"{feature_count}: " + "; ".join(labels)


def transition_summary(
    rows: list[dict[str, str]],
    comparator: str,
    comparator_column: str,
) -> dict[str, object]:
    required = {"true_author", comparator_column, "chips_r_pred"}
    missing = required - set(rows[0]) if rows else set()
    if missing:
        raise ValueError(f"prediction rows missing required columns: {sorted(missing)}")

    overrides = 0
    fixed = 0
    broken = 0
    wrong_to_wrong = 0
    unchanged_errors = 0
    comparator_correct = 0
    chips_r_correct = 0

    for row in rows:
        truth = row["true_author"]
        base_pred = row[comparator_column]
        chips_r_pred = row["chips_r_pred"]
        base_ok = base_pred == truth
        chips_ok = chips_r_pred == truth
        changed = base_pred != chips_r_pred

        comparator_correct += int(base_ok)
        chips_r_correct += int(chips_ok)
        overrides += int(changed)
        fixed += int(changed and not base_ok and chips_ok)
        broken += int(changed and base_ok and not chips_ok)
        wrong_to_wrong += int(changed and not base_ok and not chips_ok)
        unchanged_errors += int((not changed) and (not base_ok))

    return {
        "comparator": comparator,
        "comparator_column": comparator_column,
        "test_groups": len(rows),
        "comparator_correct": comparator_correct,
        "chips_r_correct": chips_r_correct,
        "delta_correct": chips_r_correct - comparator_correct,
        "overrides": overrides,
        "fixed_errors": fixed,
        "broken_correct": broken,
        "wrong_to_wrong_changes": wrong_to_wrong,
        "unchanged_comparator_errors": unchanged_errors,
    }


def selected_policy_oracle(metrics: dict, policy: str) -> dict[str, object]:
    for row in metrics.get("candidate_oracle", []):
        if row.get("split") == "test" and row.get("policy") == policy:
            return row
    return {}


def conclusion_for(role: str, transition_vs_chips_f: dict[str, object]) -> str:
    delta = int(transition_vs_chips_f["delta_correct"])
    if delta > 0:
        return f"improves CHiPS-F by +{delta} source-text groups; {role}"
    if delta == 0:
        return f"ties CHiPS-F; {role}"
    return f"does not improve CHiPS-F ({delta} source-text groups); {role}"


def detail_row(spec: RunSpec) -> tuple[dict[str, object], list[dict[str, object]]]:
    protocol = read_json(spec.run_dir / "reranker_protocol.json")
    metrics = read_json(spec.run_dir / "chips_r_metrics.json")
    predictions = read_csv(spec.run_dir / "chips_r_predictions_test.csv")
    features = read_json(spec.run_dir / "chips_r_feature_names.json")

    selected = metrics["selected_reranker"]
    mode = selected.get("mode", {})
    meta = metrics.get("_metadata", {})
    top_k = int(protocol.get("top_k", meta.get("top_k", 5)))
    feature_names = features.get("feature_names", [])
    feature_count = int(features.get("feature_count", len(feature_names)))
    family_counts = feature_family_counts(feature_names)

    transition_ch_svm = transition_summary(predictions, "CH_SVM", "ch_svm_pred")
    transition_chips_f = transition_summary(predictions, "CHiPS_F", "chips_f_pred")
    oracle = selected_policy_oracle(metrics, str(selected["policy"]))

    row = {
        "run_id": spec.run_id,
        "corpus": spec.display_name,
        "role": spec.role,
        "run_dir": display_path(spec.run_dir),
        "dataset": protocol.get("dataset", ""),
        "input": protocol.get("input", ""),
        "split_json": protocol.get("split_json", ""),
        "base_run": protocol.get("base_run", ""),
        "train_work_groups": meta.get("train_work_groups", ""),
        "validation_work_groups": meta.get("validation_work_groups", ""),
        "trainval_work_groups": meta.get("trainval_work_groups", ""),
        "test_work_groups": meta.get("test_work_groups", transition_chips_f["test_groups"]),
        "test_documents": meta.get("test_documents", ""),
        "top_k": top_k,
        "oof_folds_requested": protocol.get("oof_folds_requested", ""),
        "meta_folds_requested": protocol.get("meta_folds_requested", ""),
        "seed": protocol.get("seed", ""),
        "model_class": "LogisticRegression(lbfgs)",
        "scaler": "StandardScaler",
        "policy": selected["policy"],
        "policy_display": policy_display(str(selected["policy"])),
        "C": selected["C"],
        "class_weight": selected.get("class_weight") or "none",
        "non_anchor_weight": selected["non_anchor_weight"],
        "gate_mode": mode.get("mode", ""),
        "anchor_margin_max": mode.get("anchor_margin_max", ""),
        "meta_margin_min": mode.get("meta_margin_min", ""),
        "gate_display": format_gate(mode),
        "feature_count": feature_count,
        "feature_families": feature_family_phrase(feature_count, family_counts),
        "feature_family_counts_json": json.dumps(family_counts, sort_keys=True),
        "cv_accuracy": selected["cv_metrics"]["accuracy"],
        "cv_macro_f1": selected["cv_metrics"]["macro_f1"],
        "anchor_cv_accuracy": selected["anchor_cv_metrics"]["accuracy"],
        "anchor_cv_macro_f1": selected["anchor_cv_metrics"]["macro_f1"],
        "net_accuracy_gain": selected["net_accuracy_gain"],
        "cv_overrides": selected["overrides"],
        "ch_svm_correct": transition_ch_svm["comparator_correct"],
        "chips_f_correct": transition_chips_f["comparator_correct"],
        "chips_r_correct": transition_chips_f["chips_r_correct"],
        "chips_r_accuracy": metrics["CHiPS_R"]["accuracy"],
        "chips_r_macro_f1": metrics["CHiPS_R"]["macro_f1"],
        "chips_r_balanced_accuracy": metrics["CHiPS_R"]["balanced_accuracy"],
        "delta_correct_vs_ch_svm": transition_ch_svm["delta_correct"],
        "delta_correct_vs_chips_f": transition_chips_f["delta_correct"],
        "fixes_vs_ch_svm": transition_ch_svm["fixed_errors"],
        "breaks_vs_ch_svm": transition_ch_svm["broken_correct"],
        "wrong_to_wrong_vs_ch_svm": transition_ch_svm["wrong_to_wrong_changes"],
        "fixes_vs_chips_f": transition_chips_f["fixed_errors"],
        "breaks_vs_chips_f": transition_chips_f["broken_correct"],
        "wrong_to_wrong_vs_chips_f": transition_chips_f["wrong_to_wrong_changes"],
        "selected_policy_test_oracle_accuracy": oracle.get("candidate_oracle_accuracy", ""),
        "selected_policy_test_true_author_in_candidates": oracle.get("true_author_in_candidates", ""),
        "selected_policy_test_fixable_anchor_errors": oracle.get("fixable_anchor_errors", ""),
        "selected_policy_test_unfixable_anchor_errors": oracle.get("unfixable_anchor_errors", ""),
        "improves_chips_f": int(transition_chips_f["delta_correct"]) > 0,
        "conclusion": conclusion_for(spec.role, transition_chips_f),
    }
    transitions = []
    for summary in (transition_ch_svm, transition_chips_f):
        transitions.append({"run_id": spec.run_id, "corpus": spec.display_name, **summary})
    return row, transitions


def write_latex_table(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table}",
        r"  \tbl{\caption{Selected CHiPS-R reranker details from the authors' grouped source-text experiments. LR denotes logistic regression fitted after feature standardization. Transition counts compare CHiPS-R against CHiPS-F on held-out source-text groups; fixed and broken are counted relative to CHiPS-F.}\label{tab:chips-r-detail}}",
        r"  {\tablefont\begin{tabular}{@{}p{0.13\textwidth}p{0.20\textwidth}p{0.22\textwidth}p{0.17\textwidth}p{0.13\textwidth}@{}}",
        r"    \hline",
        r"    Corpus & Selected model and candidate set & Gate and feature families & Held-out transition vs CHiPS-F & Interpretation\\",
        r"    \hline",
    ]
    for row in rows:
        model_cell = (
            f"Standardized LR, C={row['C']}, class weight={row['class_weight']}, "
            f"non-anchor w={row['non_anchor_weight']}; "
            f"{row['policy_display']}; top-{row['top_k']}"
        )
        latex_gate = str(row["gate_display"]).replace(" <= ", " at most ").replace(" >= ", " at least ")
        gate_cell = f"{latex_gate}; {row['feature_families']}"
        transition_cell = (
            f"{row['chips_r_correct']}/{row['test_work_groups']} vs "
            f"{row['chips_f_correct']}/{row['test_work_groups']}; "
            f"fixed {row['fixes_vs_chips_f']}, broke {row['breaks_vs_chips_f']}, "
            f"wrong-to-wrong {row['wrong_to_wrong_vs_chips_f']}"
        )
        line = "    " + " & ".join(
            [
                latex_escape(row["corpus"]),
                latex_escape(model_cell),
                latex_escape(gate_cell),
                latex_escape(transition_cell),
                latex_escape(row["conclusion"]),
            ]
        ) + r"\\"
        lines.append(line)
    lines.extend(
        [
            r"    \hline",
            r"  \end{tabular}}",
            r"  {\begin{tabnote}Protocol: all rows use grouped 70/15/15 source-text splits with seed 42. Candidate policy, model settings, and gates are selected only from grouped OOF train/validation predictions; generated by scripts/chips\_r\_detail.py.\end{tabnote}}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_metadata(path: Path, rows: list[dict[str, object]], transitions: list[dict[str, object]]) -> None:
    payload = {
        "created_by": "scripts/chips_r_detail.py",
        "unit": "held-out source-text group",
        "model_class": "StandardScaler + LogisticRegression(lbfgs, max_iter=2000)",
        "source_files_per_run": [
            "reranker_protocol.json",
            "chips_r_metrics.json",
            "chips_r_predictions_test.csv",
            "chips_r_feature_names.json",
        ],
        "feature_family_labels": FEATURE_FAMILY_LABELS,
        "transition_note": "Fixed, broken, and wrong-to-wrong counts are recomputed from held-out prediction CSVs, not copied from prose.",
        "runs": [
            {
                "run_id": row["run_id"],
                "corpus": row["corpus"],
                "role": row["role"],
                "run_dir": row["run_dir"],
                "policy": row["policy"],
                "feature_count": row["feature_count"],
                "conclusion": row["conclusion"],
            }
            for row in rows
        ],
        "transition_rows": len(transitions),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export compact CHiPS-R detail tables from saved run artifacts.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "results" / "chips_r_detail",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detail_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    for spec in DEFAULT_RUNS:
        row, transitions = detail_row(spec)
        detail_rows.append(row)
        transition_rows.extend(transitions)

    write_csv(args.output / "chips_r_detail.csv", detail_rows, DETAIL_FIELDS)
    write_csv(args.output / "chips_r_transition_summary.csv", transition_rows, TRANSITION_FIELDS)
    write_latex_table(args.output / "table_chips_r_detail.tex", detail_rows)
    write_metadata(args.output / "chips_r_detail_metadata.json", detail_rows, transition_rows)
    print(args.output)


if __name__ == "__main__":
    main()
