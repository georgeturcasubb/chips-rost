#!/usr/bin/env python3
"""Generate LaTeX table snippets from a CHiPS metrics.json file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = ["majority_baseline", "CH_SVM", "FFT12_LR", "CHiPS_F", "CHiPS_R", "CHAR_NGRAM_2_5_SVM"]
DISPLAY = {
    "majority_baseline": "Majority baseline",
    "CH_SVM": "CH-SVM",
    "FFT12_LR": "FFT12-LR",
    "CHiPS_F": "CHiPS-F",
    "CHiPS_R": "CHiPS-R",
    "CHAR_NGRAM_2_5_SVM": "Char 2--5 TF-IDF SVM",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--caption", default="ROST grouped source-level results.")
    parser.add_argument("--label", default="tab:rost-results")
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{args.caption}}}",
        rf"\label{{{args.label}}}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & Accuracy & Macro-F1 & Balanced acc. \\",
        r"\midrule",
    ]
    for key in ORDER:
        if key not in metrics:
            continue
        m = metrics[key]
        lines.append(f"{DISPLAY.get(key, key)} & {m['accuracy']:.4f} & {m['macro_f1']:.4f} & {m['balanced_accuracy']:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)

if __name__ == "__main__":
    main()
