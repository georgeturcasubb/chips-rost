#!/usr/bin/env python3
"""Check public-release inventory for CHiPS/ROST."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "MANIFEST.md",
    "LICENSE",
    "DATA_LICENSES.md",
    "CITATION.cff",
    "REPRODUCIBILITY.md",
    "requirements.txt",
    "requirements-freeze.txt",
    "pyproject.toml",
    "Makefile",
    ".github/workflows/tests.yml",
    "data/README.md",
]

FORBIDDEN_TRACKED_PREFIXES = [
    ".codex/",
    ".vscode/",
    "latex/",
    "logs/",
    "models/",
    "reports/",
    "workspace/",
    "data/ROST-NormElip/",
    "data/ro-stories-original/",
    "data/ro-storiesAP.zip",
    "data/ro-storiesAP-ROST.zip",
    "experiments/reproduce/",
    "experiments/remote_jobs/",
    "experiments/results/remote_python/",
    "experiments/runs/rostories_cleaned_norm_input/",
    "workspace/source_material/datasets/ROST-NormElip.zip",
    "workspace/source_material/datasets/ro-storiesAP.zip",
    "workspace/source_material/datasets/ro-storiesAP-ROST.zip",
    "workspace/source_material/original_article/chips-source.zip",
    "workspace/source_material/original_article/chips_rost_nlp.pdf",
    "workspace/source_material/legacy_experiments/scripts_results_2026-05-21.zip",
]

FORBIDDEN_TRACKED_EXACT = [
    "AGENTS.md",
    "latex/article/chips_rost_nlp.tex",
    "workspace/source_material/original_article/chips_rost_nlp.tex",
]

FORBIDDEN_TRACKED_SUFFIXES = [
    ".joblib",
    ".pkl",
    ".zip",
]

OLD_REPOSITORY_URL = "https://github.com/" + "sanda" + "-avram/" + "fingerprint" + "OnROST"

REQUIRED_TEXT = {
    "README.md": [
        "https://github.com/georgeturcasubb/chips-rost",
        "does not vendor",
        "full-text corpora by default",
    ],
    "data/README.md": [
        "This public repository does not vendor full-text corpora by default",
    ],
}

FORBIDDEN_TEXT = {
    "README.md": [
        "This public branch is intentionally reviewer-focused",
        "This public GitHub repository includes:",
        OLD_REPOSITORY_URL,
        "latex/article",
        "working article source",
    ],
    "MANIFEST.md": [
        "working article source",
    ],
}


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.splitlines()


def main() -> None:
    problems: list[str] = []

    for rel in REQUIRED_FILES:
        if not (ROOT / rel).is_file():
            problems.append(f"missing required release file: {rel}")

    tracked = tracked_files()
    for rel in tracked:
        if rel in FORBIDDEN_TRACKED_EXACT:
            problems.append(f"internal/manuscript file is tracked: {rel}")
        for suffix in FORBIDDEN_TRACKED_SUFFIXES:
            if rel.endswith(suffix):
                problems.append(f"binary/local archive artifact is tracked: {rel}")
        for forbidden in FORBIDDEN_TRACKED_PREFIXES:
            if rel == forbidden or rel.startswith(forbidden):
                problems.append(f"local-only or internal file is tracked: {rel}")

    for rel, needles in REQUIRED_TEXT.items():
        path = ROOT / rel
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        for needle in needles:
            if needle not in text:
                problems.append(f"{rel} missing required text: {needle}")

    for rel, needles in FORBIDDEN_TEXT.items():
        path = ROOT / rel
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        for needle in needles:
            if needle in text:
                problems.append(f"{rel} still contains stale text: {needle}")

    data_tracked = [rel for rel in tracked if rel.startswith("data/")]
    if data_tracked != ["data/README.md"]:
        problems.append(
            "tracked data inventory should be exactly ['data/README.md']; "
            f"found {data_tracked}"
        )

    if problems:
        print("Release inventory check failed:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)

    print("Release inventory check passed.")


if __name__ == "__main__":
    main()
