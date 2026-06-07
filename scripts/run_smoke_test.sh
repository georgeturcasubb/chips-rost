#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/validate_workspace.py
python3 scripts/chips.py describe --input data/ROST-NormElip --output experiments/runs/describe_rost
python3 scripts/chips.py train-all --input data/ROST-NormElip --output experiments/runs/rost_quick --quick --limit-per-author 8
python3 scripts/generate_paper_tables.py --metrics experiments/runs/rost_quick/metrics.json --output experiments/runs/rost_quick/table_rost_results.tex
