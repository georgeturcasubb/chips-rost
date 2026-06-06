# CHiPS-ROST Reviewer Release

This repository contains the minimal public code and reproducibility artifacts
for:

> CHiPS: Character Histograms and Positional Signals for Lightweight Authorship
> Attribution in Romanian Texts

CHiPS is a lightweight, character-level authorship-attribution pipeline for
Romanian literary texts. The core models are:

- `CH-SVM`: one-character marginal distributions plus scalar character
  statistics.
- `FFT12-LR`: twelve positional character channels summarized with
  Fourier/Welch descriptors.
- `CHiPS-F`: validation or grouped out-of-fold selected decision-level fusion.
- `CHiPS-R`: optional top-5 reranking selected from grouped out-of-fold
  predictions, not tuned on held-out test labels.

This public branch is intentionally reviewer-focused. It includes the public
ro-stories text copy and the cleaned ro-stories AP archive needed for the
secondary public corpus checks.

## Contents

- `src/chips/` — reusable Python package for loading, splitting, features, and
  modeling.
- `scripts/` — command-line scripts needed to describe datasets, make grouped
  splits, train CHiPS models, run CHiPS-R, audit corpora, and run the
  ROSTories-cleaned shortcut-risk sensitivity check.
- `experiments/configs/` — locked grouped split/configuration JSON files.
- `experiments/runs/` — selected metrics, predictions, CV grids, audit outputs,
  and software-version manifests used to inspect the reported results.
- `tests/` — small unit/smoke tests for release-critical helpers.
- `data/ro-stories-original/` — public ro-stories text files used for the
  contextual grouped run.
- `data/ro-storiesAP.zip` — cleaned paragraph-aggregated ro-stories archive
  used for the public secondary cleaned-corpus run.
- `data/README.md` — data-placement and redistribution policy for public and
  local-only inputs.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests
```

The code requires Python 3.10 or newer.

## Data Included

This public GitHub repository includes:

- `data/ro-stories-original/`
- `data/ro-storiesAP.zip`

The locked ROST and ROSTories-cleaned checks require local inputs that are not
public GitHub redistributable:

- `data/ROST-NormElip/` — normalized ROST text directory.
- `data/ro-storiesAP-ROST.zip` — ROSTories-cleaned archive containing ROST
  material.

See `data/README.md` for details.

## Public Smoke Checks

These commands run from a fresh clone using only files included in this
repository:

```bash
python -m unittest discover -s tests

python scripts/chips.py describe \
  --input data/ro-stories-original \
  --dataset rostories \
  --output experiments/reproduce/describe_ro_stories_original

python scripts/audit_ro_stories.py \
  --input data/ro-stories-original \
  --dataset rostories \
  --output experiments/reproduce/audit_ro_stories_original

python scripts/chips.py train-all-cv \
  --input data/ro-storiesAP.zip \
  --dataset rostories \
  --split-json experiments/configs/ro_stories_ap_clean_source_split_seed42.json \
  --output experiments/reproduce/ro_stories_ap_clean_quick \
  --cv-folds 5 \
  --quick \
  --seed 42
```

The quick training command is a smoke test of the pipeline, not a replacement
for the reported full grid. To rerun a reported full result, use the commands
below without `--quick` and compare the new JSON/CSV files with the checked-in
artifacts under `experiments/runs/`.

## Reproduce The Main ROST Checks

These commands require the local-only `data/ROST-NormElip/` directory.

Locked CHiPS grouped-CV run:

```bash
python scripts/chips.py train-all-cv \
  --input data/ROST-NormElip \
  --dataset rost \
  --split-json experiments/configs/rost_source_split_seed42.json \
  --output experiments/reproduce/rost_cv5_full \
  --cv-folds 5 \
  --seed 42
```

Matched character 2--5 gram TF-IDF SVM comparator:

```bash
python scripts/chips.py train-char-ngram-cv \
  --input data/ROST-NormElip \
  --dataset rost \
  --split-json experiments/configs/rost_source_split_seed42.json \
  --output experiments/reproduce/rost_char_ngram_2_5_cv5 \
  --cv-folds 5 \
  --seed 42
```

Optional CHiPS-R reranker:

```bash
python scripts/chips_rerank.py \
  --input data/ROST-NormElip \
  --dataset rost \
  --split-json experiments/configs/rost_source_split_seed42.json \
  --base-run experiments/reproduce/rost_cv5_full \
  --output experiments/reproduce/rost_cv5_chips_r \
  --top-k 5 \
  --oof-folds 5 \
  --meta-folds 5 \
  --seed 42
```

## Secondary Corpus Checks

Public ro-stories contextual run:

```bash
python scripts/chips.py train-all-cv \
  --input data/ro-stories-original \
  --dataset rostories \
  --split-json experiments/configs/ro_stories_original_source_split_seed42.json \
  --output experiments/reproduce/ro_stories_original_cv5_full \
  --cv-folds 5 \
  --seed 42

python scripts/chips_rerank.py \
  --input data/ro-stories-original \
  --dataset rostories \
  --split-json experiments/configs/ro_stories_original_source_split_seed42.json \
  --base-run experiments/reproduce/ro_stories_original_cv5_full \
  --output experiments/reproduce/ro_stories_original_cv5_chips_r \
  --top-k 5 \
  --oof-folds 5 \
  --meta-folds 5 \
  --seed 42
```

Cleaned ro-stories AP run:

```bash
python scripts/chips.py train-all-cv \
  --input data/ro-storiesAP.zip \
  --dataset rostories \
  --split-json experiments/configs/ro_stories_ap_clean_source_split_seed42.json \
  --output experiments/reproduce/ro_stories_ap_clean_cv5_full \
  --cv-folds 5 \
  --seed 42

python scripts/chips_rerank.py \
  --input data/ro-storiesAP.zip \
  --dataset rostories \
  --split-json experiments/configs/ro_stories_ap_clean_source_split_seed42.json \
  --base-run experiments/reproduce/ro_stories_ap_clean_cv5_full \
  --output experiments/reproduce/ro_stories_ap_clean_cv5_chips_r \
  --top-k 5 \
  --oof-folds 5 \
  --meta-folds 5 \
  --seed 42
```

ROSTories-cleaned run, using a local archive and normalized export:

```bash
python scripts/export_normalized_corpus.py \
  --input data/ro-storiesAP-ROST.zip \
  --dataset rostories \
  --output experiments/reproduce/rostories_cleaned_norm_input/texts \
  --overwrite

python scripts/chips.py train-all-cv \
  --input experiments/reproduce/rostories_cleaned_norm_input/texts \
  --dataset rostories \
  --split-json experiments/configs/rostories_cleaned_source_split_seed42.json \
  --output experiments/reproduce/rostories_cleaned_cv5_full \
  --cv-folds 5 \
  --seed 42
```

Shortcut-risk sensitivity for author-name strings in ROSTories-cleaned:

```bash
python scripts/rostories_shortcut_sensitivity.py \
  --input experiments/reproduce/rostories_cleaned_norm_input/texts \
  --dataset rostories \
  --split-json experiments/configs/rostories_cleaned_source_split_seed42.json \
  --base-run experiments/reproduce/rostories_cleaned_cv5_full \
  --output experiments/reproduce/rostories_cleaned_shortcut_sensitivity \
  --top-k 5 \
  --oof-folds 5 \
  --meta-folds 5 \
  --seed 42
```

## Inspect Existing Result Artifacts

The checked-in `experiments/runs/` files are plain CSV/JSON artifacts. Useful
entry points are:

- `experiments/runs/rost_cv5_full/metrics.json`
- `experiments/runs/rost_char_ngram_2_5_cv5/metrics.json`
- `experiments/runs/rost_cv5_chips_r/chips_r_metrics.json`
- `experiments/runs/rostories_cleaned_shortcut_sensitivity/summary.json`
- `experiments/runs/audit_rostories_cleaned/combined_source_audit.json`

These files are enough to inspect split discipline, selected configurations,
held-out predictions, and the shortcut-risk sensitivity result. Public
ro-stories and ro-stories AP-clean reruns can be performed directly from this
repository. ROST and ROSTories-cleaned reruns require the local-only inputs
listed above.

## AI-Assisted Preparation

The coding work in this repository, mainly the Python scripts and release
helpers, was prepared with assistance from Codex agents configured to use OpenAI
GPT-5.5. The human authors retain responsibility for the claims, experiments,
source attribution, code, and final text. We disclose this tool use and follow
the spirit of the Leiden Declaration on Artificial Intelligence and Mathematics:
transparent tool use, human responsibility, careful attribution, reviewability,
and open-science practice where corpus licenses allow it.
