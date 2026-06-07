# CHiPS/ROST Reproducibility Package

Canonical public repository:

`https://github.com/georgeturcasubb/chips-rost`

This repository contains the code and release artifacts for:

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

The release is intentionally data-careful. It provides code, split files,
result artifacts, diagnostics, checksums, and documentation. It does not vendor
full-text corpora by default.

## Contents

- `src/chips/` - reusable Python package for loading, splitting, features, and
  modeling.
- `scripts/` - command-line tools for dataset description, normalization,
  grouped splits, model training, reranking, corpus audits, diagnostics, and
  paper tables.
- `experiments/configs/` - locked grouped split/configuration JSON files.
- `experiments/runs/` and `experiments/results/` - selected metrics,
  predictions, CV grids, audit outputs, tables, and software-version manifests.
- `tests/` - unit and smoke tests for release-critical helpers.
- `data/README.md` - corpus placement and redistribution policy.
- `DATA_LICENSES.md` - corpus provenance and release restrictions.
- `REPRODUCIBILITY.md` - commands and expected outputs for reruns.
- `CITATION.cff` - citation metadata for the software/reproducibility package.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests
```

The package requires Python 3.10 or newer. The exact local environment used in
the current workspace is recorded in `requirements-freeze.txt`; individual
experiment folders also include `software_versions.json` where applicable.

## Public Data Policy

No full-text corpus directory or corpus ZIP is tracked in the public release by
default. This includes the public ro-stories files, the cleaned ro-stories
AP-clean archive, ROST, and the ROST-containing ROSTories-cleaned archive.

Users who have lawful access to the corpora should place them locally as
follows:

- `data/ro-stories-original/` - public ro-stories text files, obtained from the
  upstream Hugging Face release.
- `data/ro-storiesAP.zip` - cleaned paragraph-aggregated ro-stories export used
  for the AP-clean secondary run.
- `data/ROST-NormElip/` - local normalized ROST text directory.
- `data/ro-storiesAP-ROST.zip` - local ROSTories-cleaned archive containing
  ROST material.

ROST and ROST-containing ROSTories-cleaned full texts are not released through
the public GitHub repository because of attribution and redistribution
constraints. The article reports those results from local/confidential inputs;
the public repository keeps the code and non-text artifacts needed to inspect
or rerun the experiments once the user has local access to the corpora.

## Public Smoke Checks

These commands run without full-text corpora:

```bash
python -m unittest discover -s tests
python scripts/check_release_inventory.py
python scripts/rost_per_author.py
python scripts/chips_r_detail.py
```

With `make` available, the same checks are:

```bash
make smoke-public
make audit-public
```

The GitHub Actions workflow in `.github/workflows/tests.yml` runs the same
public smoke checks on push and pull request events.

## Main ROST Reruns

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

## Secondary Corpus Reruns

Public ro-stories contextual run, after placing `data/ro-stories-original/`
locally:

```bash
python scripts/chips.py train-all-cv \
  --input data/ro-stories-original \
  --dataset rostories \
  --split-json experiments/configs/ro_stories_original_source_split_seed42.json \
  --output experiments/reproduce/ro_stories_original_cv5_full \
  --cv-folds 5 \
  --seed 42
```

Cleaned ro-stories AP run, after placing `data/ro-storiesAP.zip` locally:

```bash
python scripts/chips.py train-all-cv \
  --input data/ro-storiesAP.zip \
  --dataset rostories \
  --split-json experiments/configs/ro_stories_ap_clean_source_split_seed42.json \
  --output experiments/reproduce/ro_stories_ap_clean_cv5_full \
  --cv-folds 5 \
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

## Inspect Existing Result Artifacts

The checked-in experiment artifacts are plain CSV/JSON files. Useful entry
points are:

- `experiments/runs/rost_cv5_full/metrics.json`
- `experiments/runs/rost_char_ngram_2_5_cv5/metrics.json`
- `experiments/runs/rost_cv5_chips_r/chips_r_metrics.json`
- `experiments/results/rost_uncertainty/uncertainty_table.csv`
- `experiments/results/rost_ngram_audit/diagnostic_table.csv`
- `experiments/results/per_author/rost_per_author_metrics.csv`
- `experiments/results/per_author/rost_confusion_matrix.csv`
- `experiments/results/chips_r_detail/chips_r_detail.csv`
- `experiments/results/chips_r_detail/chips_r_transition_summary.csv`
- `experiments/results/interpretability/ch_svm_top_features.csv`
- `experiments/results/surface_cue_ablations/ch_svm_surface_cue_ablation.csv`
- `experiments/runs/rostories_cleaned_shortcut_sensitivity/summary.json`

These files are enough to inspect split discipline, selected configurations,
held-out predictions, uncertainty calculations, shortcut audits, and
interpretability diagnostics. Full reruns require the local data inputs listed
above.

## Release Checklist

Before publishing a paper-ready release:

1. Confirm that `git ls-files data` lists only `data/README.md`.
2. Run `python scripts/check_release_inventory.py`.
3. Run `python -m unittest discover -s tests`.
4. Confirm the manuscript Data Availability section uses the same repository
   URL, release tag, commit SHA, archive DOI, and data policy.
5. Create the GitHub release tag only after the article text and repository
   contents agree.
6. Archive the GitHub release with Zenodo and add the version DOI to the
   manuscript before final submission if a DOI is promised.

## AI-Assisted Preparation

The coding work in this repository, mainly the Python scripts and release
helpers, was prepared with assistance from Codex agents configured to use OpenAI
GPT-5.5. The human authors retain responsibility for the claims, experiments,
source attribution, code, and final text. We disclose this tool use and follow
the spirit of the Leiden Declaration on Artificial Intelligence and Mathematics:
transparent tool use, human responsibility, careful attribution, reviewability,
and open-science practice where corpus licenses allow it.
