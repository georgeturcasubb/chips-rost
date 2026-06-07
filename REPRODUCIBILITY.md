# Reproducibility

This file records the commands and data requirements for inspecting and rerunning
the CHiPS/ROST experiments.

## Environment

Minimum install:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

The current local release environment used for validation is:

- Python: `3.14.2`
- Exact package freeze: `requirements-freeze.txt`

Per-run software manifests are stored in the corresponding experiment folders
where the training script emitted `software_versions.json`.

## Public Checks

These checks do not require full-text corpora:

```bash
python -m unittest discover -s tests
python scripts/check_release_inventory.py
python scripts/rost_per_author.py
python scripts/chips_r_detail.py
```

Expected result:

- unit tests finish with `OK`;
- release inventory reports that required release files are present and no
  local-only corpus paths are tracked by Git.
- the per-author script writes locked ROST support and confusion artifacts under
  `experiments/results/per_author/`.
- the CHiPS-R detail script writes reranker selection and held-out transition
  artifacts under `experiments/results/chips_r_detail/`.

## Locked ROST Runs

Required local input:

- `data/ROST-NormElip/`

Main CHiPS run:

```bash
python scripts/chips.py train-all-cv \
  --input data/ROST-NormElip \
  --dataset rost \
  --split-json experiments/configs/rost_source_split_seed42.json \
  --output experiments/reproduce/rost_cv5_full \
  --cv-folds 5 \
  --seed 42
```

Expected locked held-out metrics:

- CH-SVM: `53/58`, accuracy `0.9138`, macro-F1 `0.9172`
- FFT12-LR: `49/58`, accuracy `0.8448`, macro-F1 `0.8266`
- CHiPS-F: `54/58`, accuracy `0.9310`, macro-F1 `0.9341`

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

Expected locked held-out metric:

- character 2--5 gram TF-IDF SVM: `58/58`, accuracy `1.0000`, macro-F1 `1.0000`

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

Expected locked held-out metric:

- CHiPS-R: `53/58`, accuracy `0.9138`, macro-F1 `0.9110`

## Secondary ROSTories-Cleaned Run

Required local input:

- `data/ro-storiesAP-ROST.zip`

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

Expected held-out metrics under the supplied-archive secondary framing:

- CH-SVM: accuracy `0.8757`, macro-F1 `0.8361`
- FFT12-LR: accuracy `0.6757`, macro-F1 `0.6309`
- CHiPS-F: accuracy `0.8811`, macro-F1 `0.8362`
- CHiPS-R: accuracy `0.8919`, macro-F1 `0.8708`

## Notes

The commands write rerun outputs under `experiments/reproduce/`, which is
ignored by Git. Reported result artifacts under `experiments/runs/` and
`experiments/results/` are the release-side records used by the manuscript.
