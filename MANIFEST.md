# Manifest

This is the public, referee-oriented CHiPS/ROST reproducibility package.

## Included

- `README.md` - human entry point, data policy, smoke checks, and rerun
  commands.
- `REPRODUCIBILITY.md` - compact command record and expected result summaries.
- `DATA_LICENSES.md` - corpus provenance and redistribution policy.
- `CITATION.cff` - citation metadata for the software/reproducibility package.
- `LICENSE` - MIT license for the release code.
- `requirements.txt` and `requirements-freeze.txt` - dependency records.
- `pyproject.toml` - package metadata.
- `Makefile` - public smoke and audit targets.
- `.github/workflows/tests.yml` - public CI smoke checks.
- `data/README.md` - local corpus placement instructions.
- `src/chips/` - reusable CHiPS Python package.
- `scripts/` - public command-line tools for loading, splitting, training,
  reranking, audits, diagnostics, and table generation.
- `tests/` - release-critical unit tests.
- `experiments/configs/` - locked grouped split/configuration files.
- `experiments/runs/` - selected non-text run artifacts: metrics, predictions,
  CV grids, split summaries, selected configurations, protocols, and software
  manifests.
- `experiments/results/` - result tables, uncertainty summaries, audit
  diagnostics, per-author summaries, CHiPS-R detail summaries, and
  interpretability/ablation artifacts.

## Not Included

- No manuscript source or article build files.
- No `latex/`, `workspace/`, `.codex/`, `reports/`, or local process logs.
- No full-text corpus directories or corpus ZIP files.
- No trained model binaries such as `.joblib` or `.pkl` files.
- No local reproduction outputs under `experiments/reproduce/`.

The public tree is intended to be easy for a referee to browse: code, protocol
files, predictions, metrics, and diagnostics are public; manuscript preparation
and restricted text corpora remain outside the GitHub release.
