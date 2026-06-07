# Data Policy

This public repository does not vendor full-text corpora by default. The `data/`
directory is a local placement area for users who have obtained the corpora
through the appropriate upstream or author-controlled channels.

Tracked in the public repository:

- `data/README.md`

Local inputs used by reproducibility commands:

- `data/ro-stories-original/` - public ro-stories text files, obtained from the
  upstream Hugging Face release.
- `data/ro-storiesAP.zip` - cleaned paragraph-aggregated ro-stories archive
  derived from public ro-stories.
- `data/ROST-NormElip/` - local normalized ROST text directory.
- `data/ro-storiesAP-ROST.zip` - local ROSTories-cleaned archive containing
  ROST material.

ROST and ROST-containing ROSTories-cleaned full-text files are not released
through the public GitHub repository because of attribution and redistribution
constraints. They are used locally, included in confidential referee/submission
packages when approved by the authors, and may be made available for reasonable
research requests subject to the relevant constraints.

The public ro-stories corpus and the cleaned ro-stories AP-clean component are
also not vendored in this release by default. The repository instead provides
scripts, split files, audit outputs, checksums, metrics, prediction CSVs, and
documentation so that users can inspect the reported results and rerun them
after placing the required local corpus files.

Before publishing a release, `git ls-files data` should list only this README
unless the human authors explicitly decide to vendor a corpus file and update
`DATA_LICENSES.md`, `README.md`, and the manuscript Data Availability statement
accordingly.
