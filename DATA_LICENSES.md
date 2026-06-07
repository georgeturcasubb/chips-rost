# Data Licenses and Redistribution Policy

This repository is a reproducibility package for the CHiPS/ROST article. It
does not vendor full-text corpora by default. Corpus files are local inputs and
must be obtained through the relevant upstream or author-controlled channels.

## Corpus Inputs

| Corpus/input | Public release status | Notes |
| --- | --- | --- |
| ROST / `ROST-NormElip` | Not vendored in public GitHub | Used as the locked benchmark. The repository provides split files, metadata, checksums, predictions, metrics, and scripts, but not full texts. |
| Public ro-stories | Not vendored by default | Available from the upstream Hugging Face release cited in the article. Users may place files locally under `data/ro-stories-original/`. |
| ro-stories AP-clean | Not vendored by default | Cleaned paragraph-aggregated component derived from public ro-stories. Users may place the archive locally as `data/ro-storiesAP.zip`. |
| ROSTories-cleaned | Not vendored in public GitHub | Secondary ROST-overlapping supplied archive. Because it contains ROST material, full texts are local/confidential only. |

## Public Release Contents

The public release may include:

- code under `src/` and `scripts/`;
- grouped split/configuration JSON files;
- metrics, prediction CSVs, CV grids, and selected model settings;
- audit diagnostics and checksum outputs;
- generated result-table snippets and reproducibility notes;
- tests and release-inventory checks.

The public release should not include manuscript source, full-text corpus
directories, or corpus ZIP files unless the human authors explicitly make a
later licensing decision and update this file, `README.md`, `data/README.md`,
and the manuscript Data Availability statement.

## Referee and Research Access

ROST and ROST-containing ROSTories-cleaned full texts may be provided in a
confidential referee/submission package when approved by the authors. They may
also be available from the authors upon reasonable request for research on
Romanian authorship attribution, subject to attribution and redistribution
constraints.
