# Data Policy

This public repository vendors only the public ro-stories materials needed for
the secondary public checks.

Included in the repository:

- `data/ro-stories-original/` — public ro-stories text files.
- `data/ro-storiesAP.zip` — cleaned paragraph-aggregated ro-stories archive.

Local-only inputs needed for the locked ROST and ROSTories-cleaned reruns:

- `data/ROST-NormElip/` — local normalized ROST text directory.
- `data/ro-storiesAP-ROST.zip` — local ROSTories-cleaned archive containing
  ROST material.

ROST and ROSTories-cleaned full-text files are not released through this public
GitHub repository because of attribution and redistribution constraints. The
repository keeps split files, metadata, checksums/audit outputs, metrics,
prediction CSVs, and scripts needed to inspect or rerun the experiments once
the user has local access to those corpora.

The ROSTories-cleaned result is a related-corpus result, not an independent
benchmark, because it intentionally contains ROST material.
