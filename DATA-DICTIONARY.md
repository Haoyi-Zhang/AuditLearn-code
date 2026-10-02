# Data dictionary

The exact set of files can grow as campaigns are added. All CSV files are UTF-8, have one header row, and are deterministically sorted before release.

Common identifiers:

- `campaign`: experiment family.
- `truth` / `truth_id`: finite benchmark truth identifier.
- `seed`: explicit Monte Carlo seed.
- `method`: algorithm or diagnostic comparator; see `ARTIFACT-EVALUATION.md`.
- `delay` / `delay_mechanism`: audit-return mechanism.
- `horizon`: number of episodes.
- `episode`: one-based or zero-based episode index as declared in the file-specific README/header.

Common outcomes:

- `pseudo_regret` / `regret`: cumulative expected-loss difference relative to the configured oracle.
- `truth_excluded`: indicator that the valid truth/cover point is outside a reported confidence set.
- `prefix_debt`: chronological-prefix delay descriptor.
- `unresolved_mass` / `gamma_missing`: launch-normalized unresolved-audit descriptor.
- `outstanding`: number of unresolved launched audits at the episode boundary.
- `launches`: realized or expected task launches, as indicated by the table name.

The CSV header is authoritative for each file. `reviewer_requirements.py`, `reviewer_invariants.py`, and `validate_release.py` check campaign coverage, key uniqueness, finite numeric values, and scope consistency without relying on a shipped checksum manifest.
