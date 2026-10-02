# Artifact evaluation guide

## What this artifact establishes

The artifact independently recomputes finite policy values, oracle minima,
launch-sensitive coupling checks, completion-set geometry, delayed-exposure
inequalities, selector information boundaries, and the three frozen synthetic
campaigns: pilot, stress, and scaling. It also rebuilds the paper tables and
plot inputs.

## What it does not establish

It does not mechanically prove the general theorems, execute a language model,
benchmark serving hardware, establish population-level generalization, or
certify novelty or acceptance. It contains no off-grid empirical campaign and
no inflated-confidence selector. The known-radius dictionary result is a hand
proof plus an exact binary arithmetic regression.

## Quick path

```bash
python run.py check --output check-results
python run.py smoke --output smoke-results
python reproduce_release.py quick-results --skip-long
```

The quick path does not regenerate stress or scaling.

## Complete path

```bash
python reproduce_release.py repro-results
```

This runs the documented eight stress shards and merge, four scaling shards and
merge, and all remaining checks. Missing or duplicate shards cause failure.

## Method information sets

- **prefix (`prefix`)**: valid chronological-prefix confidence information.
- **intersection (`intersection`)**: intersection of prefix confidence with the
  valid completion-polytope confidence description.
- **completed control (`completed`)**: deliberately invalid negative control
  that treats the selected returned records as if they were iid. The legacy
  serialized name is retained for frozen-result identity; it is not a
  completion-polytope-only learner.
- **plug-in (`plugin`)**: nonconservative point-estimate diagnostic.
- **known, product, safe**: privileged or fixed comparators, not deployable
  confidence learners.

## Determinism and integrity

Every campaign uses explicit stable integer seeds. Merge operations require the
complete configured key set, reject duplicates, and sort rows before
serialization. The validators check syntax, finite numeric outputs, exact
row duplication, bibliography consistency, and the absence of invented
off-grid executable assets. A successful run is an internal-consistency check,
not a formal proof or independent review.
