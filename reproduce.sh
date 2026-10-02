#!/bin/sh
# Deterministic bounded reproduction for the standalone artifact.
set -eu

OUT=${1:-repro-results}
PYTHON=${PYTHON:-python3}

"$PYTHON" run.py check --output "$OUT"
"$PYTHON" run.py pilot --output "$OUT"

i=0
while [ "$i" -lt 8 ]; do
  "$PYTHON" run.py stress --output "$OUT" --shard-index "$i" --shard-count 8
  i=$((i + 1))
done
"$PYTHON" run.py merge-stress --output "$OUT" --shard-count 8

i=0
while [ "$i" -lt 4 ]; do
  "$PYTHON" run.py scale --output "$OUT" --shard-index "$i" --shard-count 4
  i=$((i + 1))
done
"$PYTHON" run.py merge-scale --output "$OUT" --shard-count 4

"$PYTHON" run.py verify --output "$OUT"
"$PYTHON" report.py --results "$OUT"
