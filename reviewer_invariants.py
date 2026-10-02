#!/usr/bin/env python3
"""Independent invariants over the serialized current-scope result tables."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path


def read(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    root = Path(args.results)
    errors: list[str] = []
    checks: list[dict] = []

    tables = sorted(root.glob("*.csv"))
    bad_numeric = []
    for path in tables:
        for line, row in enumerate(read(path), 2):
            for key, value in row.items():
                if value in ("", None):
                    continue
                try:
                    number = float(value)
                except ValueError:
                    continue
                if not math.isfinite(number):
                    bad_numeric.append(f"{path.name}:{line}:{key}={value}")
    checks.append({"name": "finite_numeric_outputs", "executed": True, "passed": not bad_numeric, "details": bad_numeric[:50]})
    if bad_numeric:
        errors.append("nonfinite numeric outputs")

    # Pilot confidence exclusions are a direct serialized diagnostic. Prefix and
    # intersection are the two valid learners; completed is the intentionally
    # invalid returned-sample control.
    pilot = read(root / "runs.csv")
    if pilot:
        exclusions = {}
        for row in pilot:
            exclusions[row["algorithm"]] = exclusions.get(row["algorithm"], 0) + int(row["confidence_exclusions"])
        check = {
            "name": "pilot_confidence_exclusions",
            "executed": True,
            "prefix": exclusions.get("prefix"),
            "intersection": exclusions.get("intersection"),
            "completed_negative_control": exclusions.get("completed"),
            "passed": exclusions.get("prefix") == 0 and exclusions.get("intersection") == 0 and exclusions.get("completed", 0) > 0,
        }
        checks.append(check)
        if not check["passed"]:
            errors.append("pilot confidence-exclusion diagnostic changed")
    else:
        checks.append({"name": "pilot_confidence_exclusions", "executed": False, "passed": False, "reason": "runs.csv absent"})

    scale = read(root / "scale_runs.csv")
    if scale:
        zero_rows = sum(any(int(row[f"launches{i}"]) == 0 for i in range(3)) for row in scale)
        check = {
            "name": "scale_zero_launch_rows",
            "executed": True,
            "rows": zero_rows,
            "passed": zero_rows == 80,
            "meaning": "zero-launch types remain in the frozen scaling evidence and contribute Psi_U(0)=0",
        }
        checks.append(check)
        if not check["passed"]:
            errors.append(f"expected 80 zero-launch scaling rows, got {zero_rows}")
    else:
        checks.append({"name": "scale_zero_launch_rows", "executed": False, "passed": False, "reason": "scale_runs.csv absent"})

    report = {"status": "PASS" if not errors else "FAIL", "checks": checks, "errors": errors}
    root.mkdir(parents=True, exist_ok=True)
    (root / "reviewer-invariants.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
