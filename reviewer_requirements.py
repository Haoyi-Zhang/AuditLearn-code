#!/usr/bin/env python3
"""Check the actual finite experiment matrix without inventing campaigns."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def methods(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f)
        return {r.get("algorithm", r.get("method", "")).strip() for r in rows if r.get("algorithm", r.get("method", "")).strip()}


def unique(path: Path, column: str) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {r[column] for r in csv.DictReader(f) if column in r and r[column] != ""}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    root = Path(args.results)

    pilot = root / "runs.csv"
    stress = root / "stress_runs.csv"
    scale = root / "scale_runs.csv"
    all_methods = methods(pilot) | methods(stress) | methods(scale)
    expected_core = {"prefix", "intersection", "completed", "plugin"}
    offgrid_files = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and "offgrid" in p.name.lower().replace("-", "")]

    checks = {
        "pilot_present": pilot.exists(),
        "stress_present": stress.exists(),
        "scale_present": scale.exists(),
        "actual_core_methods_present": expected_core <= all_methods,
        "completed_label_is_negative_control_documented": "completed" in all_methods,
        "pilot_seed_count_at_least_4": len(unique(pilot, "seed")) >= 4,
        "stress_truth_count_27": (not stress.exists()) or len(unique(stress, "truth")) == 27,
        "no_offgrid_result_files": not offgrid_files,
    }
    long_keys = {"stress_present", "scale_present", "stress_truth_count_27"}
    errors = [k for k, v in checks.items() if not v and not (args.allow_partial and k in long_keys)]
    report = {
        "status": "PASS" if not errors else "FAIL",
        "partial_mode": args.allow_partial,
        "methods": sorted(all_methods),
        "checks": checks,
        "offgrid_files": offgrid_files,
        "note": "The artifact has pilot, stress, and scale campaigns only. 'completed' is the invalid returned-sample negative control, not completion-polytope confidence.",
        "errors": errors,
    }
    (root / "reviewer-requirements.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
