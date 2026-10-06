"""Benign synthetic regressions for shard and checkpoint identity validation."""
from __future__ import annotations
import argparse
from copy import deepcopy
import importlib.util
from itertools import product
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch


def check(artifact: Path, output: Path | None = None):
    sys.path.insert(0, str(artifact))
    spec = importlib.util.spec_from_file_location("campaign_driver", artifact / "run.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    conf = {"truth_indices": [0], "seeds": [111], "algorithms": ["prefix", "intersection"],
            "scenarios": [{"name": "single-blocker"}], "horizon": 64, "curve_stride": 32}
    runs = [dict(truth="0", seed="111", scenario="single-blocker", algorithm=method,
                 episodes="64") for method in conf["algorithms"]]
    curves = [dict(truth="0", seed="111", scenario="single-blocker", algorithm=method,
                   episode=str(episode)) for method, episode in product(conf["algorithms"], (32, 64))]
    metrics = ("pseudo_regret", "expected_waste", "prefix_exposure", "completion_exposure",
               "intersection_exposure", "unresolved_mass_exposure")
    for row in runs:
        row.update({metric: "2" for metric in metrics})
    for row in curves:
        row.update({metric: "1" if row["episode"] == "32" else "2" for metric in metrics})
    variants = [("valid", runs, curves, False)]
    duplicate = deepcopy(curves)
    duplicate[1] = deepcopy(duplicate[0])
    variants.append(("duplicate-replaces-missing-checkpoint", runs, duplicate, True))
    wrong_episode = deepcopy(curves)
    wrong_episode[0]["episode"] = "33"
    variants.append(("off-grid-checkpoint", runs, wrong_episode, True))
    bad_run = deepcopy(runs)
    bad_run[0]["episodes"] = "63"
    variants.append(("wrong-run-horizon", bad_run, curves, True))
    variants.append(("missing-checkpoint", runs, curves[:-1], True))
    duplicate_run = [runs[0], runs[0]]
    variants.append(("duplicate-run", duplicate_run, curves, True))
    foreign_key = deepcopy(curves)
    foreign_key[0]["truth"] = "99"
    variants.append(("unconfigured-checkpoint", runs, foreign_key, True))
    final_mismatch = deepcopy(curves)
    final_mismatch[1]["pseudo_regret"] = "3"
    variants.append(("final-checkpoint-disagrees", runs, final_mismatch, True))
    nonfinite = deepcopy(curves)
    nonfinite[0]["pseudo_regret"] = "nan"
    variants.append(("nonfinite-checkpoint", runs, nonfinite, True))
    decreasing = deepcopy(curves)
    decreasing[0]["pseudo_regret"] = "3"
    variants.append(("decreasing-cumulative-checkpoint", runs, decreasing, True))
    results = []
    root = output if output is not None else Path(tempfile.mkdtemp(prefix="auditlearn-keys-"))
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    (root / "inputs" / "stress.json").write_text(json.dumps(conf), encoding="utf-8")
    for name, r, c, must_reject in variants:
        destination = root / name
        destination.mkdir(exist_ok=True)
        with patch.object(driver, "ROOT", root), patch.object(
            driver, "_read_csv", side_effect=lambda p: deepcopy(r if p.stem == "stress_runs" else c)
        ):
            try:
                driver.merge_stress(destination, 1)
                rejected = False
            except (ValueError, KeyError):
                rejected = True
        results.append(dict(case=name, expected_rejection=must_reject, rejected=rejected,
                            passed=rejected == must_reject))
    print(json.dumps(results, indent=2))
    assert all(r["passed"] for r in results), "campaign identity validation failed"
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    check(args.artifact, args.output)
