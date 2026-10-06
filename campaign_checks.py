"""Configured campaign identities and cumulative-checkpoint consistency."""
from __future__ import annotations
from itertools import product
from math import isfinite

CURVE_METRICS = ("pseudo_regret", "expected_waste", "prefix_exposure",
                 "completion_exposure", "intersection_exposure", "unresolved_mass_exposure")


def _require_rows(rows, fields, expected, name):
    keys = [tuple(row[field] for field in fields) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(f"incomplete, duplicate, or unconfigured {name} keys")


def _check_curves(runs, curves, fields, expected_runs, horizon, stride, name):
    checkpoints = tuple(range(stride, horizon, stride)) + (horizon,)
    _require_rows(curves, (*fields, "episode"),
                  {(*key, str(t)) for key in expected_runs for t in checkpoints},
                  f"{name} checkpoint")
    by_run = {tuple(r[field] for field in fields): r for r in runs}
    by_curve = {tuple(r[field] for field in (*fields, "episode")): r for r in curves}
    for key in expected_runs:
        previous = {metric: 0.0 for metric in CURVE_METRICS}
        for t in checkpoints:
            row = by_curve[(*key, str(t))]
            for metric in CURVE_METRICS:
                value = float(row[metric])
                if not isfinite(value) or value < previous[metric] - 1e-9:
                    raise ValueError(f"nonfinite or decreasing {name} checkpoint {metric}")
                previous[metric] = value
        for metric, value in previous.items():
            total = float(by_run[key][metric])
            if not isfinite(total) or abs(value - total) > 1e-9:
                raise ValueError(f"{name} final checkpoint disagrees with run {metric}")


def validate_pilot(runs, curves, config):
    fields = ("truth", "lag", "seed", "algorithm")
    expected = {tuple(map(str, key)) for key in product(
        config["truth_indices"], config["audit_lags"], config["seeds"], config["algorithms"])}
    _require_rows(runs, fields, expected, "pilot run")
    horizon = config["horizon"]
    if any(int(row["episodes"]) != horizon for row in runs):
        raise ValueError("pilot run horizon disagrees with configuration")
    _check_curves(runs, curves, fields, expected, horizon, 32, "pilot")


def validate_stress(runs, curves, config):
    fields = ("truth", "seed", "scenario", "algorithm")
    expected = {tuple(map(str, key)) for key in product(
        config["truth_indices"], config["seeds"],
        [s["name"] for s in config["scenarios"]], config["algorithms"])}
    _require_rows(runs, fields, expected, "stress run")
    horizon = config["horizon"]
    if any(int(row["episodes"]) != horizon for row in runs):
        raise ValueError("stress run horizon disagrees with configuration")
    _check_curves(runs, curves, fields, expected, horizon, config["curve_stride"], "stress")


def validate_scale(runs, config):
    fields = ("truth", "seed", "horizon", "scenario", "algorithm")
    expected = {tuple(map(str, key)) for key in product(
        config["truth_indices"], config["seeds"], config["horizons"],
        [s["name"] for s in config["scenarios"]], config["algorithms"])}
    _require_rows(runs, fields, expected, "scale run")
    if any(int(row["episodes"]) != int(row["horizon"]) for row in runs):
        raise ValueError("scale run horizon disagrees with episode count")
