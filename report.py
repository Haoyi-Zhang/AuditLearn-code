#!/usr/bin/env python3
"""Reconcile retained campaigns and derive paper tables/plot data.

This script performs deterministic descriptive aggregation only.  It does not
run another experiment, access a network, or use external packages.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parent


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarize_pilot(results: Path):
    config = json.loads((ROOT / "inputs" / "pilot.json").read_text())
    runs = read_csv(results / "runs.csv")
    curves = read_csv(results / "curves.csv")
    expected = (
        len(config["truth_indices"])
        * len(config["audit_lags"])
        * len(config["seeds"])
        * len(config["algorithms"])
    )
    if len(runs) != expected:
        raise ValueError(f"Expected {expected} complete pilot runs, got {len(runs)}")

    targets = {
        (r["truth"], r["lag"], r["seed"], r["algorithm"], int(r["episode"])): r
        for r in curves
    }
    if len(targets) != len(curves):
        raise ValueError("Duplicate pilot learning-curve keys")
    sums = defaultdict(lambda: [0.0] * 6 + [0])
    with (results / "episode_trace.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = (r["truth"], r["lag"], r["seed"], r["algorithm"])
            a = sums[k]
            for j, name in enumerate(
                (
                    "regret",
                    "expected_waste",
                    "prefix_exposure",
                    "completion_exposure",
                    "intersection_exposure",
                    "unresolved_mass_exposure",
                )
            ):
                a[j] += float(r[name])
            a[6] += 1
            if int(r["episode"]) != a[6]:
                raise ValueError("Noncontiguous pilot episode sequence")
            target = targets.pop((*k, a[6]), None)
            if target:
                names = (
                    "pseudo_regret",
                    "expected_waste",
                    "prefix_exposure",
                    "completion_exposure",
                    "intersection_exposure",
                    "unresolved_mass_exposure",
                )
                if any(abs(a[j] - float(target[name])) > 1e-9 for j, name in enumerate(names)):
                    raise ValueError("Pilot curve and episode trace disagree")
    if targets:
        raise ValueError("Unmatched pilot curve rows")

    groups = defaultdict(list)
    for r in runs:
        groups[(int(r["truth"]), int(r["lag"]), r["algorithm"])].append(r)
    aggregate = []
    numeric_metrics = (
        "pseudo_regret",
        "expected_waste",
        "realized_cost",
        "prefix_exposure",
        "completion_exposure",
        "intersection_exposure",
        "unresolved_mass_exposure",
    )
    for (truth, lag, algorithm), rows in sorted(groups.items()):
        if sorted(int(r["seed"]) for r in rows) != sorted(config["seeds"]):
            raise ValueError("Missing or duplicate pilot seed")
        row = dict(
            truth=truth,
            lag=lag,
            algorithm=algorithm,
            seeds=len(rows),
            episodes_per_seed=config["horizon"],
        )
        for metric in numeric_metrics:
            numbers = [float(r[metric]) for r in rows]
            row[metric + "_mean"] = mean(numbers)
            row[metric + "_sample_sd"] = stdev(numbers)
        for metric in ("confidence_exclusions", "safe_fallbacks"):
            row[metric + "_sum"] = sum(int(r[metric]) for r in rows)
        for metric in ("maximum_prefix_debt", "maximum_unreturned"):
            row[metric] = max(int(r[metric]) for r in rows)
        aggregate.append(row)
    write_csv(results / "aggregate.csv", aggregate)
    index = {(r["truth"], r["lag"], r["algorithm"]): r for r in aggregate}

    methods = ("prefix", "intersection", "completed", "plugin")
    cg = defaultdict(list)
    for r in curves:
        if int(r["truth"]) == 20 and int(r["lag"]) == 64 and r["algorithm"] in methods:
            cg[(int(r["episode"]), r["algorithm"])].append(float(r["pseudo_regret"]))
    curve_rows = []
    for t in sorted({t for t, _ in cg}):
        row = {"episode": t}
        for method in methods:
            if len(cg[(t, method)]) != len(config["seeds"]):
                raise ValueError("Pilot curve does not contain all seeds")
            row[method] = mean(cg[(t, method)])
        curve_rows.append(row)
    with (results / "regret_curve.dat").open("w", encoding="utf-8") as f:
        f.write("episode prefix intersection completed plugin\n")
        f.write("0 0 0 0 0\n")
        for r in curve_rows:
            f.write(" ".join(str(r[k]) for k in ("episode", *methods)) + "\n")

    lines = []
    for truth in config["truth_indices"]:
        for lag in config["audit_lags"]:
            cells = [str(truth), str(lag)]
            for method in methods:
                r = index[(truth, lag, method)]
                cells.append(f"${r['pseudo_regret_mean']:.2f} \\pm {r['pseudo_regret_sample_sd']:.2f}$")
            lines.append(" & ".join(cells) + r" \\")
        if truth != config["truth_indices"][-1]:
            lines.append(r"\midrule")
    (results / "regret_table.tex").write_text("\n".join(lines) + "\n")

    lines = []
    for truth in config["truth_indices"]:
        cells = [str(truth)]
        for method in ("safe", "known", "product"):
            cells.append(f"{index[(truth, 0, method)]['pseudo_regret_mean']:.0f}")
        lines.append(" & ".join(cells) + r" \\")
    (results / "oracle_table.tex").write_text("\n".join(lines) + "\n")

    lines = []
    labels = {
        "prefix": "Prefix",
        "intersection": "Intersection",
        "completed": "Completed",
        "plugin": "Plug-in",
        "known": "Known law",
    }
    for method in (*methods, "known"):
        cells = [labels[method]]
        for truth in config["truth_indices"]:
            r = index[(truth, 64, method)]
            cells.append(f"${r['expected_waste_mean']:.2f} \\pm {r['expected_waste_sample_sd']:.2f}$")
        lines.append(" & ".join(cells) + r" \\")
    (results / "waste_table.tex").write_text("\n".join(lines) + "\n")

    T = config["horizon"]
    delta = config["confidence_delta"]
    L = 9 / 4
    B = 2
    m = 3
    d = 4
    c = (d * math.log(2) + math.log(m * B * T / delta)) / 2
    bounds = [
        dict(
            lag=lag,
            c_i=c,
            trivial_bound=L * T,
            displayed_prefix_upper_bound=L * (m * (lag + 1) + 4 * math.sqrt(B * T * m * c)) + L * T * delta,
        )
        for lag in config["audit_lags"]
    ]
    write_csv(results / "bound_values.csv", bounds)
    summary = dict(
        reconciled_curve_rows=len(curves),
        reconciled_runs=len(runs),
        episode_records=sum(a[6] for a in sums.values()),
        completed_exclusion_records=sum(
            r["confidence_exclusions_sum"] for r in aggregate if r["algorithm"] == "completed"
        ),
        prefix_exclusion_records=sum(
            r["confidence_exclusions_sum"] for r in aggregate if r["algorithm"] == "prefix"
        ),
        intersection_exclusion_records=sum(
            r["confidence_exclusions_sum"] for r in aggregate if r["algorithm"] == "intersection"
        ),
        status="pilot aggregation and curve reconciliation passed",
    )
    (results / "report_checks.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def summarize_stress(results: Path):
    config = json.loads((ROOT / "inputs" / "stress.json").read_text())
    runs = read_csv(results / "stress_runs.csv")
    curves = read_csv(results / "stress_curves.csv")
    expected = (
        len(config["truth_indices"])
        * len(config["seeds"])
        * len(config["algorithms"])
        * len(config["scenarios"])
    )
    if len(runs) != expected:
        raise ValueError(f"Expected {expected} stress runs, got {len(runs)}")
    expected_curve_points = math.ceil(config["horizon"] / config["curve_stride"])
    expected_curves = expected * expected_curve_points
    if len(curves) != expected_curves:
        raise ValueError(f"Expected {expected_curves} stress curve rows, got {len(curves)}")

    keys = {(r["scenario"], r["truth"], r["seed"], r["algorithm"]) for r in runs}
    if len(keys) != len(runs):
        raise ValueError("Duplicate stress run keys")

    groups = defaultdict(list)
    for r in runs:
        groups[(r["scenario"], int(r["truth"]), r["algorithm"])].append(r)
    aggregate = []
    numeric_metrics = (
        "pseudo_regret",
        "expected_waste",
        "realized_cost",
        "prefix_exposure",
        "completion_exposure",
        "intersection_exposure",
        "unresolved_mass_exposure",
    )
    for (scenario, truth, algorithm), rows in sorted(groups.items()):
        if sorted(int(r["seed"]) for r in rows) != sorted(config["seeds"]):
            raise ValueError("Missing or duplicate stress seed")
        row = dict(
            scenario=scenario,
            truth=truth,
            algorithm=algorithm,
            seeds=len(rows),
            episodes_per_seed=config["horizon"],
        )
        for metric in numeric_metrics:
            values = [float(r[metric]) for r in rows]
            row[metric + "_mean"] = mean(values)
            row[metric + "_sample_sd"] = stdev(values)
        for metric in ("confidence_exclusions", "safe_fallbacks"):
            row[metric + "_sum"] = sum(int(r[metric]) for r in rows)
        for metric in ("maximum_prefix_debt", "maximum_unreturned"):
            row[metric] = max(int(r[metric]) for r in rows)
        aggregate.append(row)
    write_csv(results / "stress_aggregate.csv", aggregate)

    summary_rows = []
    for scenario in [x["name"] for x in config["scenarios"]]:
        for algorithm in config["algorithms"]:
            rows = [r for r in runs if r["scenario"] == scenario and r["algorithm"] == algorithm]
            regrets = [float(r["pseudo_regret"]) for r in rows]
            summary_rows.append(
                dict(
                    scenario=scenario,
                    algorithm=algorithm,
                    cells=len(rows),
                    pseudo_regret_mean=mean(regrets),
                    pseudo_regret_median=median(regrets),
                    pseudo_regret_sample_sd=stdev(regrets),
                    confidence_exclusions=sum(int(r["confidence_exclusions"]) for r in rows),
                    safe_fallbacks=sum(int(r["safe_fallbacks"]) for r in rows),
                    maximum_prefix_debt=max(int(r["maximum_prefix_debt"]) for r in rows),
                    maximum_unreturned=max(int(r["maximum_unreturned"]) for r in rows),
                    prefix_exposure_mean=mean(float(r["prefix_exposure"]) for r in rows),
                    completion_exposure_mean=mean(float(r["completion_exposure"]) for r in rows),
                    intersection_exposure_mean=mean(float(r["intersection_exposure"]) for r in rows),
                    unresolved_mass_exposure_mean=mean(float(r["unresolved_mass_exposure"]) for r in rows),
                )
            )
    write_csv(results / "stress_dictionary_summary.csv", summary_rows)

    by_key = {(r["scenario"], r["truth"], r["seed"], r["algorithm"]): r for r in runs}
    pairwise = []
    for scenario in [x["name"] for x in config["scenarios"]]:
        diffs = []
        better = equal = worse = 0
        for truth in config["truth_indices"]:
            for seed in config["seeds"]:
                p = float(by_key[(scenario, str(truth), str(seed), "prefix")]["pseudo_regret"])
                q = float(by_key[(scenario, str(truth), str(seed), "intersection")]["pseudo_regret"])
                d = q - p
                diffs.append(d)
                if d < -1e-12:
                    better += 1
                elif d > 1e-12:
                    worse += 1
                else:
                    equal += 1
        pairwise.append(
            dict(
                scenario=scenario,
                cells=len(diffs),
                intersection_minus_prefix_mean=mean(diffs),
                intersection_minus_prefix_median=median(diffs),
                intersection_better=better,
                equal=equal,
                intersection_worse=worse,
            )
        )
    write_csv(results / "stress_pairwise.csv", pairwise)

    labels = {
        "prefix": "Prefix",
        "intersection": "Intersection",
        "completed": "Completed",
        "plugin": "Plug-in",
        "safe": "Safe",
        "known": "Known law",
        "product": "Product law",
    }
    summary_index = {(r["scenario"], r["algorithm"]): r for r in summary_rows}
    lines = []
    for method in ("prefix", "intersection", "completed", "plugin", "safe", "known", "product"):
        cells = [labels[method]]
        for scenario in ("outcome-lag-64", "single-blocker"):
            r = summary_index[(scenario, method)]
            cells.append(f"${float(r['pseudo_regret_mean']):.2f} \\pm {float(r['pseudo_regret_sample_sd']):.2f}$")
        lines.append(" & ".join(cells) + r" \\")
    (results / "stress_table.tex").write_text("\n".join(lines) + "\n")

    methods = ("prefix", "intersection", "completed", "plugin")
    curve_groups = defaultdict(list)
    for r in curves:
        if r["scenario"] == "single-blocker" and r["algorithm"] in methods:
            curve_groups[(int(r["episode"]), r["algorithm"])].append(float(r["pseudo_regret"]))
    with (results / "blocker_curve.dat").open("w", encoding="utf-8") as f:
        f.write("episode prefix intersection completed plugin\n")
        f.write("0 0 0 0 0\n")
        for episode in sorted({k[0] for k in curve_groups}):
            values = [episode]
            for method in methods:
                cell = curve_groups[(episode, method)]
                if len(cell) != len(config["truth_indices"]) * len(config["seeds"]):
                    raise ValueError("Incomplete single-blocker curve cell")
                values.append(mean(cell))
            f.write(" ".join(map(str, values)) + "\n")

    T = config["horizon"]
    c = (4 * math.log(2) + math.log(3 * 2 * T / config["confidence_delta"])) / 2
    with (results / "width_curve.dat").open("w", encoding="utf-8") as f:
        f.write("launches prefix completion intersection single_blocker_envelope\n")
        for n in range(1, T + 1):
            prefix = 1.0
            completion = min(1.0, 2 * min(1.0, math.sqrt(c / n)) + 1 / n)
            f.write(f"{n} {prefix} {completion} {min(prefix, completion)} {1+math.log(n)}\n")

    checks = dict(
        stress_runs=len(runs),
        stress_curve_rows=len(curves),
        truth_seed_cells=len(config["truth_indices"]) * len(config["seeds"]),
        scenarios=[x["name"] for x in config["scenarios"]],
        single_blocker_max_unreturned=max(
            int(r["maximum_unreturned"]) for r in runs if r["scenario"] == "single-blocker"
        ),
        single_blocker_max_prefix_debt=max(
            int(r["maximum_prefix_debt"]) for r in runs if r["scenario"] == "single-blocker"
        ),
        pairwise=pairwise,
        status="stress aggregation passed",
    )
    (results / "stress_checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
    return checks



def summarize_scale(results: Path):
    config = json.loads((ROOT / "inputs" / "scaling.json").read_text())
    runs = read_csv(results / "scale_runs.csv")
    expected = (
        len(config["horizons"])
        * len(config["truth_indices"])
        * len(config["seeds"])
        * len(config["algorithms"])
        * len(config["scenarios"])
    )
    if len(runs) != expected:
        raise ValueError(f"Expected {expected} scale runs, got {len(runs)}")
    keys = {
        (r["scenario"], r["horizon"], r["truth"], r["seed"], r["algorithm"])
        for r in runs
    }
    if len(keys) != len(runs):
        raise ValueError("Duplicate scale run keys")

    scenario_blockers = {x["name"]: int(x["blockers"]) for x in config["scenarios"]}
    rows_with_unlaunched_type = 0
    unlaunched_type_occurrences = 0
    for r in runs:
        U = scenario_blockers[r["scenario"]]
        if int(r["maximum_unreturned"]) > U:
            raise ValueError("Front-blocker run exceeded configured outstanding count")
        launches = [int(r[f"launches{i}"]) for i in range(3)]
        exact_envelope = sum(
            sum(min(U, s) / s for s in range(1, n)) for n in launches
        )
        if abs(float(r["sharp_unresolved_mass_envelope"]) - exact_envelope) > 1e-9:
            raise ValueError("Stored sharp envelope disagrees with launch counts")
        if abs(float(r["unresolved_mass_exposure"]) - exact_envelope) > 1e-9:
            raise ValueError("Front-blocker descriptor did not attain its sharp envelope")
        zero_count = sum(n == 0 for n in launches)
        rows_with_unlaunched_type += int(zero_count > 0)
        unlaunched_type_occurrences += zero_count
    # Frozen-regression count: zero-launch types are part of the reported data,
    # not an excluded edge case.
    if rows_with_unlaunched_type != 80:
        raise ValueError(
            f"Expected 80 scaling rows with an unlaunched type, got {rows_with_unlaunched_type}"
        )

    groups = defaultdict(list)
    for r in runs:
        groups[(r["scenario"], int(r["horizon"]), r["algorithm"])].append(r)
    summary = []
    for (scenario, horizon, algorithm), rows in sorted(groups.items()):
        expected_cells = len(config["truth_indices"]) * len(config["seeds"])
        if len(rows) != expected_cells:
            raise ValueError("Incomplete scale cell")
        regrets = [float(r["pseudo_regret"]) for r in rows]
        summary.append(
            dict(
                scenario=scenario,
                blockers=scenario_blockers[scenario],
                horizon=horizon,
                algorithm=algorithm,
                cells=len(rows),
                pseudo_regret_mean=mean(regrets),
                pseudo_regret_median=median(regrets),
                pseudo_regret_sample_sd=stdev(regrets),
                regret_per_sqrt_horizon=mean(regrets) / math.sqrt(horizon),
                confidence_exclusions=sum(int(r["confidence_exclusions"]) for r in rows),
                safe_fallbacks=sum(int(r["safe_fallbacks"]) for r in rows),
                maximum_prefix_debt=max(int(r["maximum_prefix_debt"]) for r in rows),
                maximum_unreturned=max(int(r["maximum_unreturned"]) for r in rows),
                prefix_exposure_mean=mean(float(r["prefix_exposure"]) for r in rows),
                completion_exposure_mean=mean(float(r["completion_exposure"]) for r in rows),
                intersection_exposure_mean=mean(float(r["intersection_exposure"]) for r in rows),
                unresolved_mass_exposure_mean=mean(float(r["unresolved_mass_exposure"]) for r in rows),
                sharp_unresolved_mass_envelope_mean=mean(
                    float(r["sharp_unresolved_mass_envelope"]) for r in rows
                ),
            )
        )
    write_csv(results / "scale_summary.csv", summary)

    by_key = {
        (r["scenario"], int(r["horizon"]), int(r["truth"]), int(r["seed"]), r["algorithm"]): r
        for r in runs
    }
    pairwise = []
    for scenario in scenario_blockers:
        for horizon in config["horizons"]:
            diffs = []
            better = equal = worse = 0
            for truth in config["truth_indices"]:
                for seed in config["seeds"]:
                    p = float(by_key[(scenario, horizon, truth, seed, "prefix")]["pseudo_regret"])
                    q = float(by_key[(scenario, horizon, truth, seed, "intersection")]["pseudo_regret"])
                    d = q - p
                    diffs.append(d)
                    if d < -1e-12:
                        better += 1
                    elif d > 1e-12:
                        worse += 1
                    else:
                        equal += 1
            pairwise.append(
                dict(
                    scenario=scenario,
                    blockers=scenario_blockers[scenario],
                    horizon=horizon,
                    cells=len(diffs),
                    intersection_minus_prefix_mean=mean(diffs),
                    intersection_minus_prefix_median=median(diffs),
                    intersection_better=better,
                    equal=equal,
                    intersection_worse=worse,
                )
            )
    write_csv(results / "scale_pairwise.csv", pairwise)

    index = {(r["scenario"], int(r["horizon"]), r["algorithm"]): r for r in summary}
    labels = {"prefix": "Prefix", "intersection": "Intersection", "plugin": "Plug-in"}
    lines = []
    selected_horizons = (128, 512, 2048)
    for scenario in scenario_blockers:
        for horizon in selected_horizons:
            cells = [str(scenario_blockers[scenario]), str(horizon)]
            for method in ("prefix", "intersection", "plugin"):
                r = index[(scenario, horizon, method)]
                cells.append(f"${float(r['pseudo_regret_mean']):.2f} \\pm {float(r['pseudo_regret_sample_sd']):.2f}$")
            r = index[(scenario, horizon, "intersection")]
            cells.append(f"${float(r['unresolved_mass_exposure_mean']):.2f}$")
            lines.append(" & ".join(cells) + r" \\")
        if scenario != list(scenario_blockers)[-1]:
            lines.append(r"\midrule")
    (results / "scale_table.tex").write_text("\n".join(lines) + "\n")

    checks = dict(
        scale_runs=len(runs),
        total_episodes=sum(int(r["episodes"]) for r in runs),
        horizons=config["horizons"],
        scenarios=scenario_blockers,
        exact_envelope_rows=len(runs),
        rows_with_unlaunched_type=rows_with_unlaunched_type,
        unlaunched_type_occurrences=unlaunched_type_occurrences,
        zero_launch_convention="Psi_U(0)=0 and unlaunched types contribute zero",
        pairwise=pairwise,
        status="horizon/backlog scaling aggregation passed",
    )
    (results / "scale_checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
    return checks

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=ROOT / "results")
    p.add_argument("--skip-stress", action="store_true")
    p.add_argument("--skip-scale", action="store_true")
    args = p.parse_args()
    result = {"pilot": summarize_pilot(args.results)}
    if not args.skip_stress and (args.results / "stress_runs.csv").exists():
        result["stress"] = summarize_stress(args.results)
    if not args.skip_scale and (args.results / "scale_runs.csv").exists():
        result["scale"] = summarize_scale(args.results)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
