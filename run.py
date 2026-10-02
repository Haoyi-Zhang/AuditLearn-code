#!/usr/bin/env python3
"""One-worker bounded reproduction. Python standard library only.

Commands:
  python run.py check
  python run.py pilot
  python run.py stress [--shard-index I --shard-count S]
  python run.py merge-stress --shard-count S
  python run.py scale [--shard-index I --shard-count S]
  python run.py merge-scale --shard-count S
  python run.py verify

No network, subprocesses, external models, or GPU libraries are used.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from model import models, POLICIES, ROOT_MARKS, CHILD_MARKS, evaluate_world, draw_index, exact_value
from learning import AuditTape, confidence_constant, select
from check_exact import check


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty results table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def prepare(output: Path):
    report, values, wastes = check()
    write_json(output / "exact_checks.json", report)
    rows = []
    for i, m in enumerate(models()):
        for j, policy in enumerate(POLICIES):
            rows.append(
                dict(
                    model=i,
                    correlation=str(m.correlation),
                    success0=str(m.success0),
                    success1=str(m.success1),
                    policy=j,
                    actions=" ".join(map(str, policy)),
                    residual=str(values[i][j]),
                    waste=str(wastes[i][j]),
                )
            )
    write_csv(output / "oracle_values.csv", rows)
    return values, wastes


def campaign_objects():
    """Build the finite dictionary without rerunning the exhaustive checker.

    The documented reproduction sequence runs ``check`` first.  Campaigns then
    reconstruct the same exact 27-by-27 value table directly from the model,
    avoiding several seconds of unrelated checker work in every long command.
    ``report.py`` and ``verify`` still require the independently generated
    ``exact_checks.json`` and ``oracle_values.csv`` from ``check``.
    """
    all_models = models()
    triplets = [[exact_value(model, policy) for policy in POLICIES] for model in all_models]
    values = [[item[0] for item in row] for row in triplets]
    wastes = [[item[1] for item in row] for row in triplets]
    float_values = [[float(x) for x in row] for row in values]
    float_wastes = [[float(x) for x in row] for row in wastes]
    laws = [tuple(tuple(map(float, law)) for law in model.laws) for model in all_models]
    best_policy = [min(range(len(POLICIES)), key=lambda j: (values[i][j], j)) for i in range(len(all_models))]
    best_value = [float_values[i][best_policy[i]] for i in range(len(all_models))]
    return float_values, float_wastes, laws, best_policy, best_value


def _validate_shard(shard_index: int, shard_count: int, item_count: int) -> None:
    if shard_count < 1:
        raise ValueError("shard-count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    if shard_count > item_count:
        raise ValueError("shard-count may not exceed the number of truth/seed cells")


def _shard_pairs(truths, seeds, shard_index: int, shard_count: int):
    pairs = [(truth, seed) for truth in truths for seed in seeds]
    _validate_shard(shard_index, shard_count, len(pairs))
    return [pair for position, pair in enumerate(pairs) if position % shard_count == shard_index]


def _part_path(output: Path, stem: str, shard_index: int, shard_count: int) -> Path:
    if shard_count == 1:
        return output / f"{stem}.csv"
    return output / f"{stem}.part-{shard_index:03d}-of-{shard_count:03d}.csv"


def _part_execution_path(output: Path, command: str, shard_index: int, shard_count: int) -> Path:
    if shard_count == 1:
        return output / f"{command}_execution.json"
    return output / f"{command}_execution.part-{shard_index:03d}-of-{shard_count:03d}.json"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing shard file: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty shard file: {path}")
    return rows


def schedule_due(
    kind: str,
    lag: int,
    signal: int,
    tape: AuditTape,
    episode_zero: int,
    horizon: int,
    blockers: int = 1,
) -> int:
    """Return the boundary at which an audit becomes visible.

    `episode_zero` is the zero-based current episode. A due value of t+1 is
    available at the next episode boundary, matching the original pilot.
    """
    if kind == "outcome_dependent":
        return episode_zero + (lag if signal == 1 else 0) + 1
    if kind in ("single_blocker", "front_blockers"):
        # Withhold the first `blockers` launched marks of each type until after
        # the learning horizon; all later marks return at the next boundary.
        # `single_blocker` is retained as the blockers=1 named stress case.
        count = 1 if kind == "single_blocker" else blockers
        if count < 1:
            raise ValueError("front blocker count must be positive")
        return horizon + 1 if tape.n < count else episode_zero + 1
    if kind == "immediate":
        return episode_zero + 1
    raise ValueError(f"unknown audit schedule: {kind}")


def boundary_metrics(views, c: float, active: tuple[int, ...]) -> tuple[float, float, float, float]:
    prefix = sum(views[i].prefix_width(c) for i in active)
    completion = sum(views[i].completion_width(c) for i in active)
    intersection = sum(views[i].intersection_width(c) for i in active)
    unresolved_mass = sum(
        (views[i].n - views[i].observed) / views[i].n if views[i].n else 0.0
        for i in active
    )
    return prefix, completion, intersection, unresolved_mass


def pilot(output: Path, smoke: bool = False):
    conf = json.loads((ROOT / "inputs" / "pilot.json").read_text())
    float_values, float_wastes, laws, best_policy, best_value = campaign_objects()
    T = 128 if smoke else conf["horizon"]
    truths = conf["truth_indices"][:1] if smoke else conf["truth_indices"]
    lags = conf["audit_lags"][:1] if smoke else conf["audit_lags"]
    seeds = conf["seeds"][:1] if smoke else conf["seeds"]
    algorithms = conf["algorithms"][:2] if smoke else conf["algorithms"]
    c = confidence_constant(T, conf["confidence_delta"])
    final, curves = [], []
    trace_path = output / "episode_trace.csv"
    output.mkdir(parents=True, exist_ok=True)
    fields = [
        "truth", "lag", "seed", "algorithm", "episode", "policy", "regret",
        "expected_waste", "realized_cost", "excluded", "fallback",
        "prefix_exposure", "completion_exposure", "intersection_exposure",
        "unresolved_mass_exposure", "n0", "n1", "n2", "m0", "m1", "m2",
        "u0", "u1", "u2",
    ]
    with trace_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for truth in truths:
            product_index = 9 + truth % 9
            for lag in lags:
                for seed in seeds:
                    rng = random.Random(seed + 10000 * truth)
                    worlds = [
                        tuple(draw_index(laws[truth][i], rng.random()) for i in range(3))
                        for _ in range(T)
                    ]
                    for method in algorithms:
                        tapes = [AuditTape() for _ in range(3)]
                        regret = waste = cost = 0.0
                        prefix_exposure = completion_exposure = intersection_exposure = 0.0
                        unresolved_mass_exposure = 0.0
                        exclusions = fallbacks = 0
                        max_debt = max_pending = 0
                        for t, world in enumerate(worlds):
                            for tape in tapes:
                                tape.receive(t)
                            views = [x.snapshot() for x in tapes]
                            if method in ("known", "product"):
                                policy_id = best_policy[truth if method == "known" else product_index]
                                excluded, fallback = False, False
                            else:
                                policy_id, feasible, fallback = select(
                                    method, views, laws, best_value, best_policy, c
                                )
                                excluded = feasible is not None and truth not in feasible
                            instantaneous = float_values[truth][policy_id] - best_value[truth]
                            assert instantaneous >= -1e-12
                            regret += instantaneous
                            waste += float_wastes[truth][policy_id]
                            exclusions += int(excluded)
                            fallbacks += int(fallback)
                            residual, _, active, _ = evaluate_world(POLICIES[policy_id], world)
                            root_delay = ROOT_MARKS[world[0]][1]
                            realized = float(residual) + root_delay
                            cost += realized
                            pe, ce, ie, hc = boundary_metrics(views, c, active)
                            prefix_exposure += pe
                            completion_exposure += ce
                            intersection_exposure += ie
                            unresolved_mass_exposure += hc
                            ns = [x.n for x in tapes]
                            ms = [x.prefix for x in tapes]
                            us = [x.n - x.observed for x in tapes]
                            max_debt = max(max_debt, max(n - m for n, m in zip(ns, ms)))
                            max_pending = max(max_pending, max(us))
                            writer.writerow(
                                [
                                    truth, lag, seed, method, t + 1, policy_id,
                                    instantaneous, float_wastes[truth][policy_id], realized,
                                    int(excluded), int(fallback), pe, ce, ie, hc,
                                    *ns, *ms, *us,
                                ]
                            )
                            for i in active:
                                signal = ROOT_MARKS[world[i]][0] if i == 0 else CHILD_MARKS[world[i]][0]
                                due = schedule_due("outcome_dependent", lag, signal, tapes[i], t, T)
                                tapes[i].launch(world[i], due)
                            if (t + 1) % 32 == 0 or t + 1 == T:
                                curves.append(
                                    dict(
                                        truth=truth,
                                        lag=lag,
                                        seed=seed,
                                        algorithm=method,
                                        episode=t + 1,
                                        pseudo_regret=regret,
                                        expected_waste=waste,
                                        prefix_exposure=prefix_exposure,
                                        completion_exposure=completion_exposure,
                                        intersection_exposure=intersection_exposure,
                                        unresolved_mass_exposure=unresolved_mass_exposure,
                                    )
                                )
                        for tape in tapes:
                            for due in sorted(tape.ready):
                                tape.receive(due)
                            assert tape.observed == tape.prefix == tape.n
                            assert tape.histogram == tape.prefix_histogram
                        final.append(
                            dict(
                                truth=truth,
                                lag=lag,
                                seed=seed,
                                algorithm=method,
                                episodes=T,
                                pseudo_regret=regret,
                                expected_waste=waste,
                                realized_cost=cost,
                                confidence_exclusions=exclusions,
                                safe_fallbacks=fallbacks,
                                maximum_prefix_debt=max_debt,
                                maximum_unreturned=max_pending,
                                prefix_exposure=prefix_exposure,
                                completion_exposure=completion_exposure,
                                intersection_exposure=intersection_exposure,
                                unresolved_mass_exposure=unresolved_mass_exposure,
                            )
                        )
    write_csv(output / "runs.csv", final)
    write_csv(output / "curves.csv", curves)
    return {
        "runs": len(final),
        "episodes": len(final) * T,
        "dictionary_models": 27,
        "policies": 27,
        "smoke": smoke,
    }


def stress(output: Path, shard_index: int = 0, shard_count: int = 1):
    conf = json.loads((ROOT / "inputs" / "stress.json").read_text())
    float_values, float_wastes, laws, best_policy, best_value = campaign_objects()
    T = conf["horizon"]
    c = confidence_constant(T, conf["confidence_delta"])
    stride = conf["curve_stride"]
    runs, curves = [], []
    selected_pairs = _shard_pairs(
        conf["truth_indices"], conf["seeds"], shard_index, shard_count
    )
    for truth, seed in selected_pairs:
        product_index = 9 + truth % 9
        rng = random.Random(seed + 10000 * truth)
        worlds = [
            tuple(draw_index(laws[truth][i], rng.random()) for i in range(3))
            for _ in range(T)
        ]
        for scenario in conf["scenarios"]:
            name = scenario["name"]
            kind = scenario["kind"]
            lag = int(scenario.get("lag", 0))
            for method in conf["algorithms"]:
                tapes = [AuditTape() for _ in range(3)]
                regret = waste = cost = 0.0
                prefix_exposure = completion_exposure = intersection_exposure = 0.0
                unresolved_mass_exposure = 0.0
                exclusions = fallbacks = 0
                max_debt = max_pending = 0
                for t, world in enumerate(worlds):
                    for tape in tapes:
                        tape.receive(t)
                    views = [x.snapshot() for x in tapes]
                    if method in ("known", "product"):
                        policy_id = best_policy[truth if method == "known" else product_index]
                        excluded, fallback = False, False
                    else:
                        policy_id, feasible, fallback = select(
                            method, views, laws, best_value, best_policy, c
                        )
                        excluded = feasible is not None and truth not in feasible
                    instantaneous = float_values[truth][policy_id] - best_value[truth]
                    assert instantaneous >= -1e-12
                    regret += instantaneous
                    waste += float_wastes[truth][policy_id]
                    exclusions += int(excluded)
                    fallbacks += int(fallback)
                    residual, _, active, _ = evaluate_world(POLICIES[policy_id], world)
                    cost += float(residual) + ROOT_MARKS[world[0]][1]
                    pe, ce, ie, hc = boundary_metrics(views, c, active)
                    prefix_exposure += pe
                    completion_exposure += ce
                    intersection_exposure += ie
                    unresolved_mass_exposure += hc
                    ns = [x.n for x in tapes]
                    ms = [x.prefix for x in tapes]
                    us = [x.n - x.observed for x in tapes]
                    max_debt = max(max_debt, max(n - m for n, m in zip(ns, ms)))
                    max_pending = max(max_pending, max(us))
                    for i in active:
                        signal = ROOT_MARKS[world[i]][0] if i == 0 else CHILD_MARKS[world[i]][0]
                        due = schedule_due(
                            kind, lag, signal, tapes[i], t, T, int(scenario.get("blockers", 1))
                        )
                        tapes[i].launch(world[i], due)
                    if (t + 1) % stride == 0 or t + 1 == T:
                        curves.append(
                            dict(
                                scenario=name,
                                truth=truth,
                                seed=seed,
                                algorithm=method,
                                episode=t + 1,
                                pseudo_regret=regret,
                                expected_waste=waste,
                                prefix_exposure=prefix_exposure,
                                completion_exposure=completion_exposure,
                                intersection_exposure=intersection_exposure,
                                unresolved_mass_exposure=unresolved_mass_exposure,
                            )
                        )
                for tape in tapes:
                    for due in sorted(tape.ready):
                        tape.receive(due)
                    assert tape.observed == tape.prefix == tape.n
                    assert tape.histogram == tape.prefix_histogram
                runs.append(
                    dict(
                        scenario=name,
                        truth=truth,
                        seed=seed,
                        algorithm=method,
                        episodes=T,
                        pseudo_regret=regret,
                        expected_waste=waste,
                        realized_cost=cost,
                        confidence_exclusions=exclusions,
                        safe_fallbacks=fallbacks,
                        maximum_prefix_debt=max_debt,
                        maximum_unreturned=max_pending,
                        prefix_exposure=prefix_exposure,
                        completion_exposure=completion_exposure,
                        intersection_exposure=intersection_exposure,
                        unresolved_mass_exposure=unresolved_mass_exposure,
                    )
                )
    write_csv(_part_path(output, "stress_runs", shard_index, shard_count), runs)
    write_csv(_part_path(output, "stress_curves", shard_index, shard_count), curves)
    return {
        "runs": len(runs),
        "episodes": len(runs) * T,
        "scenarios": [x["name"] for x in conf["scenarios"]],
        "truth_seed_cells": len(selected_pairs),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "workers": 1,
    }

def merge_stress(output: Path, shard_count: int):
    conf = json.loads((ROOT / "inputs" / "stress.json").read_text())
    _validate_shard(0, shard_count, len(conf["truth_indices"]) * len(conf["seeds"]))
    runs, curves = [], []
    for shard_index in range(shard_count):
        runs.extend(_read_csv(_part_path(output, "stress_runs", shard_index, shard_count)))
        curves.extend(_read_csv(_part_path(output, "stress_curves", shard_index, shard_count)))
    truth_rank = {str(x): i for i, x in enumerate(conf["truth_indices"])}
    seed_rank = {str(x): i for i, x in enumerate(conf["seeds"])}
    scenario_rank = {x["name"]: i for i, x in enumerate(conf["scenarios"])}
    algorithm_rank = {x: i for i, x in enumerate(conf["algorithms"])}
    run_key = lambda r: (
        truth_rank[r["truth"]], seed_rank[r["seed"]], scenario_rank[r["scenario"]],
        algorithm_rank[r["algorithm"]]
    )
    curve_key = lambda r: run_key(r) + (int(r["episode"]),)
    runs.sort(key=run_key)
    curves.sort(key=curve_key)
    expected_runs = (
        len(conf["truth_indices"]) * len(conf["seeds"]) *
        len(conf["scenarios"]) * len(conf["algorithms"])
    )
    expected_curves = expected_runs * ((conf["horizon"] + conf["curve_stride"] - 1) // conf["curve_stride"])
    if len(runs) != expected_runs or len(curves) != expected_curves:
        raise ValueError(
            f"incomplete stress shards: {len(runs)}/{expected_runs} runs and "
            f"{len(curves)}/{expected_curves} curve rows"
        )
    if len({(r["truth"], r["seed"], r["scenario"], r["algorithm"]) for r in runs}) != expected_runs:
        raise ValueError("duplicate or missing stress run keys")
    write_csv(output / "stress_runs.csv", runs)
    write_csv(output / "stress_curves.csv", curves)
    return {
        "status": "stress shards merged",
        "shard_count": shard_count,
        "runs": len(runs),
        "curve_rows": len(curves),
        "episodes": len(runs) * conf["horizon"],
    }

def saturated_outstanding_envelope(blockers: int, launches: int) -> float:
    """Sharp unit-batch envelope sum_{s=1}^{n-1} min(U,s)/s."""
    if blockers < 0 or launches < 0:
        raise ValueError("blockers and launches must be nonnegative")
    return sum(min(blockers, s) / s for s in range(1, launches))


def scale(output: Path, shard_index: int = 0, shard_count: int = 1):
    """Prespecified horizon/backlog sensitivity campaign.

    Common worlds are nested across horizons for each truth/seed pair.  The
    front-blocker schedules attain the saturated outstanding-record envelope
    exactly, so this campaign checks both decision behavior and descriptor
    accounting without tuning a delay pattern after seeing outcomes.
    """
    conf = json.loads((ROOT / "inputs" / "scaling.json").read_text())
    float_values, float_wastes, laws, best_policy, best_value = campaign_objects()
    max_horizon = max(conf["horizons"])
    rows = []
    total_episodes = 0
    selected_pairs = _shard_pairs(
        conf["truth_indices"], conf["seeds"], shard_index, shard_count
    )
    for truth, seed in selected_pairs:
        rng = random.Random(seed + 10000 * truth)
        all_worlds = [
            tuple(draw_index(laws[truth][i], rng.random()) for i in range(3))
            for _ in range(max_horizon)
        ]
        for T in conf["horizons"]:
            c = confidence_constant(T, conf["confidence_delta"])
            worlds = all_worlds[:T]
            for scenario in conf["scenarios"]:
                name = scenario["name"]
                kind = scenario["kind"]
                lag = int(scenario.get("lag", 0))
                blockers = int(scenario.get("blockers", 1))
                for method in conf["algorithms"]:
                    tapes = [AuditTape() for _ in range(3)]
                    regret = waste = cost = 0.0
                    prefix_exposure = completion_exposure = intersection_exposure = 0.0
                    unresolved_mass_exposure = 0.0
                    exclusions = fallbacks = 0
                    max_debt = max_pending = 0
                    for t, world in enumerate(worlds):
                        for tape in tapes:
                            tape.receive(t)
                        views = [x.snapshot() for x in tapes]
                        if method == "known":
                            policy_id = best_policy[truth]
                            excluded, fallback = False, False
                        else:
                            policy_id, feasible, fallback = select(
                                method, views, laws, best_value, best_policy, c
                            )
                            excluded = feasible is not None and truth not in feasible
                        instantaneous = float_values[truth][policy_id] - best_value[truth]
                        assert instantaneous >= -1e-12
                        regret += instantaneous
                        waste += float_wastes[truth][policy_id]
                        exclusions += int(excluded)
                        fallbacks += int(fallback)
                        residual, _, active, _ = evaluate_world(POLICIES[policy_id], world)
                        cost += float(residual) + ROOT_MARKS[world[0]][1]
                        pe, ce, ie, ue = boundary_metrics(views, c, active)
                        prefix_exposure += pe
                        completion_exposure += ce
                        intersection_exposure += ie
                        unresolved_mass_exposure += ue
                        ns = [x.n for x in tapes]
                        ms = [x.prefix for x in tapes]
                        us = [x.n - x.observed for x in tapes]
                        max_debt = max(max_debt, max(n - m for n, m in zip(ns, ms)))
                        max_pending = max(max_pending, max(us))
                        for i in active:
                            signal = ROOT_MARKS[world[i]][0] if i == 0 else CHILD_MARKS[world[i]][0]
                            due = schedule_due(kind, lag, signal, tapes[i], t, T, blockers)
                            tapes[i].launch(world[i], due)
                    launch_counts = [tape.n for tape in tapes]
                    envelope = sum(
                        saturated_outstanding_envelope(blockers, n) for n in launch_counts
                    )
                    if kind == "front_blockers":
                        assert abs(unresolved_mass_exposure - envelope) < 1e-9
                    for tape in tapes:
                        for due in sorted(tape.ready):
                            tape.receive(due)
                        assert tape.observed == tape.prefix == tape.n
                        assert tape.histogram == tape.prefix_histogram
                    rows.append(
                        dict(
                            scenario=name,
                            blockers=blockers,
                            horizon=T,
                            truth=truth,
                            seed=seed,
                            algorithm=method,
                            episodes=T,
                            pseudo_regret=regret,
                            expected_waste=waste,
                            realized_cost=cost,
                            confidence_exclusions=exclusions,
                            safe_fallbacks=fallbacks,
                            maximum_prefix_debt=max_debt,
                            maximum_unreturned=max_pending,
                            launches0=launch_counts[0],
                            launches1=launch_counts[1],
                            launches2=launch_counts[2],
                            prefix_exposure=prefix_exposure,
                            completion_exposure=completion_exposure,
                            intersection_exposure=intersection_exposure,
                            unresolved_mass_exposure=unresolved_mass_exposure,
                            sharp_unresolved_mass_envelope=envelope,
                        )
                    )
                    total_episodes += T
    write_csv(_part_path(output, "scale_runs", shard_index, shard_count), rows)
    return {
        "runs": len(rows),
        "episodes": total_episodes,
        "horizons": conf["horizons"],
        "scenarios": [x["name"] for x in conf["scenarios"]],
        "truth_seed_cells": len(selected_pairs),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "workers": 1,
    }


def merge_scale(output: Path, shard_count: int):
    conf = json.loads((ROOT / "inputs" / "scaling.json").read_text())
    _validate_shard(0, shard_count, len(conf["truth_indices"]) * len(conf["seeds"]))
    rows = []
    for shard_index in range(shard_count):
        rows.extend(_read_csv(_part_path(output, "scale_runs", shard_index, shard_count)))
    truth_rank = {str(x): i for i, x in enumerate(conf["truth_indices"])}
    seed_rank = {str(x): i for i, x in enumerate(conf["seeds"])}
    horizon_rank = {str(x): i for i, x in enumerate(conf["horizons"])}
    scenario_rank = {x["name"]: i for i, x in enumerate(conf["scenarios"])}
    algorithm_rank = {x: i for i, x in enumerate(conf["algorithms"])}
    rows.sort(key=lambda r: (
        truth_rank[r["truth"]], seed_rank[r["seed"]], horizon_rank[r["horizon"]],
        scenario_rank[r["scenario"]], algorithm_rank[r["algorithm"]]
    ))
    expected = (
        len(conf["truth_indices"]) * len(conf["seeds"]) * len(conf["horizons"]) *
        len(conf["scenarios"]) * len(conf["algorithms"])
    )
    keys = {(r["truth"], r["seed"], r["horizon"], r["scenario"], r["algorithm"]) for r in rows}
    if len(rows) != expected or len(keys) != expected:
        raise ValueError(f"incomplete or duplicate scale shards: {len(rows)}/{expected} rows")
    write_csv(output / "scale_runs.csv", rows)
    return {
        "status": "scale shards merged",
        "shard_count": shard_count,
        "runs": len(rows),
        "episodes": sum(int(r["episodes"]) for r in rows),
    }

def verify(output: Path):
    from collections import defaultdict

    totals = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0])
    with (output / "episode_trace.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            k = (r["truth"], r["lag"], r["seed"], r["algorithm"])
            a = totals[k]
            a[0] += 1
            a[1] += float(r["regret"])
            a[2] += float(r["expected_waste"])
            a[3] += float(r["realized_cost"])
            a[4] += int(r["excluded"])
            a[5] += int(r["fallback"])
            a[6] = max(a[6], *(int(r[f"n{i}"]) - int(r[f"m{i}"]) for i in range(3)))
            a[7] = max(a[7], *(int(r[f"u{i}"]) for i in range(3)))
            a[8] += float(r["prefix_exposure"])
            a[9] += float(r["completion_exposure"])
            a[10] += float(r["intersection_exposure"])
            a[11] += float(r["unresolved_mass_exposure"])
    count = 0
    with (output / "runs.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            k = (r["truth"], r["lag"], r["seed"], r["algorithm"])
            a = totals.pop(k)
            assert a[0] == int(r["episodes"])
            for j, name in enumerate(("pseudo_regret", "expected_waste", "realized_cost"), 1):
                assert abs(a[j] - float(r[name])) < 1e-9
            for j, name in enumerate(
                ("confidence_exclusions", "safe_fallbacks", "maximum_prefix_debt", "maximum_unreturned"), 4
            ):
                assert a[j] == int(r[name])
            for j, name in enumerate(
                ("prefix_exposure", "completion_exposure", "intersection_exposure", "unresolved_mass_exposure"), 8
            ):
                assert abs(a[j] - float(r[name])) < 1e-9
            count += 1
    assert not totals
    return {"reconciled_runs": count, "status": "passed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "check", "smoke", "pilot", "stress", "merge-stress",
            "scale", "merge-scale", "verify"
        ),
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()

    cap = 3584 * 1024 * 1024
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (min(cap, hard) if hard != -1 else cap, hard))
    args.output.mkdir(parents=True, exist_ok=True)
    start_wall = time.monotonic()
    start_cpu = time.process_time()
    if args.command == "check":
        if (args.shard_index, args.shard_count) != (0, 1):
            raise ValueError("check is not sharded")
        prepare(args.output)
        report = {"status": "exact checks passed"}
    elif args.command == "verify":
        if (args.shard_index, args.shard_count) != (0, 1):
            raise ValueError("verify is not sharded")
        report = verify(args.output)
    elif args.command == "stress":
        report = stress(args.output, args.shard_index, args.shard_count)
    elif args.command == "merge-stress":
        if args.shard_index != 0 or args.shard_count == 1:
            raise ValueError("merge-stress requires --shard-count > 1 and no shard index")
        report = merge_stress(args.output, args.shard_count)
    elif args.command == "scale":
        report = scale(args.output, args.shard_index, args.shard_count)
    elif args.command == "merge-scale":
        if args.shard_index != 0 or args.shard_count == 1:
            raise ValueError("merge-scale requires --shard-count > 1 and no shard index")
        report = merge_scale(args.output, args.shard_count)
    else:
        if (args.shard_index, args.shard_count) != (0, 1):
            raise ValueError(f"{args.command} is not sharded")
        report = pilot(args.output, smoke=args.command == "smoke")
    report.update(
        cpu_seconds=time.process_time() - start_cpu,
        wall_seconds=time.monotonic() - start_wall,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        workers=1,
    )
    if args.command in ("stress", "scale"):
        execution_path = _part_execution_path(
            args.output, args.command, args.shard_index, args.shard_count
        )
    else:
        execution_path = args.output / (args.command.replace("-", "_") + "_execution.json")
    write_json(execution_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
