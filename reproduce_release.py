#!/usr/bin/env python3
"""Run the documented three-campaign release workflow.

The artifact has exactly three empirical campaigns: pilot, stress, and scale.
There is no off-grid command or result.  ``--skip-long`` runs only the exact
checks, pilot, reconciliation, report subset, smoke test, and partial validators.
"""
from __future__ import annotations
import argparse
import json
import os
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str], log: Path) -> dict:
    start = time.perf_counter()
    cpu0 = time.process_time()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with log.open("w", encoding="utf-8") as f:
        try:
            proc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                  text=True, timeout=120)
            returncode = proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            f.write("\nCommand exceeded the 120-second wall-time budget.\n")
            returncode = 124
            timed_out = True
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "command": cmd,
        "returncode": returncode,
        "timed_out": timed_out,
        "timeout_seconds": 120,
        "wall_seconds": time.perf_counter() - start,
        "parent_cpu_seconds": time.process_time() - cpu0,
        "child_user_seconds": after.ru_utime - before.ru_utime,
        "child_system_seconds": after.ru_stime - before.ru_stime,
        "maxrss_kib_process_peak": after.ru_maxrss,
        "log": str(log),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output", nargs="?", default="repro-results")
    ap.add_argument("--skip-long", action="store_true")
    args = ap.parse_args()

    out = (ROOT / args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "command-logs"
    logs.mkdir(exist_ok=True)
    records: list[dict] = []

    def execute(role: str, cmd: list[str]) -> bool:
        rec = run(cmd, logs / f"{len(records):02d}-{role}.log")
        rec["role"] = role
        records.append(rec)
        if rec["returncode"]:
            status = {"status": "FAIL", "scope": "quick" if args.skip_long else "full",
                      "commands": records}
            (out / "reproduction-command-report.json").write_text(json.dumps(status, indent=2) + "\n")
        return rec["returncode"] == 0

    base = [sys.executable, "-B", "run.py"]
    commands: list[tuple[str, list[str]]] = [
        ("campaign-keys", [sys.executable, "-B", "tests/check_campaign_keys.py", "--output", str(out / "campaign-key-inputs")]),
        ("check", base + ["check", "--output", str(out)]),
        ("pilot", base + ["pilot", "--output", str(out)]),
    ]
    if not args.skip_long:
        for i in range(8):
            commands.append((f"stress-shard-{i}", base + ["stress", "--output", str(out), "--shard-index", str(i), "--shard-count", "8"]))
        commands.append(("merge-stress", base + ["merge-stress", "--output", str(out), "--shard-count", "8"]))
        for i in range(4):
            commands.append((f"scale-shard-{i}", base + ["scale", "--output", str(out), "--shard-index", str(i), "--shard-count", "4"]))
        commands.append(("merge-scale", base + ["merge-scale", "--output", str(out), "--shard-count", "4"]))
    commands.append(("verify", base + ["verify", "--output", str(out)]))

    for role, cmd in commands:
        if not execute(role, cmd):
            status = {"status": "FAIL", "scope": "quick" if args.skip_long else "full", "commands": records}
            (out / "reproduction-command-report.json").write_text(json.dumps(status, indent=2) + "\n")
            return records[-1]["returncode"]

    report_cmd = [sys.executable, "-B", "report.py", "--results", str(out)]
    if args.skip_long:
        report_cmd += ["--skip-stress", "--skip-scale"]
    if not execute("report", report_cmd):
        return records[-1]["returncode"]

    smoke = out / "smoke"
    smoke.mkdir(exist_ok=True)
    if not execute("smoke", base + ["smoke", "--output", str(smoke)]):
        return records[-1]["returncode"]

    checks = [
        [sys.executable, "-B", "reviewer_requirements.py", "--results", str(out)] + (["--allow-partial"] if args.skip_long else []),
        [sys.executable, "-B", "reviewer_invariants.py", "--results", str(out)],
        [sys.executable, "-B", "validate_release.py", "--results", str(out)] + (["--quick"] if args.skip_long else []),
    ]
    for cmd in checks:
        role = Path(cmd[2]).stem
        if not execute(role, cmd):
            status = {"status": "FAIL", "scope": "quick" if args.skip_long else "full", "commands": records}
            (out / "reproduction-command-report.json").write_text(json.dumps(status, indent=2) + "\n")
            return records[-1]["returncode"]

    report = {
        "schema_version": 1,
        "status": "PASS",
        "scope": "quick: long stress/scale campaigns not regenerated" if args.skip_long else "full three-campaign reproduction",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": sys.platform,
        "cpu_count_reported_by_os": os.cpu_count(),
        "campaigns_regenerated": ["pilot"] if args.skip_long else ["pilot", "stress", "scale"],
        "offgrid_campaign": "absent; no command or result is claimed",
        "commands": records,
        "wall_seconds_sum": sum(x["wall_seconds"] for x in records),
        "resource_note": "maxrss is a host-reported per-command process peak and is not additive; timing varies by environment.",
    }
    (out / "reproduction-command-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
