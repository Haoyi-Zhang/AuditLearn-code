#!/usr/bin/env python3
"""Independent standard-library validator for the current artifact scope."""
from __future__ import annotations
import argparse
import ast
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
REQUIRED_DOCS = [
    ROOT / "README.md",
    ROOT / "ARTIFACT-EVALUATION.md",
    ROOT / "DATA-DICTIONARY.md",
    ROOT / "STATISTICAL-INTERPRETATION.md",
    ROOT / "LIMITATIONS-MATRIX.md",
    ROOT / "proofs" / "analysis.md",
    ROOT / "proofs" / "adaptive-indexing-and-completion.md",
    ROOT / "proofs" / "misspecification-and-blackout.md",
]


def source_audit() -> dict:
    findings = {"builtin_hash_calls": [], "network_or_model_imports": [], "gpu_imports": []}
    for path in sorted(ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            findings.setdefault("syntax_errors", []).append(f"{path.relative_to(PROJECT)}:{exc.lineno}:{exc.msg}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "hash":
                findings["builtin_hash_calls"].append(f"{path.relative_to(PROJECT)}:{node.lineno}")
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name.split('.')[0] for a in node.names] if isinstance(node, ast.Import) else [(node.module or '').split('.')[0]]
                for name in names:
                    if name in {"requests", "httpx", "openai", "anthropic", "socket", "urllib3"}:
                        findings["network_or_model_imports"].append(f"{path.relative_to(PROJECT)}:{node.lineno}:{name}")
                    if name in {"torch", "tensorflow", "jax", "cupy"}:
                        findings["gpu_imports"].append(f"{path.relative_to(PROJECT)}:{node.lineno}:{name}")
    return findings


def bibliography_audit() -> dict:
    """Audit the paper when this artifact is inside the complete project.

    The standalone repository intentionally has no sibling ``paper/`` tree.
    That absence is therefore reported explicitly rather than treated as a
    successful bibliography check or as an artifact failure.
    """
    paper = PROJECT / "paper"
    bib_path = paper / "references.bib"
    if not bib_path.exists():
        return {
            "performed": False,
            "reason": "standalone artifact: no sibling paper/references.bib",
            "entries": None,
            "cited": None,
            "missing": [],
            "uncited": [],
            "uses_nocite": None,
        }
    bib = bib_path.read_text(encoding="utf-8")
    tex = "\n".join(p.read_text(encoding="utf-8") for p in paper.glob("*.tex"))
    entries = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib))
    cited: set[str] = set()
    for grp in re.findall(r"\\cite\w*\s*(?:\[[^\]]*\]\s*)*\{([^}]+)\}", re.sub(r"(?m)%.*$", "", tex)):
        cited.update(k.strip() for k in grp.split(",") if k.strip())
    return {
        "performed": True,
        "reason": None,
        "entries": len(entries),
        "cited": len(cited),
        "missing": sorted(cited - entries),
        "uncited": sorted(entries - cited),
        "uses_nocite": "\\nocite" in tex,
    }


def inspect_csv(path: Path, full: bool) -> dict:
    nonfinite = 0
    duplicate_rows = 0
    seen: set[tuple[str, ...]] = set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        rows = 0
        for row in reader:
            rows += 1
            if full:
                key = tuple(row)
                duplicate_rows += int(key in seen)
                seen.add(key)
            for cell in row:
                try:
                    value = float(cell)
                except ValueError:
                    continue
                if not math.isfinite(value):
                    nonfinite += 1
    return {"columns": header, "rows": rows, "duplicate_exact_rows": duplicate_rows, "nonfinite_cells": nonfinite}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    results = Path(args.results).resolve()
    errors: list[str] = []

    for path in REQUIRED_DOCS:
        if not path.exists():
            errors.append(f"missing required document: {path.relative_to(PROJECT)}")

    compile_proc = subprocess.run([sys.executable, "-m", "compileall", "-q", str(ROOT)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if compile_proc.returncode:
        errors.append("Python compileall failed")

    csv_report = {}
    if results.exists():
        for path in sorted(results.glob("*.csv")):
            info = inspect_csv(path, full=not args.quick)
            csv_report[path.name] = info
            if info["nonfinite_cells"]:
                errors.append(f"nonfinite numeric cells in {path.name}")
            if info["duplicate_exact_rows"]:
                errors.append(f"exact duplicate rows in {path.name}: {info['duplicate_exact_rows']}")
    else:
        errors.append(f"results directory does not exist: {results}")

    src = source_audit()
    for key, hits in src.items():
        if hits:
            errors.append(f"{key}: {hits[:10]}")
    bib = bibliography_audit()
    if bib["missing"] or bib["uncited"] or bib["uses_nocite"]:
        errors.append(f"bibliography mismatch: {bib}")

    run_text = (ROOT / "run.py").read_text(encoding="utf-8")
    inputs = sorted(p.name for p in (ROOT / "inputs").glob("*.json"))
    scope = {
        "run_commands": re.findall(r'choices=\[([^\]]+)\]', run_text),
        "input_files": inputs,
        "offgrid_command_present": "offgrid" in run_text.lower(),
        "offgrid_input_present": any("offgrid" in x.lower() for x in inputs),
    }
    if scope["offgrid_command_present"] or scope["offgrid_input_present"]:
        errors.append("unexpected off-grid executable asset present")

    try:
        results_display = str(results.relative_to(ROOT))
    except ValueError:
        results_display = str(results)
    report = {
        "status": "PASS" if not errors else "FAIL",
        "quick": args.quick,
        "results": results_display,
        "compileall_returncode": compile_proc.returncode,
        "csv": csv_report,
        "bibliography": bib,
        "source_audit": src,
        "campaign_scope": scope,
        "errors": errors,
    }
    if results.exists():
        (results / "release-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
