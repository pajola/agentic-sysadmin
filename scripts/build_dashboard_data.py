#!/usr/bin/env python3
"""
Build an enriched dashboard input (`summary_extended.json`) for an
experiment_logs/ run that has already finished.

The base `summary.json` produced by `core/analysis_engine.py` covers accuracy,
timing, tokens, and the rough error taxonomy. This script adds three extra
per-(model, solver) breakdowns by re-reading the per-run logs:

  - hallucination   : real vs text-encoded tool calls, count of invented names
  - duplicates      : repeated (tool, args) calls within a single run
  - empty_reasoning : how often intermediate llm_output nodes returned ""

The original `summary.json` is left untouched — output goes to
`summary_extended.json` in the same run directory. The dashboard reads either
file (extended is preferred if present).

Usage:
    python scripts/build_dashboard_data.py                       # latest run
    python scripts/build_dashboard_data.py --run 20260521_074715
    python scripts/build_dashboard_data.py --all                 # every run
    python scripts/build_dashboard_data.py --out path/to/file.json --run <id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _log_analytics import (  # noqa: E402
    aggregate_duplicates,
    aggregate_empty_reasoning,
    aggregate_hallucination,
    analyze_run_duplicates,
    analyze_run_empty_reasoning,
    analyze_run_hallucination,
    iter_run_files,
    parse_solver_dir,
)

EXPERIMENT_LOGS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiment_logs",
)


def pick_run_dirs(args) -> list[str]:
    if args.run:
        return [os.path.join(EXPERIMENT_LOGS_DIR, args.run)]
    all_runs = sorted(
        d for d in os.listdir(EXPERIMENT_LOGS_DIR)
        if os.path.isdir(os.path.join(EXPERIMENT_LOGS_DIR, d))
    )
    if args.all:
        return [os.path.join(EXPERIMENT_LOGS_DIR, d) for d in all_runs]
    if not all_runs:
        return []
    return [os.path.join(EXPERIMENT_LOGS_DIR, all_runs[-1])]


def _model_name_with_provider(model_part: str) -> str:
    """Reverse the filename sanitization done by analysis_engine.

    Solver dirs use ``<model>__<provider>`` where ``model`` has had ``:``,
    ``.``, ``/`` etc collapsed to ``_`` and the (provider) parentheses
    stripped. We can't recover the exact original string ("granite4:latest
    (chat-ollama)") from the filename alone, so this returns the safe form
    used inside the experiment_logs tree. The dashboard already deals with
    both shapes via its MODEL_SHORT map / fallback.
    """
    if "__" in model_part:
        model, provider = model_part.rsplit("__", 1)
        return f"{model} ({provider})"
    return model_part


def enrich_run(run_dir: str) -> dict[str, Any] | None:
    """Build the analytics payload for one experiment_logs/<id> directory."""
    summary_path = os.path.join(run_dir, "summary.json")
    if not os.path.isfile(summary_path):
        print(f"WARN: no summary.json in {run_dir}, skipping", file=sys.stderr)
        return None

    with open(summary_path, "r", encoding="utf-8") as f:
        base_summary = json.load(f)

    # Bucket per-run analyses by (model, solver) using the directory naming
    # convention. We re-use the same filename derivation as the legacy
    # scripts so identifiers line up.
    hallu_runs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    dup_runs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    empty_runs: dict[tuple[str, str], list[dict]] = defaultdict(list)

    files_scanned = 0
    files_with_planner_steps = 0
    for _run_id, _lab, _question, solver_dir, path in iter_run_files(run_dir):
        model_part, solver = parse_solver_dir(solver_dir)
        try:
            with open(path, "r", encoding="utf-8") as f:
                run = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"WARN: skipping {path}: {e}", file=sys.stderr)
            continue
        files_scanned += 1

        h = analyze_run_hallucination(run)
        d = analyze_run_duplicates(run)
        e = analyze_run_empty_reasoning(run)
        if h["planner_turns"] > 0:
            files_with_planner_steps += 1

        # Use the same (model, solver) identifier shape as the summary.json
        # pair_summary key so the dashboard can join the two.
        full_model = _model_name_with_provider(model_part)
        key = (full_model, solver)
        hallu_runs[key].append(h)
        dup_runs[key].append(d)
        empty_runs[key].append(e)

    per_pair: dict[str, dict[str, Any]] = {}
    for key in set(hallu_runs) | set(dup_runs) | set(empty_runs):
        model, solver = key
        per_pair[f"{model}|{solver}"] = {
            "model": model,
            "solver": solver,
            "hallucination": aggregate_hallucination(hallu_runs.get(key, [])),
            "duplicates": aggregate_duplicates(dup_runs.get(key, [])),
            "empty_reasoning": aggregate_empty_reasoning(empty_runs.get(key, [])),
        }

    enriched = dict(base_summary)  # shallow copy preserves all original fields
    enriched["schema_version"] = 2
    enriched["log_analytics"] = {
        "source_run_dir": os.path.basename(os.path.normpath(run_dir)),
        "files_scanned": files_scanned,
        "files_with_planner_steps": files_with_planner_steps,
        "per_pair": per_pair,
    }
    return enriched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="Specific run id under experiment_logs/")
    parser.add_argument("--all", action="store_true",
                        help="Build summary_extended.json for every run")
    parser.add_argument(
        "--out",
        help="Output path. Default: experiment_logs/<run>/summary_extended.json. "
             "With --all this is ignored (each run gets its own file).",
    )
    args = parser.parse_args()

    run_dirs = pick_run_dirs(args)
    if not run_dirs:
        print(f"No runs found under {EXPERIMENT_LOGS_DIR}", file=sys.stderr)
        return 1

    failures = 0
    for run_dir in run_dirs:
        payload = enrich_run(run_dir)
        if payload is None:
            failures += 1
            continue
        out_path = (
            args.out
            if (args.out and not args.all)
            else os.path.join(run_dir, "summary_extended.json")
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        per_pair_count = len(payload.get("log_analytics", {}).get("per_pair", {}))
        scanned = payload.get("log_analytics", {}).get("files_scanned", 0)
        print(
            f"Wrote {out_path}  "
            f"(scanned {scanned} run files, {per_pair_count} (model,solver) pairs)"
        )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
