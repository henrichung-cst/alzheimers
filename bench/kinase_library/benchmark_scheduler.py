"""Benchmark sequential vs process-parallel MEA workloads.

This script is bench-only. It runs ``run_source_benchmark.py`` in subprocesses
so each workload gets its own Python process and its own ``ALZ_MEA_THREADS``
setting. Use it to evaluate exact ``gseapy.prerank_rs`` scheduling without
touching canonical analysis outputs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


BENCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCH_DIR.parents[1]
DEFAULT_RUNS_DIR = BENCH_DIR / "runs"


@dataclass(frozen=True)
class Case:
    workload: str
    donor: str
    track: str
    state: str

    @property
    def label(self) -> str:
        return f"{self.workload}_{self.donor}_{self.track}_{self.state}"


def _parse_case(value: str) -> Case:
    fields = {}
    for part in value.split(","):
        if "=" not in part:
            raise argparse.ArgumentTypeError(
                "case must be comma-separated KEY=VALUE pairs"
            )
        key, val = part.split("=", 1)
        fields[key.strip()] = val.strip()
    workload = fields.get("workload", "projected-state")
    donor = fields.get("donor", "donor1")
    track = fields.get("track", "st")
    state = fields.get("state", "CD8Naive")
    if workload not in {"projected-state", "tcells-timecourse"}:
        raise argparse.ArgumentTypeError(f"unsupported workload: {workload}")
    if donor not in {"donor1", "donor2"}:
        raise argparse.ArgumentTypeError(f"unsupported donor: {donor}")
    if track not in {"st", "py"}:
        raise argparse.ArgumentTypeError(f"unsupported track: {track}")
    return Case(workload=workload, donor=donor, track=track, state=state)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--source", choices=["public", "local"], default="local")
    parser.add_argument("--local-src", type=Path, default=BENCH_DIR / "local_src")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--mode", choices=["sequential", "parallel"], default="parallel")
    parser.add_argument("--threads-per-process", type=int, default=8)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument(
        "--case",
        action="append",
        type=_parse_case,
        required=True,
        help=(
            "Workload spec as KEY=VALUE pairs, e.g. "
            "workload=projected-state,donor=donor1,track=st,state=CD8Naive. "
            "Repeat for multiple cases."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def _run_case(
    args: argparse.Namespace,
    case: Case,
    *,
    run_root: Path,
    index: int,
) -> dict:
    case_root = run_root / f"{index:02d}_{case.label}"
    out_dir = case_root / "outputs"
    summary_path = case_root / "summary.json"
    cmd = [
        args.python,
        str(BENCH_DIR / "run_source_benchmark.py"),
        "--source",
        args.source,
        "--local-src",
        str(args.local_src),
        "--out-dir",
        str(out_dir),
        "--summary-json",
        str(summary_path),
        "--workload",
        case.workload,
        "--donor",
        case.donor,
        "--track",
        case.track,
        "--state",
        case.state,
    ]
    if args.verbose:
        cmd.append("--verbose")

    env = os.environ.copy()
    env["ALZ_MEA_THREADS"] = str(args.threads_per_process)
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=not args.verbose,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, file=sys.stderr)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(completed.returncode, cmd)
    summary = json.loads(summary_path.read_text())
    summary["case_label"] = case.label
    return summary


def _run_all(args: argparse.Namespace, run_root: Path) -> list[dict]:
    cases: list[Case] = args.case
    if args.mode == "sequential":
        return [
            _run_case(args, case, run_root=run_root, index=i)
            for i, case in enumerate(cases, start=1)
        ]

    max_workers = args.max_workers or len(cases)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_run_case, args, case, run_root=run_root, index=i)
            for i, case in enumerate(cases, start=1)
        ]
        return [future.result() for future in futures]


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.threads_per_process <= 0:
        raise SystemExit("--threads-per-process must be positive")
    if args.max_workers is not None and args.max_workers <= 0:
        raise SystemExit("--max-workers must be positive")

    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_root = (args.runs_dir / f"scheduler_{run_id}").resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    summaries = _run_all(args, run_root)
    elapsed = time.perf_counter() - start
    result = {
        "run_root": str(run_root),
        "mode": args.mode,
        "source": args.source,
        "threads_per_process": args.threads_per_process,
        "max_workers": args.max_workers or (len(args.case) if args.mode == "parallel" else 1),
        "elapsed_seconds": elapsed,
        "case_elapsed_seconds": [
            {
                "case_label": summary["case_label"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "output_files": len(summary["outputs"]),
            }
            for summary in summaries
        ],
        "summaries": summaries,
    }
    result_path = run_root / "scheduler_summary.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
