"""Compare public-package MEA output against a local kinase-library source copy."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path


BENCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCH_DIR.parents[1]
DEFAULT_LOCAL_SRC = BENCH_DIR / "local_src"
DEFAULT_RUNS_DIR = BENCH_DIR / "runs"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--local-src", type=Path, default=DEFAULT_LOCAL_SRC)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--donor", choices=["donor1", "donor2"], default="donor1")
    parser.add_argument("--track", choices=["st", "py"], default="st")
    parser.add_argument("--state", default="CD8Naive")
    parser.add_argument(
        "--profile-source",
        choices=["none", "public", "local", "both"],
        default="none",
    )
    parser.add_argument("--profile-limit", type=int, default=40)
    parser.add_argument("--verbose", action="store_true")
    return parser


def _run_source(
    args: argparse.Namespace,
    *,
    source: str,
    run_root: Path,
) -> dict:
    out_dir = run_root / source / "outputs"
    summary_path = run_root / source / "summary.json"
    profile = args.profile_source in {source, "both"}

    cmd = [
        args.python,
        str(BENCH_DIR / "run_source_benchmark.py"),
        "--source",
        source,
        "--local-src",
        str(args.local_src),
        "--out-dir",
        str(out_dir),
        "--summary-json",
        str(summary_path),
        "--donor",
        args.donor,
        "--track",
        args.track,
        "--state",
        args.state,
        "--profile-limit",
        str(args.profile_limit),
    ]
    if profile:
        cmd.append("--profile")
    if args.verbose:
        cmd.append("--verbose")

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=not args.verbose,
        text=True,
    )
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, file=sys.stderr)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(completed.returncode, cmd)
    return json.loads(summary_path.read_text())


def _output_map(summary: dict) -> dict[str, tuple[str, int]]:
    return {
        row["path"]: (row["sha256"], int(row["bytes"]))
        for row in summary["outputs"]
    }


def _compare_outputs(public: dict, local: dict) -> dict[str, object]:
    public_map = _output_map(public)
    local_map = _output_map(local)
    all_paths = sorted(set(public_map) | set(local_map))
    mismatches = []
    for path in all_paths:
        if public_map.get(path) != local_map.get(path):
            mismatches.append({
                "path": path,
                "public": public_map.get(path),
                "local": local_map.get(path),
            })
    return {
        "exact_match": not mismatches,
        "mismatches": mismatches,
        "public_file_count": len(public_map),
        "local_file_count": len(local_map),
    }


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    package_dir = args.local_src / "kinase_library"
    if not package_dir.exists():
        raise SystemExit(
            f"Missing local source tree: {package_dir}. "
            "Run bench/kinase_library/bootstrap_local_source.py first."
        )

    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_root = (args.runs_dir / f"compare_{run_id}").resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    public = _run_source(args, source="public", run_root=run_root)
    local = _run_source(args, source="local", run_root=run_root)
    comparison = _compare_outputs(public, local)
    result = {
        "run_root": str(run_root),
        "workload": {
            "donor": args.donor,
            "track": args.track,
            "state": args.state,
        },
        "public": public,
        "local": local,
        "comparison": comparison,
    }
    result_path = run_root / "comparison.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

    if not comparison["exact_match"]:
        raise SystemExit(f"public/local output mismatch; see {result_path}")


if __name__ == "__main__":
    main()
