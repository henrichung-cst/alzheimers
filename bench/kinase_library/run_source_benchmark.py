"""Run projected-state MEA with either public or local kinase-library source.

The command is intentionally bench-scoped. It modifies ``sys.path`` only inside
this process so a local source copy can shadow the installed package for parity
and profiling checks without changing production imports.
"""

from __future__ import annotations

import argparse
import contextlib
import cProfile
import hashlib
import io
import json
import pstats
import sys
import tempfile
import time
from pathlib import Path


BENCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCH_DIR.parents[1]
DEFAULT_LOCAL_SRC = BENCH_DIR / "local_src"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["public", "local"], required=True)
    parser.add_argument("--local-src", type=Path, default=DEFAULT_LOCAL_SRC)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--donor", choices=["donor1", "donor2"], default="donor1")
    parser.add_argument("--track", choices=["st", "py"], default="st")
    parser.add_argument("--state", default="CD8Naive")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-limit", type=int, default=40)
    parser.add_argument("--verbose", action="store_true")
    return parser


def _prepare_import_path(source: str, local_src: Path) -> None:
    if source == "local":
        local_src = local_src.resolve()
        package_dir = local_src / "kinase_library"
        if not package_dir.exists():
            raise SystemExit(
                f"Missing local source tree: {package_dir}. "
                "Run bench/kinase_library/bootstrap_local_source.py first."
            )
        sys.path.insert(0, str(local_src))
    sys.path.insert(0, str(PROJECT_ROOT))


def _fingerprint_outputs(out_dir: Path) -> list[dict[str, object]]:
    fingerprints = []
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in {".csv", ".json"}:
            continue
        rel = path.relative_to(out_dir).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        fingerprints.append({
            "path": rel,
            "sha256": digest,
            "bytes": path.stat().st_size,
        })
    return fingerprints


def _run_case(args: argparse.Namespace, out_dir: Path) -> tuple[float, str | None]:
    from alz.cohorts.tcells import state_mea

    profile_text = None
    start = time.perf_counter()

    def run() -> None:
        state_mea.run_projected_state_mea(
            args.donor,
            args.track,
            out_dir,
            states=[args.state],
        )

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        if args.verbose:
            run()
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                run()
        profiler.disable()
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
            "cumtime"
        ).print_stats(args.profile_limit)
        profile_text = stream.getvalue()
    elif args.verbose:
        run()
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            run()

    return time.perf_counter() - start, profile_text


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    _prepare_import_path(args.source, args.local_src)

    import kinase_library

    out_dir = args.out_dir
    temp_dir = None
    if out_dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        out_dir = Path(temp_dir.name)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    elapsed_seconds, profile_text = _run_case(args, out_dir)
    profile_path = None
    if profile_text is not None:
        profile_path = out_dir / "cprofile.txt"
        profile_path.write_text(profile_text)

    summary = {
        "source": args.source,
        "kinase_library_file": str(Path(kinase_library.__file__).resolve()),
        "kinase_library_version": getattr(kinase_library, "__version__", "unknown"),
        "donor": args.donor,
        "track": args.track,
        "state": args.state,
        "out_dir": str(out_dir),
        "elapsed_seconds": elapsed_seconds,
        "profile_path": str(profile_path) if profile_path is not None else None,
        "outputs": _fingerprint_outputs(out_dir),
    }

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))

    if temp_dir is not None:
        temp_dir.cleanup()


if __name__ == "__main__":
    main()
