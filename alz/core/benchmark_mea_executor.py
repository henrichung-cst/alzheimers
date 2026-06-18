"""Benchmark MEA executor overhead with optional cProfile output.

The default ``noop`` caller exercises projected-state MEA input preparation,
contrast construction, output stacking, aggregate writing, and mechanism
attribution without requiring ``kinase_library``. Use ``--caller real`` inside
the project environment to include the actual enrichment runtime.
"""

from __future__ import annotations

import argparse
import contextlib
import cProfile
import io
import json
import pstats
import statistics
import tempfile
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from alz.cohorts.tcells import state_mea


MeaCaller = Callable[..., tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]


def _noop_mea_caller(**kwargs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return contract-shaped deterministic MEA outputs without enrichment work."""
    contrast = next(iter(kwargs["results_by_contrast"]))
    track = kwargs["track"]
    residue = "Y" if track == "py" else "ST"
    return (
        pd.DataFrame(
            {
                "kinase": ["K1"],
                "ES": [1.0],
                "NES": [0.5],
                "p-value": [0.1],
                "FDR": [0.2],
                "Subs fraction": ["1/1"],
                "Leading substrates": ["benchmark"],
                "residue_type": [residue],
                "track": [track],
                "contrast": [contrast],
            }
        ),
        pd.DataFrame({"contrast": [contrast], "median_shift": [0.0]}),
        pd.DataFrame({"contrast": [contrast]}),
        pd.DataFrame(
            {
                "kinase": ["K1"],
                "contrast": [contrast],
                "motif": ["benchmark"],
                "residue_type": [residue],
                "track": [track],
                "kl_percentile": [1.0],
            }
        ),
    )


def _real_mea_caller() -> MeaCaller:
    from alz.bulk_mea import enrich as kinase_enrich

    return kinase_enrich._run_mea


def _run_once(
    donor: str,
    track: str,
    caller: MeaCaller,
    *,
    state: str | None,
    quiet: bool,
) -> float:
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        t0 = time.perf_counter()
        if quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                state_mea.run_projected_state_mea(
                    donor,
                    track,
                    out_dir,
                    states=[state] if state is not None else None,
                    mea_caller=caller,
                )
        else:
            state_mea.run_projected_state_mea(
                donor,
                track,
                out_dir,
                states=[state] if state is not None else None,
                mea_caller=caller,
            )
        return time.perf_counter() - t0


def _benchmark_case(
    donor: str,
    track: str,
    caller: MeaCaller,
    *,
    state: str | None,
    iterations: int,
    quiet: bool,
) -> dict:
    timings: list[float] = []
    try:
        for _ in range(iterations):
            timings.append(_run_once(donor, track, caller, state=state, quiet=quiet))
    except FileNotFoundError as exc:
        return {
            "donor": donor,
            "track": track,
            "status": "skipped_missing_input",
            "detail": str(exc),
        }

    return {
        "donor": donor,
        "track": track,
        "state": state,
        "status": "ok",
        "iterations": iterations,
        "median_seconds": statistics.median(timings),
        "best_seconds": min(timings),
        "worst_seconds": max(timings),
        "timings_seconds": timings,
    }


def _profile_case(
    donor: str,
    track: str,
    caller: MeaCaller,
    *,
    state: str | None,
    quiet: bool,
    limit: int,
) -> str:
    profile = cProfile.Profile()
    profile.enable()
    _run_once(donor, track, caller, state=state, quiet=quiet)
    profile.disable()

    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumtime").print_stats(limit)
    return stream.getvalue()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor", choices=["donor1", "donor2", "both"], default="both")
    parser.add_argument("--track", choices=["st", "py", "both"], default="both")
    parser.add_argument("--state", help="Optional projected T-cell state filter.")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--caller", choices=["noop", "real"], default="noop")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-limit", type=int, default=25)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")

    caller = _noop_mea_caller if args.caller == "noop" else _real_mea_caller()
    donors = ["donor1", "donor2"] if args.donor == "both" else [args.donor]
    tracks = ["st", "py"] if args.track == "both" else [args.track]
    quiet = not args.verbose

    results = [
        _benchmark_case(
            donor,
            track,
            caller,
            state=args.state,
            iterations=args.iterations,
            quiet=quiet,
        )
        for donor in donors
        for track in tracks
    ]
    print(json.dumps({"caller": args.caller, "results": results}, indent=2))

    if args.profile:
        donor = donors[0]
        track = tracks[0]
        print(f"\n# cProfile {donor}/{track} caller={args.caller}")
        print(_profile_case(
            donor,
            track,
            caller,
            state=args.state,
            quiet=quiet,
            limit=args.profile_limit,
        ))


if __name__ == "__main__":
    main()
