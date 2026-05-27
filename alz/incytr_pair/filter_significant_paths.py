"""Apply the collaborator significance filter to pair-mode Incytr output.

The driver emits ALL paths (parity override #5: ``cutoff_SigProb=0``,
``cutoff_PDS=0`` inside ``Cal_SigProb`` / ``Cal_PDS``) so the wide parquets are
the unfiltered superset (~54.7M rows/contrast at nboot=0). The analysis filter
the upstream group actually uses is the *downstream* half of that design:

    (SigProb_<disease> > 0.1 OR SigProb_<WTyp> > 0.1) AND abs(PDS) >= 0.2

mapping the email's ``cutoff_SigProb=0.1`` (``Cal_SigProb``: ``SigProb > c``,
strict, OR across conditions) and ``cutoff_PDS=0.2`` (``Cal_PDS``:
``abs(PDS) >= c``). Both cutoffs are pure row subsets — they drop rows, never
recompute SigProb/PDS for survivors — so filtering the existing parquets is
mathematically identical to having re-run the driver with these cutoffs. No
re-run needed. The filter is parity-preserving: sce4's Top300 reference itself
satisfies both cutoffs (Ndnf ref min |PDS| = 0.2122, sitting on the 0.2 gate).

Operates on a single parquet or every ``*_incytr_output.parquet`` under
``--dir``. DuckDB-streamed (predicate pushdown, capped memory, spill to
``~/.cache/duckdb``), atomic (``.tmp`` + rename), idempotent (re-running a
filtered file is a no-op). Reports before/after row counts.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import duckdb

SIGPROB_CUTOFF = 0.1   # Cal_SigProb cutoff_SigProb: SigProb > cutoff (strict)
ABS_PDS_CUTOFF = 0.2   # Cal_PDS cutoff_PDS: abs(PDS) >= cutoff


def _detect_columns(path: str) -> tuple[str, str]:
    """Return (disease_sigprob_col, wtyp_sigprob_col) for a pair-mode parquet."""
    names = list(
        duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{path}') LIMIT 0")
        .fetchnumpy()["column_name"]
    )
    sigprob = [n for n in names if n.startswith("SigProb_")]
    if len(sigprob) != 2:
        raise SystemExit(
            f"{path}: expected exactly 2 SigProb_* columns, found {sigprob}"
        )
    if "PDS" not in names:
        raise SystemExit(f"{path}: no PDS column (found {names[:10]}...)")
    wtyp = [n for n in sigprob if n.endswith("_WTyp")]
    disease = [n for n in sigprob if not n.endswith("_WTyp")]
    if len(wtyp) != 1 or len(disease) != 1:
        raise SystemExit(
            f"{path}: cannot split SigProb cols into disease/WTyp: {sigprob}"
        )
    return disease[0], wtyp[0]


def filter_one(path: str) -> tuple[int, int]:
    """Filter one parquet in place. Returns (rows_before, rows_after)."""
    disease_col, wtyp_col = _detect_columns(path)
    tmp_out = path + ".tmp"
    spill = os.environ.get(
        "DUCKDB_TEMP_DIR", os.path.join(os.path.expanduser("~"), ".cache", "duckdb")
    )
    os.makedirs(spill, exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='10GB'")
    con.execute(f"SET temp_directory='{spill}'")

    n_before = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{path}')"
    ).fetchone()[0]

    where = (
        f'(CAST("{disease_col}" AS DOUBLE) > {SIGPROB_CUTOFF} '
        f'OR CAST("{wtyp_col}" AS DOUBLE) > {SIGPROB_CUTOFF}) '
        f"AND ABS(CAST(PDS AS DOUBLE)) >= {ABS_PDS_CUTOFF}"
    )
    con.execute(
        f"COPY (SELECT * FROM read_parquet('{path}') WHERE {where}) "
        f"TO '{tmp_out}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    n_after = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{tmp_out}')"
    ).fetchone()[0]
    con.close()

    os.replace(tmp_out, path)
    return n_before, n_after


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", help="single pair-mode parquet to filter in place")
    g.add_argument("--dir", help="directory of *_incytr_output.parquet to filter")
    args = ap.parse_args()

    if args.file:
        targets = [args.file]
    else:
        targets = sorted(glob.glob(os.path.join(args.dir, "*_incytr_output.parquet")))
        if not targets:
            raise SystemExit(f"no *_incytr_output.parquet under {args.dir!r}")

    print(
        f"filter: (SigProb_disease > {SIGPROB_CUTOFF} OR SigProb_WTyp > "
        f"{SIGPROB_CUTOFF}) AND |PDS| >= {ABS_PDS_CUTOFF}  ({len(targets)} file(s))",
        flush=True,
    )
    grand_before = grand_after = 0
    for path in targets:
        n_before, n_after = filter_one(path)
        grand_before += n_before
        grand_after += n_after
        pct = (100.0 * n_after / n_before) if n_before else 0.0
        print(
            f"  {os.path.basename(path)}: {n_before:,} -> {n_after:,} "
            f"({pct:.1f}% kept)",
            flush=True,
        )
    if len(targets) > 1:
        pct = (100.0 * grand_after / grand_before) if grand_before else 0.0
        print(
            f"total: {grand_before:,} -> {grand_after:,} ({pct:.1f}% kept)",
            flush=True,
        )


if __name__ == "__main__":
    main()
