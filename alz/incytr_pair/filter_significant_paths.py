"""Apply the Incytr paper's published significance gate to pair-mode output.

The driver emits ALL paths (overrides ``cutoff_SigProb=0``, ``cutoff_PDS=0`` in
``Cal_SigProb`` / ``Cal_PDS``) so the wide parquets are the unfiltered superset.
This script applies the conjunction the paper uses in Figure 2 / Materials &
Methods ("Identification of the Statistically Significant L-T Pathways"):

    (SigProb_<A> > 0.1  OR  SigProb_<B> > 0.1)
      AND
    (p_adj_<A>   < 0.05 OR  p_adj_<B>   < 0.05)
      AND
    |PDS| >= 0.2

where p_adj is the Benjamini-Hochberg-adjusted p-value computed per p_value
column over the **full unfiltered set** (the driver's permutation p, M=100
permutations, one column per condition). All three are pure row subsets; BH
adjustment is a rank-based monotone transform of the raw p, so applying it then
thresholding is mathematically equivalent to computing the BH cutoff τ and
keeping rows with raw p <= τ. We use the cutoff form below — cheaper than
materializing p_adj per row, identical result.

DuckDB-streamed (predicate pushdown, spill to ``~/.cache/duckdb``), atomic
(``.tmp`` + rename), idempotent (re-running an already-filtered file is a no-op
because the surviving rows still satisfy the same gate). Reports row counts
plus the BH cutoff τ used for each contrast (sanity gauge: τ should be small,
typically 0 or 0.01 at nboot=100).
"""

from __future__ import annotations

import argparse
import glob
import os

import duckdb

SIGPROB_CUTOFF = 0.1   # paper: SigProb > 0.1 in at least one condition
ABS_PDS_CUTOFF = 0.2   # paper: |PDS| >= 0.2 (minimum for "significant")
BH_Q          = 0.05   # paper: BH-adjusted p-value < 0.05 in at least one condition


def _detect_columns(con: duckdb.DuckDBPyConnection, path: str) -> tuple[str, str, str, str]:
    """Return (sigprob_a, sigprob_b, pval_a, pval_b) for a pair-mode parquet.

    The two conditions are name-matched: SigProb_<cond> pairs with p_value_<cond>.
    Sorted by condition name so the call site is cohort-agnostic.
    """
    names = [r[0] for r in con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{path}') LIMIT 0"
    ).fetchall()]
    sigprob = sorted(n for n in names if n.startswith("SigProb_"))
    pval    = sorted(n for n in names if n.startswith("p_value_"))
    if len(sigprob) != 2:
        raise SystemExit(f"{path}: expected 2 SigProb_* cols, found {sigprob}")
    if len(pval) != 2:
        raise SystemExit(f"{path}: expected 2 p_value_* cols, found {pval}")
    if "PDS" not in names:
        raise SystemExit(f"{path}: no PDS column")
    cond_a = sigprob[0].removeprefix("SigProb_")
    cond_b = sigprob[1].removeprefix("SigProb_")
    if pval != sorted([f"p_value_{cond_a}", f"p_value_{cond_b}"]):
        raise SystemExit(
            f"{path}: SigProb conditions {[cond_a, cond_b]} do not match "
            f"p_value conditions {pval}"
        )
    return sigprob[0], sigprob[1], f"p_value_{cond_a}", f"p_value_{cond_b}"


def _bh_cutoff(con: duckdb.DuckDBPyConnection, path: str, p_col: str) -> float:
    """BH cutoff τ at q=BH_Q: largest p_(i) such that p_(i) <= q*i/N.

    Reject rows with raw p <= τ. Returns -1.0 if no rows pass (no rejections).
    Rows with NULL p are excluded from the BH calculation (counted neither in
    N nor in ranks) — they cannot be rejected.
    """
    row = con.execute(f"""
        WITH src AS (
            SELECT CAST({p_col} AS DOUBLE) AS p
            FROM read_parquet('{path}')
            WHERE {p_col} IS NOT NULL
        ),
        ordered AS (
            SELECT p, ROW_NUMBER() OVER (ORDER BY p ASC) AS i,
                   COUNT(*) OVER () AS N
            FROM src
        )
        SELECT COALESCE(MAX(p), -1.0)
        FROM ordered
        WHERE p <= {BH_Q} * i / N
    """).fetchone()
    return float(row[0])


def filter_one(path: str) -> tuple[int, int, float, float]:
    """Filter one parquet in place. Returns (rows_before, rows_after, τ_a, τ_b)."""
    spill = os.environ.get(
        "DUCKDB_TEMP_DIR", os.path.join(os.path.expanduser("~"), ".cache", "duckdb")
    )
    os.makedirs(spill, exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='10GB'")
    con.execute(f"SET temp_directory='{spill}'")

    sp_a, sp_b, pv_a, pv_b = _detect_columns(con, path)

    n_before = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{path}')"
    ).fetchone()[0]

    tau_a = _bh_cutoff(con, path, pv_a)
    tau_b = _bh_cutoff(con, path, pv_b)

    where = (
        f'(CAST("{sp_a}" AS DOUBLE) > {SIGPROB_CUTOFF} '
        f'OR CAST("{sp_b}" AS DOUBLE) > {SIGPROB_CUTOFF}) '
        f'AND (CAST("{pv_a}" AS DOUBLE) <= {tau_a} '
        f'OR CAST("{pv_b}" AS DOUBLE) <= {tau_b}) '
        f"AND ABS(CAST(PDS AS DOUBLE)) >= {ABS_PDS_CUTOFF}"
    )
    tmp_out = path + ".tmp"
    con.execute(
        f"COPY (SELECT * FROM read_parquet('{path}') WHERE {where}) "
        f"TO '{tmp_out}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    n_after = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{tmp_out}')"
    ).fetchone()[0]
    con.close()

    os.replace(tmp_out, path)
    return n_before, n_after, tau_a, tau_b


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
        f"paper gate: SigProb > {SIGPROB_CUTOFF} (either) AND "
        f"p_adj < {BH_Q} (either, BH on full set) AND "
        f"|PDS| >= {ABS_PDS_CUTOFF}  ({len(targets)} file(s))",
        flush=True,
    )
    grand_before = grand_after = 0
    for path in targets:
        n_before, n_after, tau_a, tau_b = filter_one(path)
        grand_before += n_before
        grand_after += n_after
        pct = (100.0 * n_after / n_before) if n_before else 0.0
        print(
            f"  {os.path.basename(path)}: {n_before:,} -> {n_after:,} "
            f"({pct:.1f}% kept)  τ=({tau_a:.4g}, {tau_b:.4g})",
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
