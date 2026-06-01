"""Apply the canonical significance floor to pair-mode output.

The driver emits ALL paths (overrides ``cutoff_SigProb=0``, ``cutoff_PDS=0`` in
``Cal_SigProb`` / ``Cal_PDS``) so the wide parquets are the unfiltered superset.
The canonical production gate is the floor that reproduces sce4's pre-cap
``Allpathway`` universe:

    (SigProb_<A> > 0.1  OR  SigProb_<B> > 0.1)      # at least one condition
      AND
    |PDS| >= 0.2                                     # minimum "significant" PDS

The per-pair Top300 up/down cap is intentionally NOT the default. Our sce4
forensics showed that Top300 membership is rank-sensitive to PDS drift even
when the pre-cap row universe is nearly identical. Use ``--top300`` only for
explicit sce4 table-compatibility diagnostics, not canonical Song or T-cell
viewer outputs.

For AD transgene-excluded sensitivity analyses, pass ``--exclude-transgenes``.
That removes paths touching ``App``, ``Psen1``, or ``Mapt`` in any pathway node
before any optional per-pair cap. This is an analysis rule, not a hidden
default.

The two floors match sce4's ``Allpathway_table`` floor definition (it has zero
rows with ``|PDS| < 0.2`` and zero rows with both SigProb <= 0.1). Top300 table
reconstruction remains available via ``--top300``.

NO p-value / FDR arm: sce4 never ran the permutation test — none of its
artifacts (420 per-pair CSVs, ``Allpathway``, ``Top300``, the two pairwise
``.rds`` tables) carry a ``p_value`` / ``p_adj`` / ``fdr`` / ``q-value``
column. A p_adj gate is therefore foreign to the reference and drops paths sce4
kept (notably cell-sparse pairs whose permutation p is NA, e.g.
Microglia -> Cholinergic-Neurons at 2mo). The nboot=100 permutation ``p_value_*``
columns stay in ``wide/`` as informational columns; they do not gate.

DuckDB-streamed (spill to ``~/.cache/duckdb``), atomic (``.tmp`` + rename),
idempotent for a given mode when re-running from an uncapped or already
floor-gated file. A previously Top300-capped or p-adj-filtered file cannot be
expanded; rerun the R driver to regenerate those rows.
"""

from __future__ import annotations

import argparse
import glob
import os

import duckdb

SIGPROB_CUTOFF = 0.1   # sce4: SigProb > 0.1 in at least one condition
ABS_PDS_CUTOFF = 0.2   # sce4: |PDS| >= 0.2 (minimum for "significant")
TOP_N          = 300   # sce4: per pair, top-N by PDS up UNION top-N by PDS down
TRANSGENES     = ("App", "Psen1", "Mapt")


def _detect_sigprob(con: duckdb.DuckDBPyConnection, path: str) -> tuple[str, str]:
    """Return the two ``SigProb_<cond>`` column names for a pair-mode parquet."""
    names = [r[0] for r in con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{path}') LIMIT 0"
    ).fetchall()]
    sigprob = sorted(n for n in names if n.startswith("SigProb_"))
    if len(sigprob) != 2:
        raise SystemExit(f"{path}: expected 2 SigProb_* cols, found {sigprob}")
    if "PDS" not in names:
        raise SystemExit(f"{path}: no PDS column")
    for c in ("Sender.group", "Receiver.group"):
        if c not in names:
            raise SystemExit(f"{path}: no {c!r} column (needed for the per-pair cap)")
    return sigprob[0], sigprob[1]


def filter_one(path: str, exclude_transgenes: bool = False, top300: bool = False) -> tuple[int, int]:
    """Filter one parquet in place. Returns (rows_before, rows_after)."""
    spill = os.environ.get(
        "DUCKDB_TEMP_DIR", os.path.join(os.path.expanduser("~"), ".cache", "duckdb")
    )
    os.makedirs(spill, exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='10GB'")
    con.execute(f"SET temp_directory='{spill}'")

    sp_a, sp_b = _detect_sigprob(con, path)

    n_before = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{path}')"
    ).fetchone()[0]

    floors = (
        f'(CAST("{sp_a}" AS DOUBLE) > {SIGPROB_CUTOFF} '
        f'OR CAST("{sp_b}" AS DOUBLE) > {SIGPROB_CUTOFF}) '
        f"AND ABS(CAST(PDS AS DOUBLE)) >= {ABS_PDS_CUTOFF}"
    )
    if exclude_transgenes:
        tg_csv = ", ".join(f"'{g}'" for g in TRANSGENES)
        floors = (
            f"{floors} AND NOT ("
            f"COALESCE(Ligand, '') IN ({tg_csv}) OR "
            f"COALESCE(Receptor, '') IN ({tg_csv}) OR "
            f"COALESCE(EM, '') IN ({tg_csv}) OR "
            f"COALESCE(Target, '') IN ({tg_csv})"
            f")"
        )
    tmp_out = path + ".tmp"
    if top300:
        query = f"""
            WITH gated AS (
                SELECT * FROM read_parquet('{path}') WHERE {floors}
            ),
            ranked AS (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY "Sender.group", "Receiver.group"
                        ORDER BY CAST(PDS AS DOUBLE) DESC) AS rn_up,
                    ROW_NUMBER() OVER (
                        PARTITION BY "Sender.group", "Receiver.group"
                        ORDER BY CAST(PDS AS DOUBLE) ASC) AS rn_dn
                FROM gated
            )
            SELECT * EXCLUDE (rn_up, rn_dn) FROM ranked
            WHERE rn_up <= {TOP_N} OR rn_dn <= {TOP_N}
        """
    else:
        query = f"SELECT * FROM read_parquet('{path}') WHERE {floors}"

    con.execute(f"""
        COPY ({query})
        TO '{tmp_out}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
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
    ap.add_argument(
        "--exclude-transgenes",
        action="store_true",
        help=(
            "Drop paths containing App, Psen1, or Mapt before any optional cap. "
            "Use only for the AD transgene-excluded sensitivity analysis."
        ),
    )
    ap.add_argument(
        "--top300",
        action="store_true",
        help=(
            "After the SigProb/PDS floors, keep the per-pair top-300 PDS up "
            "union top-300 PDS down. Use only for explicit sce4 Top300 "
            "compatibility diagnostics."
        ),
    )
    args = ap.parse_args()

    if args.file:
        targets = [args.file]
    else:
        targets = sorted(glob.glob(os.path.join(args.dir, "*_incytr_output.parquet")))
        if not targets:
            raise SystemExit(f"no *_incytr_output.parquet under {args.dir!r}")

    print(
        f"canonical gate: SigProb > {SIGPROB_CUTOFF} (either) AND "
        f"|PDS| >= {ABS_PDS_CUTOFF}"
        f"{', THEN per-pair top-' + str(TOP_N) + ' PDS up ∪ down' if args.top300 else ', uncapped'} "
        f"{'(excluding App/Psen1/Mapt paths) ' if args.exclude_transgenes else ''}"
        f"({len(targets)} file(s))",
        flush=True,
    )
    grand_before = grand_after = 0
    for path in targets:
        n_before, n_after = filter_one(
            path,
            exclude_transgenes=args.exclude_transgenes,
            top300=args.top300,
        )
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
