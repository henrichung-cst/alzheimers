"""R-EM-T backbone reduction for pair-mode Incytr paths (B2 sankey input).

Reads the 9 contrast-level wide parquets from
``outputs/reports/incytr_pair_mode/wide/`` (one file per contrast),
applies the canonical significance floor (same as ``filter_significant_paths.py``),
and aggregates each unique R-EM-T path across contrasts into a ranked backbone
table.

Grouping
--------
A path's "backbone" identity is the **Receptor-EM-Target** spine:

    R-EM-T   GROUP BY (Sender.group, Receiver.group, Receptor, EM, Target)

This is the only grouping any consumer needs — B2's sankey reads it lazily.
(The complete-path 6-tuple is NOT a backbone grouping: Target already fans a
Receptor-EM spine ~547×, and the per-kinase ``#Backbones`` participation count
— a full-path count — lives in the B4 bridge, not here.)  ``Ligand`` is not in
the R-EM-T key and is therefore absent from the output; ``Receptor``/``EM`` are
always present, ``Target`` is present (it is in the key) but may be NULL where a
path has no target.

**Output schema** — ``outputs/reports/incytr_pair_mode/backbone/backbone_rem_t.parquet``

    Sender.group       VARCHAR  — sending cluster (Levy-t5 spine)
    Receiver.group     VARCHAR  — receiving cluster
    Receptor           VARCHAR
    EM                 VARCHAR  — enzymatic mediator; NULL when absent from a path
    Target             VARCHAR  — downstream effector; NULL when absent from a path
    PDS                DOUBLE   — representative: signed PDS at the max-|PDS| occurrence
    n_timepoints_present  INTEGER — max, *over conditions*, of (distinct timepoints
                                    that condition appears in); within-genotype
                                    temporal stability (1–3).  A path at 2/4/6mo of
                                    App scores 3; a path scattered one-timepoint each
                                    across 3 genotypes scores 1.
    n_conditions_present  INTEGER — number of distinct genotype conditions (1–3)
    backbone_rank      BIGINT   — dense rank: n_timepoints_present DESC →
                                  n_conditions_present DESC → |PDS| DESC;
                                  dense and gap-free (ties share a rank).
    is_cholinergic_target  BOOLEAN — TRUE when Receiver.group == 'Cholinergic-Neurons';
                                      never filtered, only flagged (B2 display concern).
    conditions_present VARCHAR  — comma-joined sorted condition labels, e.g. "App,Tau"
    contrasts_present  VARCHAR  — comma-joined sorted contrast labels,
                                  e.g. "App_2mo,App_4mo,Tau_2mo"

Canonical floor (idempotent re-application)
-------------------------------------------
    (SigProb_<disease> > 0.1 OR SigProb_<WTyp> > 0.1) AND |PDS| >= 0.2

The wide parquets are unfiltered (driver emits all paths); the floor is
re-applied here to remain correct whether or not ``filter_significant_paths.py``
has run.  Raising the cutoffs is explicitly forbidden (CLAUDE.md).

Memory safety
-------------
All reduction runs through DuckDB (spill to ``DUCKDB_TEMP_DIR``, default
``~/.cache/duckdb``).  No whole-file pandas read of the wide parquets.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from pathlib import Path

import duckdb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIGPROB_CUTOFF: float = 0.1
ABS_PDS_CUTOFF: float = 0.2

_GENO_NORMALIZE: dict[str, str] = {
    "AppP": "App", "App": "App",
    "Ttau": "Tau", "Tau": "Tau", "TtauP": "Tau",
    "ApTt": "ApTt", "AppPTtau": "ApTt", "AppTtau": "ApTt",
}

# The R-EM-T backbone identity key.
_KEY_COLS: tuple[str, ...] = ("Sender.group", "Receiver.group", "Receptor", "EM", "Target")

_REPO_ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

_FNAME_RE = re.compile(
    r"ma_(\d+)mo_([A-Za-z]+)_ma_\1mo_WTyp_incytr_output\.parquet$"
)


def _parse_filename(fname: str) -> tuple[str, str, str] | None:
    """Return (contrast, condition, timepoint) or None if not parseable.

    ``ma_2mo_AppP_ma_2mo_WTyp_incytr_output.parquet`` → ``('App_2mo', 'App', '2mo')``
    """
    m = _FNAME_RE.match(os.path.basename(fname))
    if not m:
        return None
    age, geno_token = m.group(1), m.group(2)
    geno = _GENO_NORMALIZE.get(geno_token)
    if geno is None:
        return None
    contrast = f"{geno}_{age}mo"
    return contrast, geno, f"{age}mo"


def _detect_sigprob(con: duckdb.DuckDBPyConnection, path: str) -> tuple[str, str]:
    """Return the two ``SigProb_*`` column names for one wide parquet."""
    names = [r[0] for r in con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{path}') LIMIT 0"
    ).fetchall()]
    sigprob = sorted(n for n in names if n.startswith("SigProb_"))
    if len(sigprob) != 2:
        raise SystemExit(f"{path}: expected 2 SigProb_* cols, found {sigprob!r}")
    if "PDS" not in names:
        raise SystemExit(f"{path}: no PDS column")
    return sigprob[0], sigprob[1]


# ---------------------------------------------------------------------------
# SQL builder
# ---------------------------------------------------------------------------

def _q(col: str) -> str:
    """Double-quote a column name for DuckDB SQL."""
    return f'"{col}"'


def _build_backbone_query(
    files_info: list[tuple[str, str, str, str, str, str]],
) -> str:
    """Build the full DuckDB SQL: gate → R-EM-T reduce → rank."""
    key_sel = ", ".join(_q(c) for c in _KEY_COLS)
    # NULL == NULL join (EM/Target nullable — USING would silently drop them).
    join_cond = " AND\n        ".join(
        f"cs.{_q(c)} IS NOT DISTINCT FROM pp.{_q(c)}" for c in _KEY_COLS
    )

    # --- gated per-file subqueries ----------------------------------------
    file_parts: list[str] = []
    for abs_path, sp1, sp2, contrast, condition, timepoint in files_info:
        safe = abs_path.replace("'", "''")
        file_parts.append(f"""
    SELECT
        "Sender.group", "Receiver.group",
        Receptor, EM, Target, PDS,
        '{contrast}' AS contrast,
        '{condition}' AS condition,
        '{timepoint}' AS timepoint
    FROM read_parquet('{safe}')
    WHERE ({_q(sp1)} > {SIGPROB_CUTOFF} OR {_q(sp2)} > {SIGPROB_CUTOFF})
      AND ABS(PDS) >= {ABS_PDS_CUTOFF}""")
    gated_union = "\n    UNION ALL".join(file_parts)

    return f"""
WITH
gated AS ({gated_union}
),
pc AS (
    SELECT
        {key_sel},
        contrast, condition, timepoint,
        ARG_MAX(PDS, ABS(PDS)) AS rep_pds
    FROM gated
    GROUP BY {key_sel}, contrast, condition, timepoint
),
pcond AS (
    SELECT {key_sel}, condition, COUNT(DISTINCT timepoint) AS n_tp
    FROM pc
    GROUP BY {key_sel}, condition
),
cs AS (
    SELECT
        {key_sel},
        COUNT(*) AS n_conditions_present,
        MAX(n_tp) AS n_timepoints_present,
        STRING_AGG(condition, ',' ORDER BY condition) AS conditions_present
    FROM pcond
    GROUP BY {key_sel}
),
pp AS (
    SELECT
        {key_sel},
        ARG_MAX(rep_pds, ABS(rep_pds)) AS PDS,
        STRING_AGG(contrast, ',' ORDER BY contrast) AS contrasts_present
    FROM pc
    GROUP BY {key_sel}
),
bb AS (
    SELECT
        cs."Sender.group",
        cs."Receiver.group",
        cs.Receptor,
        cs.EM,
        cs.Target,
        pp.PDS,
        cs.n_timepoints_present::INTEGER AS n_timepoints_present,
        cs.n_conditions_present::INTEGER AS n_conditions_present,
        (cs."Receiver.group" = 'Cholinergic-Neurons') AS is_cholinergic_target,
        cs.conditions_present,
        pp.contrasts_present
    FROM cs
    JOIN pp ON (
        {join_cond}
    )
),
ranked AS (
    SELECT *,
        DENSE_RANK() OVER (
            ORDER BY n_timepoints_present DESC,
                     n_conditions_present DESC,
                     ABS(PDS) DESC
        ) AS backbone_rank
    FROM bb
)
SELECT
    "Sender.group", "Receiver.group",
    Receptor, EM, Target,
    PDS,
    n_timepoints_present,
    n_conditions_present,
    backbone_rank,
    is_cholinergic_target,
    conditions_present,
    contrasts_present
FROM ranked
ORDER BY backbone_rank,
         "Sender.group", "Receiver.group", Receptor, EM, Target"""


# ---------------------------------------------------------------------------
# Main reduction entry point
# ---------------------------------------------------------------------------

def reduce(
    wide_dir: str,
    out_path: str,
    memory_limit: str = "8GB",
    verbose: bool = True,
) -> dict:
    """Run R-EM-T backbone reduction and write ``backbone_rem_t.parquet``.

    Returns a summary dict with row count and timing.
    """
    # --- locate wide parquets --------------------------------------------
    pattern = os.path.join(wide_dir, "*_incytr_output.parquet")
    all_files = sorted(glob.glob(pattern))
    if not all_files:
        raise FileNotFoundError(f"No *_incytr_output.parquet in {wide_dir!r}")

    spill = os.environ.get(
        "DUCKDB_TEMP_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "duckdb"),
    )
    os.makedirs(spill, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute(f"SET temp_directory='{spill}'")

    # --- parse each file -------------------------------------------------
    files_info: list[tuple[str, str, str, str, str, str]] = []
    skipped: list[str] = []
    for fpath in all_files:
        parsed = _parse_filename(fpath)
        if parsed is None:
            skipped.append(fpath)
            continue
        contrast, condition, timepoint = parsed
        sp1, sp2 = _detect_sigprob(con, fpath)
        files_info.append((fpath, sp1, sp2, contrast, condition, timepoint))

    if not files_info:
        raise FileNotFoundError(
            f"No parseable wide parquets in {wide_dir!r}. Skipped: {skipped}"
        )
    if skipped and verbose:
        print(f"  (warn) skipped {len(skipped)} unparseable files: {skipped}", flush=True)

    if verbose:
        print(f"  grouping=R-EM-T  files={len(files_info)}", flush=True)
        for abs_path, sp1, sp2, contrast, condition, timepoint in files_info:
            print(f"    {os.path.basename(abs_path):55s}  "
                  f"contrast={contrast}  condition={condition}  timepoint={timepoint}",
                  flush=True)

    sql = _build_backbone_query(files_info)

    if verbose:
        print(f"  running R-EM-T reduction query (DuckDB {duckdb.__version__})...",
              flush=True)

    t0 = time.perf_counter()
    result = con.execute(sql).to_arrow_table()
    elapsed = time.perf_counter() - t0

    n_rows = result.num_rows
    if verbose:
        print(f"  reduction done: {n_rows:,} R-EM-T backbones in {elapsed:.1f}s", flush=True)

    # --- write output (atomic .tmp + rename) -----------------------------
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".tmp"
    import pyarrow.parquet as pq
    pq.write_table(result, tmp_path, compression="snappy")
    os.replace(tmp_path, out_path)

    if verbose:
        sz = os.path.getsize(out_path)
        print(f"  written: {out_path}  ({sz / 1024 / 1024:.1f} MB)", flush=True)

    return {
        "n_backbone_paths": n_rows,
        "n_contrasts_input": len(files_info),
        "elapsed_s": round(elapsed, 2),
        "out_path": out_path,
    }


# ---------------------------------------------------------------------------
# Verification helpers (light, in-memory on the already-small output)
# ---------------------------------------------------------------------------

def verify(out_path: str, wide_dir: str, verbose: bool = True) -> bool:
    """Run light sanity checks on the R-EM-T backbone table.

    Returns True if all checks pass, False otherwise.
    """
    import pyarrow.parquet as pq

    df = pq.read_table(out_path).to_pandas()
    ok = True

    def _fail(msg: str) -> None:
        nonlocal ok
        ok = False
        print(f"  FAIL: {msg}", flush=True)

    def _pass(msg: str) -> None:
        print(f"  OK:   {msg}", flush=True)

    if verbose:
        print(f"  rows: {len(df):,}", flush=True)

    if len(df) == 0:
        _fail("no rows")
        return False

    # 1. n_* ranges in [1, 3]
    for col in ("n_timepoints_present", "n_conditions_present"):
        mn, mx = int(df[col].min()), int(df[col].max())
        if mn < 1 or mx > 3:
            _fail(f"{col} out of range [1,3]: min={mn}, max={mx}")
        else:
            _pass(f"{col} in [1,3]: min={mn}, max={mx}")

    # 2. backbone_rank dense + gap-free
    ranks = sorted(df["backbone_rank"].unique())
    if ranks != list(range(1, len(ranks) + 1)):
        _fail(f"backbone_rank not dense: {ranks[:10]}...")
    else:
        _pass(f"backbone_rank dense: 1..{max(ranks)}")

    # 3. rows ≤ distinct R-EM-T key-tuples in gated wide/
    con = duckdb.connect()
    spill = os.environ.get(
        "DUCKDB_TEMP_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "duckdb"),
    )
    con.execute(f"SET temp_directory='{spill}'")
    pattern = os.path.join(wide_dir, "*_incytr_output.parquet")
    key_expr = ", ".join(_q(c) for c in _KEY_COLS)
    try:
        raw_n = con.execute(
            f"""SELECT COUNT(*) FROM (
                    SELECT DISTINCT {key_expr}
                    FROM read_parquet('{pattern}', union_by_name=true)
                )"""
        ).fetchone()[0]
        if len(df) <= raw_n:
            _pass(f"rows ({len(df):,}) ≤ raw distinct key-tuples ({raw_n:,})")
        else:
            _fail(f"rows ({len(df):,}) > raw distinct key-tuples ({raw_n:,})")
    except Exception as exc:
        print(f"  (warn) could not check raw wide count: {exc}", flush=True)

    # 4. is_cholinergic_target consistent + Cholinergic paths PRESENT (not dropped)
    chol = df[df["Receiver.group"] == "Cholinergic-Neurons"]
    if len(chol) > 0:
        if chol["is_cholinergic_target"].all():
            _pass(f"is_cholinergic_target=True for all {len(chol):,} Cholinergic-Neurons rows (present, not dropped)")
        else:
            _fail("is_cholinergic_target=False for some Cholinergic-Neurons rows")
    else:
        print("  (info) no Cholinergic-Neurons rows in backbone", flush=True)

    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_paths() -> tuple[str, str]:
    repo = str(_REPO_ROOT)
    wide_dir = os.path.join(repo, "outputs", "reports", "incytr_pair_mode", "wide")
    out_path = os.path.join(repo, "outputs", "reports", "incytr_pair_mode",
                            "backbone", "backbone_rem_t.parquet")
    return wide_dir, out_path


def main(argv: list[str] | None = None) -> None:
    wide_dir_def, out_def = _default_paths()

    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--wide-dir", default=wide_dir_def,
                   help="Directory containing *_incytr_output.parquet files "
                        f"(default: {wide_dir_def})")
    p.add_argument("--out", default=out_def,
                   help=f"Output parquet path (default: {out_def})")
    p.add_argument("--memory-limit", default="8GB",
                   help="DuckDB memory limit (default: 8GB)")
    p.add_argument("--verify", action="store_true",
                   help="Run sanity checks on the output after writing")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress progress output")
    args = p.parse_args(argv)

    verbose = not args.quiet
    if verbose:
        print("=== backbone_reduction.py (R-EM-T) ===", flush=True)

    summary = reduce(
        wide_dir=args.wide_dir,
        out_path=args.out,
        memory_limit=args.memory_limit,
        verbose=verbose,
    )

    if verbose:
        print(f"\n  Summary:", flush=True)
        for k, v in summary.items():
            print(f"    {k}: {v}", flush=True)

    if args.verify:
        print("\n  Running verification...", flush=True)
        ok = verify(args.out, args.wide_dir, verbose=verbose)
        if not ok:
            print("\n  VERIFICATION FAILED", flush=True)
            sys.exit(1)
        print("\n  Verification passed.", flush=True)

    if verbose:
        print("\n=== done ===", flush=True)


if __name__ == "__main__":
    main()
