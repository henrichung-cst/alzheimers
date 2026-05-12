"""Filter Incytr factorial output to significant pathways.

Applies the per-row gate from He et al. 2025 Figure 2D, evaluated on
**raw** p-values rather than BH-adjusted q-values:

    raw pvalue < 0.05  AND  |PDS| > 0.76  AND  sigprob_max > 0.10

BH adjustment on the 135,206-row long-form family per contrast is too
stringent on this dataset (q-values collapse, surviving-row count
drops to single digits). The gate is applied **per contrast** so each
(sender, receiver, Path) row is tested independently in each of the 9
disease x timepoint contrasts (App / Tau / ApTt x 2mo / 4mo / 6mo).

A path is retained iff it passes the gate in >= 1 contrast. All 9
contrast rows for a qualifying path are emitted (long form), so
downstream consumers can read which contrasts the path lit up in.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import time
from pathlib import Path

import duckdb


DEFAULT_P_THRESHOLD = 0.05      # raw p-value
DEFAULT_PDS_THRESHOLD = 0.76    # |PDS| > 0.76 ~ one-fold-change difference (k=2)
DEFAULT_SIGPROB_THRESHOLD = 0.10  # max SigProb across the two regimes compared

CONTRASTS = [
    "App_2mo", "App_4mo", "App_6mo",
    "Tau_2mo", "Tau_4mo", "Tau_6mo",
    "ApTt_2mo", "ApTt_4mo", "ApTt_6mo",
]

DEFAULT_INPUT_DIR = Path("data/incytr_factorial_outputs")
DEFAULT_CACHE_GLOB = "receiver_cache/receiver=*/data.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
        help="directory containing receiver_cache/ (default: %(default)s)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="output parquet (default: significant_pathways_<YYYYMMDD>.parquet "
             "alongside the cache)",
    )
    parser.add_argument("--p-threshold", type=float, default=DEFAULT_P_THRESHOLD)
    parser.add_argument("--pds-threshold", type=float, default=DEFAULT_PDS_THRESHOLD)
    parser.add_argument("--sigprob-threshold", type=float,
                        default=DEFAULT_SIGPROB_THRESHOLD)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-limit", default="12GB")
    args = parser.parse_args()

    cache_glob = args.input_dir / DEFAULT_CACHE_GLOB
    stamp = _dt.date.today().strftime("%Y%m%d")
    output_path = args.output or args.input_dir / f"significant_pathways_{stamp}.parquet"

    print(f"input  : {cache_glob}")
    print(f"output : {output_path}")
    print(f"filter : pvalue<{args.p_threshold}  "
          f"|PDS|>{args.pds_threshold}  "
          f"sigprob_max>{args.sigprob_threshold}  (raw p)")
    print()

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={args.threads}; "
                f"PRAGMA memory_limit='{args.memory_limit}';")
    con.execute("SET temp_directory='/home/hchung/.cache/duckdb';")

    # Stage the long-form cache with a sigprob_max derived column.
    t0 = time.time()
    con.execute(f"""
        CREATE TEMP TABLE src AS
        SELECT
          sender, receiver, Path, Ligand, Receptor, EM, Target,
          "Ligand.label", "Receptor.label", "EM.label", "Target.label",
          contrast, pvalue, PDS, log2FC, SigProb_ref, SigProb_alt,
          GREATEST(COALESCE(SigProb_ref, 0.0), COALESCE(SigProb_alt, 0.0))
            AS sigprob_max
        FROM read_parquet('{cache_glob}', hive_partitioning = true)
    """)
    n_src = con.execute("SELECT COUNT(*) FROM src").fetchone()[0]
    contrasts_seen = sorted(
        r[0] for r in con.execute(
            "SELECT DISTINCT contrast FROM src"
        ).fetchall()
    )
    print(f"input rows: {n_src:,}  ({time.time() - t0:.1f}s)")
    print(f"contrasts in input: {contrasts_seen}")
    missing = set(CONTRASTS) - set(contrasts_seen)
    if missing:
        print(f"  WARNING missing contrasts: {sorted(missing)}")
    print()

    # Annotate each row with is_sig (raw p, |PDS|, sigprob_max).
    t0 = time.time()
    con.execute(f"""
        CREATE TEMP TABLE annotated AS
        SELECT
          s.*,
          (s.pvalue IS NOT NULL
           AND s.pvalue < {args.p_threshold}
           AND ABS(s.PDS) > {args.pds_threshold}
           AND s.sigprob_max > {args.sigprob_threshold}) AS is_significant
        FROM src s
    """)
    print(f"annotate: {time.time() - t0:.1f}s")

    # Path-level aggregate: signature + n_contrasts_sig.
    # We materialize a path-level lookup, then join back to keep long form.
    t0 = time.time()
    signature_expr = " || ".join(
        f"MAX(CASE WHEN contrast = '{c}' AND is_significant "
        f"THEN '1' ELSE '0' END)"
        for c in CONTRASTS
    )
    con.execute(f"""
        CREATE TEMP TABLE path_summary AS
        SELECT
          sender, receiver, Path,
          {signature_expr} AS signature,
          SUM(CAST(is_significant AS INTEGER)) AS n_contrasts_sig
        FROM annotated
        GROUP BY sender, receiver, Path
    """)
    print(f"path summary: {time.time() - t0:.1f}s")

    # Output: long form rows of paths with >=1 sig contrast, with signature
    # and n_contrasts_sig joined back.
    t0 = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"""
        COPY (
          SELECT
            a.*,
            ps.signature,
            ps.n_contrasts_sig
          FROM annotated a
          JOIN path_summary ps
            USING (sender, receiver, Path)
          WHERE ps.n_contrasts_sig >= 1
        ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"write: {time.time() - t0:.1f}s")
    print()

    # Summary.
    summary = con.execute("""
        SELECT
          COUNT(DISTINCT (sender, receiver, Path)) AS n_paths_input,
          COUNT(DISTINCT (sender, receiver, Path)) FILTER (WHERE n_contrasts_sig >= 1)
            AS n_paths_kept
        FROM path_summary
    """).fetchone()
    n_paths_input, n_paths_kept = summary
    n_kept_rows = con.execute(
        "SELECT COUNT(*) FROM annotated a JOIN path_summary ps "
        "USING (sender, receiver, Path) WHERE ps.n_contrasts_sig >= 1"
    ).fetchone()[0]
    print(f"  input paths      : {n_paths_input:>10,}")
    print(f"  surviving paths  : {n_paths_kept:>10,}  "
          f"({n_paths_kept / max(n_paths_input,1):.2%})")
    print(f"  surviving rows   : {n_kept_rows:>10,}  "
          f"(9 contrasts x surviving paths, minus missing pvalues)")
    print()

    per_contrast = con.execute("""
        SELECT contrast, COUNT(*) FILTER (WHERE is_significant) AS n_sig
        FROM annotated
        GROUP BY contrast
        ORDER BY contrast
    """).fetchdf()
    print("per-contrast significant rows:")
    for _, r in per_contrast.iterrows():
        print(f"  {r['contrast']:<10} {int(r['n_sig']):>10,}")
    print()

    breadth = con.execute("""
        SELECT n_contrasts_sig, COUNT(*) AS n_paths
        FROM path_summary
        WHERE n_contrasts_sig >= 1
        GROUP BY n_contrasts_sig
        ORDER BY n_contrasts_sig
    """).fetchdf()
    print("breadth (paths significant in N contrasts):")
    print(breadth.to_string(index=False))
    print()

    out_size_mb = output_path.stat().st_size / 1e6
    print(f"output: {output_path}  ({out_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
