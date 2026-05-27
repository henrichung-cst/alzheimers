"""Reshape Incytr pair-mode output (9 wide parquets) into the long-form
`receiver_cache/` layout the unified viewer already consumes.

Pair-mode emits one parquet per (genotype, age) comparison at
``outputs/reports/incytr_pair_mode/wide/ma_<age>_<geno>_ma_<age>_WTyp_incytr_output.parquet``
with two-condition columns (``SigProb_<c1>``, ``p_value_<c1>``,
``SiK_score_<c1>``, plus the WTyp twin) and no ``contrast`` column. The
factorial path the viewer was originally wired against is long-form with a
``contrast`` column and a single ``pvalue`` / ``SigProb`` / ``SiK_score``
per row. This script bridges the two: read each of the 9 inputs, derive
``contrast`` from the filename, collapse two-condition columns down to the
treatment side, NULL out columns pair-mode does not produce (per-node
``log2FC`` and ``.label`` columns), then write Hive-partitioned by
receiver under ``<out>/receiver_cache/receiver=<sanitized>/data.parquet``
plus a ``pair_metadata.parquet`` matching ``alz/integration/persist.R``'s
contract.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from viewer.paths import INCYTR_PAIR_MODE_OUTPUTS_DIR  # noqa: E402

DEFAULT_INPUT_DIR = os.path.normpath(
    os.path.join(HERE, "..", "..", "outputs", "reports", "incytr_pair_mode", "wide")
)

# Pair-mode filename → factorial contrast. Pair-mode uses AppP/Ttau/ApTt;
# the live viewer uses App/Tau/ApTt.
_GENO_MAP = {"AppP": "App", "Ttau": "Tau", "ApTt": "ApTt"}
_FILENAME_RE = re.compile(
    r"^ma_(?P<age>\dmo)_(?P<geno>AppP|Ttau|ApTt)_ma_\dmo_WTyp_incytr_output\.parquet$"
)

# Per-node × per-metric log2FC columns. As of 2026-05-15 the pair-mode driver
# emits the full 4×4 grid: 4 sc cells (via `Cal_scFC`) and 4 cells per omics
# layer pr/ps/py (with the `_aFC` mirrors stripped by the driver's `drop_pat`).
# We pass through when present and NULL-fill on older parquets so the viewer
# schema stays stable across reshape runs against pre-patch outputs.
# Per-node .label columns: pair-mode emits these (driver assigns DEG / prG
# inline in runIncytr); pass them through. Kept in sync with
# _INCYTR_FC_NODES / _INCYTR_FC_METRICS / _INCYTR_LABEL_SRC in alz/build_unified_viewer.py.
_FC_NODES = ("Ligand", "Receptor", "EM", "Target")
_FC_METRICS = ("sclog2FC", "pr_log2FC", "ps_log2FC", "py_log2FC")
_FC_COLS = [f"{n}_{m}" for n in _FC_NODES for m in _FC_METRICS]
_LABEL_COLS = [f"{n}.label" for n in _FC_NODES]

# Direction note: Incytr's `Cal_foldchange` (incytr/R/math.R) computes
# `log2(condition1 / condition2)`. The pair-mode driver
# (alz/incytr_pair/incytr_commandline.R) passes
# `condition1 = <disease>`, `condition2 = WTyp` — so the raw `*_sclog2FC`
# values are already disease/WT (positive = up in disease), matching the
# viewer tooltip and the proteomics layers. No sign flip needed.


def _sanitize_celltype(name: str) -> str:
    """Mirror ``sanitize_celltype`` in alz/integration/load.R."""
    return name.replace("/", "-").replace(" ", "_")


def _parse_filename(path: str) -> tuple[str, str]:
    """Return (suffix, contrast). Suffix is the ``ma_<age>_<geno>`` slug used
    as a column suffix inside the file; contrast is the viewer-facing label."""
    m = _FILENAME_RE.match(os.path.basename(path))
    if m is None:
        raise ValueError(f"unrecognized pair-mode filename: {path!r}")
    age, geno = m.group("age"), m.group("geno")
    return f"ma_{age}_{geno}", f"{_GENO_MAP[geno]}_{age}"


def _build_select(path: str, suffix: str, contrast: str) -> str:
    """SQL for one pair-mode file → unified long-form schema."""
    src_cols = set(
        duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{path}') LIMIT 0")
        .fetchnumpy()["column_name"]
    )
    # Per-(node, metric) log2FC: pass through when present, NULL-fill against
    # older parquets (pre Cal_scFC + drop_pat narrowing on 2026-05-15).
    fc_exprs = ",\n          ".join(
        f'CAST("{c}" AS DOUBLE) AS "{c}"' if c in src_cols
        else f'CAST(NULL AS DOUBLE) AS "{c}"'
        for c in _FC_COLS
    )
    # Pass `<Node>.label` through if present in the source file; otherwise
    # fall back to NULL so older parquets (pre-label driver patch) still load.
    label_exprs = ",\n          ".join(
        f'"{c}" AS "{c}"' if c in src_cols
        else f'CAST(NULL AS VARCHAR) AS "{c}"'
        for c in _LABEL_COLS
    )
    # nboot=0 runs skip the permutation test, so there is no p_value_<suffix>
    # column. NULL-fill it then (rank/filter on |PDS|, not pvalue) rather than
    # erroring on a missing column.
    pvalue_col = f"p_value_{suffix}"
    pvalue_expr = (f'CAST("{pvalue_col}" AS DOUBLE)' if pvalue_col in src_cols
                   else "CAST(NULL AS DOUBLE)")
    return f"""
        SELECT
          Sender AS sender,
          Receiver AS receiver,
          Path, Ligand, Receptor, EM, Target,
          '{contrast}' AS contrast,
          {pvalue_expr} AS pvalue,
          CAST(PDS AS DOUBLE) AS PDS,
          CAST(TPDS AS DOUBLE) AS TPDS,
          CAST(PPDS AS DOUBLE) AS PPDS,
          CAST(PhPDS_ps AS DOUBLE) AS PhPDS_ps,
          CAST(PhPDS_py AS DOUBLE) AS PhPDS_py,
          CAST("SiK_score_{suffix}" AS DOUBLE) AS SiK_score,
          {fc_exprs},
          {label_exprs}
        FROM read_parquet('{path}')
    """


def reshape(input_dir: str, out_dir: str, *, require_all_nine: bool = False) -> None:
    files = sorted(glob.glob(os.path.join(input_dir, "*_incytr_output.parquet")))
    if not files:
        raise SystemExit(f"no pair-mode parquets under {input_dir!r}")
    parsed = [(f, *_parse_filename(f)) for f in files]
    if require_all_nine and len(parsed) != 9:
        raise SystemExit(
            f"expected 9 pair-mode parquets, got {len(parsed)}: "
            f"{sorted(c for _, _, c in parsed)}"
        )
    print(f"reshape: {len(parsed)} pair-mode files → {out_dir}", flush=True)

    cache_dir = os.path.join(out_dir, "receiver_cache")
    os.makedirs(cache_dir, exist_ok=True)
    # Wipe any prior receiver= partitions so a partial converter run can't
    # leave stale rows behind. pair_metadata is overwritten below.
    for d in glob.glob(os.path.join(cache_dir, "receiver=*")):
        for f in glob.glob(os.path.join(d, "*")):
            os.remove(f)
        os.rmdir(d)

    con = duckdb.connect()
    con.execute("PRAGMA threads=4; PRAGMA memory_limit='8GB';")
    con.execute(f"SET temp_directory='{os.path.expanduser('~/.cache/duckdb')}';")
    # Hard spill cap so a regression can't fill the shared disk (the original
    # TEMP TABLE blew it to 110 GiB). With the streaming VIEW below the spill
    # stays small; this just fails fast instead of exhausting the drive.
    con.execute("SET max_temp_directory_size='40GiB';")
    # PARTITION_BY over the full (181M-row at nboot=0) union must NOT buffer the
    # whole table: a TEMP TABLE + insertion-order preservation spilled >110 GiB
    # and exhausted the disk. A VIEW streams the union per-partition, and
    # disabling insertion-order preservation lets DuckDB flush partitions
    # incrementally (the fix DuckDB's OOM message itself recommends). Row order
    # within a shard is irrelevant — the viewer re-sorts on read.
    con.execute("SET preserve_insertion_order=false;")

    union_sql = "\n        UNION ALL\n".join(
        _build_select(path, suffix, contrast) for path, suffix, contrast in parsed
    )
    # VIEW (not TEMP TABLE) carrying both raw and sanitized receiver so we can
    # (a) partition on the sanitized name to get the directory layout the viewer
    # expects and (b) emit pair_metadata. The union is re-scanned per consumer
    # (count, COPY, pair_metadata) — I/O-bound but memory-flat.
    con.execute(f"""
        CREATE VIEW staged AS
        SELECT *, replace(replace(receiver, '/', '-'), ' ', '_') AS receiver_part
        FROM ({union_sql})
    """)
    n_rows = con.execute("SELECT COUNT(*) FROM staged").fetchone()[0]
    print(f"  staged {n_rows:,} rows across {len(parsed)} contrasts", flush=True)

    # COPY ... PARTITION_BY drops the partition column from the parquet,
    # leaving `receiver` (raw) inside. The viewer reads with
    # hive_partitioning=true and uses the partition value (sanitized) as the
    # `receiver` column, which is what existing build_unified_viewer.py
    # expects (sanitized_to_display map in _write_incytr_pathways).
    con.execute(f"""
        COPY (SELECT * EXCLUDE (receiver), receiver_part AS receiver
              FROM staged)
        TO '{cache_dir}'
        (FORMAT PARQUET, PARTITION_BY (receiver), OVERWRITE_OR_IGNORE TRUE,
         COMPRESSION 'zstd')
    """)

    pm_path = os.path.join(out_dir, "pair_metadata.parquet")
    con.execute(f"""
        COPY (
          SELECT DISTINCT sender, receiver,
                 CAST(NULL AS INTEGER) AS n_post,
                 CAST(NULL AS INTEGER) AS n_pre,
                 'ok' AS status
          FROM staged
          ORDER BY receiver, sender
        ) TO '{pm_path}' (FORMAT PARQUET)
    """)
    n_pairs = con.execute(f"SELECT COUNT(*) FROM read_parquet('{pm_path}')").fetchone()[0]
    print(f"  wrote pair_metadata.parquet ({n_pairs} pairs)", flush=True)

    n_partitions = len(glob.glob(os.path.join(cache_dir, "receiver=*")))
    print(f"  wrote receiver_cache/ ({n_partitions} receiver partitions)", flush=True)
    con.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", default=DEFAULT_INPUT_DIR,
                    help=f"pair-mode parquet directory (default: {DEFAULT_INPUT_DIR})")
    ap.add_argument("--out-dir", default=INCYTR_PAIR_MODE_OUTPUTS_DIR,
                    help=f"output root (default: {INCYTR_PAIR_MODE_OUTPUTS_DIR})")
    ap.add_argument("--strict", action="store_true",
                    help="require all 9 (genotype × age) comparisons present")
    args = ap.parse_args()
    reshape(args.input_dir, args.out_dir, require_all_nine=args.strict)


if __name__ == "__main__":
    main()
