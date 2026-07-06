"""
Build per-cohort long-form kinase -> cell_type attribution CSVs for KsG admission.

Outputs (columns exactly: kinase,cell_type):
  data/derived/ksg/song_attribution_long.csv
  data/derived/ksg/5xfad_cortex_attribution_long.csv
  data/derived/ksg/5xfad_hippocampus_attribution_long.csv
  data/derived/ksg/tcells_donor1_attribution_long.csv

Usage:
  python alz/incytr_pair/build_ksg_attribution.py [--cohort {song,5xfad,tcells}] [--tissue {cortex,hippocampus}]
  Default: build all cohorts / all tissues.
"""

import argparse
import os
import sys
import duckdb

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUT_DIR = os.path.join(REPO_ROOT, "data", "derived", "ksg")

SONG_SRC = os.path.join(
    REPO_ROOT, "outputs", "reports", "attribution_recovery", "kinase_hypothesis_table.csv"
)
FIVEXFAD_SRC = os.path.join(
    REPO_ROOT,
    "outputs",
    "reports",
    "kinase_attribution_5xfad",
    "fivexfad_snrna_attribution.csv",
)
TCELLS_DONOR1_SRC = os.path.join(
    REPO_ROOT,
    "outputs",
    "reports",
    "kinase_attribution_tcells",
    "donor1",
    "unified_attribution_tcells.csv",
)


def check_source(path: str, label: str) -> bool:
    if not os.path.exists(path):
        print(f"[skip] {label}: source not found at {path}", file=sys.stderr)
        return False
    return True


def build_song(con: duckdb.DuckDBPyConnection) -> int:
    """Melt top_celltype_1/2/3 into long (kinase, cell_type). Drop NA/empty. Dedup."""
    if not check_source(SONG_SRC, "song"):
        return 0

    # Discover top_celltype_N columns dynamically
    cols = [
        row[0]
        for row in con.execute(
            f"DESCRIBE SELECT * FROM read_csv_auto('{SONG_SRC}')"
        ).fetchall()
    ]
    celltype_cols = sorted(c for c in cols if c.startswith("top_celltype_") and not any(
        c.endswith(suf) for suf in ("_wmb_tier", "_sea_ad_lfc", "_evidence", "_song_lfc")
    ))

    if not celltype_cols:
        print("[error] song: no top_celltype_N columns found", file=sys.stderr)
        return 0

    # Build UNION of one SELECT per top_celltype_N column
    unions = " UNION ALL ".join(
        f"SELECT kinase, {col} AS cell_type FROM read_csv_auto('{SONG_SRC}')"
        for col in celltype_cols
    )
    sql = f"""
        SELECT DISTINCT kinase, cell_type
        FROM ({unions})
        WHERE kinase IS NOT NULL
          AND kinase != ''
          AND cell_type IS NOT NULL
          AND cell_type != ''
        ORDER BY kinase, cell_type
    """
    out_path = os.path.join(OUT_DIR, "song_attribution_long.csv")
    con.execute(f"COPY ({sql}) TO '{out_path}' (HEADER TRUE, DELIMITER ',')")
    n = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{out_path}')").fetchone()[0]
    print(f"[ok] song -> {out_path}  ({n} rows)")
    return n


def build_5xfad(con: duckdb.DuckDBPyConnection, tissues: list[str]) -> dict[str, int]:
    """Per-tissue distinct (kinase, cell_type) from fivexfad_snrna_attribution.csv."""
    if not check_source(FIVEXFAD_SRC, "5xfad"):
        return {}

    counts = {}
    for tissue in tissues:
        sql = f"""
            SELECT DISTINCT kinase, cell_type
            FROM read_csv_auto('{FIVEXFAD_SRC}')
            WHERE tissue = '{tissue}'
              AND kinase IS NOT NULL AND kinase != ''
              AND cell_type IS NOT NULL AND cell_type != ''
            ORDER BY kinase, cell_type
        """
        out_path = os.path.join(OUT_DIR, f"5xfad_{tissue}_attribution_long.csv")
        con.execute(f"COPY ({sql}) TO '{out_path}' (HEADER TRUE, DELIMITER ',')")
        n = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{out_path}')").fetchone()[0]
        print(f"[ok] 5xfad/{tissue} -> {out_path}  ({n} rows)")
        counts[tissue] = n
    return counts


def build_tcells_donor1(con: duckdb.DuckDBPyConnection) -> int:
    """Distinct (kinase, cell_type) from donor1 unified_attribution_tcells.csv."""
    if not check_source(TCELLS_DONOR1_SRC, "tcells/donor1"):
        return 0

    sql = f"""
        SELECT DISTINCT kinase, cell_type
        FROM read_csv_auto('{TCELLS_DONOR1_SRC}')
        WHERE kinase IS NOT NULL AND kinase != ''
          AND cell_type IS NOT NULL AND cell_type != ''
        ORDER BY kinase, cell_type
    """
    out_path = os.path.join(OUT_DIR, "tcells_donor1_attribution_long.csv")
    con.execute(f"COPY ({sql}) TO '{out_path}' (HEADER TRUE, DELIMITER ',')")
    n = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{out_path}')").fetchone()[0]
    print(f"[ok] tcells/donor1 -> {out_path}  ({n} rows)")
    return n


def main():
    parser = argparse.ArgumentParser(description="Build KsG attribution CSVs")
    parser.add_argument(
        "--cohort",
        choices=["song", "5xfad", "tcells"],
        default=None,
        help="Build only this cohort (default: all)",
    )
    parser.add_argument(
        "--tissue",
        choices=["cortex", "hippocampus"],
        default=None,
        help="5xFAD tissue filter (default: both)",
    )
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    con = duckdb.connect()

    cohorts = [args.cohort] if args.cohort else ["song", "5xfad", "tcells"]
    tissues = [args.tissue] if args.tissue else ["cortex", "hippocampus"]

    totals: dict[str, int] = {}

    if "song" in cohorts:
        totals["song"] = build_song(con)

    if "5xfad" in cohorts:
        tissue_counts = build_5xfad(con, tissues)
        for t, n in tissue_counts.items():
            totals[f"5xfad/{t}"] = n

    if "tcells" in cohorts:
        totals["tcells/donor1"] = build_tcells_donor1(con)

    print("\nRow counts:")
    for label, n in totals.items():
        print(f"  {label}: {n}")


if __name__ == "__main__":
    main()
